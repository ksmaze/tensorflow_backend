## 2026-03-09 - [Triton Output Resolution Hot Path]
**Learning:** `std::find` was checking if an output `const char*` was requested by searching an array using string comparisons inside the per-request output execution loop. Also, dynamically allocating a `new std::string()` for every request and string output tensor was happening on the hot path.
**Action:** Replaced the string comparison loop with an index array (`std::vector<uint32_t> request_required_outputs`) that performs integer comparisons. Pre-reserved `std::vector<std::string>` to securely place strings on a stable buffer array without needing `std::unique_ptr` dynamic allocations per-request, preventing CUDA async pointer invalidation and eliminating multiple `O(N * M)` heap allocations per execute call.

## 2026-03-09 - [Hidden Vector Allocations in ProcessRequests]
**Learning:** Found a performance bottleneck in the `ProcessRequests` hot-path. Unnecessary internal memory allocations were made due to repeated implicit temporary `std::string` copies when calling `std::find(required_outputs.begin(), required_outputs.end(), const char*)`. The `batchn_shape` vector was also needlessly allocated and reallocated inside a loop iteration. The 2D vector `request_required_outputs` resulted in O(N) allocations for small dynamic collections.
**Action:** Next time, avoid implicitly constructing short-lived temporaries like `std::string` in hot loops. Preallocate linear containers outside loops via `assign()` or `clear()` to reuse capacity. Flatten vector-of-vectors to reduce dynamic heap allocations in favor of contiguous memory caches.

## 2026-03-10 - [Avoid VLA and String Allocations in Output Request Iteration]
**Learning:** Non-standard variable-length arrays (VLAs) (`const char* output_names_cstr[required_outputs.size()]`) were being used, and multiple `std::string` objects were instantiated redundantly inside the outputs loop just to hold the names to find in `model_.output_name_map_`.
**Action:** Always replace VLAs with `std::vector` to be standard compliant and avoid stack overflow issues. Furthermore, you can search `IONameMap` directly with `const char*` or cache `const char*` pointers inside the pre-allocated `std::vector<const char*> output_names_cstr` to eliminate dynamic allocations inside the loop.

## 2026-03-10 - Removed String Allocation Overhead in Hot Path
**Learning:** `std::unordered_map` with `std::string` keys requires constructing a `std::string` for queries using `const char*` prior to C++20 transparent comparisons. In Triton, tensors names are queried as `const char*` during `ProcessRequests`.
**Action:** Replaced `std::unordered_map` with a custom `std::vector<std::pair<std::string, std::string>>` and linear search on `const char*` directly. Since the number of inputs/outputs per model is extremely small (<10), linear search is significantly more cache friendly and removes heap allocations entirely.

## 2026-03-10 - Fast Lookups for Large Tensor Counts without Allocation
**Learning:** Some models can have up to ~800 inputs, but overall small 1-2 outputs. A linear search on `std::vector` isn't scalable enough, but using `std::unordered_map` with `std::string` allocates heap memory when queried with `const char*` due to lack of transparent comparators.
**Action:** By maintaining a sorted `std::vector` and using `std::lower_bound` combined with a custom lambda comparing `a.first.c_str()` to the searched `const char*` using `std::strcmp(..., ...) < 0`, we achieve both `O(log N)` lookup times and zero `std::string` allocations!

## 2026-03-11 - Verbose logging
**Learning:** `LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE)` was called witht level check, and can involve many `std::string` constructions.
**Action:** Use `TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)` to check if verbose logging is enabled before constructing expensive `std::string` objects.

## 2026-03-12 - High-Concurrency Heap Contention Optimizations
**Learning:** At concurrency=8 with 800+ string inputs, ~7200+ heap allocations per execute (×8 threads = ~57,600 malloc/free calls) create severe allocator lock contention. Prior optimizations improved single-thread overhead but didn't address multi-thread malloc contention. The dominant sources were: TensorImpl name_ string copies (800+), TRITONTF_Shape double-allocations (1600), TRITONTF_TensorList node allocations (800+), per-element flat<tstring>() recomputation, and vector<string> construction in TRITONTF_ModelRun.
**Actions:**
1. **Skip tensor name_ in callable path** — In `ProcessRequests`, pass `nullptr` to `TRITONTF_TensorNew` when `input_device_id_ != MODEL_DEVICE` (callable path uses positional inputs, names are unused). TensorImpl now handles nullptr name via SSO empty string. Eliminates 800+ string heap allocs/frees per execute.
2. **Batch TRITONTF_TensorSetStrings API** — Added `TRITONTF_TensorSetStrings()` that gets `flat<tstring>()` once and sets all strings in a batch. Called from `SetStringInputTensor` instead of per-element `TRITONTF_TensorSetString`. Eliminates redundant flat-tensor recomputation per string element.
3. **Combined TRITONTF_Shape allocation** — `TRITONTF_ShapeNew` now allocates the shape struct and dims array in a single `operator new` call instead of two separate allocations. `TRITONTF_ShapeDelete` uses a single `operator delete`. Eliminates 800+ extra malloc calls per execute.
4. **Thread-local TensorList node free list** — `TRITONTF_TensorListNew`/`Delete` now use a thread-local free list to recycle nodes. After warmup, zero malloc/free calls for TensorList nodes. Eliminates ~800 malloc + 800 free calls per execute per thread.
5. **Eliminated vector<string> in TRITONTF_ModelRun** — `ModelImpl::Run` now accepts `const char**` + counts directly. Input count is passed from `TRITONTF_ModelRun` to avoid re-traversing the 800+ node linked list. `vector<string>` for output names is only built in the non-callable path where `session_->Run` requires it.

## 2026-03-13 - Callable on CPU: No Measurable Benefit — Reverted
**Discovery:** Enabling `RunCallable` on CPU and all associated optimizations (feed_index stamping, feed_index_map_ keyed by config name, zero-string Run(), nullptr RunMetadata) produced **zero measurable latency improvement** over baseline `session_->Run` at concurrency=8, batch=25, 800+ string inputs.
**Root cause:** TF's `session_->Run` already caches the execution plan after the first call via `GetOrCreateExecutors` in `direct_session.cc`. After warmup, `Run()` is just a hash-map lookup for the cached executor + positional tensor placement via `input_name_to_index`. Tensor names are short enough (~20 chars) for SSO — no heap allocation. The wrapper layer accounts for <1% of total latency; TF graph execution and string DATA copying (800 × 25 = 20,000 strings per inference) dominate.
**Evidence from TF source:**
- `direct_session.cc`: `executors_.find(key)` — O(1) cached executor lookup after first call
- `direct_session.cc`: `feed_args[input_name_to_index[it.first]]` — TF already uses positional placement internally
- `session.h`: `run_metadata` may be nullptr (documented, confirmed in implementation)
**Action:** Reverted commits `7873ae4`..`ea0ede9` (callable-on-CPU, feed_index, ordering fixes). Kept this commit (`f14a917`) which has general-purpose optimizations that still reduce overhead for the `session_->Run` path: batch string API, combined shape allocation, thread-local TensorList free list, eliminated vector<string> in ModelRun.

## 2026-03-19 - O(N^2) Bottleneck in Request Input Retrieval
**Learning:** Iterating over 800+ model inputs using `TRITONBACKEND_RequestInput` by name for every request inside `ProcessRequests` results in O(N^2) string comparisons, because `TRITONBACKEND_RequestInput` internally performs a linear search over the request's provided inputs.
**Action:** Introduce a fast-path lookup using `TRITONBACKEND_RequestInputByIndex` to guess the input by index (O(1)) and verify the name using `TRITONBACKEND_InputProperties` (O(1)). Fall back to the linear `TRITONBACKEND_RequestInput` if the name doesn't match or the index lookup fails. Always remember to `TRITONSERVER_ErrorDelete` the error objects returned by failed fast-path API calls to avoid memory leaks.

## 2026-03-20 - [Redundant Triton API Calls in Hot Path]
**Learning:** Discovered that `ProcessRequests` was making redundant C API calls across the Triton boundary. `GetInput` was calling `TRITONBACKEND_InputProperties` just to verify the tensor name, and then the caller was invoking it *again* to get the shape and dimensions. Similarly, `GetContiguousInputContent` was iterating over buffers using `TRITONBACKEND_InputBufferForHostPolicy` and then calling it a second time to fetch the exact same pointer when `chunk_count == 1`.
**Action:** Always fuse API calls where possible. Modify helper functions like `GetInput` to return all necessary properties (shape, dims_count, etc.) during their internal verification step. Cache outputs of iteration loops (like the first valid buffer pointer) to avoid re-querying the API for the exact same data immediately afterward.

## 2026-03-21 - [Caching buffer iterations to avoid redundant C API calls]
**Learning:** Found a performance bottleneck where `GetContiguousInputContent` iterated over `buffer_count` using `TRITONBACKEND_InputBufferForHostPolicy` and then redundantly iterated over it again when `chunk_count > 1`, resulting in extra C API cross-boundary overheads in the execution hot-path. Furthermore, the second loop incorrectly used `chunk_count` as the index instead of the original buffer index, introducing a latent bug if there were null pointers in the middle of the buffers.
**Action:** Always fuse redundant iterations into a single pass when possible. Here, caching the results of the first API call inside a small stack-allocated `inline_buffers` array allows subsequent passes to avoid API overhead, prevent heap allocations (since typical buffer counts are small), and maintain correct logical mappings.

## 2026-03-23 - Revert for regression
**Learning:** Reverted commits between 2026-03-19 and 2026-03-21. They introduced a prediction performance regression, while did have latency improvements.
## 2026-05-24 - [Fix Latent Bug and Redundant Loop in GetContiguousInputContent]
**Learning:** Found a performance bottleneck and a latent bug in `GetContiguousInputContent`. It redundantly invoked the `TRITONBACKEND_InputBufferForHostPolicy` C API twice to evaluate contiguous chunks. Furthermore, the copy iteration loop iterated over `chunk_count` instead of `buffer_count`, which causes erroneous behavior if earlier buffers were `nullptr` and later buffers were valid.
**Action:** Always cache the buffer properties struct during the first loop validation pass to prevent redundant C API invocations later. Using a stack-allocated buffer (e.g., `BufferInfo[32]`) perfectly optimizes out heap allocations for target high-concurrency batch workloads up to 32 items. Iterate over `buffer_count` when applying operations sequentially.
