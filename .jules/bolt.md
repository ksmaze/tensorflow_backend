## 2024-03-09 - [Triton Output Resolution Hot Path]
**Learning:** `std::find` was checking if an output `const char*` was requested by searching an array using string comparisons inside the per-request output execution loop. Also, dynamically allocating a `new std::string()` for every request and string output tensor was happening on the hot path.
**Action:** Replaced the string comparison loop with an index array (`std::vector<uint32_t> request_required_outputs`) that performs integer comparisons. Pre-reserved `std::vector<std::string>` to securely place strings on a stable buffer array without needing `std::unique_ptr` dynamic allocations per-request, preventing CUDA async pointer invalidation and eliminating multiple `O(N * M)` heap allocations per execute call.

## 2023-10-27 - [Hidden Vector Allocations in ProcessRequests]
**Learning:** Found a performance bottleneck in the `ProcessRequests` hot-path. Unnecessary internal memory allocations were made due to repeated implicit temporary `std::string` copies when calling `std::find(required_outputs.begin(), required_outputs.end(), const char*)`. The `batchn_shape` vector was also needlessly allocated and reallocated inside a loop iteration. The 2D vector `request_required_outputs` resulted in O(N) allocations for small dynamic collections.
**Action:** Next time, avoid implicitly constructing short-lived temporaries like `std::string` in hot loops. Preallocate linear containers outside loops via `assign()` or `clear()` to reuse capacity. Flatten vector-of-vectors to reduce dynamic heap allocations in favor of contiguous memory caches.

## 2024-03-10 - [Avoid VLA and String Allocations in Output Request Iteration]
**Learning:** Non-standard variable-length arrays (VLAs) (`const char* output_names_cstr[required_outputs.size()]`) were being used, and multiple `std::string` objects were instantiated redundantly inside the outputs loop just to hold the names to find in `model_.output_name_map_`.
**Action:** Always replace VLAs with `std::vector` to be standard compliant and avoid stack overflow issues. Furthermore, you can search `IONameMap` directly with `const char*` or cache `const char*` pointers inside the pre-allocated `std::vector<const char*> output_names_cstr` to eliminate dynamic allocations inside the loop.

## 2024-05-19 - Removed String Allocation Overhead in Hot Path
**Learning:** `std::unordered_map` with `std::string` keys requires constructing a `std::string` for queries using `const char*` prior to C++20 transparent comparisons. In Triton, tensors names are queried as `const char*` during `ProcessRequests`.
**Action:** Replaced `std::unordered_map` with a custom `std::vector<std::pair<std::string, std::string>>` and linear search on `const char*` directly. Since the number of inputs/outputs per model is extremely small (<10), linear search is significantly more cache friendly and removes heap allocations entirely.

## 2024-05-19 - Fast Lookups for Large Tensor Counts without Allocation
**Learning:** Some models can have up to ~800 inputs/outputs. A linear search on `std::vector` isn't scalable enough, but using `std::unordered_map` with `std::string` allocates heap memory when queried with `const char*` due to lack of transparent comparators.
**Action:** By maintaining a sorted `std::vector` and using `std::lower_bound` combined with a custom lambda comparing `a.first.c_str()` to the searched `const char*` using `std::strcmp(..., ...) < 0`, we achieve both `O(log N)` lookup times and zero `std::string` allocations!
