1. **Optimize duplicate output checking logic**
   - The current code uses `std::find` to check if an output is already in `required_outputs`: `if (std::find(required_outputs.begin(), required_outputs.end(), output_name) == required_outputs.end()) { required_outputs.emplace_back(output_name); }`.
   - Since `required_outputs` is a `std::vector`, this is an O(N) operation per output.
   - We can either switch to an `std::unordered_set<std::string>` or `std::set<std::string>` for the existence check to make this O(1) or O(log N). Alternatively, since the number of outputs per request is small, we could sort the vector and use `std::unique` after collecting all outputs across all requests.
   - However, since `required_outputs` vector is small, maybe we can use an `std::set<std::string>` alongside the `std::vector<std::string>`, or just change it to `std::set<std::string>`. Wait, order might matter. Let's see if we can use a `std::unordered_set` or `std::set` `seen_outputs` alongside the vector to avoid the `O(N)` find.

Wait, another issue: `request_required_outputs[idx].push_back(output_name)` inside the loop! `output_name` is a `const char*` pointer from `TRITONBACKEND_RequestOutputName`. Then later when we check if the requested output is needed we do `std::find` on it. Actually, `request_required_outputs` is a `std::vector<std::vector<const char*>>` which makes it fast to copy pointers but the check `std::find(request_required_outputs[idx].begin(), request_required_outputs[idx].end(), name)` compares `const char*` against `std::string name`. This might use `operator==(std::string, const char*)` which is string comparison.

Wait, looking at `request_required_outputs` which is `std::vector<std::vector<const char*>> request_required_outputs(request_count);`
`request_required_outputs[idx].push_back(output_name)` stores `const char*`.
Later, when processing outputs:
```cpp
            if ((response != nullptr) &&
                (std::find(request_required_outputs[idx].begin(), request_required_outputs[idx].end(), name) !=
                 request_required_outputs[idx].end())) {
```
`name` is of type `std::string` (from `model_output_names`).
The `std::find` on `std::vector<const char*>` will search for `std::string`.
Wait, does `std::find` work correctly here? It compares elements using `==`. So it does `const char* == std::string`, which performs string comparison (allocates? no, `std::string::operator==(const char*)` is fast).
However, doing `std::find` on each output across requests can be slow. But `request_required_outputs[idx]` is typically very small.

Wait, in `ProcessRequests`, there's another allocation:
```cpp
        // Custom handling for string/bytes tensor...
        if (datatype == TRITONSERVER_TYPE_BYTES) {
...
              string_buffer.emplace_back(new std::string());
              cuda_copy |= SetStringOutputBuffer(
                  output_tensor, &response, response_output, tensor_element_cnt,
                  tensor_offset, CudaStream(), string_buffer.back().get());
```
`string_buffer` is `std::vector<std::unique_ptr<std::string>> string_buffer;`. It allocates a `std::string` on the heap for each output tensor per request! That's a heap allocation (`new std::string()`) in the hot path.

Let's look at `SetStringOutputBuffer` in `src/tensorflow.cc`:
```cpp
bool
SetStringOutputBuffer(
    TRITONTF_Tensor* tensor, TRITONBACKEND_Response** response,
    TRITONBACKEND_Output* response_output, const size_t tensor_element_count,
    const size_t tensor_offset, cudaStream_t stream, std::string* serialized)
{
  bool cuda_copy = false;

  // Serialize the output tensor strings. Each string is serialized as
  // a 4-byte length followed by the string itself with no
  // null-terminator.
  serialized->clear();
  for (size_t e = 0; e < tensor_element_count; ++e) {
    size_t len;
    const char* cstr = TRITONTF_TensorString(tensor, tensor_offset + e, &len);
    serialized->append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
    if (len > 0) {
      serialized->append(cstr, len);
    }
  }

  // Allocate a buffer large enough to hold the serialized tensor.
  TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t actual_memory_type_id = 0;

  void* buffer;
  auto err = TRITONBACKEND_OutputBuffer(
      response_output, &buffer, serialized->size(), &actual_memory_type,
      &actual_memory_type_id);
...
```
It serializes strings into a `std::string* serialized`, which is `new std::string()`.
Why does `string_buffer` need to be a vector of pointers to strings?
```cpp
  // The serialized string buffer must be valid until output copies are done
  std::vector<std::unique_ptr<std::string>> string_buffer;
```
It says "The serialized string buffer must be valid until output copies are done". But wait, the copy is done in `SetStringOutputBuffer` by calling `CopyBuffer`:
```cpp
  err = CopyBuffer(
      "String output", TRITONSERVER_MEMORY_CPU /* src_memory_type */,
      0 /* src_memory_type_id */, actual_memory_type, actual_memory_type_id,
      serialized->size(), reinterpret_cast<const void*>(serialized->c_str()),
      buffer, stream, &cuda_used);
```
If `cuda_used` is true, the copy is asynchronous. If asynchronous, the source buffer `serialized->c_str()` must outlive the copy, which completes when we synchronize the stream later!
Ah, yes, `cudaStreamSynchronize` is called after the loop.
But wait! We don't need `new std::string()`. We can use `std::list<std::string>` or `std::vector<std::string>` to manage the strings without `unique_ptr`, but wait, `std::vector<std::string>` could reallocate and invalidate the `c_str()` pointers!
`std::list<std::string>` or `std::deque<std::string>` would not invalidate existing elements, avoiding the heap allocation of the `std::string` object itself (though the string buffer still heap allocates, it's unavoidable, but we save one heap allocation for the `std::string` object).
Actually, `std::string` allocates internally anyway. Is `new std::string()` a big deal?

What about `required_outputs` collection:
```cpp
            if (std::find(required_outputs.begin(), required_outputs.end(), output_name) == required_outputs.end()) {
              required_outputs.emplace_back(output_name);
            }
```
If `required_outputs` grows, it is slow. But usually a model has a few outputs (like < 10). So `std::find` is fast enough because of cache locality, probably faster than `std::set`.
Wait, looking closely at `required_outputs`: `output_name` is `const char*` from `TRITONBACKEND_RequestOutputName`.
Wait, `required_outputs` is `std::vector<std::string>`.
So `required_outputs.emplace_back(output_name)` creates a new `std::string`.
And `std::find` checks if the `const char* output_name` is in `required_outputs`. This will invoke `operator==(const std::string&, const char*)`.

Is there a better optimization?
"Minimize data copying (strive for zero-copy between Triton and TensorFlow)."
"Memory Allocations: Unnecessary std::string, std::vector, or std::set heap allocations in the hot path."

Let's look at `ProcessRequests`.
```cpp
      if (response != nullptr) {
        request_required_outputs[idx].reserve(output_count);
        for (uint32_t output_idx = 0; output_idx < output_count; output_idx++) {
          const char* output_name;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &response, TRITONBACKEND_RequestOutputName(
                             request, output_idx, &output_name));
          if (response != nullptr) {
            if (std::find(required_outputs.begin(), required_outputs.end(), output_name) == required_outputs.end()) {
              required_outputs.emplace_back(output_name);
            }
            request_required_outputs[idx].push_back(output_name);
          }
        }
      }
```

Wait, `required_outputs` only needs to be unique across all requests.
Instead of `std::find` for every single output of every single request (which means if we have 100 requests, and each has 2 outputs, we do `std::find` 200 times), we can use a `std::set<const char*>` or `std::unordered_set<const char*>`? But wait, the strings returned by `TRITONBACKEND_RequestOutputName` are they guaranteed to have the same pointer? Triton typically stores the strings in the request, they might not be the same pointer for the same name.
Wait, if we use a `std::vector<const char*>` and then sort and unique it, we could avoid many `std::string` heap allocations! Wait, Triton returns `const char*`.
If we collect all `const char*` into a `std::vector<const char*>`, we can't sort them by pointer. We'd have to sort by string value `strcmp`.

Wait, the number of outputs per model is very small. Number of requests might be up to batch size.
If max batch size is 128, and each request asks for the same output "logits", `required_outputs` will be checked 128 times, doing `std::find` and string comparisons.

Can we optimize `request_required_outputs` ?
```cpp
            if ((response != nullptr) &&
                (std::find(request_required_outputs[idx].begin(), request_required_outputs[idx].end(), name) !=
                 request_required_outputs[idx].end())) {
```
For every response, and for every model output, it does `std::find` in `request_required_outputs[idx]`.
If we have 128 requests, and 5 model outputs, that's `128 * 5` calls to `std::find`. In `std::find`, it compares `const char*` against `std::string name`!
Wait, `name` is `const std::string&`.
So `std::find` on `std::vector<const char*>` looking for `std::string` will do:
`*it == name` which invokes `operator==(const char*, const std::string&)`. This does a string comparison.
`128 * 5 = 640` string comparisons!

If we change `request_required_outputs` to store something faster or if we pre-calculate this?
Wait, if `request_required_outputs` stores `std::vector<std::string>`, it allocates.
What if `request_required_outputs` stored boolean flags?
For example, `std::vector<std::vector<bool>> request_required_outputs(request_count);`
Wait, how do we map model output names to indices?
The model output names are collected into `required_outputs`.
What if we change `required_outputs` to `std::vector<std::string> model_output_names;`
And for each request, we can just track if they requested it.
But wait, during the first loop:
```cpp
  std::vector<std::string> required_outputs;
  std::vector<std::vector<const char*>> request_required_outputs(request_count);
```
We are dynamically discovering `required_outputs`.

Wait, another big thing:
```cpp
      const char* input_tensor_name = name;
      const auto& tn_itr = model_.input_name_map_.find(input_tensor_name);
      if (tn_itr != model_.input_name_map_.end()) {
        input_tensor_name = tn_itr->second.c_str();
      }
```
Wait! `model_.input_name_map_.find(input_tensor_name)` creates a `std::string` from `const char*`! Because `find` on `std::unordered_map<std::string, std::string>` takes a `std::string` key!
In C++11/C++14, `std::unordered_map::find` does NOT support heterogeneous lookup (that's C++20). So passing a `const char*` will allocate a `std::string` to do the find!
This is inside the `input_count` loop, for every input of the request!
```cpp
    for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
      TRITONBACKEND_Input* input;
      TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input);
      const char* name;
...
      const char* input_tensor_name = name;
      const auto& tn_itr = model_.input_name_map_.find(input_tensor_name);
      if (tn_itr != model_.input_name_map_.end()) {
        input_tensor_name = tn_itr->second.c_str();
      }
```
Wait! This loop is only over the inputs of `requests[0]`!
```cpp
    TRITONBACKEND_RequestInputCount(requests[0], &input_count);
    for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
```
So it's only executed `input_count` times (once per input of the first request). That's a very small number of allocations per batch, not per request!

Wait, what about batch inputs?
```cpp
    // Process batch input if any
    for (const auto& batch_input : StateForModel()->BatchInputs()) {
...
      for (const auto& input_name : batch_input.TargetNames()) {
        // The name of the input in the model can be different...
        const char* input_tensor_name = input_name.c_str();
        const auto& tn_itr = model_.input_name_map_.find(input_name);
```
This is also per batch input.

Let's look at outputs:
```cpp
  std::vector<std::string> required_outputs;
  std::vector<std::vector<const char*>> request_required_outputs(request_count);
  for (size_t idx = 0; idx < request_count; idx++) {
...
        for (uint32_t output_idx = 0; output_idx < output_count; output_idx++) {
          const char* output_name;
...
            if (std::find(required_outputs.begin(), required_outputs.end(), output_name) == required_outputs.end()) {
              required_outputs.emplace_back(output_name);
            }
            request_required_outputs[idx].push_back(output_name);
```
Here we loop over ALL requests, and ALL outputs.
If `request_count` is 128, and each request has 2 outputs. We loop 256 times.
In each loop iteration:
`std::find` checks if `output_name` is in `required_outputs` (which does `std::string == const char*`, no allocation).
If not, `required_outputs.emplace_back(output_name)` (allocates a `std::string`, but only once per unique output name, e.g. 2 times total).
Then `request_required_outputs[idx].push_back(output_name)` (adds to a vector, which might allocate/resize, but `reserve(output_count)` is called!).
So inside the loop over all requests, there are NO heap allocations (except the first few unique outputs)!

Wait, what about:
```cpp
            if ((response != nullptr) &&
                (std::find(request_required_outputs[idx].begin(), request_required_outputs[idx].end(), name) !=
                 request_required_outputs[idx].end())) {
```
Here `name` is `const std::string&`.
`request_required_outputs` contains `const char*`.
`std::find` compares `const char*` against `std::string`, which doesn't allocate.
But it does a lot of string comparisons. Is there a better way?

Wait! In `ProcessRequests`, there is a big one:
```cpp
  // Create the vector of required output names using the names
  // expected by the model.
  std::vector<std::string> model_output_names;
  model_output_names.reserve(required_outputs.size());
  const char* output_names_cstr[required_outputs.size()];
  {
    size_t oidx = 0;
    for (const auto& name : required_outputs) {
      model_output_names.push_back(name);
      const auto& tn_itr = model_.output_name_map_.find(name);
      if (tn_itr == model_.output_name_map_.end()) {
        output_names_cstr[oidx] = name.c_str();
      } else {
        output_names_cstr[oidx] = tn_itr->second.c_str();
      }
      oidx++;
    }
  }
```
This is only over `required_outputs` which is the number of unique outputs (e.g. 2). So it's very fast.

Let's look at `BatchOutput`:
```cpp
    TRITONTF_TensorList* output_tensor_itr = output_tensors.get();
    for (const auto& name : model_output_names) {
      TRITONTF_Tensor* output_tensor = output_tensor_itr->tensor_;

      const BatchOutput* batch_output = StateForModel()->FindBatchOutput(name);
```
`name` is `std::string`.
What does `FindBatchOutput` do?
`StateForModel()` is `ModelState*`. It inherits from `BackendModel`.
`FindBatchOutput` takes a `std::string`. Is it doing an unordered_map lookup? Probably.

Wait, is there any unnecessary allocation in the hot path?
Look at `std::vector<int64_t> batchn_shape;`
Inside the output loop:
```cpp
        // batchn_shape holds the shape of the entire tensor batch, but
        // is overwritten below and used as the shape for each response
        // output.
        std::vector<int64_t> batchn_shape;
        batchn_shape.reserve(tf_shape->rank_);
        for (size_t itr = 0; itr < tf_shape->rank_; itr++) {
          const int64_t dim = tf_shape->dims_[itr];
          batchn_shape.push_back(dim);
        }
```
`batchn_shape` is recreated inside the `for (const auto& name : model_output_names)` loop. But that's per unique output, not per request.

Wait, inside the loop over requests for `TRITONSERVER_TYPE_BYTES`:
```cpp
          for (size_t idx = 0; idx < responses.size(); idx++) {
            auto& request = requests[idx];
            auto& response = responses[idx];

            if (max_batch_size != 0) {
              batchn_shape[0] = request_batch_sizes[idx];
            }

            const size_t tensor_element_cnt = GetElementCount(batchn_shape);

            // Only need an response tensor for requested outputs.
            if ((response != nullptr) &&
                (std::find(request_required_outputs[idx].begin(), request_required_outputs[idx].end(), name) !=
                 request_required_outputs[idx].end())) {
              TRITONBACKEND_Output* response_output;
              RESPOND_AND_SET_NULL_IF_ERROR(
                  &response,
                  TRITONBACKEND_ResponseOutput(
                      response, &response_output, name.c_str(), datatype,
                      batchn_shape.data(), batchn_shape.size()));
              string_buffer.emplace_back(new std::string());
              cuda_copy |= SetStringOutputBuffer(
                  output_tensor, &response, response_output, tensor_element_cnt,
                  tensor_offset, CudaStream(), string_buffer.back().get());
            }

            tensor_offset += tensor_element_cnt;
          }
```
Here, `string_buffer.emplace_back(new std::string())` is allocating a `std::string` object on the heap, for every request, and for every string output tensor.
We can avoid the `new std::string()` by using `std::deque<std::string> string_buffer;` or `std::list<std::string> string_buffer;` instead of `std::vector<std::unique_ptr<std::string>>`.
`std::deque` never invalidates references to its elements when we append to the end. Wait, `std::deque::emplace_back` guarantees pointers/references to elements are NOT invalidated. (Wait, iterators are invalidated, but pointers and references are NOT!).
Actually, `std::list<std::string>` is even safer.
But `std::list` allocates nodes on the heap.
What about `std::vector<std::string> string_buffer;`
We know exactly how many elements will be added! We can `reserve()` the vector ahead of time!
```cpp
  size_t total_string_outputs = 0;
  // wait, we would need to count them.
```
Actually, we can just use `std::list<std::string>` to avoid `new std::string()`, wait, `std::list` node allocation is exactly the same as `new std::string()` (it allocates the node).
If we use `std::vector<std::string>`, and we DO NOT push_back. We can resize it beforehand or reserve.
Wait, can we just declare `std::vector<std::string> string_buffer;` at the beginning, and reserve enough capacity?
How many string outputs? At most `request_count * model_output_names.size()`.
If we `string_buffer.reserve(request_count * model_output_names.size());`
Then `string_buffer.emplace_back();`
Since it's reserved, `emplace_back` won't reallocate, so pointers to elements won't be invalidated!
Wait, `std::vector::reserve` guarantees that no reallocation happens as long as size doesn't exceed capacity. This means pointers are perfectly safe!
And `std::vector` allocates its array once!
This saves `request_count * num_string_outputs` heap allocations!

Let's check `string_buffer`:
Currently:
```cpp
  // The serialized string buffer must be valid until output copies are done
  std::vector<std::unique_ptr<std::string>> string_buffer;
...
              string_buffer.emplace_back(new std::string());
              cuda_copy |= SetStringOutputBuffer(
                  output_tensor, &response, response_output, tensor_element_cnt,
                  tensor_offset, CudaStream(), string_buffer.back().get());
```
If we change it to:
```cpp
  // The serialized string buffer must be valid until output copies are done
  std::vector<std::string> string_buffer;
  string_buffer.reserve(request_count * model_output_names.size());
...
              string_buffer.emplace_back();
              cuda_copy |= SetStringOutputBuffer(
                  output_tensor, &response, response_output, tensor_element_cnt,
                  tensor_offset, CudaStream(), &string_buffer.back());
```
This is a single heap allocation for the vector's storage, instead of one `new` for each request and each output.
And `std::string` also has a small-string optimization, though these strings will be large so they'll allocate anyway inside `SetStringOutputBuffer`.
Wait, `SetStringOutputBuffer` appends to the string:
```cpp
  serialized->clear();
  for (size_t e = 0; e < tensor_element_count; ++e) {
...
      serialized->append(cstr, len);
```
Yes, the `std::string` internally allocates. But we save the allocation of the `std::string` control block!

Wait, is there anything else?
What about `request_required_outputs[idx]`?
```cpp
            if ((response != nullptr) &&
                (std::find(request_required_outputs[idx].begin(), request_required_outputs[idx].end(), name) !=
                 request_required_outputs[idx].end())) {
```
The comparison does `name == request_required_outputs[idx][i]`. `name` is a `std::string`, `request_required_outputs[idx][i]` is `const char*`. This is a string comparison.
If we have 128 requests, and 5 outputs. 640 string comparisons.
If we instead check a boolean array or just compare the pointer?
Triton passes the SAME `const char*` pointer if the string is identical? NO, we cannot assume that.
But we CAN compare `const char*` using `!strcmp` or we could store booleans!
Notice `model_output_names` is exactly the list of `required_outputs`.
Why do we need to search?
Because some requests might not request all outputs.
When building `required_outputs`:
```cpp
          const char* output_name;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &response, TRITONBACKEND_RequestOutputName(
                             request, output_idx, &output_name));
          if (response != nullptr) {
            if (std::find(required_outputs.begin(), required_outputs.end(), output_name) == required_outputs.end()) {
              required_outputs.emplace_back(output_name);
            }
            request_required_outputs[idx].push_back(output_name);
          }
```
If we change `request_required_outputs` to store the INDEX of the required output instead of the `const char*`?
Wait, if we find the output in `required_outputs`, we get its index!
And then `request_required_outputs` can be `std::vector<std::vector<uint32_t>>`.
But wait, we can just do:
```cpp
  std::vector<std::string> required_outputs;
  // request_required_outputs[idx][req_out_idx] indicates if required_outputs[req_out_idx] is requested by request idx
  std::vector<std::vector<bool>> request_required_outputs(request_count);
```
Wait, we don't know the size of `required_outputs` beforehand.
Instead, we can just use a `std::vector<std::vector<bool>>`? No.
How about:
```cpp
  std::vector<std::string> required_outputs;
  std::vector<std::vector<uint32_t>> request_required_outputs(request_count);
...
        for (uint32_t output_idx = 0; output_idx < output_count; output_idx++) {
          const char* output_name;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &response, TRITONBACKEND_RequestOutputName(
                             request, output_idx, &output_name));
          if (response != nullptr) {
            auto it = std::find(required_outputs.begin(), required_outputs.end(), output_name);
            uint32_t req_out_idx;
            if (it == required_outputs.end()) {
              req_out_idx = required_outputs.size();
              required_outputs.emplace_back(output_name);
            } else {
              req_out_idx = std::distance(required_outputs.begin(), it);
            }
            request_required_outputs[idx].push_back(req_out_idx);
          }
        }
```
Then, later, when iterating over `required_outputs`:
```cpp
    uint32_t out_idx = 0;
    for (const auto& name : model_output_names) {
        ...
          for (size_t idx = 0; idx < responses.size(); idx++) {
            auto& request = requests[idx];
            auto& response = responses[idx];
            ...
            // Only need an response tensor for requested outputs.
            bool is_requested = false;
            if (response != nullptr) {
                for (auto req_out_idx : request_required_outputs[idx]) {
                    if (req_out_idx == out_idx) {
                        is_requested = true;
                        break;
                    }
                }
            }

            if (is_requested) {
                // ...
            }
```
This replaces string comparisons with integer comparisons! Integer comparisons are lightning fast.
Let's see: `out_idx` is the index of the output in `model_output_names`.
Since `model_output_names` is populated exactly from `required_outputs` in the same order:
```cpp
  std::vector<std::string> model_output_names;
  model_output_names.reserve(required_outputs.size());
  const char* output_names_cstr[required_outputs.size()];
  {
    size_t oidx = 0;
    for (const auto& name : required_outputs) {
      model_output_names.push_back(name);
```
Yes! `model_output_names[out_idx]` corresponds to `required_outputs[out_idx]`.
So we can just use integer index!

Wait, `request_required_outputs` is used in two places.
One for `TRITONSERVER_TYPE_BYTES`:
```cpp
            if ((response != nullptr) &&
                (std::find(request_required_outputs[idx].begin(), request_required_outputs[idx].end(), name) !=
                 request_required_outputs[idx].end())) {
```
The other place is ... wait, where else is it used?
Let's search for `request_required_outputs` in `src/tensorflow.cc`.
