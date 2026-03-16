## 2024-03-16 - Hoisting Vector Allocations in Hot Paths
**Learning:** In the `ProcessRequests` hot path, local `std::vector` allocations within inner loops (e.g. for batch input processing) can cause significant overhead. Furthermore, string concatenation in `SetStringOutputBuffer` without pre-reserving capacity results in multiple hidden heap re-allocations as strings grow.
**Action:** Always hoist vector variables out of tight inner loops and clear them for reuse. For string concatenation where the final size can be determined in advance, iterate twice (once to sum required bytes, and once to append data) and call `reserve()` to eliminate dynamic re-allocation overhead.

## 2024-03-16 - Avoid Over-Optimizing Small Array Lookups
**Learning:** Trying to optimize `O(N)` linear searches over small arrays (like requested model outputs, typically 1-5) using `O(log N)` binary search on a sorted vector is a pessimization. The overhead of setting up the sorted vector (including heap allocation) and inserting elements (`O(N)` shifts) dwarfs the minimal cost of a contiguous linear scan for typical workloads.
**Action:** Stick to `std::vector` + linear search for small collections in the hot path. Only consider more complex data structures if the element count guarantees a measurable improvement that offsets the initialization cost.
