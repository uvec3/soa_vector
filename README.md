# soa_vector

A header-only C++20 library providing **Structure-of-Arrays (SoA)** containers with a familiar `std::vector`-like API, including full CUDA device support.

Instead of storing objects contiguously (*Array of Structures*, AoS), this library splits each field into its own packed array — a layout that dramatically improves cache efficiency and SIMD/GPU throughput for field-heavy workloads.

---

## Features

- **Zero boilerplate** — define your element type as a plain struct of references; the library reflects its fields at compile time using C++20 structured bindings
- **Nested structs** — fields can be nested aggregates, flattened automatically (up to 32 members per level)
- **`std::vector`-like API** — `operator[]`, `at()`, `front()`, `back()`, `begin()`/`end()`, `size()`, `capacity()`, `reserve()`, `resize()`, `clear()`, `shrink_to_fit()`
- **Random-access iterators** — compatible with `std::sort`, range-based `for`, and other STL algorithms
- **`soa_view`** — lightweight non-owning view / slice into any SoA container
- **`soa_pool`** — SoA container with a free-list allocator for stable slot allocation and reuse
- **`aosoa_vector`** — Array-of-Structures-of-Arrays hybrid with a configurable inner block size (default 32), balancing locality and vectorisation
- **Custom allocators** — swap in any allocator via a simple static interface (`alloc`, `free`, `copy`, optional `realloc`)
- **CUDA support** — `soa_view` and iterators carry `__host__ __device__` annotations; `soa_cuda.hpp` provides `cuda_device_vector` for seamless GPU transfers

---

## Quick Start

### 1. Define your reference type

Each element is described by a struct whose members are **references** to scalar or nested-aggregate types:

```cpp
#include "src/soa_vector.hpp"

struct Point {
    float& x;
    float& y;
    float& z;
};

struct VertexRef {
    Point   position;   // nested aggregate — reflected automatically
    Point   normal;
    float&  u;
    float&  v;
};
```

### 2. Create a vector and access elements

```cpp
soa::soa_vector<VertexRef> vertices(1'000'000);

// Write through a proxy reference
vertices[0].position.x = 1.f;
vertices[0].u = 0.5f;

// Read back — same syntax
float x = vertices[0].position.x;

// Bounds-checked access
vertices.at(999999).v = 1.f;
```

### 3. Iterate and use STL algorithms

```cpp
// Range-based for
for (auto&& v : vertices) {
    v.position.x += 1.f;
}

// std::sort by a field
std::sort(vertices.begin(), vertices.end(),
    [](VertexRef a, VertexRef b) { return a.u < b.u; });
```

### 4. Use on the GPU (CUDA)

```cpp
#include "src/soa_cuda.hpp"

// Copy host → device
soa::cuda_device_vector<VertexRef> d_vertices(vertices);

// Launch a kernel — pass the view by value
__global__ void move_kernel(soa::soa_view<VertexRef> verts, //<-- use soa_view to pass shallow copy
                            float dx, float dy, float dz) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < verts.size) {
        verts[i].position.x += dx;
        verts[i].position.y += dy;
        verts[i].position.z += dz;
    }
}

move_kernel<<<(N+127)/128, 128>>>(d_vertices, 1.f, 0.f, 0.f);
```

---

## API Reference

### `soa::soa_vector<TReference, Allocator>`

| Member | Description |
|---|---|
| `soa_vector(size_t n)` | Construct with `n` default-initialised elements |
| `operator[](i)` | Proxy reference to element `i` (no bounds check) |
| `at(i)` | Proxy reference with bounds check (`std::out_of_range`) |
| `front()` / `back()` | First / last element proxy |
| `size()` / `capacity()` / `empty()` | Capacity query |
| `reserve(n)` | Pre-allocate without changing size |
| `resize(n)` | Change size, reallocate if needed |
| `allocate(n)` | Append `n` slots, return start index; grows by ×2 |
| `clear()` | Set size to 0, keep allocation |
| `shrink_to_fit()` | Release excess memory |
| `view()` | Return a lightweight `soa_view<TReference>` |
| `range(offset, size)` | Sub-view into `[offset, offset+size)` |
| `begin()` / `end()` | Random-access iterators |

### `soa::soa_view<TReference>`

Non-owning, copyable, `__device__`-safe slice. Passed by value to GPU kernels. Supports `operator[]`, `size()`, `empty()`, `begin()`/`end()`, and `copy_from()`.

### `soa_pool<TReference, Allocator>`

Extends `soa_vector` with a free-list (`LogicalAllocatorOffset`) for stable O(1) slot allocation and reuse:

```cpp
soa_pool<VertexRef> pool(1024);
size_t slot = pool.alloc_element();   // allocate one slot
pool.free(slot);                      // return slot to pool
```

### `soa::aosoa_vector<TReference, BlockSize, Allocator>`

AoSoA layout — data is split into fixed-size blocks of `BlockSize` (default 32) elements. Each block stores its fields contiguously, providing SIMD-friendly inner loops while limiting cache-line waste at block boundaries.

---

## Custom Allocators

Implement the following static interface to use any memory backend:

```cpp
struct MyAllocator {
    static void* alloc(size_t bytes);
    static void  free(void* ptr);
    static void  copy(void* dst, const void* src, size_t bytes);
    // Optional — enables in-place realloc path:
    static void  realloc(void** ptr, size_t new_size, size_t old_size);
};

soa::soa_vector<VertexRef, MyAllocator> vec(1024);
```

A CUDA-managed-memory allocator lives in `src/soa_cuda.hpp`.

---



## Project Layout

```
src/
  soa_vector.hpp          # Core SoA container & reflection engine
  aosoa_vector.hpp        # AoSoA container
  soa_pool.hpp            # Free-list SoA pool
  soa_cuda.hpp            # CUDA device vector & managed allocator
  cuda_common.hpp         # CUDA_HOST_DEVICE macro & CUDA_CHECK helper
  LogicalAllocators/      # Offset-based logical allocator (used by soa_pool)
tests/
  soa_vector_basic_tests.cpp
  soa_vector_cuda_tests.cu
benchmarks/               # MoveVertex benchmarks: AoS, SoA, AoSoA, aligned, random access
examples/
```

---

## How It Works

The library reflects the fields of `TReference` entirely at compile time using C++20:

1. **Arity detection** — probes `T{AnyLeaf{}, ...}` with increasing argument counts to find the number of leaf fields (supports nested aggregates by flattening recursively).
2. **`init_struct_recursive`** — constructs a `TReference` by supplying each leaf slot with an `IndexedLeafProvider<I>` that routes the `operator T&()` conversion to user-provided storage (e.g. the `I`-th raw array at position `index`).
3. **`tie_flat`** — structured-binding unpack for up to 32 members, recursing into sub-aggregates, yielding a flat `std::tuple` of references used by iterators and copy helpers.

No macros, no code generation, no external dependencies beyond the standard library (and optionally CUDA).

---

## License

See [LICENSE](LICENSE).
