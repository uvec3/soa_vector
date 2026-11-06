# soa_vector

A header-only C++20 library providing **configurable memory-layout containers** (**SoA**, **AoS**, **Hybrid SoA/AoS**, and **AoSoA**) with a familiar `std::vector`-like API and full CUDA device support.

Whether you want every field split into its own packed array (*Structure of Arrays*, SoA), a classic contiguous buffer (*Array of Structures*, AoS), or a fine-tuned mix where hot co-accessed fields stay grouped together while cold fields are separated — **you control the exact memory layout solely by adding or removing `&` in your struct definition, without changing a single line of container, algorithm, or GPU kernel code.**

---

## Features

- **Zero boilerplate** — define your element type as a plain C++ struct; the library reflects its fields at compile time using C++20 structured bindings
- **Effortless SoA ↔ Hybrid ↔ AoS tuning** — reference members (`T&`) are split into separate arrays; value structs (with no references inside) are stored contiguously as a single AoS unit. Switching a type between SoA, Hybrid, and pure AoS requires only toggling `&` on its fields
- **Nested structs** — fields can be nested aggregates, flattened or grouped automatically (up to 32 members per level)
- **Methods on element types** — add computed properties, mutating helpers, and operators directly to your struct; OOP style is fully preserved
- **`std::vector`-like API** — `operator[]`, `at()`, `front()`, `back()`, `begin()`/`end()`, `size()`, `capacity()`, `reserve()`, `resize()`, `clear()`, `shrink_to_fit()`
- **Random-access iterators** — compatible with `std::sort`, `thrust::sort_by_key`, range-based `for`, and other STL/Thrust algorithms
- **`soa_view` & `soa_span`** — lightweight `__host__ __device__` non-owning view (`soa_view`) for kernels, plus allocator-aware `soa_span` (`range(offset, size)`) for cross-allocator subrange copies
- **`aosoa_vector`** — Array-of-Structures-of-Arrays hybrid with a configurable inner block size (default 32), balancing locality and vectorisation
- **Custom allocators** — swap in any allocator via a simple static interface (`alloc`, `free`, `copy`, optional `realloc`)
- **CUDA support** — `soa_view` and iterators carry `__host__ __device__` annotations; `soa_cuda.hpp` provides `cuda_device_vector`, `cuda_host_vector`, and `cuda_unified_vector` for seamless GPU transfers

---

## Effortless Layout Tuning: SoA, Hybrid, and AoS

In real-world rendering and simulation workloads, the optimal memory layout is rarely "all SoA" or "all AoS" — you often need to **experiment and profile** on real hardware. Usually, changing data layout means rewriting every container, kernel signature, indexing expression, and copy call.

With `soa_vector`, **the struct definition *is* the layout specification**:

- **Reference fields (`T&`)** tell `soa_vector` to allocate a dedicated contiguous array for that field.
- **Pure value structs** (aggregates containing no reference members anywhere inside) are treated as a single leaf type and stored contiguously in one array.
- When the **top-level struct itself has no references**, `soa_vector<T>` acts as a standard AoS container: it allocates a single contiguous `T*` buffer and `operator[]` returns `T&` directly.

### Side-by-Side: 3 Layouts, 0 Code Changes

```cpp
// 1. Pure SoA — 4 separate allocations (v1[], v2[], v3[], material_id[])
// Ideal when kernels stream only vertex indices without touching material_id.
struct Triangle {
    uint32_t& v1;
    uint32_t& v2;
    uint32_t& v3;
    uint32_t& material_id;
};

// 2. Hybrid (Grouped SoA) — 2 allocations (indices[], material_id[])
// v1, v2, v3 are always fetched together in a single cache line, while
// material_id is isolated into a separate stream for shading.
struct TriIndices { uint32_t v1, v2, v3; }; // no '&' -> contiguous AoS leaf
struct Triangle {
    TriIndices& indices;                    // '&' -> separate stream from material_id
    uint32_t&   material_id;
};

// 3. Pure AoS — 1 contiguous allocation (Triangle[])
// operator[] returns Triangle& directly, just like std::vector<Triangle>.
struct Triangle {
    uint32_t v1;
    uint32_t v2;
    uint32_t v3;
    uint32_t material_id;
};
```

Regardless of whether `Triangle` is Pure SoA, Hybrid, or Pure AoS, **all of your application and kernel code stays identical**:

```cpp
soa::soa_vector<Triangle> h_tris(N);
soa::cuda_device_vector<Triangle> d_tris = h_tris;

// Same kernel for SoA and AoS:
__global__ void remap_vertices(soa::soa_view<Triangle> tris, uint32_t offset) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tris.size()) {
        tris[i].v1 += offset;
        tris[i].v2 += offset;
        tris[i].v3 += offset;
    }
}
```

### Writing Layout-Independent Code

Follow these four idioms so you can freely toggle `&` on any field at any time:

1. **Bind element references with `auto&&`** — For SoA types, `vec[i]` returns a proxy (`RefWrapper<T>`) by value; for AoS types, `vec[i]` returns an lvalue reference (`T&`). Using `auto&&` binds cleanly to both and always mutates in place:
   ```cpp
   auto&& tri = triangles[i];
   tri.v1 = 10; // mutates in-place for both SoA and AoS
   ```
2. **Pass `soa_view<T>` (via `.view()` or `.view(offset, size)`) to kernels** — Avoid taking `&vec[start].field` and assuming a stride-1 array of that field; pass `vec.view(start, count)` and index `view[i].field`.
3. **Copy subranges with `.range(offset, size).copy_from(...)`** — `range()` returns an allocator-aware `soa_span` that copies all underlying buffers (1 buffer for AoS, $N$ buffers for SoA) across host and device:
   ```cpp
   d_triangles.range(dst_offset, count).copy_from(d_local_triangles);
   h_triangles.range(dst_offset, count).copy_from(d_triangles.range(dst_offset, count));
   ```
4. **Use `.get_tuple(offset)` with Thrust `zip_iterator`** — `get_tuple()` returns a tuple of pointers to each underlying buffer (`tuple<T*>` for AoS, `tuple<Field0*, Field1*, ...>` for SoA), so `thrust::sort_by_key` and `thrust::fill` work transparently on any layout:
   ```cpp
   auto vals = thrust::make_zip_iterator(d_triangles.get_tuple(start));
   thrust::sort_by_key(thrust::device, keys.begin(), keys.end(), vals);
   ```

---

## Quick Start

### 1. Define your element type

Each element is described by a struct whose members are **references** (for SoA separation) or **values** (for AoS grouping):

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

// Write through a proxy reference (or direct reference in AoS mode)
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
    [](const auto& a, const auto& b) { return a.u < b.u; });
```

### 4. Use on the GPU (CUDA)

```cpp
#include "src/soa_cuda.hpp"

// Copy host → device
soa::cuda_device_vector<VertexRef> d_vertices(vertices);

// Launch a kernel — pass the view by value
__global__ void move_kernel(soa::soa_view<VertexRef> verts,
                            float dx, float dy, float dz) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < verts.size()) {
        verts[i].position.x += dx;
        verts[i].position.y += dy;
        verts[i].position.z += dz;
    }
}

move_kernel<<<(d_vertices.size()-1)/128+1, 128>>>(d_vertices, 1.f, 0.f, 0.f);
```

---

## OOP with Reference Types

Element structs are plain C++ structs — they can carry **methods, operators, and static helpers** just like any other type. The proxy (or direct reference) returned by `operator[]` *is* that struct, so calling a method on it directly manipulates the in-place container data.

Three patterns are demonstrated in [`examples/oop_methods.cpp`](examples/oop_methods.cpp):

### Pattern 1 — computed properties and mutation helpers

```cpp
struct Vec3Ref {
    float& x;
    float& y;
    float& z;

    float length() const { return std::sqrt(x*x + y*y + z*z); }

    void normalize() {
        float inv = 1.f / length();
        x *= inv; y *= inv; z *= inv;
    }

    Vec3Ref& operator+=(Vec3Ref other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }
};

vecs[0].normalize();           // mutates the SoA arrays directly
vecs[1] += vecs[2];            // operator+= composes across slots
```

### Pattern 2 — domain logic on nested reference types

```cpp
struct ParticleRef {
    Vec3Ref position;
    Vec3Ref velocity;
    float&  mass;
    float&  charge;

    float kinetic_energy() const {
        float v2 = velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
        return 0.5f * mass * v2;
    }

    // Symplectic Euler step
    void integrate(float dt, float fx, float fy, float fz) {
        position.x += velocity.x * dt;
        position.y += velocity.y * dt;
        position.z += velocity.z * dt;
        velocity.x += (fx / mass) * dt;
        velocity.y += (fy / mass) * dt;
        velocity.z += (fz / mass) * dt;
    }

    float coulomb_force(ParticleRef other) const {
        constexpr float k = 8.99e9f;
        float dx = other.position.x - position.x;
        float dy = other.position.y - position.y;
        float dz = other.position.z - position.z;
        return k * charge * other.charge / (dx*dx + dy*dy + dz*dz + 1e-10f);
    }
};

soa::soa_vector<ParticleRef> particles(N);
particles[0].integrate(dt, 0.f, -9.81f, 0.f);   // reads/writes SoA in-place
float F = particles[0].coulomb_force(particles[1]);
```

### Pattern 3 — thin domain wrapper with named constructors

```cpp
class ParticleSystem {
public:
    // Named factory — hides the SoA initialisation loop
    static ParticleSystem uniform(size_t n, float mass, float charge,
                                  float vx, float vy, float vz);

    void  step(float dt, float fx, float fy, float fz); // bulk integrate
    float total_kinetic_energy() const;                  // aggregate query

private:
    soa::soa_vector<ParticleRef> m_particles;
};

auto sys = ParticleSystem::uniform(1'000'000, 1.f, 1.f, 0.f, 1.f, 0.f);
sys.step(0.01f, 0.f, -9.81f, 0.f);
float ke = sys.total_kinetic_energy();
```

> [!NOTE]
> For SoA structs, `operator[]` returns a `RefWrapper<TReference>` proxy by value whose members are references into the arrays; for AoS structs, `operator[]` returns `TReference&` directly. In both cases, methods called on `vec[i]` operate directly on the underlying storage — no copies, no indirection.

---

## API Reference

### `soa::soa_vector<TReference, Allocator>`

| Member | Description |
|---|---|
| `soa_vector(size_t n)` | Construct with `n` uninitialised element slots |
| `operator[](i)` | `RefWrapper<TReference>` (SoA) or `TReference&` (AoS) for element `i` (no bounds check) |
| `at(i)` | Element reference/proxy with bounds check (`std::out_of_range`) |
| `front()` / `back()` | First / last element reference/proxy |
| `size()` / `capacity()` / `empty()` | Capacity query |
| `reserve(n)` | Pre-allocate without changing size |
| `resize(n)` | Change size, reallocate if needed |
| `allocate(n)` | Append `n` slots, return start index; grows by ×2 |
| `clear()` | Set size to 0, keep allocation |
| `shrink_to_fit()` | Release excess memory |
| `view()` | Return a lightweight `soa_view<TReference>` over the full container |
| `view(offset, size)` | Return a lightweight `soa_view<TReference>` over `[offset, offset+size)` |
| `span()` | Return an allocator-aware `soa_span<TReference, Allocator>` over the full container |
| `range(offset, size)` | Return an allocator-aware `soa_span<TReference, Allocator>` over `[offset, offset+size)` |
| `begin()` / `end()` | Random-access iterators |

### `soa::soa_view<TReference>`

Non-owning, copyable, `__host__ __device__`-safe slice. Passed by value to GPU kernels. Supports `operator[]`, `size()`, `empty()`, `begin()`/`end()`, and element-wise `copy_from()`.

### `soa::soa_span<TReference, Allocator>`

Inherits from `soa_view<TReference>` (and converts implicitly to it) while retaining the container's `Allocator`. Its `copy_from(other_span_or_vector)` method uses bulk allocator copies (`Allocator::copy` or cross-allocator `transfer_data`) across all underlying field buffers regardless of whether `TReference` is SoA, Hybrid, or AoS.

### `soa::aosoa_vector<TReference, BlockSize, Allocator>`

AoSoA layout — data is split into fixed-size blocks of `BlockSize` (default 32) elements. Each block stores its fields contiguously, providing SIMD-friendly inner loops while limiting cache-line waste at block boundaries.

### CUDA Containers (`src/soa_cuda.hpp`)

Type aliases backed by `CudaDeviceAllocator`, `CudaHostAllocator` (pinned host memory), and `CudaUnifiedAllocator` (managed memory with device-preferred residency hints):

- `soa::cuda_device_vector<TReference>` / `soa::cuda_device_aosoa<TReference, BlockSize>`
- `soa::cuda_host_vector<TReference>` / `soa::cuda_host_aosoa<TReference, BlockSize>`
- `soa::cuda_unified_vector<TReference>` / `soa::cuda_unified_aosoa<TReference, BlockSize>`
- `.get_tuple(offset = 0)` — returns a `cuda::std::tuple` of typed pointers into each underlying array (1 pointer for AoS, $N$ pointers for SoA), ready for `thrust::make_zip_iterator`.

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

Device, pinned-host, and unified-memory CUDA allocators live in `src/soa_cuda.hpp`.

---

## Project Layout

```
src/
  soa_vector.hpp          # Core SoA/AoS container, views, spans & reflection engine
  aosoa_vector.hpp        # AoSoA container
  soa_cuda.hpp            # CUDA device/host/unified vectors & allocators
  cuda_common.hpp         # CUDA_HOST_DEVICE macro & CUDA_CHECK helper
tests/
  soa_vector_basic_tests.cpp
  soa_vector_cuda_tests.cu
benchmarks/               # MoveVertex benchmarks: AoS, SoA, AoSoA, aligned, random access
examples/
  oop_methods.cpp         # Methods / operators / domain wrappers on reference types
```

---

## How It Works

The library reflects the fields of `TReference` entirely at compile time using C++20:

1. **Leaf vs. SoA detection (`IsLeaf`)** — Any non-aggregate type or default-constructible aggregate (containing no reference members anywhere inside) is detected as a contiguous leaf (`IsLeaf<T> == true`).
2. **Arity detection (`find_arity`)** — Returns `1` if `IsLeaf<T>` is true (single AoS buffer), or probes `T{AnyLeaf{}, ...}` recursively to count all separated reference leaves.
3. **`init_struct_recursive`** — For AoS types (`IsLeaf<T>`), returns a direct `T&` into the single buffer; for SoA types, constructs `TReference` by supplying each leaf slot with an `IndexedLeafProvider<I>` that binds the `I`-th reference to the `I`-th array at `index`.
4. **`tie_flat`** — Structured-binding unpack for up to 32 members, recursing only into non-leaf sub-aggregates, yielding a flat `std::tuple` of references used by snapshots, iterators, and `get_tuple()`.

No macros, no code generation, no external dependencies beyond the standard library (and optionally CUDA).

---

## License

See [LICENSE](LICENSE).
