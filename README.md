# soa_vector

A header-only C++20 library providing **Structure-of-Arrays (SoA)** containers with a familiar `std::vector`-like API, including full CUDA device support.

Instead of storing objects contiguously (*Array of Structures*, AoS), this library splits each field into its own packed array — a layout that dramatically improves cache efficiency and SIMD/GPU throughput for field-heavy workloads.

---

## Features

- **Zero boilerplate** — define your element type as a plain struct of references; the library reflects its fields at compile time using C++20 structured bindings
- **Nested structs** — fields can be nested aggregates, flattened automatically (up to 32 members per level)
- **Methods on reference types** — add computed properties, mutating helpers, and operators directly to your reference struct; OOP style is fully preserved
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

Reference structs are plain C++ structs — they can carry **methods, operators, and static helpers** just like any other type. The proxy returned by `operator[]` *is* that struct (with each reference bound into the SoA arrays), so calling a method on it directly manipulates the in-place SoA data.

Three patterns are demonstrated in [`examples/oop_methods.cpp`](examples/oop_methods.cpp):

### Pattern 1 — computed properties and mutation helpers

```cpp
struct Vec3Ref {
    float& x, &y, &z;

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

soa::soa_vector<Vec3Ref> vecs(1000);
vecs[0] = {3.f, 4.f, 0.f};   // length == 5
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
> Because `operator[]` returns the reference struct by value (its members are references into the arrays), methods called on it operate directly on the SoA storage — no copies, no indirection.

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
  oop_methods.cpp         # Methods / operators / domain wrappers on reference types
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
