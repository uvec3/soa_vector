// examples/oop_methods.cpp
//
// Demonstrates that reference structs used with soa_vector can carry ordinary
// C++ methods, keeping OOP-style encapsulation while the data sits in the
// cache-friendly SoA layout underneath.
//
// Three progressively richer patterns are shown:
//   1. Plain methods          – computed properties, mutation helpers
//   2. Nested reference type  – sub-object methods compose naturally
//   3. Static factory / named constructor on the *vector* via a thin wrapper

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "../src/soa_vector.hpp"

// ---------------------------------------------------------------------------
// Pattern 1 – a Vec3 sub-object with full arithmetic helpers
// ---------------------------------------------------------------------------

struct Vec3Ref {
    float& x;
    float& y;
    float& z;

    // Read-only computed property — works just like a regular member function
    float length() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    // Mutating helper
    void normalize() {
        float inv = 1.f / length();
        x *= inv;
        y *= inv;
        z *= inv;
    }

    // In-place arithmetic
    Vec3Ref& operator+=(Vec3Ref other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    void scale(float s) {
        x *= s; y *= s; z *= s;
    }

    // Named constructor via a static helper that writes into an existing slot
    static void set(Vec3Ref out, float ax, float ay, float az) {
        out.x = ax; out.y = ay; out.z = az;
    }
};

// ---------------------------------------------------------------------------
// Pattern 2 – a Particle that composes Vec3Ref and adds domain logic
// ---------------------------------------------------------------------------

struct ParticleRef {
    Vec3Ref position;   // nested aggregate — reflected automatically
    Vec3Ref velocity;
    float&  mass;
    float&  charge;

    // Derived property — reads multiple fields
    float kinetic_energy() const {
        float v2 = velocity.x*velocity.x
                 + velocity.y*velocity.y
                 + velocity.z*velocity.z;
        return 0.5f * mass * v2;
    }

    // Symplectic Euler step: p += v*dt, v += (F/m)*dt
    void integrate(float dt, float fx, float fy, float fz) {
        position.x += velocity.x * dt;
        position.y += velocity.y * dt;
        position.z += velocity.z * dt;

        velocity.x += (fx / mass) * dt;
        velocity.y += (fy / mass) * dt;
        velocity.z += (fz / mass) * dt;
    }

    // Coulomb force magnitude from another particle (read-only, const refs)
    float coulomb_force(ParticleRef other) const {
        constexpr float k = 8.99e9f;
        float dx = other.position.x - position.x;
        float dy = other.position.y - position.y;
        float dz = other.position.z - position.z;
        float r2 = dx*dx + dy*dy + dz*dz + 1e-10f; // epsilon avoids /0
        return k * charge * other.charge / r2;
    }
};

// ---------------------------------------------------------------------------
// Pattern 3 – a thin domain type that wraps soa_vector and exposes
//             named constructors / bulk operations
// ---------------------------------------------------------------------------

class ParticleSystem {
public:
    explicit ParticleSystem(std::size_t n) : m_particles(n) {}

    // Named factory: fill all slots with a uniform initial state
    static ParticleSystem uniform(std::size_t n,
                                  float mass, float charge,
                                  float vx, float vy, float vz)
    {
        ParticleSystem sys(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto p = sys[i];
            Vec3Ref::set(p.position, float(i), 0.f, 0.f);
            Vec3Ref::set(p.velocity, vx, vy, vz);
            p.mass   = mass;
            p.charge = charge;
        }
        return sys;
    }

    ParticleRef operator[](std::size_t i) { return m_particles[i]; }

    std::size_t size() const { return m_particles.size(); }

    // Bulk method: advance every particle by dt under a uniform field
    void step(float dt, float fx, float fy, float fz) {
        for (auto&& p : m_particles)
            p.integrate(dt, fx, fy, fz);
    }

    // Aggregate query — total kinetic energy
    float total_kinetic_energy() const {
        float total = 0.f;
        for (auto&& p : m_particles)
            total += p.kinetic_energy();
        return total;
    }

private:
    soa::soa_vector<ParticleRef> m_particles;
};

// ---------------------------------------------------------------------------
// main — exercise all three patterns
// ---------------------------------------------------------------------------

int main()
{
    // ---- Pattern 1: Vec3Ref methods ----------------------------------------
    {
        soa::soa_vector<Vec3Ref> vecs(4);

        Vec3Ref::set(vecs[0], 3.f, 4.f, 0.f);
        assert(std::abs(vecs[0].length() - 5.f) < 1e-5f);

        vecs[0].normalize();
        assert(std::abs(vecs[0].length() - 1.f) < 1e-5f);

        Vec3Ref::set(vecs[1], 1.f, 0.f, 0.f);
        Vec3Ref::set(vecs[2], 0.f, 1.f, 0.f);
        vecs[1] += vecs[2];                  // operator+= via proxy reference
        assert(vecs[1].x == 1.f && vecs[1].y == 1.f);

        std::cout << "Pattern 1 OK: Vec3 methods work on SoA proxies\n";
    }

    // ---- Pattern 2: ParticleRef domain methods ------------------------------
    {
        soa::soa_vector<ParticleRef> particles(3);

        // Init via methods on the proxy — no special syntax needed
        auto p0 = particles[0];
        Vec3Ref::set(p0.position, 0.f, 0.f, 0.f);
        Vec3Ref::set(p0.velocity, 1.f, 0.f, 0.f);
        p0.mass   = 2.f;
        p0.charge = 1.f;

        float ke_before = p0.kinetic_energy(); // 0.5 * 2 * 1 = 1.0
        assert(std::abs(ke_before - 1.f) < 1e-5f);

        // Integrate one step under a force in x-direction
        p0.integrate(0.1f, 10.f, 0.f, 0.f);  // a = F/m = 5, dv = 0.5
        // position.x moved, velocity.x increased
        assert(particles[0].position.x > 0.f);
        assert(particles[0].velocity.x > 1.f);

        // Coulomb force between two slots
        auto p1 = particles[1];
        Vec3Ref::set(p1.position, 1.f, 0.f, 0.f);
        p1.mass = 1.f; p1.charge = -1.f;
        float F = particles[0].coulomb_force(particles[1]);
        assert(F < 0.f); // attractive (opposite charges)

        std::cout << "Pattern 2 OK: ParticleRef domain methods work\n";
        std::cout << "  KE before integrate: " << ke_before << " J\n";
        std::cout << "  Coulomb force p0→p1: " << F << " N\n";
    }

    // ---- Pattern 3: ParticleSystem thin wrapper ----------------------------
    {
        // Named constructor builds the SoA internally
        auto sys = ParticleSystem::uniform(
            1000,
            /*mass=*/   1.f,
            /*charge=*/ 1.f,
            /*v=*/      0.f, 1.f, 0.f
        );

        float ke0 = sys.total_kinetic_energy();   // 1000 * 0.5*1*1 = 500
        assert(std::abs(ke0 - 500.f) < 1e-2f);

        // Advance under gravity-like force in -y
        constexpr int   STEPS = 100;
        constexpr float DT    = 0.01f;
        for (int i = 0; i < STEPS; ++i)
            sys.step(DT, 0.f, -9.81f, 0.f);

        // After 100 × 0.01 s the y-velocity should have decreased by ~9.81
        float vy_final = sys[0].velocity.y;
        assert(std::abs(vy_final - (1.f - 9.81f)) < 1e-2f);

        std::cout << "Pattern 3 OK: ParticleSystem bulk methods work\n";
        std::cout << "  Total KE at t=0: " << ke0 << " J\n";
        std::cout << "  Vy after " << STEPS << " steps: " << vy_final << " m/s\n";
    }

    std::cout << "\nAll assertions passed.\n";
    return 0;
}
