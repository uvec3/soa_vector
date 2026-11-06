#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include "../src/soa_cuda.hpp"


struct Position
{
    float& x;
    float& y;
    float& z;

    __device__ __host__
    void move(float dx, float dy, float dz)
    {
        x+=dx;
        y+=dy;
        z+=dz;
    }
};

struct TypeRef
{
    float& a;
    int& b;
    Position pos;
};


__global__ void kernel(soa::soa_view<TypeRef> vec)
{
    auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<vec.size())
    {
        vec[i].pos.move(1,1,1);
    }
}

TEST (SOA_VECTOR_CUDA, Basic)
{
    soa::soa_vector<TypeRef> vec_h(1000);

    for (int i=0;i<vec_h.size();++i)
    {
        vec_h[i].pos.x=i;
        vec_h[i].pos.y=2*i;
        vec_h[i].pos.z=3*i;
    }

    soa::cuda_device_vector vec_d=vec_h;//copy data to device

    //run kernel
    kernel<<<(vec_d.size()-1)/128+1,128>>>(vec_d);

    //copy data back to host
    vec_h=vec_d;

    //verify
    for (int i=0;i<vec_h.size();++i)
    {
        GTEST_ASSERT_EQ(vec_h[i].pos.x,i+1);
        GTEST_ASSERT_EQ(vec_h[i].pos.y,2*i+1);
        GTEST_ASSERT_EQ(vec_h[i].pos.z,3*i+1);
    }
}

TEST(SOA_VECTOR_CUDA, ThrustSortByKey)
{
    constexpr int N = 1000;
    soa::soa_vector<TypeRef> vec_h(N);
    std::vector<int> keys_h(N);

    for (int i = 0; i < N; ++i)
    {
        // Reverse keys so sorting puts elements in reversed order (N - 1 - i)
        keys_h[i] = N - 1 - i;
        vec_h[i].a = static_cast<float>(i);
        vec_h[i].b = i * 10;
        vec_h[i].pos.x = static_cast<float>(i * 2);
        vec_h[i].pos.y = static_cast<float>(i * 3);
        vec_h[i].pos.z = static_cast<float>(i * 4);
    }

    soa::cuda_device_vector<TypeRef> vec_d = vec_h;
    thrust::device_vector<int> keys_d(keys_h.begin(), keys_h.end());

    auto values_begin = thrust::make_zip_iterator(vec_d.get_tuple());
    thrust::sort_by_key(thrust::device, keys_d.begin(), keys_d.end(), values_begin);

    vec_h = vec_d;

    for (int i = 0; i < N; ++i)
    {
        int orig = N - 1 - i;
        GTEST_ASSERT_EQ(vec_h[i].a, static_cast<float>(orig));
        GTEST_ASSERT_EQ(vec_h[i].b, orig * 10);
        GTEST_ASSERT_EQ(vec_h[i].pos.x, static_cast<float>(orig * 2));
        GTEST_ASSERT_EQ(vec_h[i].pos.y, static_cast<float>(orig * 3));
        GTEST_ASSERT_EQ(vec_h[i].pos.z, static_cast<float>(orig * 4));
    }
}

struct PositionAoS
{
    float x;
    float y;
    float z;

    __device__ __host__
    void move(float dx, float dy, float dz)
    {
        x += dx;
        y += dy;
        z += dz;
    }
};

struct TypeAoS
{
    float a;
    int b;
    PositionAoS pos;
};

__global__ void kernel_aos(soa::soa_view<TypeAoS> vec)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vec.size())
    {
        vec[i].pos.move(1, 1, 1);
    }
}

TEST(SOA_VECTOR_CUDA, BasicAoS)
{
    soa::soa_vector<TypeAoS> vec_h(1000);

    for (int i = 0; i < vec_h.size(); ++i)
    {
        vec_h[i].pos.x = i;
        vec_h[i].pos.y = 2 * i;
        vec_h[i].pos.z = 3 * i;
    }

    soa::cuda_device_vector vec_d = vec_h;

    kernel_aos<<<(vec_d.size() - 1) / 128 + 1, 128>>>(vec_d);

    vec_h = vec_d;

    for (int i = 0; i < vec_h.size(); ++i)
    {
        GTEST_ASSERT_EQ(vec_h[i].pos.x, i + 1);
        GTEST_ASSERT_EQ(vec_h[i].pos.y, 2 * i + 1);
        GTEST_ASSERT_EQ(vec_h[i].pos.z, 3 * i + 1);
    }
}

TEST(SOA_VECTOR_CUDA, ThrustSortByKeyAoS)
{
    constexpr int N = 1000;
    soa::soa_vector<TypeAoS> vec_h(N);
    std::vector<int> keys_h(N);

    for (int i = 0; i < N; ++i)
    {
        keys_h[i] = N - 1 - i;
        vec_h[i].a = static_cast<float>(i);
        vec_h[i].b = i * 10;
        vec_h[i].pos.x = static_cast<float>(i * 2);
        vec_h[i].pos.y = static_cast<float>(i * 3);
        vec_h[i].pos.z = static_cast<float>(i * 4);
    }

    soa::cuda_device_vector<TypeAoS> vec_d = vec_h;
    thrust::device_vector<int> keys_d(keys_h.begin(), keys_h.end());

    auto values_begin = thrust::make_zip_iterator(vec_d.get_tuple());
    thrust::sort_by_key(thrust::device, keys_d.begin(), keys_d.end(), values_begin);

    vec_h = vec_d;

    for (int i = 0; i < N; ++i)
    {
        int orig = N - 1 - i;
        GTEST_ASSERT_EQ(vec_h[i].a, static_cast<float>(orig));
        GTEST_ASSERT_EQ(vec_h[i].b, orig * 10);
        GTEST_ASSERT_EQ(vec_h[i].pos.x, static_cast<float>(orig * 2));
        GTEST_ASSERT_EQ(vec_h[i].pos.y, static_cast<float>(orig * 3));
        GTEST_ASSERT_EQ(vec_h[i].pos.z, static_cast<float>(orig * 4));
    }
}

TEST(SOA_VECTOR_CUDA, SpanCrossAllocatorCopy)
{
    soa::cuda_device_vector<TypeAoS> d_src(10);
    soa::soa_vector<TypeAoS> h_src(10);
    for (int i = 0; i < 10; ++i)
    {
        h_src[i] = TypeAoS{static_cast<float>(i), i * 10, {static_cast<float>(i), static_cast<float>(i * 2), static_cast<float>(i * 3)}};
    }
    d_src = h_src;

    // Modify subrange [2, 7) on device using view(offset, size)
    kernel_aos<<<1, 32>>>(d_src.view(2, 5));

    // Copy d_src into a larger device vector at offset 20 via soa_span operator= and range(offset)
    soa::cuda_device_vector<TypeAoS> d_dst(40);
    d_dst.range(20) = d_src;

    // Download subrange [20, 30) back to host pool at offset 5 via soa_span operator=
    soa::soa_vector<TypeAoS> h_dst(20);
    h_dst.range(5) = d_dst.range(20, 10);

    // Also test assigning and constructing a vector directly from a span
    soa::soa_vector<TypeAoS> h_slice;
    h_slice = d_dst.range(20, 10);
    GTEST_ASSERT_EQ(h_slice.size(), 10u);

    soa::soa_vector<TypeAoS> h_constructed = d_dst.range(22, 5);
    GTEST_ASSERT_EQ(h_constructed.size(), 5u);

    // Same-allocator span-to-span assignment
    h_dst.range(15) = h_slice.range(0, 5);

    for (int i = 0; i < 10; ++i)
    {
        float delta = (i >= 2 && i < 7) ? 1.0f : 0.0f;
        GTEST_ASSERT_EQ(h_dst[5 + i].b, i * 10);
        GTEST_ASSERT_EQ(h_dst[5 + i].pos.x, static_cast<float>(i) + delta);
        GTEST_ASSERT_EQ(h_dst[5 + i].pos.y, static_cast<float>(i * 2) + delta);
        GTEST_ASSERT_EQ(h_dst[5 + i].pos.z, static_cast<float>(i * 3) + delta);
        GTEST_ASSERT_EQ(h_slice[i].pos.x, static_cast<float>(i) + delta);
    }
    for (int i = 0; i < 5; ++i)
    {
        GTEST_ASSERT_EQ(h_dst[15 + i].b, h_slice[i].b);
        GTEST_ASSERT_EQ(h_constructed[i].b, h_slice[2 + i].b);
    }
}
