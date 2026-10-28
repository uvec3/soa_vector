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
