#include <gtest/gtest.h>
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
