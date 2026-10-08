#include <chrono>
#include <iostream>
#include <format>
#include <string>
#include "common.hpp"

#include "../src/soa_cuda.hpp"

struct Point
{
    float& x;
    float& y;
    float& z;

    std::string print() const
    {
        return std::to_string(x)+"\t"+std::to_string(y)+"\t"+std::to_string(z);
    }
};

struct VertexRef
{
    Point position;
    Point normal;
    float& r_color;
    float& g_color;
    float& b_color;
    float& a_color;
    float& u;
    float& v;

    void Init()
    {
        position.x=0;
        position.y=0;
        position.z=0;

        normal.x=0;
        normal.y=0;
        normal.z=0;

        r_color=1;
        g_color=1;
        b_color=1;
        a_color=1;

        u=0;
        v=0;
    }

    __device__ __host__
    inline void move(float dx, float dy, float dz)
    {
        position.x+=dx;
        position.y+=dy;
        position.z+=dz;
    }
};

__global__ void move_kernel(soa::aosoa_iterator<VertexRef,32> vertices, size_t size, float dx, float dy, float dz)
{
    auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<size)
    {
        i=random_id(i,size);
        vertices[i].move(dx,dy,dz);
    }
}


int main()
{
    std::cout<<std::setprecision(10);
    const int number_of_samples=NUMBER_OF_SAMPLES;
    const int max_N=100'000'000;

    for (size_t N=10;N<=max_N;N*=10)
    {
        std::cout<<"\n--------------- N="<<N<<" -------------\n";
        double cpu_sum_ms=0;
        double gpu_sum_ms=0;
        for (int sample=0;sample<number_of_samples;++sample)
        {
            std::cout<<"Sample "<<sample<<"\n";

            soa::aosoa_vector<VertexRef,32> vertices(N);

            //init position
            for (int i = 0; i < N; ++i)
            {
                vertices[i].Init();
                vertices[i].position.x=i;
                vertices[i].position.y=i*2;
                vertices[i].position.z=i*3;
            }
            //device copy
            soa::cuda_device_aosoa<VertexRef,32> vertices_d(vertices);
            CUDA_CHECK(cudaDeviceSynchronize());

            float dx=10;
            float dy=20;
            float dz=30;

            auto start_time=std::chrono::high_resolution_clock::now();
            for (int i = 0; i < N; ++i)
            {
                vertices[random_id(i,N)].move(dx,dy,dz);
            }

            auto passed=std::chrono::high_resolution_clock::now()-start_time;
            std::cout<<"CPU time: "<<passed.count()/1'000'000.0<<"ms\n";
            cpu_sum_ms+=passed.count()/1'000'000.0;


            cudaEvent_t start;
            cudaEvent_t finish;
            cudaEventCreate(&start);
            cudaEventCreate(&finish);

            cudaDeviceSynchronize();
            CUDA_CHECK(cudaEventRecord(start));
            move_kernel<<<(N-1)/128+1,128>>>(vertices_d.begin(),N,dx,dy,dz);
            CUDA_CHECK(cudaEventRecord(finish));
            CUDA_CHECK(cudaDeviceSynchronize());

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, finish);
            std::cout<<"GPU time: "<<milliseconds<<"ms\n";
            gpu_sum_ms+=milliseconds;

            cudaEventDestroy(start);
            cudaEventDestroy(finish);
        }
        std::cout<<"CPU average time: "<< cpu_sum_ms/number_of_samples <<"ms\n";
        std::cout<<"GPU average time: "<< gpu_sum_ms/number_of_samples <<"ms\n\n";
    }

    return 0;
}
