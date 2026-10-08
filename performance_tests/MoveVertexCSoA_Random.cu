#include <chrono>
#include <iostream>
#include <format>
#include "common.hpp"

#include "../src/soa_cuda.hpp"

struct Point
{
    float* x;
    float* y;
    float* z;
};

struct VertexSoA
{
    Point position;
    Point normal;
    float* r_color;
    float* g_color;
    float* b_color;
    float* a_color;
    float* u;
    float* v;
};


inline void Init(VertexSoA vertex_view, size_t index)
{
    vertex_view.position.x[index]=0;
    vertex_view.position.y[index]=0;
    vertex_view.position.z[index]=0;

    vertex_view.normal.x[index]=0;
    vertex_view.normal.y[index]=0;
    vertex_view.normal.z[index]=0;

    vertex_view.r_color[index]=1;
    vertex_view.g_color[index]=1;
    vertex_view.b_color[index]=1;
    vertex_view.a_color[index]=1;

    vertex_view.u[index]=0;
    vertex_view.v[index]=0;
}

__host__ __device__
__forceinline__ void move(VertexSoA& vertices, size_t index, float dx, float dy, float dz)
{
    vertices.position.x[index]+=dx;
    vertices.position.y[index]+=dy;
    vertices.position.z[index]+=dz;
}


__global__ void move_kernel(VertexSoA vertices, size_t size, float dx,float dy,float dz)
{
    auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<size)
    {
        i=random_id(i,size);
        move(vertices,i,dx,dy,dz);
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

            VertexSoA vertices;
            vertices.position.x=    (float*)malloc(sizeof(*vertices.position.x)*N);
            vertices.position.y=    (float*)malloc(sizeof(*vertices.position.y)*N);
            vertices.position.z=    (float*)malloc(sizeof(*vertices.position.z)*N);
            vertices.normal.x=      (float*)malloc(sizeof(*vertices.normal.x)*N);
            vertices.normal.y=      (float*)malloc(sizeof(*vertices.normal.y)*N);
            vertices.normal.z=      (float*)malloc(sizeof(*vertices.normal.z)*N);
            vertices.r_color=       (float*)malloc(sizeof(*vertices.r_color)*N);
            vertices.g_color=       (float*)malloc(sizeof(*vertices.g_color)*N);
            vertices.b_color=       (float*)malloc(sizeof(*vertices.b_color)*N);
            vertices.a_color=       (float*)malloc(sizeof(*vertices.a_color)*N);
            vertices.u=             (float*)malloc(sizeof(*vertices.b_color)*N);
            vertices.v=             (float*)malloc(sizeof(*vertices.a_color)*N);

            //init position
            for (int i = 0; i < N; ++i)
            {
                Init(vertices,i);
                vertices.position.x[i]=i;
                vertices.position.y[i]=i*2;
                vertices.position.z[i]=i*3;
            }


            //create device copy
            VertexSoA vertices_d{};
            cudaMalloc( &vertices_d.position.x ,sizeof(*vertices.position.x)*N);
            cudaMalloc( &vertices_d.position.y ,sizeof(*vertices.position.y)*N);
            cudaMalloc( &vertices_d.position.z ,sizeof(*vertices.position.z)*N);
            cudaMalloc( &vertices_d.normal.x   ,sizeof(*vertices.normal.x)*N);
            cudaMalloc( &vertices_d.normal.y   ,sizeof(*vertices.normal.y)*N);
            cudaMalloc( &vertices_d.normal.z   ,sizeof(*vertices.normal.z)*N);
            cudaMalloc( &vertices_d.r_color    ,sizeof(*vertices.r_color)*N);
            cudaMalloc( &vertices_d.g_color    ,sizeof(*vertices.g_color)*N);
            cudaMalloc( &vertices_d.b_color    ,sizeof(*vertices.b_color)*N);
            cudaMalloc( &vertices_d.a_color    ,sizeof(*vertices.a_color)*N);
            cudaMalloc( &vertices_d.u          ,sizeof(*vertices.b_color)*N);
            cudaMalloc( &vertices_d.v          ,sizeof(*vertices.a_color)*N);

            cudaMemcpy( vertices_d.position.x ,vertices.position.x,sizeof(*vertices.position.x)*N,cudaMemcpyHostToDevice);
            cudaMemcpy( vertices_d.position.y ,vertices.position.y,sizeof(*vertices.position.y)*N,cudaMemcpyHostToDevice);
            cudaMemcpy( vertices_d.position.z ,vertices.position.z,sizeof(*vertices.position.z)*N,cudaMemcpyHostToDevice);
            cudaMemcpy( vertices_d.normal.x   ,vertices.normal.x  ,sizeof(*vertices.normal.x)*N  ,cudaMemcpyHostToDevice);
            cudaMemcpy( vertices_d.normal.y   ,vertices.normal.y  ,sizeof(*vertices.normal.y)*N  ,cudaMemcpyHostToDevice);
            cudaMemcpy( vertices_d.normal.z   ,vertices.normal.z  ,sizeof(*vertices.normal.z)*N  ,cudaMemcpyHostToDevice);
            cudaMemcpy( vertices_d.r_color    ,vertices.r_color   ,sizeof(*vertices.r_color)*N   ,cudaMemcpyHostToDevice);
            cudaMemcpy( vertices_d.g_color    ,vertices.g_color   ,sizeof(*vertices.g_color)*N   ,cudaMemcpyHostToDevice);
            cudaMemcpy( vertices_d.b_color    ,vertices.b_color   ,sizeof(*vertices.b_color)*N   ,cudaMemcpyHostToDevice);
            cudaMemcpy( vertices_d.a_color    ,vertices.a_color   ,sizeof(*vertices.a_color)*N   ,cudaMemcpyHostToDevice);
            cudaMemcpy( vertices_d.u          ,vertices.u         ,sizeof(*vertices.b_color)*N   ,cudaMemcpyHostToDevice);
            cudaMemcpy( vertices_d.v          ,vertices.v         ,sizeof(*vertices.a_color)*N   ,cudaMemcpyHostToDevice);


            float dx=10;
            float dy=20;
            float dz=30;

            auto start_time=std::chrono::high_resolution_clock::now();
            for (int i=0;i<N;++i)
            {
                move(vertices,random_id(i,N),dx,dy,dz);
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
            move_kernel<<<(N-1)/128+1,128>>>(vertices_d,N,dx,dy,dz);
            CUDA_CHECK(cudaEventRecord(finish));
            CUDA_CHECK(cudaDeviceSynchronize());

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, finish);
            std::cout<<"GPU time: "<<milliseconds<<"ms\n";
            gpu_sum_ms+=milliseconds;

            cudaEventDestroy(start);
            cudaEventDestroy(finish);

            free(vertices.position.x);
            free(vertices.position.y);
            free(vertices.position.z);
            free(vertices.normal.x);
            free(vertices.normal.y);
            free(vertices.normal.z);
            free(vertices.r_color);
            free(vertices.g_color);
            free(vertices.b_color);
            free(vertices.a_color);
            free(vertices.u);
            free(vertices.v);

            cudaFree(vertices_d.position.x);
            cudaFree(vertices_d.position.y);
            cudaFree(vertices_d.position.z);
            cudaFree(vertices_d.normal.x);
            cudaFree(vertices_d.normal.y);
            cudaFree(vertices_d.normal.z);
            cudaFree(vertices_d.r_color);
            cudaFree(vertices_d.g_color);
            cudaFree(vertices_d.b_color);
            cudaFree(vertices_d.a_color);
            cudaFree(vertices_d.u);
            cudaFree(vertices_d.v);
        }
        std::cout<<"CPU average time: "<< cpu_sum_ms/number_of_samples <<"ms\n";
        std::cout<<"GPU average time: "<< gpu_sum_ms/number_of_samples <<"ms\n\n";
    }

    return 0;
}
