#pragma once

inline __host__ __device__ size_t random_id(size_t i, size_t N)
{
    return (i*1111+123)% N;
}

#define NUMBER_OF_SAMPLES 10