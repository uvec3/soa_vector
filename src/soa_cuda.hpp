#pragma once

#include <cuda_runtime_api.h>
#include <cuda/std/tuple>
#include <iostream>

#include "aosoa_vector.hpp"
#include "soa_vector.hpp"

namespace soa
{
    struct CudaDeviceAllocator
    {
        inline static  void* alloc(size_t bytes)
        {
            void *ptr;
            CUDA_CHECK(cudaMalloc(&ptr,bytes));
            return ptr;
        }

        inline static void free(void* ptr)
        {
            CUDA_CHECK(cudaFree(ptr));
        }

        inline static void copy(void* dst, const void* src, size_t bytes)
        {
            CUDA_CHECK(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice));
        }
    };


    struct CudaHostAllocator
    {
        inline static  void* alloc(size_t bytes)
        {
            void *ptr;
            CUDA_CHECK(cudaMallocHost(&ptr,bytes));
            return ptr;
        }

        inline static void free(void* ptr)
        {
            CUDA_CHECK(cudaFreeHost(ptr));
        }

        inline static void copy(void* dst, const void* src, size_t bytes)
        {
            CUDA_CHECK(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToHost));
        }
    };


    struct CudaUnifiedAllocator
    {
        inline static  void* alloc(size_t bytes)
        {
            void *ptr;
            CUDA_CHECK(cudaMallocManaged(&ptr,bytes));

            int current_dev;
            CUDA_CHECK(cudaGetDevice(&current_dev));

            // Check if the device supports Concurrent Managed Access (Page Migration / Advise)
            int concurrentManagedAccess = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, current_dev));

            if (concurrentManagedAccess)
            {
                // 1. Hint to keep the physical memory on the GPU
                cudaMemLocation deviceLoc = {.type = cudaMemLocationTypeDevice, .id = current_dev};
                CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, deviceLoc));

                // 2. Hint to set up direct CPU mapping to avoid page migrations upon access
                cudaMemLocation hostLoc = {.type = cudaMemLocationTypeHost, .id = 0};
                CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, hostLoc));

                // // (Optional) Proactively move the memory to the GPU before the kernel starts
                // cudaMemPrefetchAsync(ptr, bytes, deviceLoc, 0, stream);
            }

            return ptr;
        }

        inline static void free(void* ptr)
        {
            CUDA_CHECK(cudaFree(ptr));
        }

        inline static void copy(void* dst, const void* src, size_t bytes)
        {
            CUDA_CHECK(cudaMemcpy(dst,src,bytes,cudaMemcpyDefault));
        }
    };


    // Transfer from device to host (DefaultAllocator / CudaHostAllocator)
    inline void transfer_data(void* ptr_dst, const void* ptr_src, size_t bytes, DefaultAllocator, CudaDeviceAllocator)
    {
        CUDA_CHECK(cudaMemcpy(ptr_dst, ptr_src, bytes, cudaMemcpyDeviceToHost));
    }

    inline void transfer_data(void* ptr_dst, const void* ptr_src, size_t bytes, CudaHostAllocator, CudaDeviceAllocator)
    {
        CUDA_CHECK(cudaMemcpy(ptr_dst, ptr_src, bytes, cudaMemcpyDeviceToHost));
    }

    // Transfer from host (DefaultAllocator / CudaHostAllocator) to device
    inline void transfer_data(void* ptr_dst, const void* ptr_src, size_t bytes, CudaDeviceAllocator, DefaultAllocator)
    {
        CUDA_CHECK(cudaMemcpy(ptr_dst, ptr_src, bytes, cudaMemcpyHostToDevice));
    }

    inline void transfer_data(void* ptr_dst, const void* ptr_src, size_t bytes, CudaDeviceAllocator, CudaHostAllocator)
    {
        CUDA_CHECK(cudaMemcpy(ptr_dst, ptr_src, bytes, cudaMemcpyHostToDevice));
    }

    // Transfer between host allocators (DefaultAllocator <-> CudaHostAllocator)
    inline void transfer_data(void* ptr_dst, const void* ptr_src, size_t bytes, DefaultAllocator, CudaHostAllocator)
    {
        CUDA_CHECK(cudaMemcpy(ptr_dst, ptr_src, bytes, cudaMemcpyHostToHost));
    }

    inline void transfer_data(void* ptr_dst, const void* ptr_src, size_t bytes, CudaHostAllocator, DefaultAllocator)
    {
        CUDA_CHECK(cudaMemcpy(ptr_dst, ptr_src, bytes, cudaMemcpyHostToHost));
    }

    // Fallback for transfers involving CudaUnifiedAllocator (or any other UVA-compatible pair)
    template <typename AllocatorDst, typename AllocatorSrc>
    inline void transfer_data(void* ptr_dst, const void* ptr_src, size_t bytes, AllocatorDst, AllocatorSrc)
    {
        CUDA_CHECK(cudaMemcpy(ptr_dst, ptr_src, bytes, cudaMemcpyDefault));
    }

    template <typename TReference, typename Allocator>
    class cuda_soa_vector : public soa_vector<TReference, Allocator>
    {
        using Base = soa_vector<TReference, Allocator>;

    public:
        cuda_soa_vector(size_t size = 0, Allocator&& allocator = Allocator{})
            : Base(size, std::forward<Allocator>(allocator))
        {
        }

        cuda_soa_vector(const cuda_soa_vector& other) = default;
        cuda_soa_vector(cuda_soa_vector&& other) noexcept = default;
        cuda_soa_vector& operator=(const cuda_soa_vector& other) = default;
        cuda_soa_vector& operator=(cuda_soa_vector&& other) noexcept = default;

        template <typename OtherAllocator>
        cuda_soa_vector(const soa_vector<TReference, OtherAllocator>& other)
            : Base(other)
        {
        }

        using Base::operator=;

        auto get_tuple(size_t offset = 0)
        {
            using FlatTuple = decltype(detail::tie_flat(std::declval<TReference&>()));
            constexpr std::size_t N = std::tuple_size_v<FlatTuple>;

            return [&]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                return cuda::std::make_tuple(
                    (static_cast<std::remove_reference_t<std::tuple_element_t<Is, FlatTuple>>*>(this->m_data[Is]) + offset)...
                );
            }(std::make_index_sequence<N>{});
        }

        auto get_tuple(size_t offset = 0) const
        {
            using FlatTuple = decltype(detail::tie_flat(std::declval<TReference&>()));
            constexpr std::size_t N = std::tuple_size_v<FlatTuple>;

            return [&]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                return cuda::std::make_tuple(
                    (static_cast<const std::remove_reference_t<std::tuple_element_t<Is, FlatTuple>>*>(this->m_data[Is]) + offset)...
                );
            }(std::make_index_sequence<N>{});
        }
    };

    template<typename View>
    using cuda_device_vector=cuda_soa_vector<View,CudaDeviceAllocator>;

    template<typename View>
    using cuda_host_vector=cuda_soa_vector<View,CudaHostAllocator>;

    template<typename View>
    using cuda_unified_vector=cuda_soa_vector<View,CudaUnifiedAllocator>;

    template<typename View,size_t internal_array_size=32>
    using cuda_device_aosoa = aosoa_vector<View,internal_array_size, CudaDeviceAllocator>;

    template<typename View,size_t internal_array_size=32>
    using cuda_host_aosoa = aosoa_vector<View,internal_array_size, CudaHostAllocator>;

    template<typename View,size_t internal_array_size=32>
    using cuda_unified_aosoa = aosoa_vector<View,internal_array_size, CudaUnifiedAllocator>;

}