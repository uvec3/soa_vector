include(FetchContent)
FetchContent_Declare(   googletest   URL https://github.com/google/googletest/archive/refs/heads/main.zip )
FetchContent_MakeAvailable(googletest)
enable_testing()


add_executable(soa_vector_basic_tests soa_vector_basic_tests.cpp)
target_link_libraries(soa_vector_basic_tests PRIVATE gtest_main)

add_executable(soa_vector_cuda_tests soa_vector_cuda_tests.cu)
target_link_libraries(soa_vector_cuda_tests PRIVATE gtest_main)

target_compile_options(soa_vector_cuda_tests
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)