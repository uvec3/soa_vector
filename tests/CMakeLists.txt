include(FetchContent)
FetchContent_Declare(   googletest   URL https://github.com/google/googletest/archive/refs/heads/main.zip )
FetchContent_MakeAvailable(googletest)
enable_testing()


add_executable(soa_vector_basic_tests soa_vector_basic_tests.cpp)
target_link_libraries(soa_vector_basic_tests PRIVATE gtest_main)

add_executable(soa_pool_tests soa_pool_tests.cpp ../src/LogicalAllocators/OffsetAllocator/offsetAllocator.cpp)
target_link_libraries(soa_pool_tests PRIVATE gtest_main )