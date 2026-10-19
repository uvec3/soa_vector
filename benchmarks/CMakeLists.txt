add_executable(MoveVertexAoS MoveVertexAoS.cu)
add_executable(MoveVertexAoSAligned MoveVertexAoSAligned.cu)
add_executable(MoveVertexCSoA MoveVertexCSoA.cu)
add_executable(MoveVertexSoA MoveVertexSoA.cu)
add_executable(MoveVertexAoSoA MoveVertexAoSoA.cu)

add_executable(MoveVertexAoS_Random MoveVertexAoS_Random.cu)
add_executable(MoveVertexAoSAligned_Random MoveVertexAoSAligned_Random.cu)
add_executable(MoveVertexCSoA_Random MoveVertexCSoA_Random.cu)
add_executable(MoveVertexSoA_Random MoveVertexSoA_Random.cu)
add_executable(MoveVertexAoSoA_Random MoveVertexAoSoA_Random.cu)


target_compile_options(MoveVertexAoS
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_compile_options(MoveVertexAoSAligned
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_compile_options(MoveVertexCSoA
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_compile_options(MoveVertexSoA
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_compile_options(MoveVertexAoSoA
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_compile_options(MoveVertexAoS_Random
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_compile_options(MoveVertexAoSAligned_Random
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_compile_options(MoveVertexCSoA_Random
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_compile_options(MoveVertexSoA_Random
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_compile_options(MoveVertexAoSoA_Random
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)