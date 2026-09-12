llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-adapter.h
    llama-arch.cpp
    llama-arch.h
    llama-batch.cpp
    llama-batch.h
    llama-chat.cpp
    llama-chat.h
    llama-context-memory.cpp
    llama-context-memory.h
    llama-context-workspace.cpp
    llama-context-workspace.h
    llama-context.cpp
    llama-context.h
    llama-cparams.cpp
    llama-cparams.h
    llama-ext.h
    llama-grammar.cpp
    llama-grammar.h
    llama-graph.cpp
    llama-graph.h
    llama-hparams.cpp
    llama-hparams.h
    llama-impl.cpp
    llama-impl.h
    llama-io.cpp
    llama-io.h
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-dsa-iswa.h
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa.h
    llama-kv-cache-dsv4.cpp
    llama-kv-cache-dsv4.h
    llama-kv-cache-iswa.cpp
    llama-kv-cache-iswa.h
    llama-kv-cache-msa.cpp
    llama-kv-cache-msa.h
    llama-kv-cache.cpp
    llama-kv-cache.h
    llama-kv-cells.h
    llama-kv-stream-binding.cpp
    llama-kv-stream-binding.h
    llama-kv-stream-config.cpp
    llama-kv-stream-config.h
    llama-kv-stream-content.cpp
    llama-kv-stream-content.h
    llama-kv-stream-host.cpp
    llama-kv-stream-host.h
    llama-kv-stream-policy.cpp
    llama-kv-stream-policy.h
    llama-kv-stream-resident.cpp
    llama-kv-stream-resident.h
    llama-memory-executor-cuda.cpp
    llama-memory-executor-cuda.h
    llama-memory-executor.cpp
    llama-memory-executor.h
    llama-memory-hybrid-idx.cpp
    llama-memory-hybrid-idx.h
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-iswa.h
    llama-memory-hybrid.cpp
    llama-memory-hybrid.h
    llama-memory-layout.cpp
    llama-memory-layout.h
    llama-memory-plan.cpp
    llama-memory-plan.h
    llama-memory-recurrent.cpp
    llama-memory-recurrent.h
    llama-memory-requirements.cpp
    llama-memory-requirements.h
    llama-memory-transition.cpp
    llama-memory-transition.h
    llama-memory-workspace.cpp
    llama-memory-workspace.h
    llama-memory.cpp
    llama-memory.h
    llama-moe-cache.cpp
    llama-mmap.cpp
    llama-mmap.h
    llama-model-loader.cpp
    llama-model-loader.h
    llama-model-saver.cpp
    llama-model-saver.h
    llama-model.cpp
    llama-model.h
    llama-quant.cpp
    llama-quant.h
    llama-sampler.cpp
    llama-sampler.h
    llama-vocab.cpp
    llama-vocab.h
    unicode-data.cpp
    unicode-data.h
    unicode.cpp
    unicode.h
)

add_library(llama
            ../include/llama.h
            ${LLAMA_CORE_SOURCES}
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ../include/llama.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

target_include_directories(llama PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
