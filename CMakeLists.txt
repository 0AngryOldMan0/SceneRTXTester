cmake_minimum_required(VERSION 3.27)

project(SceneRTXTester LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------
# Флаги для конкретной сборки
# -----------------------

option(USE_CUDA_RENDERER  "Enable CUDA renderer for NVIDIA GPUs" OFF)
option(USE_HIP_RENDERER   "Enable HIP renderer for AMD GPUs" OFF)
option(USE_METAL_RENDERER "Enable Metal renderer for APPLE GPU" ON)

# -----------------------
# Языки по платформам
# -----------------------
if(APPLE AND USE_METAL_RENDERER)
    enable_language(OBJCXX)
endif()

if(WIN32 AND USE_HIP_RENDERER)
    # Включаем язык HIP только на Windows, когда он реально нужен.
    enable_language(HIP)

    # RX 6900 XT (Navi 21 / RDNA2) -> архитектура gfx1030.
    # Если хочешь другую, переопредели CMAKE_HIP_ARCHITECTURES
    # через -D CMAKE_HIP_ARCHITECTURES=gfxXXXX при конфигурации.
    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx1030")
    endif()
endif()

if(WIN32 AND USE_CUDA_RENDERER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# -----------------------
# Общие исходники (C++)
# -----------------------
set(SRC_COMMON
    Source/main.cpp
    Source/Common/SceneMetaLoader.cpp
    Source/Common/BVHBuilder.cpp
    Source/Common/Camera.cpp
    Source/Common/OBJLoader.cpp
    Source/Common/Renderer.cpp
    Source/Common/RenderManager.cpp
    Source/Common/Scene.cpp
    Source/Common/SceneLoader.cpp
    Source/Common/SceneLoaderFactory.cpp
    Source/Common/SceneObject.cpp
    Source/Common/SceneJSONLoader.cpp
)

# -----------------------
# Metal-часть (APPLE macOS)
# -----------------------

if(APPLE AND USE_METAL_RENDERER)
    list(APPEND SRC_COMMON
        Source/macOSSpecific/MetalRenderer.cpp
        Source/macOSSpecific/MetalRenderer.mm
    )

    # Подготовка сборки Metal-шейдера
    set(METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/Shader/macOS/RayTrace.metal)
    set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/RayTrace.air)
    set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/RayTrace.metallib)

    add_custom_command(
        OUTPUT ${METAL_AIR}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SRC} -o ${METAL_AIR}
        DEPENDS ${METAL_SRC}
        COMMENT "Компиляция Metal шейдера в AIR: ${METAL_SRC}"
        VERBATIM
    )

    add_custom_command(
        OUTPUT ${METAL_LIB}
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
        DEPENDS ${METAL_AIR}
        COMMENT "Сборка RayTrace.metallib"
        VERBATIM
    )

    add_custom_target(Shader ALL DEPENDS ${METAL_LIB})
endif()

# -----------------------
# HIP-часть (Windows / AMD)
# -----------------------
if(WIN32 AND USE_HIP_RENDERER)
    set(SRC_HIP
        Shader/windows/AMD/RayTraceHIP.hip.cpp
        Source/windowsSpecific/AMD/HIPRenderer.cpp
        Source/windowsSpecific/AMD/HIPRendererFunctions.cpp
        Headers/HIPRendererFunctions.h
    )

    set_source_files_properties(Shader/windows/AMD/RayTraceHIP.hip.cpp
        PROPERTIES LANGUAGE HIP
    )
endif()

# -----------------------
# CUDA-часть (Windows / NVIDIA)
# -----------------------
if(WIN32 AND USE_CUDA_RENDERER)
    set(SRC_CUDA
        Headers/CUDARendererFunctions.h
        Shader/windows/NVIDIA/RayTraceCuda.cu
        Source/windowsSpecific/NVIDIA/CUDARendererFunctions.cpp
        Source/windowsSpecific/NVIDIA/CUDARenderer.cpp
    )

    set_source_files_properties(Shader/windows/NVIDIA/RayTraceCuda.cu
        PROPERTIES LANGUAGE CUDA
    )

    list(APPEND SRC_COMMON ${SRC_CUDA})
endif()

# -----------------------
# Главный executable
# -----------------------
add_executable(SceneRTXTester
    ${SRC_COMMON}
    $<$<AND:$<BOOL:${WIN32}>,$<BOOL:${USE_HIP_RENDERER}>>:${SRC_HIP}>
)

target_compile_definitions(SceneRTXTester PRIVATE
    $<$<BOOL:${USE_METAL_RENDERER}>:USE_METAL_RENDERER>
    $<$<BOOL:${USE_HIP_RENDERER}>:USE_HIP_RENDERER>
    $<$<BOOL:${USE_CUDA_RENDERER}>:USE_CUDA_RENDERER>
)

# -----------------------
# Заголовки и сторонние библиотеки
# -----------------------
target_include_directories(SceneRTXTester PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Headers
    ${CMAKE_CURRENT_SOURCE_DIR}/Headers/Classes
    ${CMAKE_CURRENT_SOURCE_DIR}/Headers/Structures
    ${CMAKE_CURRENT_SOURCE_DIR}/Headers/ShaderSpecific
    ${CMAKE_CURRENT_SOURCE_DIR}/ExternalLibs
    ${CMAKE_CURRENT_SOURCE_DIR}/ExternalLibs/json-develop/single_include
    ${CMAKE_CURRENT_SOURCE_DIR}/ExternalLibs/stb-master
)

# -----------------------
# Компиляционные флаги
# -----------------------
if(MSVC)
    target_compile_options(SceneRTXTester PRIVATE /W4 /permissive-)
else()
    target_compile_options(SceneRTXTester PRIVATE -Wall -Wextra -Wpedantic)
endif()

# -----------------------
# Metal: линковка и дефайны (macOS)
# -----------------------
if(APPLE AND USE_METAL_RENDERER)
    target_link_libraries(SceneRTXTester PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework AppKit"
        "-framework MetalKit"
    )

    target_compile_definitions(SceneRTXTester PRIVATE USE_METAL_RENDERER=1)
endif()

# -----------------------
# HIP: линковка и дефайны (Windows / AMD)
# -----------------------
if(WIN32 AND USE_HIP_RENDERER)
    # HIP SDK / ROCm ставит CMake-конфиг "hip".
    # На Windows он обычно ищется через HIP_PATH, проброшенный в CMAKE_PREFIX_PATH:
    #   cmake -D CMAKE_PREFIX_PATH=%HIP_PATH% ...
    find_package(hip REQUIRED CONFIG)

    target_link_libraries(SceneRTXTester PRIVATE hip::device)
    target_compile_definitions(SceneRTXTester PRIVATE USE_HIP_RENDERER=1)
endif()

# -----------------------
# CUDA: дефайны (Windows / NVIDIA)
# -----------------------
if(WIN32 AND USE_CUDA_RENDERER)
    target_compile_definitions(SceneRTXTester PRIVATE USE_CUDA_RENDERER=1)
endif()
