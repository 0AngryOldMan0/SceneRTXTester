cmake_minimum_required(VERSION 3.27)

project(SceneRTXTester LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SCENERTX_COMPAT_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")

# -----------------------
# Флаги для конкретной сборки
# -----------------------

set(SCENERTX_DEFAULT_CUDA OFF)
set(SCENERTX_DEFAULT_HIP OFF)
set(SCENERTX_DEFAULT_METAL OFF)

if(APPLE)
    set(SCENERTX_DEFAULT_METAL ON)
endif()

option(USE_CUDA_RENDERER  "Enable CUDA renderer for NVIDIA GPUs" ${SCENERTX_DEFAULT_CUDA})
option(USE_HIP_RENDERER   "Enable HIP renderer for AMD GPUs" ${SCENERTX_DEFAULT_HIP})
option(USE_METAL_RENDERER "Enable Metal renderer for APPLE GPU" ${SCENERTX_DEFAULT_METAL})

if(USE_HIP_RENDERER AND USE_CUDA_RENDERER)
    message(WARNING "Building with both HIP and CUDA enabled is not validated. Prefer one Windows GPU backend per build preset.")
endif()

# -----------------------
# Языки по платформам
# -----------------------
if(APPLE AND USE_METAL_RENDERER)
    enable_language(OBJCXX)
endif()

if(WIN32 AND USE_HIP_RENDERER)
    # Включаем язык HIP только на Windows, когда он реально нужен.

    # RX 6900 XT (Navi 21 / RDNA2) -> архитектура gfx1030.
    # Если хочешь другую, переопредели CMAKE_HIP_ARCHITECTURES
    # через -D CMAKE_HIP_ARCHITECTURES=gfxXXXX при конфигурации.
    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx1030" CACHE STRING "AMD GPU architectures for HIP compilation")
    endif()
endif()

if(WIN32 AND USE_CUDA_RENDERER)
    if(NOT CMAKE_CUDA_COMPILER)
        find_program(SCENERTX_NVCC
            NAMES nvcc nvcc.exe
            HINTS "$ENV{CUDA_PATH}/bin"
        )

        if(NOT SCENERTX_NVCC)
            message(FATAL_ERROR "CUDA toolkit was not found. Install CUDA and make nvcc available via PATH or set CMAKE_CUDA_COMPILER.")
        endif()

        set(CMAKE_CUDA_COMPILER "${SCENERTX_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()

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

    set(METAL_LIB_COMPAT ${SCENERTX_COMPAT_BUILD_DIR}/RayTrace.metallib)

    add_custom_command(
        OUTPUT ${METAL_LIB_COMPAT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SCENERTX_COMPAT_BUILD_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${METAL_LIB} ${METAL_LIB_COMPAT}
        DEPENDS ${METAL_LIB}
        COMMENT "Copying RayTrace.metallib to build/RayTrace.metallib"
        VERBATIM
    )

    add_custom_target(Shader ALL DEPENDS ${METAL_LIB_COMPAT})
endif()

# -----------------------
# HIP-часть (Windows / AMD)
# -----------------------
if(WIN32 AND USE_HIP_RENDERER)
    set(SRC_HIP
        Source/windowsSpecific/AMD/HIPRenderer.cpp
        Source/windowsSpecific/AMD/HIPRendererFunctions.cpp
        Headers/ShaderSpecific/HIPRendererFunctions.h
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

add_custom_command(TARGET SceneRTXTester POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SCENERTX_COMPAT_BUILD_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:SceneRTXTester>"
        "${SCENERTX_COMPAT_BUILD_DIR}/SceneRTXTester${CMAKE_EXECUTABLE_SUFFIX}"
    COMMENT "Copying SceneRTXTester to build/SceneRTXTester${CMAKE_EXECUTABLE_SUFFIX}"
    VERBATIM
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
    if(NOT HIP_PATH)
        set(HIP_PATH "$ENV{HIP_PATH}")
    endif()
    if(NOT HIP_PATH)
        message(FATAL_ERROR "HIP_PATH is not set. Open VSCode from the Visual Studio developer environment or define HIP_PATH explicitly.")
    endif()

    file(TO_CMAKE_PATH "${HIP_PATH}" HIP_PATH)
    set(SCENERTX_HIPCC "${HIP_PATH}/bin/hipcc.bat")
    if(NOT EXISTS "${SCENERTX_HIPCC}")
        message(FATAL_ERROR "hipcc was not found at ${SCENERTX_HIPCC}")
    endif()

    set(SCENERTX_HIP_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/Shader/windows/AMD/RayTraceHIP.hip.cpp")
    set(SCENERTX_HIP_KERNEL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/hip/RayTraceHIP.obj")
    set(SCENERTX_HIP_OFFLOAD_ARGS)
    set(SCENERTX_HIP_HOST_ARGS)
    foreach(arch IN LISTS CMAKE_HIP_ARCHITECTURES)
        list(APPEND SCENERTX_HIP_OFFLOAD_ARGS "--offload-arch=${arch}")
    endforeach()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND SCENERTX_HIP_HOST_ARGS
            -D_DEBUG
            -D_DLL
            -D_MT
            -D_ITERATOR_DEBUG_LEVEL=2
            -g
            -Xclang
            -gcodeview
            -Xclang
            --dependent-lib=msvcrtd
        )
    else()
        list(APPEND SCENERTX_HIP_HOST_ARGS
            -DNDEBUG
            -D_DLL
            -D_MT
            -D_ITERATOR_DEBUG_LEVEL=0
            -O2
            -Xclang
            --dependent-lib=msvcrt
        )
    endif()

    add_custom_command(
        OUTPUT "${SCENERTX_HIP_KERNEL_OBJ}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/hip"
        COMMAND "${SCENERTX_HIPCC}"
            -std=c++17
            -c "${SCENERTX_HIP_KERNEL_SRC}"
            -o "${SCENERTX_HIP_KERNEL_OBJ}"
            -I"${CMAKE_CURRENT_SOURCE_DIR}/Headers"
            -I"${CMAKE_CURRENT_SOURCE_DIR}/Headers/Classes"
            -I"${CMAKE_CURRENT_SOURCE_DIR}/Headers/Structures"
            -I"${CMAKE_CURRENT_SOURCE_DIR}/Headers/ShaderSpecific"
            -I"${CMAKE_CURRENT_SOURCE_DIR}/ExternalLibs"
            ${SCENERTX_HIP_HOST_ARGS}
            ${SCENERTX_HIP_OFFLOAD_ARGS}
        DEPENDS "${SCENERTX_HIP_KERNEL_SRC}"
        COMMENT "Compiling HIP kernel with hipcc"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )

    target_sources(SceneRTXTester PRIVATE "${SCENERTX_HIP_KERNEL_OBJ}")
    set_source_files_properties("${SCENERTX_HIP_KERNEL_OBJ}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    target_link_libraries(SceneRTXTester PRIVATE hip::host)
    target_compile_definitions(SceneRTXTester PRIVATE USE_HIP_RENDERER=1)
endif()

# -----------------------
# CUDA: дефайны (Windows / NVIDIA)
# -----------------------
if(WIN32 AND USE_CUDA_RENDERER)
    target_compile_definitions(SceneRTXTester PRIVATE USE_CUDA_RENDERER=1)
endif()
