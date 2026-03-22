Project for a custom ray tracer.

## CMake presets

The repository now includes cross-platform CMake presets:

- `windows-msvc-debug` / `windows-msvc-release`: base Windows build without a GPU backend
- `windows-hip-debug` / `windows-hip-release`: AMD/ROCm HIP build on Windows
- `windows-cuda-debug` / `windows-cuda-release`: NVIDIA CUDA build on Windows
- `macos-debug` / `macos-release`: macOS Metal build

The macOS presets still keep `USE_METAL_RENDERER=ON` and do not depend on any Windows-only files or tools.

## Windows setup

Open VSCode from the Visual Studio 2022 developer environment, or let CMake Tools use the VS developer environment automatically.

If CMake was already configured with the wrong compiler, clear the old cache first:

```powershell
Remove-Item -Recurse -Force .\build
```

Then in VSCode:

1. Run `CMake: Select Configure Preset`
2. Choose the backend you want, usually `windows-hip-debug`
3. Run `CMake: Configure`
4. Run `CMake: Build`

## Backend notes

- HIP on Windows is built through `hipcc` and linked into the normal MSVC build. This avoids fragile `enable_language(HIP)` detection issues in ROCm on Windows.
- The verified AMD preset in this workspace is `windows-hip-debug` with ROCm 6.4 and `gfx1030`.
- CUDA presets are wired up, but they require a local CUDA Toolkit installation with `nvcc` available in `PATH`, or `CMAKE_CUDA_COMPILER` set explicitly.
