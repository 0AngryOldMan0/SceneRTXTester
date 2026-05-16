Project for a custom ray tracer.

### macOS (Metal)

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUSE_METAL_RENDERER=ON \
  -DUSE_HIP_RENDERER=OFF \
  -DUSE_CUDA_RENDERER=OFF

cmake --build build
./build/SceneRTXTester Scene/UE5/SubwayTunnel/scene.json -preview 1920 1080 10
```

### Windows (HIP)

```powershell
cmake -S . -B build `
  -DCMAKE_BUILD_TYPE=Debug `
  -DUSE_HIP_RENDERER=ON `
  -DUSE_CUDA_RENDERER=OFF `
  -DUSE_METAL_RENDERER=OFF `
  -DCMAKE_PREFIX_PATH=$env:HIP_PATH `
  -DCMAKE_HIP_ARCHITECTURES=gfx1030

cmake --build build
.\build\SceneRTXTester.exe Scene/UE5/SubwayTunnel/scene.json -preview 1920 1080 10
```

## Launch Argument Order

Keep this order:

```text
<binary> <scene_path> <render_type> <resolution_and_optional_spp>
```

Examples:

```text
./build/SceneRTXTester Scene/UE5/SubwayTunnel/scene.json -preview
./build/SceneRTXTester Scene/UE5/SubwayTunnel/scene.json -progressive 1920 1080
./build/SceneRTXTester Scene/UE5/SubwayTunnel/scene.json -preview 1920 1080 10
```

## VS Code Tasks

Available tasks in `.vscode/tasks.json`:

- `Configure (Debug, platform backend)`
- `Build (cmake --build build)`
- `Clean + Build`
- `Run SceneRTXTester (integrated terminal)`
- `Run SceneRTXTester (preview)`
- `Run SceneRTXTester (progressive)`

Recommended flow:

1. Run `Configure (Debug, platform backend)` once (or just run a Run-task; it triggers build automatically).
2. Run `Build (cmake --build build)`.
3. Run any `Run SceneRTXTester ...` task.

## Notes

- On macOS, `USE_METAL_RENDERER=ON` is used.
- On Windows, the configured terminal/session must have HIP toolchain available (`HIP_PATH` must be set for HIP builds).
- The executable is copied to a stable path:
  - macOS: `./build/SceneRTXTester`
  - Windows: `.\build\SceneRTXTester.exe`
