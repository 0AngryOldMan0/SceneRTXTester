Project for a custom ray tracer.

### Сборка на macOS (Metal)
Предварительно необходимо установить xcode, стандартный для macOS, ninja для cmake, а также применить команду:
```bash
 xcodebuild -downloadComponent MetalToolchain
```
Так как на macOS используется проприетарный графический пайплайн - Metal, отличный от HIP/CUDA, то нативно, он естественно не установлен. Для сборки проекта нужно прописать следующую команду:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUSE_METAL_RENDERER=ON \
  -DUSE_HIP_RENDERER=OFF \
  -DUSE_CUDA_RENDERER=OFF
```
где:
-S - отвечает за корневой каталог с исходным кодом проекта, где находится CMakeLists.
. - исходя из правил файловых систем отвечает за текущую директорию
-B <путь до каталога> - указывает каталог, в котором CMake сгенерирует все артефакты сборки: Makefile (или проектные файлы IDE), CMakeCache.txt, CMakeFiles и т.д.
-DCMAKE_BUILD_TYPE=Debug - отвечает за тип сборки, в нашем случае пока что дебаг режим, так как до релиза еще далеко.
Далее идет набор флагов для сборки под конкретную платформу, у соответствующего флага ставится значение "ON":
-DUSE_METAL_RENDERER=ON - если сборка под Mac
-DUSE_HIP_RENDERER=OFF - если сборка под windows с видеократой от фирмы AMD
-DUSE_CUDA_RENDERER=OFF - если сборка под windows с видеократой от фирмы NVIDIA

Для запуска программы, после сборки используем следующую команду:
```bash
cmake --build build
./build/SceneRTXTester Scene/UE5/SubwayTunnel/scene.json -preview 1920 1080 10
```
Необходимо указать путь до сгенерированного бинарника, в нашем случае он находится в папке build, указать полный путь до сцены (данное решение временное и будет изменено в дальнейшем), указать один из аргументов запуска и, при необходимости, задать разрешение рендера и число лучей на кадр.

### Сборка на Windows (HIP)
Предварительно необходимо установить HIP, через профессиональный драйвер AMD, скачав его с официального сайта, cmake, и применить схожую команду:

```powershell
cmake -S . -B build `
  -DCMAKE_BUILD_TYPE=Debug `
  -DUSE_HIP_RENDERER=ON `
  -DUSE_CUDA_RENDERER=OFF `
  -DUSE_METAL_RENDERER=OFF `
  -DCMAKE_PREFIX_PATH=$env:HIP_PATH `
  -DCMAKE_HIP_ARCHITECTURES=gfx1030
```
Дополнив ее указанием пути к установленному HIP, и архитектуру установленной видеокарты, в нашем случае RX6900XT

Команда для запуска не отличается от оной на macOS
```powershell
cmake --build build
.\build\SceneRTXTester.exe Scene/UE5/SubwayTunnel/scene.json -preview 1920 1080 10
```

## Аргументы для запуска проекта

Порядок запуска подчиняется следующему условию:
```text
<binary> <scene_path> <render_type> <resolution_and_optional_spp>
```
В качестве <render_type> можно указывать следующие параметры:
-preview - кадр с полным набором PBR текстур, и освещением
-progressive - прогрессивный кадр, в коотором учитывется повторная обработка лучей и денойзинг
-textureDebug - полный кадр, и набор кадров в которых отрисовываются только определенные текстуры (альбедо, нормали, и прочее).

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

## Тесты

В проект добавлена базовая инфраструктура `ctest` и набор стартовых unit/integration/system smoke тестов.
Для автозапуска в репозитории добавлен CI workflow: `.github/workflows/autotests.yml` (запуск на `push`/`pull_request`).

Быстрый запуск:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Подробная стратегия и карта текущего покрытия: [TESTING.md](TESTING.md).

## Performance Benchmarks

В проект добавлен отдельный CPU-only benchmark таргет `SceneRTXBenchmarks` для микробенчмарков:

- `BVHBuilder` (BottomUp / TopDown / SAH) на больших синтетических мешах.
- `OBJLoader` и `SceneJSONLoader` на больших сценах.

Запуск:

```bash
cmake -S . -B build
cmake --build build --target SceneRTXBenchmarks
./build/SceneRTXBenchmarks
```

Пример с параметрами:

```bash
./build/SceneRTXBenchmarks --iterations 12 --warmup 3 --bvh-triangles 5000
```

Экспорт результатов в JSON:

```bash
./build/SceneRTXBenchmarks \
  --iterations 12 \
  --warmup 3 \
  --json-out Results/benchmarks/bench_$(date +%Y%m%d_%H%M%S).json
```
