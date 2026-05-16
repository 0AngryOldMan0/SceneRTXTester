# Testing Strategy

Этот документ задаёт базовую стратегию тестирования для `SceneRTXTester` и даёт каркас, который можно постепенно расширять.

## Цели

- Быстро ловить регрессии в CPU-логике (математика, загрузка сцен, построение BVH, управление сценой).
- Удерживать короткий feedback loop через автоматические проверки в `ctest`.
- Поддерживать «пирамиду тестов»: много быстрых unit, меньше integration, минимальный набор smoke/system.

## Структура тестов

```text
tests/
  TestFramework.h
  TestMain.cpp
  unit/
  integration/
  system/
```

- `unit/`: локальные, быстрые проверки классов/функций без файловых зависимостей и без GPU.
- `integration/`: проверка связки модулей и форматов (`.obj`, `scene.json + meshes.bin`).
- `system/`: smoke-сценарии на уровне пайплайна CPU (загрузка -> сцена -> BVH).

## Что покрыто сейчас

- `unit/MathUtilsTests.cpp`: векторная математика, AABB helper-ы, пересечение луч-треугольник.
- `unit/CameraTests.cpp`: генерация лучей, стабильность базиса камеры, перемещение.
- `unit/SceneObjectTests.cpp`: материал-мэппинг треугольников, инстансинг, reset состояния.
- `unit/RenderCommandTests.cpp`: fluent-конфигурация рендера и сброс в default.
- `unit/SceneLoaderFactoryTests.cpp`: выбор загрузчика по расширению.
- `unit/RendererRegistryTests.cpp`: регистрация/создание рендерера через реестр/фабрику.
- `integration/OBJLoaderIntegrationTests.cpp`: парсинг простого OBJ, fallback-нормали.
- `integration/SceneJSONLoaderIntegrationTests.cpp`: загрузка минимальной JSON-сцены и защита от битого `meshes.bin`.
- `system/ScenePipelineSmokeTests.cpp`: smoke-проверка сборки BLAS/TLAS на двух объектах.

## Запуск локально

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Можно запускать подмножества:

```bash
ctest --test-dir build -R SceneRTXTester.UnitOnly --output-on-failure
ctest --test-dir build -R SceneRTXTester.IntegrationOnly --output-on-failure
ctest --test-dir build -R SceneRTXTester.SystemSmoke --output-on-failure
```

## Автотесты в CI

Тесты подключены к CI-пайплайну: `.github/workflows/autotests.yml`.

Что делает pipeline:

- на каждый `push` / `pull_request` запускает сборку CPU-only конфигурации;
- выполняет `ctest` (unit + integration + system smoke);
- запускает benchmark и выгружает JSON-отчёт как артефакт (`ci_benchmark.json`).

Это переводит текущие тесты из режима «запускаем вручную» в режим регулярных автопроверок на уровне репозитория.

## Performance-слой

В проект добавлен отдельный таргет микробенчмарков: `SceneRTXBenchmarks`.

Что измеряется:

- `BVHBuilder.BottomUp` на синтетическом меше большого размера.
- `BVHBuilder.TopDown` на синтетическом меше большого размера.
- `BVHBuilder.SAH` на синтетическом меше большого размера.
- `Loader.OBJ` на большом `.obj` (синтетически сгенерированном либо внешнем).
- `Loader.SceneJSON` на большом `scene.json + meshes.bin` (синтетически сгенерированном либо внешнем).

Базовый запуск:

```bash
cmake -S . -B build
cmake --build build --target SceneRTXBenchmarks
./build/SceneRTXBenchmarks
```

Пример кастомного запуска:

```bash
./build/SceneRTXBenchmarks \
  --iterations 12 \
  --warmup 3 \
  --bvh-triangles 5000 \
  --obj-triangles 80000 \
  --json-triangles 80000
```

Запуск только конкретного набора:

```bash
./build/SceneRTXBenchmarks --filter BVHBuilder.SAH
./build/SceneRTXBenchmarks --filter Loader.OBJ
```

Экспорт в JSON для сравнения между коммитами:

```bash
./build/SceneRTXBenchmarks \
  --iterations 12 \
  --warmup 3 \
  --json-out Results/benchmarks/bench_$(date +%Y%m%d_%H%M%S).json
```

В отчёт попадают:

- параметры запуска (`iterations`, `warmup`, размеры сцен, `filter`);
- источники данных (пути до OBJ/SceneJSON, были ли они сгенерированы);
- агрегированные метрики по каждому бенчмарку (`mean/median/min/max`);
- служебные поля (`generated_at_utc`, `command_line`, `environment.commit_hint`).

Для CI можно пробрасывать commit SHA через `GIT_COMMIT`/`CI_COMMIT_SHA`/`GITHUB_SHA` — раннер добавит это значение в поле `environment.commit_hint`.

Примечание: `BVHBuilder.BottomUp` имеет квадратичную сложность `O(n^2)`, поэтому для него обычно используют меньшее число треугольников, чем для `TopDown/SAH`.

## Visual Regression (SSIM/PSNR)

Добавлен отдельный system-набор визуальных regression-тестов:

- `SystemVisualRegression/MetricsSanityCheck`: sanity-check формул SSIM/PSNR на синтетических буферах.
- `SystemVisualRegression/CompareReferenceFramesWithRenderedFrames`: сравнение эталонных кадров с актуальными кадрами рендера.

Источники кадров:

- Эталоны: `Scene/Reference` (если каталога нет, используется fallback `Scene/UE5/SubwayTonnel/Reference`).
- Актуальные кадры: `Results/GPUFrames`.
- Сопоставление кадров идёт по имени файла (например, `Reference.png` ↔ `Results/GPUFrames/Reference.png`).

Пороговые значения в тесте:

- `PSNR >= 30 dB`
- `SSIM >= 0.95`

Кратко о метриках:

- `PSNR` (Peak Signal-to-Noise Ratio) — логарифмическая оценка отличия изображений в децибелах. Чем выше `PSNR`, тем ближе изображение к эталону; бесконечность означает полное совпадение.
- `SSIM` (Structural Similarity Index) — индекс структурного сходства (контраст, яркость, структура), обычно в диапазоне `[-1; 1]`, где `1` — идентичные изображения.

Запуск только visual regression:

```bash
ctest --test-dir build -R SceneRTXTester.VisualRegression --output-on-failure
```

Режимы:

- По умолчанию тест "мягкий": если в `Results/GPUFrames` нет подходящих кадров, он сообщает об этом и не валит CI.
- Для строгого режима (обязательное наличие и сравнение кадров) включить:

```bash
SCENERTX_VISUAL_REQUIRE_ACTUAL=1 ctest --test-dir build -R SceneRTXTester.VisualRegression --output-on-failure
```

## Следующие шаги

1. Добавить end-to-end smoke на запуск бинарника с минимальной сценой и проверкой артефактов в `Results/`.
2. Добавить явный манифест сопоставления `reference -> actual` для кадров с разными именами.
3. Добавить policy по порогам деградации производительности (fail CI при выходе за threshold).
