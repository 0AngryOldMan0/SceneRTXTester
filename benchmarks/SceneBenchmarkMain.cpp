#include "BVHBuilder.h"
#include "OBJLoader.h"
#include "SceneJSONLoader.h"
#include "SceneObject.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using Clock = std::chrono::steady_clock;

    struct Options
    {
        int iterations = 8;
        int warmup = 2;
        int bvhTriangles = 2000;
        int objTriangles = 40000;
        int jsonTriangles = 40000;
        std::string filter;
        std::string objPath;
        std::string jsonPath;
        bool helpRequested = false;
    };

    struct BenchmarkResult
    {
        std::string name;
        std::string payload;
        double meanMs = 0.0;
        double medianMs = 0.0;
        double minMs = 0.0;
        double maxMs = 0.0;
        std::size_t iterations = 0;
    };

    class TempDir
    {
    public:
        TempDir()
        {
            const auto now = static_cast<unsigned long long>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::mt19937_64 rng(now);

            path_ = std::filesystem::temp_directory_path() /
                    ("scene_rtx_bench_" + std::to_string(now) + "_" + std::to_string(rng()));
            std::filesystem::create_directories(path_);
        }

        ~TempDir()
        {
            std::error_code ec;
            std::filesystem::remove_all(path_, ec);
        }

        const std::filesystem::path &path() const { return path_; }

    private:
        std::filesystem::path path_;
    };

    void printUsage(const char *exe)
    {
        std::cout
            << "Usage:\n"
            << "  " << exe << " [options]\n\n"
            << "Options:\n"
            << "  --iterations N        Measured iterations per benchmark (default: 8)\n"
            << "  --warmup N            Warmup iterations per benchmark (default: 2)\n"
            << "  --bvh-triangles N     Triangle count for BVH microbench (default: 2000)\n"
            << "  --obj-triangles N     Triangle count for generated OBJ bench (default: 40000)\n"
            << "  --json-triangles N    Triangle count for generated SceneJSON bench (default: 40000)\n"
            << "  --obj-path PATH       Use existing OBJ file instead of generated one\n"
            << "  --json-path PATH      Use existing Scene JSON file instead of generated one\n"
            << "  --filter TEXT         Run only benchmarks that contain TEXT in name\n"
            << "  --help                Show this help\n";
    }

    bool parsePositiveInt(const char *arg, int &outValue)
    {
        if (arg == nullptr || *arg == '\0')
            return false;

        char *end = nullptr;
        const long parsed = std::strtol(arg, &end, 10);
        if (end == arg || (end != nullptr && *end != '\0'))
            return false;
        if (parsed <= 0 || parsed > static_cast<long>(std::numeric_limits<int>::max()))
            return false;

        outValue = static_cast<int>(parsed);
        return true;
    }

    bool parseOptions(int argc, char **argv, Options &opts)
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string key = argv[i];

            auto requireValue = [&](const char *name) -> const char *
            {
                if (i + 1 >= argc)
                {
                    std::cerr << "Missing value for " << name << "\n";
                    return nullptr;
                }
                return argv[++i];
            };

            if (key == "--help" || key == "-h")
            {
                opts.helpRequested = true;
                return false;
            }
            else if (key == "--iterations")
            {
                const char *value = requireValue("--iterations");
                if (value == nullptr || !parsePositiveInt(value, opts.iterations))
                    return false;
            }
            else if (key == "--warmup")
            {
                const char *value = requireValue("--warmup");
                if (value == nullptr || !parsePositiveInt(value, opts.warmup))
                    return false;
            }
            else if (key == "--bvh-triangles")
            {
                const char *value = requireValue("--bvh-triangles");
                if (value == nullptr || !parsePositiveInt(value, opts.bvhTriangles))
                    return false;
            }
            else if (key == "--obj-triangles")
            {
                const char *value = requireValue("--obj-triangles");
                if (value == nullptr || !parsePositiveInt(value, opts.objTriangles))
                    return false;
            }
            else if (key == "--json-triangles")
            {
                const char *value = requireValue("--json-triangles");
                if (value == nullptr || !parsePositiveInt(value, opts.jsonTriangles))
                    return false;
            }
            else if (key == "--obj-path")
            {
                const char *value = requireValue("--obj-path");
                if (value == nullptr)
                    return false;
                opts.objPath = value;
            }
            else if (key == "--json-path")
            {
                const char *value = requireValue("--json-path");
                if (value == nullptr)
                    return false;
                opts.jsonPath = value;
            }
            else if (key == "--filter")
            {
                const char *value = requireValue("--filter");
                if (value == nullptr)
                    return false;
                opts.filter = value;
            }
            else
            {
                std::cerr << "Unknown argument: " << key << "\n";
                return false;
            }
        }

        return true;
    }

    double medianMs(std::vector<double> values)
    {
        if (values.empty())
            return 0.0;

        std::sort(values.begin(), values.end());
        const std::size_t mid = values.size() / 2;
        if (values.size() % 2 == 0)
            return 0.5 * (values[mid - 1] + values[mid]);

        return values[mid];
    }

    template <typename Fn>
    BenchmarkResult runBenchmark(const std::string &name,
                                 const std::string &payload,
                                 const Options &opts,
                                 Fn &&fn)
    {
        for (int i = 0; i < opts.warmup; ++i)
            fn();

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(opts.iterations));

        for (int i = 0; i < opts.iterations; ++i)
        {
            const auto t0 = Clock::now();
            fn();
            const auto t1 = Clock::now();

            const auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            samples.push_back(dt);
        }

        BenchmarkResult result;
        result.name = name;
        result.payload = payload;
        result.iterations = samples.size();

        double sum = 0.0;
        double minV = std::numeric_limits<double>::infinity();
        double maxV = 0.0;
        for (double s : samples)
        {
            sum += s;
            minV = std::min(minV, s);
            maxV = std::max(maxV, s);
        }

        result.meanMs = (samples.empty() ? 0.0 : sum / static_cast<double>(samples.size()));
        result.medianMs = medianMs(samples);
        result.minMs = (samples.empty() ? 0.0 : minV);
        result.maxMs = maxV;
        return result;
    }

    bool matchesFilter(const std::string &name, const std::string &filter)
    {
        return filter.empty() || name.find(filter) != std::string::npos;
    }

    Triangle makeTriangle(float x, float y, float z)
    {
        Triangle t{};
        t.v0 = Vec3{x, y, z};
        t.v1 = Vec3{x + 1.0f, y, z};
        t.v2 = Vec3{x, y + 1.0f, z};
        t.normal = Vec3{0.0f, 0.0f, 1.0f};
        return t;
    }

    SceneObject buildSyntheticObject(int triangleCount)
    {
        SceneObject object("BenchMesh");
        object.getTrianglesMutable().reserve(static_cast<std::size_t>(triangleCount));

        constexpr int kGrid = 512;
        for (int i = 0; i < triangleCount; ++i)
        {
            const int gx = i % kGrid;
            const int gy = i / kGrid;
            object.addTriangle(makeTriangle(static_cast<float>(gx), static_cast<float>(gy), 0.0f));
        }
        return object;
    }

    std::filesystem::path generateLargeObj(const std::filesystem::path &dir, int triangleCount)
    {
        const std::filesystem::path objPath = dir / "generated_large.obj";
        std::ofstream file(objPath);
        file << "o BenchLarge\n";
        file << "vn 0 0 1\n";
        file << "usemtl BenchMaterial\n";

        int idx = 1;
        constexpr int kGrid = 2048;
        for (int i = 0; i < triangleCount; ++i)
        {
            const int gx = i % kGrid;
            const int gy = i / kGrid;
            const float x = static_cast<float>(gx);
            const float y = static_cast<float>(gy);

            file << "v " << x << ' ' << y << " 0\n";
            file << "v " << (x + 1.0f) << ' ' << y << " 0\n";
            file << "v " << x << ' ' << (y + 1.0f) << " 0\n";

            file << "vt 0 0\n";
            file << "vt 1 0\n";
            file << "vt 0 1\n";

            file << "f "
                 << idx << '/' << idx << "/1 "
                 << (idx + 1) << '/' << (idx + 1) << "/1 "
                 << (idx + 2) << '/' << (idx + 2) << "/1\n";
            idx += 3;
        }

        return objPath;
    }

    template <typename T>
    void appendBytes(std::vector<std::uint8_t> &dst, const T &value)
    {
        const std::uint8_t *ptr = reinterpret_cast<const std::uint8_t *>(&value);
        dst.insert(dst.end(), ptr, ptr + sizeof(T));
    }

    std::filesystem::path generateLargeSceneJson(const std::filesystem::path &dir, int triangleCount)
    {
        const std::filesystem::path scenePath = dir / "generated_scene.json";
        const std::filesystem::path meshPath = dir / "generated_meshes.bin";

        const int vertexCount = triangleCount * 3;
        const int indexCount = triangleCount * 3;

        std::vector<std::uint8_t> data;
        data.reserve(8u + static_cast<std::size_t>(vertexCount) * sizeof(Vec3) * 2u + static_cast<std::size_t>(indexCount) * sizeof(std::uint32_t));

        const std::uint8_t magic[8] = {'S', 'R', 'T', 'X', 'M', 'S', 'H', 0};
        data.insert(data.end(), std::begin(magic), std::end(magic));

        const std::uint64_t positionsOffset = static_cast<std::uint64_t>(data.size());
        constexpr int kGrid = 1024;
        for (int i = 0; i < triangleCount; ++i)
        {
            const int gx = i % kGrid;
            const int gy = i / kGrid;
            const float x = static_cast<float>(gx);
            const float y = static_cast<float>(gy);

            appendBytes(data, Vec3{x, y, 0.0f});
            appendBytes(data, Vec3{x + 1.0f, y, 0.0f});
            appendBytes(data, Vec3{x, y + 1.0f, 0.0f});
        }

        const std::uint64_t normalsOffset = static_cast<std::uint64_t>(data.size());
        for (int i = 0; i < vertexCount; ++i)
            appendBytes(data, Vec3{0.0f, 0.0f, 1.0f});

        const std::uint64_t indicesOffset = static_cast<std::uint64_t>(data.size());
        for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(indexCount); ++i)
            appendBytes(data, i);

        {
            std::ofstream meshFile(meshPath, std::ios::binary);
            meshFile.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
        }

        std::ostringstream json;
        json << "{\n"
             << "  \"format\": \"SceneRTXSceneExport\",\n"
             << "  \"world\": {\"meshes_file\": \"" << meshPath.filename().string() << "\"},\n"
             << "  \"materials\": [\n"
             << "    {\"stable_id\": \"mat_1\", \"name\": \"BenchMaterial\"}\n"
             << "  ],\n"
             << "  \"mesh_assets\": [\n"
             << "    {\n"
             << "      \"stable_id\": \"mesh_1\",\n"
             << "      \"lods\": [\n"
             << "        {\n"
             << "          \"vertex_count\": " << vertexCount << ",\n"
             << "          \"index_count\": " << indexCount << ",\n"
             << "          \"uv_set_count\": 0,\n"
             << "          \"positions_offset\": " << positionsOffset << ",\n"
             << "          \"normals_offset\": " << normalsOffset << ",\n"
             << "          \"tangents_offset\": 0,\n"
             << "          \"colors_offset\": 0,\n"
             << "          \"uvs_offset\": 0,\n"
             << "          \"indices_offset\": " << indicesOffset << ",\n"
             << "          \"sections\": [\n"
             << "            {\"first_index\": 0, \"index_count\": " << indexCount << ", \"material_name\": \"BenchMaterial\"}\n"
             << "          ]\n"
             << "        }\n"
             << "      ]\n"
             << "    }\n"
             << "  ],\n"
             << "  \"primitives\": [\n"
             << "    {\n"
             << "      \"name\": \"BenchPrimitive\",\n"
             << "      \"mesh_asset_id\": \"mesh_1\",\n"
             << "      \"geometry_exported\": true\n"
             << "    }\n"
             << "  ]\n"
             << "}\n";

        std::ofstream sceneFile(scenePath);
        sceneFile << json.str();

        return scenePath;
    }

    void printResults(const std::vector<BenchmarkResult> &results)
    {
        std::cout << "\nBenchmark results (ms):\n";
        std::cout << std::left
                  << std::setw(42) << "Name"
                  << std::setw(16) << "Payload"
                  << std::setw(10) << "Iter"
                  << std::setw(12) << "Mean"
                  << std::setw(12) << "Median"
                  << std::setw(12) << "Min"
                  << std::setw(12) << "Max"
                  << '\n';

        std::cout << std::string(116, '-') << '\n';

        for (const BenchmarkResult &r : results)
        {
            std::cout << std::left
                      << std::setw(42) << r.name
                      << std::setw(16) << r.payload
                      << std::setw(10) << r.iterations
                      << std::setw(12) << std::fixed << std::setprecision(3) << r.meanMs
                      << std::setw(12) << std::fixed << std::setprecision(3) << r.medianMs
                      << std::setw(12) << std::fixed << std::setprecision(3) << r.minMs
                      << std::setw(12) << std::fixed << std::setprecision(3) << r.maxMs
                      << '\n';
        }
    }
}

int main(int argc, char **argv)
{
    Options opts;
    if (!parseOptions(argc, argv, opts))
    {
        if (argc > 1)
            printUsage(argv[0]);
        return opts.helpRequested ? 0 : 1;
    }

    std::cout << "SceneRTXBenchmarks starting...\n";
    std::cout << "iterations=" << opts.iterations
              << ", warmup=" << opts.warmup
              << ", bvhTriangles=" << opts.bvhTriangles
              << ", objTriangles=" << opts.objTriangles
              << ", jsonTriangles=" << opts.jsonTriangles
              << '\n';

    if (opts.bvhTriangles > 10000)
    {
        std::cout << "WARNING: BottomUp BVH is O(n^2); large --bvh-triangles may run for a long time.\n";
    }

    TempDir temp;

    const std::filesystem::path objPath = opts.objPath.empty()
                                              ? generateLargeObj(temp.path(), opts.objTriangles)
                                              : std::filesystem::path(opts.objPath);

    const std::filesystem::path sceneJsonPath = opts.jsonPath.empty()
                                                    ? generateLargeSceneJson(temp.path(), opts.jsonTriangles)
                                                    : std::filesystem::path(opts.jsonPath);

    std::vector<BenchmarkResult> results;

    const SceneObject baseObj = buildSyntheticObject(opts.bvhTriangles);

    struct BvhCase
    {
        std::string name;
        BVHBuilder::Strategy strategy;
    };

    const std::vector<BvhCase> bvhCases = {
        {"BVHBuilder.BottomUp", BVHBuilder::Strategy::BottomUp},
        {"BVHBuilder.TopDown", BVHBuilder::Strategy::TopDown},
        {"BVHBuilder.SAH", BVHBuilder::Strategy::SAH}};

    for (const BvhCase &c : bvhCases)
    {
        if (!matchesFilter(c.name, opts.filter))
            continue;

        BVHBuilder builder(c.strategy);
        results.push_back(runBenchmark(c.name,
                                       std::to_string(opts.bvhTriangles) + " tris",
                                       opts,
                                       [&]()
                                       {
                                           SceneObject working = baseObj;
                                           builder.buildForObject(working);
                                       }));
    }

    if (matchesFilter("Loader.OBJ", opts.filter))
    {
        OBJLoader loader;
        results.push_back(runBenchmark("Loader.OBJ",
                                       opts.objPath.empty()
                                           ? (std::to_string(opts.objTriangles) + " tris")
                                           : "external file",
                                       opts,
                                       [&]()
                                       {
                                           auto objects = loader.load(objPath.string());
                                           if (objects.empty())
                                               throw std::runtime_error("OBJ benchmark: loaded empty object list");
                                       }));
    }

    if (matchesFilter("Loader.SceneJSON", opts.filter))
    {
        SceneJSONLoader loader;
        results.push_back(runBenchmark("Loader.SceneJSON",
                                       opts.jsonPath.empty()
                                           ? (std::to_string(opts.jsonTriangles) + " tris")
                                           : "external file",
                                       opts,
                                       [&]()
                                       {
                                           auto objects = loader.load(sceneJsonPath.string());
                                           if (objects.empty())
                                               throw std::runtime_error("SceneJSON benchmark: loaded empty object list");
                                       }));
    }

    if (results.empty())
    {
        std::cerr << "No benchmarks selected. Check --filter value.\n";
        return 1;
    }

    printResults(results);

    std::cout << "\nOBJ source: " << objPath.string() << '\n';
    std::cout << "Scene JSON source: " << sceneJsonPath.string() << '\n';
    std::cout << "Done.\n";

    return 0;
}
