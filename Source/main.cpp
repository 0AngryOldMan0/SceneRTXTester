/////////////////////////////////////////////////////////////////////////////////////////////////////////
// SceneRTXTester
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <locale>
#include <filesystem>
#include <cstdlib>
#include <cctype>

#include "../Headers/Classes/SceneLoaderFactory.h"
#include "../Headers/Classes/Scene.h"
#include "../Headers/Classes/Camera.h"
#include "../Headers/Classes/RenderManager.h"
#include "../Headers/SceneMetaLoader.h"

#ifdef USE_METAL_RENDERER
#include "../Headers/Classes/MetalRenderer.h"
#endif

namespace fs = std::filesystem;

namespace
{
    using Clock = std::chrono::high_resolution_clock;

    static std::string MakeSafeName(std::string s)
    {
        for (char &c : s)
        {
            if (c == ' ' || c == '\t')
                c = '_';
            else if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' ||
                     c == '"' || c == '<' || c == '>' || c == '|')
                c = '-';
        }
        if (s.empty())
            s = "camera";
        return s;
    }

    static std::string ToLower(std::string s)
    {
        for (char& c : s)
            c = (char)std::tolower((unsigned char)c);
        return s;
    }

    static bool LooksLikeIntegerArg(const char* s)
    {
        if (!s || *s == '\0')
            return false;

        char* end = nullptr;
        std::strtol(s, &end, 10);
        return end && *end == '\0';
    }

    static fs::path DefaultMetaPathFromScene(const fs::path& scenePath)
    {
        const std::string ext = ToLower(scenePath.extension().string());
        if (ext == ".json")
            return scenePath; // new SceneRTXSceneExport: scene + meta are in the same file

        fs::path p = scenePath;
        p.replace_extension();
        p += "_meta.json";
        return p;
    }

    static void PrintUsage(const char* exe)
    {
        std::cerr
            << "Использование:\n"
            << "  " << exe << " <scene_path.(obj/json)> [meta.json] [width height] [spp]\n\n"
            << "Примеры:\n"
            << "  " << exe << " Scene/UE5/SubwayTonnel/Subway.obj Scene/UE5/SubwayTonnel/Subway_meta.json 1920 1080 10\n"
            << "  " << exe << " Scene/UE5/SubwayTonnel/scene.json 1920 1080 10\n";
    }

    static std::string DetectRendererName()
    {
    #if defined(USE_HIP_RENDERER)
        return "HIP GPU Ray Tracer";
    #elif defined(USE_METAL_RENDERER)
        return "Metal GPU Ray Tracer";
    #elif defined(USE_CUDA_RENDERER)
        return "CUDA GPU Ray Tracer";
    #else
    #error "Не включён ни один рендерер: USE_HIP_RENDERER / USE_METAL_RENDERER / USE_CUDA_RENDERER"
    #endif
    }
}

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::locale::global(std::locale::classic());

    if (argc < 2)
    {
        PrintUsage(argv[0]);
        return 1;
    }

    const fs::path scenePath = fs::path(argv[1]);

    fs::path metaPath = DefaultMetaPathFromScene(scenePath);
    int argIndex = 2;

    if (argc > argIndex && !LooksLikeIntegerArg(argv[argIndex]))
    {
        metaPath = fs::path(argv[argIndex]);
        ++argIndex;
    }

    int imageWidth  = 1920;
    int imageHeight = 1080;
    int samplesPerPixel = 4;

    if (argc > argIndex + 1)
    {
        imageWidth  = std::max(1, std::atoi(argv[argIndex + 0]));
        imageHeight = std::max(1, std::atoi(argv[argIndex + 1]));
        argIndex += 2;
    }
    if (argc > argIndex)
    {
        samplesPerPixel = std::max(1, std::atoi(argv[argIndex]));
    }

    try
    {
        const auto tBegin = Clock::now();

        Scene scene;

        SceneLoaderFactory factory;
        auto loader = factory.createLoader(scenePath.string());
        if (!loader)
        {
            std::cerr << "Неизвестный формат файла: " << scenePath.string() << "\n";
            return 1;
        }

        const auto tLoad0 = Clock::now();
        std::vector<SceneObject> loadedObjects = loader->load(scenePath.string());
        const auto tLoad1 = Clock::now();

        for (auto &obj : loadedObjects)
            scene.addObject(std::move(obj));

        loadedObjects.clear();
        loadedObjects.shrink_to_fit();

        SceneMetaResources metaRes;
        bool metaOkLights = LoadLightsAndMaterialsFromMeta(metaPath.string(), scene, &metaRes);
        if (!metaOkLights)
        {
            std::cerr << "WARNING: не удалось загрузить lights/materials из meta: "
                      << metaPath.string() << "\n";
        }

        std::vector<SceneMetaCameraInfo> metaCameras;
        bool metaOkCams = LoadCamerasFromMeta(metaPath.string(), metaCameras);
        if (!metaOkCams)
        {
            std::cerr << "WARNING: не удалось загрузить камеры из meta: "
                      << metaPath.string() << "\n";
        }

        auto stats = scene.getStats();
        BVHBuilder::Strategy bvhStrategy = BVHBuilder::Strategy::BottomUp;
        if (stats.totalTriangles > 20000)
            bvhStrategy = BVHBuilder::Strategy::TopDown;

        const auto tBVH0 = Clock::now();
        scene.buildGlobalBVH(bvhStrategy);
        const auto tBVH1 = Clock::now();

        RenderManager renderManager;

        for (auto &renderer : renderManager.getRenderers())
        {
    #ifdef USE_METAL_RENDERER
            if (auto *metal = dynamic_cast<MetalRenderer *>(renderer.get()))
                metal->setMetaResources(&metaRes);
    #endif
            renderer->setSamplesPerPixel(samplesPerPixel);
            renderer->setImageSize(imageWidth, imageHeight);
        }

        const std::string rendererName = DetectRendererName();

        fs::path outDir = fs::path("Results") / "GPUFrames";
        std::error_code ec;
        fs::create_directories(outDir, ec);

        const auto tRender0 = Clock::now();

        Camera cam;

        if (!metaCameras.empty())
        {
            for (std::size_t ci = 0; ci < metaCameras.size(); ++ci) // metaCameras.size()
            {
                ApplyMetaCameraToCamera(metaCameras[ci], cam, imageWidth, imageHeight);

                const std::string camName = MakeSafeName(metaCameras[ci].name);
                const fs::path base =
                    outDir /
                    (rendererName + "_TextureFrame_Cam" + std::to_string(ci) + "_" + camName);

                renderManager.renderFrameTexture(scene, cam, rendererName, base.string());
                std::cout << "Frames for camera: <" << ci << "> generated successfully\n";
            }
        }
        else
        {
            const fs::path base = outDir / (rendererName + "_TextureFrame_DefaultCamera");
            renderManager.renderFrameTexture(scene, cam, rendererName, base.string());
            std::cout << "Frames for default camera generated successfully\n";
        }

        const auto tRender1 = Clock::now();
        const auto tEnd = Clock::now();

        const double msLoad =
            std::chrono::duration<double, std::milli>(tLoad1 - tLoad0).count();
        const double msBVH =
            std::chrono::duration<double, std::milli>(tBVH1 - tBVH0).count();
        const double msRender =
            std::chrono::duration<double, std::milli>(tRender1 - tRender0).count();
        const double msTotal =
            std::chrono::duration<double, std::milli>(tEnd - tBegin).count();

        stats = scene.getStats();
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Prototype triangles:   " << stats.uniqueTriangles << "\n";
        std::cout << "Expanded triangles:    " << stats.totalTriangles << "\n";
        std::cout << "Prototype objects:     " << stats.prototypeCount << "\n";
        std::cout << "Total instances:       " << stats.instanceCount << "\n";
        std::cout << "TLAS Nodes:            " << stats.globalBVHNodes << "\n";
        std::cout << "BLAS Nodes:            " << stats.meshBVHNodes << "\n";
        std::cout << "Global BVH Depth:      " << stats.globalBVHDepth << "\n";
        std::cout << "Load time (scene):   " << msLoad   << " ms\n";
        std::cout << "Meta load time:      " << std::chrono::duration<double, std::milli>(tBVH0 - tLoad1).count() << " ms\n";
        std::cout << "BVH build time:      " << msBVH    << " ms\n";
        std::cout << "Render time:         " << msRender << " ms\n";
        std::cout << "Total time:          " << msTotal  << " ms\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }
}
