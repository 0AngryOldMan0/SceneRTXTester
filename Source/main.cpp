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
#include <limits>
#include <cmath>

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

    static bool TryParseInt(const std::string &s, int &value)
    {
        if (!LooksLikeIntegerArg(s.c_str()))
            return false;

        char *end = nullptr;
        const long parsed = std::strtol(s.c_str(), &end, 10);
        if (!end || *end != '\0')
            return false;
        if (parsed < static_cast<long>(std::numeric_limits<int>::min()) ||
            parsed > static_cast<long>(std::numeric_limits<int>::max()))
            return false;

        value = static_cast<int>(parsed);
        return true;
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
            << "  " << exe << " <scene_path.(obj/json)> [-preview] [-progressive] [meta.json] [width height] [spp]\n\n"
            << "Примеры:\n"
            << "  " << exe << " Scene/UE5/SubwayTonnel/Subway.obj Scene/UE5/SubwayTonnel/Subway_meta.json 1920 1080 10\n"
            << "  " << exe << " Scene/UE5/SubwayTonnel/scene.json 1920 1080 10\n"
            << "  " << exe << " Scene/UE5/SubwayTonnel/scene.json -preview\n";
    }

    static int PromptCameraSelection(const std::vector<SceneMetaCameraInfo> &metaCameras)
    {
        std::cout << "Доступные камеры:\n";
        std::cout << "  " << std::left << std::setw(8) << "Index" << "Name\n";
        std::cout << "  " << std::left << std::setw(8) << "-----" << "----\n";

        for (std::size_t i = 0; i < metaCameras.size(); ++i)
        {
            const std::string cameraName = metaCameras[i].name.empty() ? "<unnamed>" : metaCameras[i].name;
            std::cout << "  " << std::left << std::setw(8) << i << cameraName << "\n";
        }

        while (true)
        {
            std::cout << "Введите номер камеры для рендера (-1 = все камеры): " << std::flush;

            std::string input;
            if (!std::getline(std::cin, input))
                return -1;

            int selectedCamera = -1;
            if (!TryParseInt(input, selectedCamera))
            {
                std::cout << "Некорректный ввод. Введите целое число от -1 до "
                          << (metaCameras.empty() ? 0 : static_cast<int>(metaCameras.size() - 1))
                          << ".\n";
                continue;
            }

            if (selectedCamera == -1)
                return selectedCamera;

            if (selectedCamera >= 0 && static_cast<std::size_t>(selectedCamera) < metaCameras.size())
                return selectedCamera;

            std::cout << "Камеры с номером " << selectedCamera << " нет в списке.\n";
        }
    }

    static std::string DetectRendererName(const RenderManager &renderManager)
    {
        const auto &renderers = renderManager.getRenderers();
        if (renderers.empty() || !renderers.front())
            return {};

        return renderers.front()->getName();
    }

    struct RenderDimensions
    {
        int width = 1920;
        int height = 1080;
    };

    static RenderDimensions ComputeRenderDimensionsForCamera(const SceneMetaCameraInfo &metaCam,
                                                             int requestedWidth,
                                                             int requestedHeight)
    {
        RenderDimensions dims{
            std::max(1, requestedWidth),
            std::max(1, requestedHeight)
        };

        if (!metaCam.constrainAspectRatio ||
            !std::isfinite(metaCam.aspectRatio) ||
            metaCam.aspectRatio <= 0.0f)
        {
            return dims;
        }

        const float targetAspect =
            static_cast<float>(dims.width) / static_cast<float>(dims.height);
        const float cameraAspect = metaCam.aspectRatio;

        if (std::fabs(cameraAspect - targetAspect) <= 1.0e-4f)
            return dims;

        if (cameraAspect > targetAspect)
        {
            dims.height = std::max(1, static_cast<int>(std::lround(
                static_cast<double>(dims.width) / static_cast<double>(cameraAspect))));
        }
        else
        {
            dims.width = std::max(1, static_cast<int>(std::lround(
                static_cast<double>(dims.height) * static_cast<double>(cameraAspect))));
        }

        dims.width = std::min(dims.width, std::max(1, requestedWidth));
        dims.height = std::min(dims.height, std::max(1, requestedHeight));
        return dims;
    }

    static void SetRendererImageSize(RenderManager &renderManager,
                                     int width,
                                     int height)
    {
        for (auto &renderer : renderManager.getRenderers())
            renderer->setImageSize(width, height);
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

    TextureRenderMode renderMode = TextureRenderMode::Progressive;
    std::vector<std::string> positionalArgs;
    positionalArgs.reserve((argc > 2) ? static_cast<std::size_t>(argc - 2) : 0u);

    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "-preview" || arg == "--preview")
        {
            renderMode = TextureRenderMode::Preview;
            continue;
        }
        if (arg == "-progressive" || arg == "--progressive")
        {
            renderMode = TextureRenderMode::Progressive;
            continue;
        }
        if (!arg.empty() && arg[0] == '-')
        {
            std::cerr << "Неизвестный флаг: " << arg << "\n";
            PrintUsage(argv[0]);
            return 1;
        }
        positionalArgs.push_back(arg);
    }

    fs::path metaPath = DefaultMetaPathFromScene(scenePath);
    std::size_t argIndex = 0;

    if (argIndex < positionalArgs.size() && !LooksLikeIntegerArg(positionalArgs[argIndex].c_str()))
    {
        metaPath = fs::path(positionalArgs[argIndex]);
        ++argIndex;
    }

    int imageWidth  = 1920;
    int imageHeight = 1080;
    int samplesPerPixel = 4;

    if (argIndex + 1 < positionalArgs.size())
    {
        imageWidth  = std::max(1, std::atoi(positionalArgs[argIndex + 0].c_str()));
        imageHeight = std::max(1, std::atoi(positionalArgs[argIndex + 1].c_str()));
        argIndex += 2;
    }
    if (argIndex < positionalArgs.size())
    {
        samplesPerPixel = std::max(1, std::atoi(positionalArgs[argIndex].c_str()));
        ++argIndex;
    }

    if (argIndex != positionalArgs.size())
    {
        PrintUsage(argv[0]);
        return 1;
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

        const auto tMeta1 = Clock::now();
        RenderManager renderManager;

        if (renderManager.getRenderers().empty())
        {
            std::cerr
                << "No renderer backend is enabled for this build.\n"
                << "Enable one of USE_HIP_RENDERER, USE_CUDA_RENDERER, or USE_METAL_RENDERER in CMake.\n";
            return 1;
        }

        const auto tPreload0 = tMeta1;
        bool didMetalPreload = false;
        for (auto &renderer : renderManager.getRenderers())
        {
    #ifdef USE_METAL_RENDERER
            if (auto *metal = dynamic_cast<MetalRenderer *>(renderer.get()))
            {
                metal->setMetaResources(&metaRes);
                didMetalPreload = true;
            }
    #endif
            renderer->setSamplesPerPixel(samplesPerPixel);
            renderer->setImageSize(imageWidth, imageHeight);

    #ifdef USE_METAL_RENDERER
            if (auto *metal = dynamic_cast<MetalRenderer *>(renderer.get()))
            {
                if (!metal->preloadSceneResources())
                {
                    std::cerr << "Ошибка preload Metal resources\n";
                    return 1;
                }
            }
    #endif
        }
        const auto tPreload1 = Clock::now();

        auto stats = scene.getStats();
        BVHBuilder::Strategy bvhStrategy = BVHBuilder::Strategy::BottomUp;
        if (stats.totalTriangles > 20000)
            bvhStrategy = BVHBuilder::Strategy::TopDown;

        const auto tBVH0 = Clock::now();
        scene.buildGlobalBVH(bvhStrategy);
        const auto tBVH1 = Clock::now();

        const std::string rendererName = DetectRendererName(renderManager);

        fs::path outDir = fs::path("Results") / "GPUFrames";
        std::error_code ec;
        fs::create_directories(outDir, ec);

        const auto tRender0 = Clock::now();

        Camera cam;

        if (!metaCameras.empty())
        {
            const int selectedCamera = PromptCameraSelection(metaCameras);
            const std::size_t beginCamera = (selectedCamera < 0) ? 0u : static_cast<std::size_t>(selectedCamera);
            const std::size_t endCamera = (selectedCamera < 0) ? metaCameras.size() : (beginCamera + 1u);

            for (std::size_t ci = beginCamera; ci < endCamera; ++ci)
            {
                const RenderDimensions renderDims =
                    ComputeRenderDimensionsForCamera(metaCameras[ci], imageWidth, imageHeight);

                SetRendererImageSize(renderManager, renderDims.width, renderDims.height);
                ApplyMetaCameraToCamera(metaCameras[ci], cam, renderDims.width, renderDims.height);

                if (renderDims.width != imageWidth || renderDims.height != imageHeight)
                {
                    const std::streamsize oldPrecision = std::cout.precision();
                    const std::ios::fmtflags oldFlags = std::cout.flags();
                    std::cout << "Camera <" << ci << ">: constrained aspect "
                              << std::fixed << std::setprecision(3) << metaCameras[ci].aspectRatio
                              << ", render size " << renderDims.width << "x" << renderDims.height
                              << " instead of " << imageWidth << "x" << imageHeight << "\n";
                    std::cout.flags(oldFlags);
                    std::cout.precision(oldPrecision);
                }

                const std::string camName = MakeSafeName(metaCameras[ci].name);
                const fs::path base =
                    outDir /
                    (rendererName + "_TextureFrame_Cam" + std::to_string(ci) + "_" + camName);

                renderManager.renderFrameTexture(scene,
                                                cam,
                                                rendererName,
                                                base.string(),
                                                renderMode,
                                                samplesPerPixel);
                std::cout << "Frames for camera: <" << ci << "> generated successfully\n";
            }
        }
        else
        {
            const fs::path base = outDir / (rendererName + "_TextureFrame_DefaultCamera");
            renderManager.renderFrameTexture(scene,
                                            cam,
                                            rendererName,
                                            base.string(),
                                            renderMode,
                                            samplesPerPixel);
            std::cout << "Frames for default camera generated successfully\n";
        }

        const auto tRender1 = Clock::now();
        const auto tEnd = Clock::now();

        const double msLoad =
            std::chrono::duration<double, std::milli>(tLoad1 - tLoad0).count();
        const double msBVH =
            std::chrono::duration<double, std::milli>(tBVH1 - tBVH0).count();
        const double msPreload =
            std::chrono::duration<double, std::milli>(tPreload1 - tPreload0).count();
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
        std::cout << "Meta load time:      " << std::chrono::duration<double, std::milli>(tMeta1 - tLoad1).count() << " ms\n";
        if (didMetalPreload)
            std::cout << "Metal preload time:  " << msPreload << " ms\n";
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
