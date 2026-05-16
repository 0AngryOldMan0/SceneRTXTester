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

#include "SceneLoaderFactory.h"
#include "Scene.h"
#include "Camera.h"
#include "RenderManager.h"
#include "SceneMetaLoader.h"
#include "Renderer.h"
#include "RendererFactory.h"
#include "RenderCommand.h"
#ifdef USE_HIP_RENDERER
#include "HIPRenderer.h"
#endif
#ifdef USE_METAL_RENDERER
#include "MetalRenderer.h"
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
        for (char &c : s)
            c = (char)std::tolower((unsigned char)c);
        return s;
    }

    static bool LooksLikeIntegerArg(const char *s)
    {
        if (!s || *s == '\0')
            return false;

        char *end = nullptr;
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

    static fs::path DefaultMetaPathFromScene(const fs::path &scenePath)
    {
        const std::string ext = ToLower(scenePath.extension().string());
        if (ext == ".json")
            return scenePath; // new SceneRTXSceneExport: scene + meta are in the same file

        fs::path p = scenePath;
        p.replace_extension();
        p += "_meta.json";
        return p;
    }

    static void PrintUsage(const char *exe)
    {
        std::cerr
            << "Usage:\n"
            << "  " << exe << " <scene_path.(obj/json)> [-preview] [-progressive] [-textureDebug] [meta.json] [width height] [spp]\n\n"
            << "Examples:\n"
            << "  " << exe << " Scene/UE5/SubwayTonnel/Subway.obj Scene/UE5/SubwayTonnel/Subway_meta.json 1920 1080 10\n"
            << "  " << exe << " Scene/UE5/SubwayTonnel/scene.json 1920 1080 10\n"
            << "  " << exe << " Scene/UE5/SubwayTonnel/scene.json -preview\n"
            << "  " << exe << " Scene/UE5/SubwayTonnel/scene.json -textureDebug\n";
    }

    struct TextureDebugPass
    {
        RenderCommand::DebugView view;
        const char *suffix;
    };

    static const std::vector<TextureDebugPass> &GetTextureDebugPasses()
    {
        static const std::vector<TextureDebugPass> passes = {
            {RenderCommand::DebugView::ShadingNormals, "_DebugNs.png"},
            {RenderCommand::DebugView::AmbientOcclusion, "_DebugAO.png"},
            {RenderCommand::DebugView::NormalDifference, "_DebugNsMinusNg.png"},
            {RenderCommand::DebugView::BaseColor, "_DebugBaseColor.png"},
            {RenderCommand::DebugView::Roughness, "_DebugRoughness.png"},
            {RenderCommand::DebugView::Metallic, "_DebugMetallic.png"},
            {RenderCommand::DebugView::Emissive, "_DebugEmissive.png"},
            {RenderCommand::DebugView::VertexColor, "_DebugVertexColor.png"},
            {RenderCommand::DebugView::MaterialModel, "_DebugMaterialModel.png"}};
        return passes;
    }

    static bool RendererSupportsTextureDebug(Renderer &renderer)
    {
#ifdef USE_HIP_RENDERER
        if (dynamic_cast<HIPRenderer *>(&renderer))
            return true;
#endif
#ifdef USE_METAL_RENDERER
        if (dynamic_cast<MetalRenderer *>(&renderer))
            return true;
#endif
        return false;
    }

    static bool SetRendererDebugView(Renderer &renderer, RenderCommand::DebugView view)
    {
#ifdef USE_HIP_RENDERER
        if (auto *hip = dynamic_cast<HIPRenderer *>(&renderer))
        {
            hip->setDebugView(HIPRenderer::commandViewToHIPView(view));
            return true;
        }
#endif
#ifdef USE_METAL_RENDERER
        if (auto *metal = dynamic_cast<MetalRenderer *>(&renderer))
        {
            metal->setDebugView(view);
            return true;
        }
#endif
        return view == RenderCommand::DebugView::Disabled;
    }

    static int PromptCameraSelection(const std::vector<SceneMetaCameraInfo> &metaCameras)
    {
        std::cout << "Available cameras:\n";
        std::cout << "  " << std::left << std::setw(8) << "Index" << "Name\n";
        std::cout << "  " << std::left << std::setw(8) << "-----" << "----\n";

        for (std::size_t i = 0; i < metaCameras.size(); ++i)
        {
            const std::string cameraName = metaCameras[i].name.empty() ? "<unnamed>" : metaCameras[i].name;
            std::cout << "  " << std::left << std::setw(8) << i << cameraName << "\n";
        }

        while (true)
        {
            std::cout << "Enter the camera index to render (-1 = all cameras): " << std::flush;

            std::string input;
            if (!std::getline(std::cin, input))
                return -1;

            int selectedCamera = -1;
            if (!TryParseInt(input, selectedCamera))
            {
                std::cout << "Invalid input. Enter an integer from -1 to "
                          << (metaCameras.empty() ? 0 : static_cast<int>(metaCameras.size() - 1))
                          << ".\n";
                continue;
            }

            if (selectedCamera == -1)
                return selectedCamera;

            if (selectedCamera >= 0 && static_cast<std::size_t>(selectedCamera) < metaCameras.size())
                return selectedCamera;

            std::cout << "Camera index " << selectedCamera << " is not in the list.\n";
        }
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
            std::max(1, requestedHeight)};

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

    static void UpdateRendererDimensions(std::vector<std::unique_ptr<Renderer>> &renderers,
                                         int width,
                                         int height)
    {
        for (auto &renderer : renderers)
            renderer->setImageSize(width, height);
    }

    static fs::path MakeAbsoluteBestEffort(const fs::path &path)
    {
        std::error_code ec;
        const fs::path absolutePath = fs::absolute(path, ec);
        return ec ? path : absolutePath;
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
    bool exportTextureDebugViews = false;
    bool didWarnLegacyHipDebugFlag = false;
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
        if (arg == "-textureDebug" || arg == "--textureDebug" || arg == "--texture-debug")
        {
            exportTextureDebugViews = true;
            continue;
        }
        if (arg == "-hip-debug" || arg == "--hip-debug" || arg == "--hip-debug-all")
        {
            exportTextureDebugViews = true;
            if (!didWarnLegacyHipDebugFlag)
            {
                std::cerr << "WARNING: '-hip-debug' is deprecated, use '-textureDebug'.\n";
                didWarnLegacyHipDebugFlag = true;
            }
            continue;
        }
        if (!arg.empty() && arg[0] == '-')
        {
            std::cerr << "Unknown flag: " << arg << "\n";
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

    int imageWidth = 1920;
    int imageHeight = 1080;
    int samplesPerPixel = 4;

    if (argIndex + 1 < positionalArgs.size())
    {
        imageWidth = std::max(1, std::atoi(positionalArgs[argIndex + 0].c_str()));
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
            std::cerr << "Unknown file format: " << scenePath.string() << "\n";
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
            std::cerr << "WARNING: failed to load lights/materials from meta: "
                      << metaPath.string() << "\n";
        }

        std::vector<SceneMetaCameraInfo> metaCameras;
        bool metaOkCams = LoadCamerasFromMeta(metaPath.string(), metaCameras);
        if (!metaOkCams)
        {
            std::cerr << "WARNING: failed to load cameras from meta: "
                      << metaPath.string() << "\n";
        }

        const auto tMeta1 = Clock::now();

        // ===== NEW PATTERN: Use RendererFactory instead of RenderManager =====
        auto renderers = RendererFactory::createAllAvailableRenderers();

        if (renderers.empty())
        {
            std::cerr
                << "No renderer backend is available.\n"
                << "Check RendererRegistry for available renderers.\n";
            return 1;
        }

        // ===== NEW PATTERN: Create RenderCommand with all parameters =====
        RenderCommand renderCmd;
        renderCmd.setImageSize(imageWidth, imageHeight)
            .setSamplesPerPixel(samplesPerPixel)
            .setMetadata(&metaRes);

        // Debug export mode: generate additional texture debug passes.
        if (exportTextureDebugViews)
        {
            renderCmd.setDebugView(RenderCommand::DebugView::Disabled);
            renderCmd.setExportDebugViews(true);
        }

        // Map TextureRenderMode to RenderCommand::RenderMode
        if (renderMode == TextureRenderMode::Preview)
        {
            renderCmd.setRenderMode(RenderCommand::RenderMode::Preview);
        }
        else
        {
            renderCmd.setRenderMode(RenderCommand::RenderMode::Progressive);
        }

        // ===== NEW PATTERN: Initialize all renderers uniformly =====
        const auto tPreload0 = tMeta1;
        bool didRendererPreload = false;

        for (auto &renderer : renderers)
        {
            // Validate renderer can handle the command
            if (!renderer->validateCommand(renderCmd))
            {
                std::cerr << "Renderer " << renderer->getName()
                          << " cannot handle this command\n";
                continue;
            }

            // Initialize renderer with all parameters (no dynamic_cast!)
            if (!renderer->initializeWithCommand(renderCmd))
            {
                std::cerr << "Failed to initialize " << renderer->getName() << "\n";
                continue;
            }

            didRendererPreload = true;
        }

        const auto tPreload1 = Clock::now();

        auto stats = scene.getStats();
        BVHBuilder::Strategy bvhStrategy = BVHBuilder::Strategy::BottomUp;
        if (stats.totalTriangles > 20000)
            bvhStrategy = BVHBuilder::Strategy::TopDown;

        const auto tBVH0 = Clock::now();
        scene.buildGlobalBVH(bvhStrategy);
        const auto tBVH1 = Clock::now();

        fs::path outDir = fs::path("Results") / "GPUFrames";
        std::error_code ec;
        fs::create_directories(outDir, ec);
        if (ec)
        {
            std::cerr << "Failed to create output directory '" << outDir.string()
                      << "': " << ec.message() << "\n";
            return 1;
        }

        outDir = MakeAbsoluteBestEffort(outDir);
        std::cout << "Frame output directory: " << outDir.string() << "\n";

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

                UpdateRendererDimensions(renderers, renderDims.width, renderDims.height);
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
                bool savedAnyFrame = false;
                bool hadSaveOrRenderError = false;

                // Render with all available renderers
                for (auto &renderer : renderers)
                {
                    Renderer &rendererRef = *renderer;
                    const std::string rendererName = rendererRef.getName();
                    std::cout << "  Rendering with " << rendererName << "...\n";

                    // Update dimensions for this camera
                    rendererRef.setImageSize(renderDims.width, renderDims.height);

                    const fs::path frameBasePath =
                        outDir /
                        (MakeSafeName(rendererName) + "_TextureFrame_Cam" + std::to_string(ci) + "_" + camName);

                    auto renderAndSave = [&](const fs::path &targetPath) -> bool
                    {
                        std::vector<Vec3> framebuffer;
                        if (!rendererRef.renderTexture(scene, cam, framebuffer))
                        {
                            std::cerr << "Rendering failed for " << rendererName << "\n";
                            hadSaveOrRenderError = true;
                            return false;
                        }

                        try
                        {
                            SaveFrameBufferToPNG(framebuffer, renderDims.width, renderDims.height, targetPath.string());
                            std::cout << "    Saved frame: " << targetPath.string() << "\n";
                            return true;
                        }
                        catch (const std::exception &saveError)
                        {
                            hadSaveOrRenderError = true;
                            std::cerr << "Failed to save frame for " << rendererName
                                      << ": " << saveError.what() << "\n";
                            return false;
                        }
                    };

                    if (!SetRendererDebugView(rendererRef, RenderCommand::DebugView::Disabled))
                    {
                        hadSaveOrRenderError = true;
                        std::cerr << "Renderer does not support disabled debug view: " << rendererName << "\n";
                        continue;
                    }

                    const fs::path beautyPath = fs::path(frameBasePath.string() + "_0.png");
                    savedAnyFrame = renderAndSave(beautyPath) || savedAnyFrame;

                    if (exportTextureDebugViews)
                    {
                        if (!RendererSupportsTextureDebug(rendererRef))
                        {
                            hadSaveOrRenderError = true;
                            std::cerr << "Renderer " << rendererName
                                      << " does not support -textureDebug passes\n";
                        }
                        else
                        {
                            for (const TextureDebugPass &pass : GetTextureDebugPasses())
                            {
                                if (!SetRendererDebugView(rendererRef, pass.view))
                                {
                                    hadSaveOrRenderError = true;
                                    std::cerr << "Renderer " << rendererName
                                              << " does not support debug pass " << pass.suffix << "\n";
                                    continue;
                                }

                                const fs::path debugPath = fs::path(frameBasePath.string() + pass.suffix);
                                savedAnyFrame = renderAndSave(debugPath) || savedAnyFrame;
                            }
                        }

                        (void)SetRendererDebugView(rendererRef, RenderCommand::DebugView::Disabled);
                    }
                }

                if (savedAnyFrame)
                {
                    std::cout << "Frames for camera: <" << ci << "> generated successfully\n";
                }
                else
                {
                    std::cerr << "No frames were saved for camera: <" << ci << ">\n";
                }
                if (hadSaveOrRenderError)
                {
                    std::cerr << "Camera <" << ci << ">: one or more render/save steps failed\n";
                }
            }
        }
        else
        {
            bool savedAnyFrame = false;
            bool hadSaveOrRenderError = false;

            // Render with all available renderers
            for (auto &renderer : renderers)
            {
                Renderer &rendererRef = *renderer;
                const std::string rendererName = rendererRef.getName();
                std::cout << "Rendering with " << rendererName << "...\n";

                const fs::path frameBasePath =
                    outDir /
                    (MakeSafeName(rendererName) + "_TextureFrame_DefaultCamera");

                auto renderAndSave = [&](const fs::path &targetPath) -> bool
                {
                    std::vector<Vec3> framebuffer;
                    if (!rendererRef.renderTexture(scene, cam, framebuffer))
                    {
                        std::cerr << "Rendering failed for " << rendererName << "\n";
                        hadSaveOrRenderError = true;
                        return false;
                    }

                    try
                    {
                        SaveFrameBufferToPNG(framebuffer, imageWidth, imageHeight, targetPath.string());
                        std::cout << "  Saved frame: " << targetPath.string() << "\n";
                        return true;
                    }
                    catch (const std::exception &saveError)
                    {
                        hadSaveOrRenderError = true;
                        std::cerr << "Failed to save frame for " << rendererName
                                  << ": " << saveError.what() << "\n";
                        return false;
                    }
                };

                if (!SetRendererDebugView(rendererRef, RenderCommand::DebugView::Disabled))
                {
                    hadSaveOrRenderError = true;
                    std::cerr << "Renderer does not support disabled debug view: " << rendererName << "\n";
                    continue;
                }

                const fs::path beautyPath = fs::path(frameBasePath.string() + "_0.png");
                savedAnyFrame = renderAndSave(beautyPath) || savedAnyFrame;

                if (exportTextureDebugViews)
                {
                    if (!RendererSupportsTextureDebug(rendererRef))
                    {
                        hadSaveOrRenderError = true;
                        std::cerr << "Renderer " << rendererName
                                  << " does not support -textureDebug passes\n";
                    }
                    else
                    {
                        for (const TextureDebugPass &pass : GetTextureDebugPasses())
                        {
                            if (!SetRendererDebugView(rendererRef, pass.view))
                            {
                                hadSaveOrRenderError = true;
                                std::cerr << "Renderer " << rendererName
                                          << " does not support debug pass " << pass.suffix << "\n";
                                continue;
                            }

                            const fs::path debugPath = fs::path(frameBasePath.string() + pass.suffix);
                            savedAnyFrame = renderAndSave(debugPath) || savedAnyFrame;
                        }
                    }

                    (void)SetRendererDebugView(rendererRef, RenderCommand::DebugView::Disabled);
                }
            }

            if (savedAnyFrame)
            {
                std::cout << "Frames for default camera generated successfully\n";
            }
            else
            {
                std::cerr << "No frames were saved for default camera\n";
            }
            if (hadSaveOrRenderError)
            {
                std::cerr << "Default camera: one or more render/save steps failed\n";
            }
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
        std::cout << "Load time (scene):   " << msLoad << " ms\n";
        std::cout << "Meta load time:      " << std::chrono::duration<double, std::milli>(tMeta1 - tLoad1).count() << " ms\n";
        if (didRendererPreload)
            std::cout << "Renderer preload time: " << msPreload << " ms\n";
        std::cout << "BVH build time:      " << msBVH << " ms\n";
        std::cout << "Render time:         " << msRender << " ms\n";
        std::cout << "Total time:          " << msTotal << " ms\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
