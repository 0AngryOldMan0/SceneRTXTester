#include "RendererFactory.h"
#include "RendererRegistry.h"
#include "Renderer.h"

std::vector<std::string> RendererFactory::getAvailableRendererNames()
{
    return RendererRegistry::getInstance().getAvailableRenderers();
}

bool RendererFactory::isRendererAvailable(const std::string &name)
{
    return RendererRegistry::getInstance().hasRenderer(name);
}

std::unique_ptr<Renderer> RendererFactory::createRenderer(const std::string &name)
{
    return RendererRegistry::getInstance().createRenderer(name);
}

std::vector<std::unique_ptr<Renderer>> RendererFactory::createAllAvailableRenderers()
{
    std::vector<std::unique_ptr<Renderer>> result;
    for (const auto &name : getAvailableRendererNames())
    {
        auto renderer = createRenderer(name);
        if (renderer)
        {
            result.push_back(std::move(renderer));
        }
    }
    return result;
}

std::unique_ptr<Renderer> RendererFactory::createDefaultRenderer()
{
    // Priority order: HIP → CUDA → Metal → CPU (fallback)
    const std::vector<std::string> priority = {"HIP", "CUDA", "Metal"};

    for (const auto &name : priority)
    {
        if (isRendererAvailable(name))
        {
            return createRenderer(name);
        }
    }

    // All GPU backends failed; return nullptr and let caller decide
    return nullptr;
}

std::string RendererFactory::getRendererDisplayName(const std::string &name)
{
    if (name == "HIP")
        return "AMD HIP GPU Ray Tracer";
    if (name == "CUDA")
        return "NVIDIA CUDA GPU Ray Tracer";
    if (name == "Metal")
        return "Apple Metal GPU Ray Tracer";
    return name;
}
