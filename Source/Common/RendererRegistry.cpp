#include "RendererRegistry.h"
#include "Renderer.h"

RendererRegistry &RendererRegistry::getInstance()
{
    static RendererRegistry instance;
    return instance;
}

bool RendererRegistry::registerRenderer(const std::string &name, RendererFactory factory)
{
    auto it = factories_.find(name);
    if (it != factories_.end())
    {
        // Renderer already registered
        return false;
    }

    factories_[name] = factory;
    return true;
}

std::unique_ptr<Renderer> RendererRegistry::createRenderer(const std::string &name) const
{
    auto it = factories_.find(name);
    if (it == factories_.end())
    {
        return nullptr;
    }

    return it->second();
}

std::vector<std::string> RendererRegistry::getAvailableRenderers() const
{
    std::vector<std::string> result;
    for (const auto &pair : factories_)
    {
        result.push_back(pair.first);
    }
    return result;
}

bool RendererRegistry::hasRenderer(const std::string &name) const
{
    return factories_.find(name) != factories_.end();
}
