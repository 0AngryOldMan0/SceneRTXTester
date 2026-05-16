#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <vector>

// Forward declaration
class Renderer;


class RendererRegistry
{
public:
    using RendererFactory = std::function<std::unique_ptr<Renderer>()>;

    /**
     * @brief Get the singleton instance
     */
    static RendererRegistry &getInstance();

    /**
     * @brief Register a renderer factory
     * @param name Unique name for the renderer (e.g., "HIP", "CUDA", "Metal")
     * @param factory Function that creates a new renderer instance
     * @return true if registered successfully, false if name was already registered
     */
    bool registerRenderer(const std::string &name, RendererFactory factory);

    /**
     * @brief Create a renderer instance by name
     * @param name The renderer name
     * @return A unique pointer to the renderer, or nullptr if not found
     */
    std::unique_ptr<Renderer> createRenderer(const std::string &name) const;

    /**
     * @brief Get all registered renderer names
     * @return Vector of registered renderer names
     */
    std::vector<std::string> getAvailableRenderers() const;

    /**
     * @brief Check if a renderer is registered
     * @param name The renderer name
     * @return true if the renderer is available
     */
    bool hasRenderer(const std::string &name) const;

private:
    RendererRegistry() = default;
    RendererRegistry(const RendererRegistry &) = delete;
    RendererRegistry &operator=(const RendererRegistry &) = delete;

    std::unordered_map<std::string, RendererFactory> factories_;
};

/**
 * @brief Helper macro for self-registering renderers
 *
 * Usage in renderer implementation file:
 *   REGISTER_RENDERER("HIP", []() { return std::make_unique<HIPRenderer>(); });
 *
 * @param name Unique name for the renderer
 * @param factory Lambda or function that creates the renderer
 */
#define REGISTER_RENDERER(name, factory)                                         \
    namespace                                                                    \
    {                                                                            \
        struct RendererRegistrar                                                 \
        {                                                                        \
            RendererRegistrar()                                                  \
            {                                                                    \
                RendererRegistry::getInstance().registerRenderer(name, factory); \
            }                                                                    \
        };                                                                       \
        static RendererRegistrar renderer_registrar;                             \
    }
