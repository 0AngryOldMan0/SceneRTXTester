#pragma once

#include <memory>
#include <string>
#include <vector>
#include "../Classes/Renderer.h"
#include "RenderCommand.h"

// Forward declarations
class Scene;

/**
 * @brief Factory for creating and managing renderer instances
 *
 * This class replaces the tight coupling in main.cpp by providing a clean API
 * for:
 * 1. Discovering available renderers
 * 2. Creating renderer instances
 * 3. Initializing renderers with RenderCommand parameters
 * 4. Validating renderer capabilities
 *
 * Eliminates the need for:
 * - Conditional compilation in main.cpp
 * - dynamic_cast for per-renderer configuration
 * - Hard-coded renderer instantiation logic
 *
 * Usage:
 *   RendererFactory factory;
 *   auto renderers = factory.createAllAvailableRenderers();
 *   for (auto &r : renderers) {
 *       if (r->validateCommand(renderCmd)) {
 *           r->initializeWithCommand(renderCmd);
 *       }
 *   }
 */
class RendererFactory
{
public:
    /**
     * @brief Get available renderer names
     * @return Vector of renderer names that can be created
     */
    static std::vector<std::string> getAvailableRendererNames();

    /**
     * @brief Check if a specific renderer is available
     * @param name The renderer name (e.g., "HIP", "CUDA", "Metal")
     * @return true if the renderer can be created
     */
    static bool isRendererAvailable(const std::string &name);

    /**
     * @brief Create a single renderer by name
     * @param name The renderer name
     * @return A unique pointer to the renderer, or nullptr if not found
     */
    static std::unique_ptr<Renderer> createRenderer(const std::string &name);

    /**
     * @brief Create all available renderers
     * @return Vector of all available renderer instances
     */
    static std::vector<std::unique_ptr<Renderer>> createAllAvailableRenderers();

    /**
     * @brief Create the default renderer
     *
     * Priority order: HIP → CUDA → Metal → CPU (fallback)
     *
     * @return A unique pointer to the default renderer, or nullptr if none available
     */
    static std::unique_ptr<Renderer> createDefaultRenderer();

    /**
     * @brief Get human-friendly renderer name
     * @param name The internal renderer name
     * @return Display name for UI/logging
     */
    static std::string getRendererDisplayName(const std::string &name);

private:
    RendererFactory() = delete; // Static class only
};
