#pragma once

#include <string>
#include <memory>

// Forward declarations
struct SceneMetaResources;

/**
 * @brief Encapsulates all rendering parameters
 * 
 * Replaces scattered configuration and global state with a single command object.
 * This solves the "global state" problem by capturing all parameters upfront.
 * 
 * Advantages:
 * - Eliminates raw pointer passing (SceneMetaResources*)
 * - Centralizes all render parameters in one place
 * - Makes render operations explicitly parameterized
 * - Simplifies function signatures (one parameter instead of many)
 * - Enables easy extension with new parameters
 * 
 * Usage:
 *   RenderCommand cmd;
 *   cmd.setMetadata(&metaRes)
 *      .setSamplesPerPixel(16)
 *      .setImageSize(1920, 1080)
 *      .setAccumulationMode(HIPAccumulationMode::FinalStill);
 *   renderer->executeRender(cmd);
 */
class RenderCommand
{
public:
    /**
     * @brief Accumulation modes (moved from HIPRenderer/MetalRenderer)
     * Now platform-agnostic instead of per-renderer enums
     */
    enum class AccumulationMode
    {
        PreviewProgressive = 0, ///< Display early results with progressive refinement
        FinalStill = 1          ///< Full quality final render
    };

    /**
     * @brief Debug visualization modes (moved from HIPRenderer)
     * Now unified instead of HIP-specific
     */
    enum class DebugView
    {
        Disabled = 0,               ///< Normal rendering
        ShadingNormals = 1,         ///< Shading normal smoothness
        AmbientOcclusion = 2,       ///< Ambient occlusion visualization
        NormalDifference = 3,       ///< Geometric vs shading normal difference
        BaseColor = 4,              ///< Base color map
        Roughness = 5,              ///< Roughness map
        Metallic = 6,               ///< Metallic map
        Emissive = 7,               ///< Emissive map
        VertexColor = 8,            ///< Per-vertex colors
        MaterialModel = 9           ///< Material model ID
    };

    /**
     * @brief Render mode for progressive or preview rendering
     */
    enum class RenderMode
    {
        Progressive = 0, ///< Progressive multi-pass rendering
        Preview = 1      ///< Single quick preview pass
    };

    // ===== Core Parameters =====

    /**
     * @brief Set render resolution
     */
    RenderCommand &setImageSize(int width, int height)
    {
        imageWidth_ = width;
        imageHeight_ = height;
        return *this;
    }

    int getImageWidth() const { return imageWidth_; }
    int getImageHeight() const { return imageHeight_; }

    /**
     * @brief Set samples per pixel
     */
    RenderCommand &setSamplesPerPixel(int samples)
    {
        samplesPerPixel_ = samples;
        return *this;
    }

    int getSamplesPerPixel() const { return samplesPerPixel_; }

    /**
     * @brief Set render mode (progressive or preview)
     */
    RenderCommand &setRenderMode(RenderMode mode)
    {
        renderMode_ = mode;
        return *this;
    }

    RenderMode getRenderMode() const { return renderMode_; }

    // ===== Metadata (replaces raw pointer) =====

    /**
     * @brief Set scene metadata (textures, materials, lights, etc.)
     * Ownership remains with caller; RenderCommand stores reference copy.
     */
    RenderCommand &setMetadata(const SceneMetaResources *metaRes)
    {
        metaResources_ = metaRes;
        return *this;
    }

    const SceneMetaResources *getMetadata() const { return metaResources_; }

    bool hasMetadata() const { return metaResources_ != nullptr; }

    // ===== Accumulation Control (unified from HIP/Metal) =====

    /**
     * @brief Set accumulation mode for GPU progressive rendering
     */
    RenderCommand &setAccumulationMode(AccumulationMode mode)
    {
        accumulationMode_ = mode;
        return *this;
    }

    AccumulationMode getAccumulationMode() const { return accumulationMode_; }

    // ===== Debug Visualization (unified from HIP) =====

    /**
     * @brief Enable debug view (visualization of internal buffers)
     */
    RenderCommand &setDebugView(DebugView view)
    {
        debugView_ = view;
        return *this;
    }

    DebugView getDebugView() const { return debugView_; }

    bool isDebugViewEnabled() const { return debugView_ != DebugView::Disabled; }

    // ===== Export Control =====

    /**
     * @brief Export debug views during rendering
     * Renderer-specific: currently supported by HIP and Metal backends
     */
    RenderCommand &setExportDebugViews(bool exportDebugViews)
    {
        exportDebugViews_ = exportDebugViews;
        return *this;
    }

    bool shouldExportDebugViews() const { return exportDebugViews_; }

    // ===== Reset =====

    /**
     * @brief Reset all parameters to defaults
     */
    void reset()
    {
        imageWidth_ = 1920;
        imageHeight_ = 1080;
        samplesPerPixel_ = 1;
        renderMode_ = RenderMode::Progressive;
        metaResources_ = nullptr;
        accumulationMode_ = AccumulationMode::PreviewProgressive;
        debugView_ = DebugView::Disabled;
        exportDebugViews_ = false;
    }

private:
    // Render target
    int imageWidth_ = 1920;
    int imageHeight_ = 1080;

    // Sampling
    int samplesPerPixel_ = 1;
    RenderMode renderMode_ = RenderMode::Progressive;

    // Scene data (not owned)
    const SceneMetaResources *metaResources_ = nullptr;

    // GPU accumulation modes
    AccumulationMode accumulationMode_ = AccumulationMode::PreviewProgressive;

    // Debug visualization
    DebugView debugView_ = DebugView::Disabled;

    // Export options
    bool exportDebugViews_ = false;
};
