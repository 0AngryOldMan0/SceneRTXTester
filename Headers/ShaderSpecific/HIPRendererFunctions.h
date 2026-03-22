#pragma once

#include <cstdint>
#include <vector>

#include "Classes/Scene.h"
#include "Structures/BVHNode.h"
#include "Structures/CameraData.h"
#include "Structures/Light.h"
#include "Structures/Triangles.h"

struct SceneMetaResources;

enum class HIPAccumulationMode : std::uint32_t
{
    PreviewProgressive = 0,
    FinalStill = 1
};

bool InitHIPRenderer();
bool PreloadHIPSceneResources(const SceneMetaResources *metaRes);

bool RenderFrameHIPTexture(const std::vector<BVHNode> &tlasNodes,
                           const std::vector<BVHNode> &meshNodes,
                           const std::vector<Triangle> &tris,
                           const std::vector<SceneInstanceGPU> &instances,
                           const std::vector<Light> &lights,
                           std::uint64_t sceneRevision,
                           HIPAccumulationMode accumulationMode,
                           int rootIndex,
                           const CameraDataCPU &cameraCPU,
                           const SceneMetaResources *metaRes,
                           std::vector<Vec3> &framebuffer);

void ResetHIPAccumulation();
