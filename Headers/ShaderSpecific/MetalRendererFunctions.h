#pragma once

#include <vector>
#include <cstdint>
#include "Triangles.h"
#include "AABB.h"
#include "BVHNode.h"
#include "CameraData.h"
#include "Light.h"
#include "Scene.h"
#include "Camera.h"

struct SceneMetaResources;

enum class MetalAccumulationMode : std::uint32_t
{
    PreviewProgressive = 0,
    FinalStill = 1
};

// LevelMeta13+ (PBR):
// - SceneMetaResources может содержать дополнительные текстуры (normal/orm/rough/metal/ao).
// - В текущем «all textures» мосте мы складываем *все* текстуры в один общий массив
//   `array<texture2d<float>, MAX_ALL_TEX> [[texture(2)]]`.
//   Соответственно, индексы в материалах (baseColorTexIndex/normalTexIndex/ormTexIndex/...) указывают
//   непосредственно в этот общий пул.
// - buffer(14)/(15) используется под расширенный MaterialGPU_PBR (layout синхронизирован с MetalRenderer.mm).

// Инициализация Metal (одноразово)
bool InitMetalRenderer();

bool RenderFrameMetalTexture(const std::vector<BVHNode> &tlasNodes,
                             const std::vector<BVHNode> &meshNodes,
                             const std::vector<Triangle> &tris,
                             const std::vector<SceneInstanceGPU> &instances,
                             const std::vector<Light> &lights,
                             std::uint64_t sceneRevision,
                             MetalAccumulationMode accumulationMode,
                             int rootIndex,
                             const CameraDataCPU &cameraCPU,
                             const SceneMetaResources *metaRes,
                             std::vector<Vec3> &framebuffer);

void ResetMetalAccumulation();
