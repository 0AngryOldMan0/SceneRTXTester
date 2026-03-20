#pragma once

#include "SceneObject.h"
#include "Light.h"
#include "AABB.h"
#include "BVHNode.h"
#include "Triangles.h"
#include "BVHBuilder.h"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

struct SceneInstanceGPU
{
    float objectToWorld[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    float worldToObject[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    // 3x3 normal transform padded to 3x4 layout for GPU alignment/convenience.
    float normalToWorld[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    AABB worldBounds{};
    int blasRootIndex = -1;
    int _pad0 = 0;
    int _pad1 = 0;
    int _pad2 = 0;
};

class Scene
{
public:
    struct SceneStats
    {
        std::size_t objectCount = 0;       // logical world instances (for backward-compatible console output)
        std::size_t prototypeCount = 0;    // unique mesh+material prototypes
        std::size_t instanceCount = 0;     // same as objectCount for step 2
        std::size_t uniqueTriangles = 0;   // triangles in unique prototypes
        std::size_t totalTriangles = 0;    // expanded triangles across all instances
        std::size_t globalBVHNodes = 0;    // TLAS nodes
        std::size_t meshBVHNodes = 0;      // concatenated BLAS nodes
        int globalBVHDepth = 0;
        AABB boundingBox{};
    };

    Scene();

    void addObject(const SceneObject &object);
    void addObject(SceneObject &&object);
    void removeObject(const std::string &name);
    void clearObjects();

    const SceneObject *getObject(const std::string &name) const;
    SceneObject *getObject(const std::string &name);

    const std::vector<SceneObject> &getObjects() const { return objects_; }

    void buildObjectBVHs(BVHBuilder::Strategy strategy = BVHBuilder::Strategy::BottomUp);
    void buildGlobalBVH(BVHBuilder::Strategy strategy = BVHBuilder::Strategy::BottomUp);

    bool hasGlobalBVH() const { return globalBVHBuilt_; }

    // TLAS
    const std::vector<BVHNode> &getGlobalNodes() const { return globalNodes_; }
    int getGlobalRootIndex() const { return globalRootIndex_; }

    // BLAS + shared geometry
    const std::vector<BVHNode> &getGlobalMeshNodes() const { return globalMeshNodes_; }
    const std::vector<Triangle> &getGlobalTriangles() const { return globalTriangles_; }
    const std::vector<SceneInstanceGPU> &getGlobalInstances() const { return globalInstances_; }

    AABB getBoundingBox() const;
    Vec3 getCenter() const;
    SceneStats getStats() const;

    void addLight(const Light &light);
    void clearLights();
    const std::vector<Light> &getLights() const { return lights_; }
    const Light &getMainLight() const;

private:
    void invalidateGlobalBVH_();
    void rebuildNameIndex_();
    void invalidateBBox_() const { bboxDirty_ = true; }

    std::vector<SceneObject> objects_;
    std::unordered_map<std::string, std::size_t> objectIndexByName_;
    std::vector<Light> lights_;

    // Step 2 runtime representation:
    // - globalTriangles_  : unique prototype triangles concatenated once
    // - globalMeshNodes_  : concatenated BLAS nodes rebased into one array
    // - globalInstances_  : instance transforms + BLAS roots
    // - globalNodes_      : TLAS nodes over instances
    std::vector<Triangle> globalTriangles_;
    std::vector<BVHNode> globalMeshNodes_;
    std::vector<SceneInstanceGPU> globalInstances_;
    std::vector<BVHNode> globalNodes_;
    int globalRootIndex_ = -1;
    bool globalBVHBuilt_ = false;

    BVHBuilder bvhBuilder_;

    mutable bool bboxDirty_ = true;
    mutable AABB bboxCache_{};
};
