#pragma once

#include "Triangles.h"
#include "BVHNode.h"

#include <array>
#include <string>
#include <vector>
#include <unordered_map>

using SceneTransformMatrix = std::array<float, 16>;

struct SceneObjectInstanceMeta
{
    bool castsShadow = true;
    std::string actorPath;
    std::string componentPath;
};

class SceneObject
{
public:
    SceneObject() = default;
    explicit SceneObject(const std::string &name);

    // ---- Геттеры ----
    const std::string &getName() const;
    const std::vector<Triangle> &getTriangles() const;
    std::vector<Triangle> &getTrianglesMutable();
    const std::vector<BVHNode> &getBVHNodes() const;
    int getRootIndex() const;

    const std::string *getTriangleMaterialName(std::size_t triIndex) const;
    int registerMaterialName(const std::string &matName);

    const std::vector<SceneTransformMatrix> &getInstanceTransforms() const;
    std::vector<SceneTransformMatrix> &getInstanceTransformsMutable();
    const std::vector<SceneObjectInstanceMeta> &getInstanceMetadata() const;
    std::vector<SceneObjectInstanceMeta> &getInstanceMetadataMutable();
    std::size_t getInstanceCount() const;
    bool hasInstances() const;

    const std::string &getMeshAssetId() const;
    const std::string &getMeshSpace() const;

    // ---- Сеттеры / модификация ----
    void setName(const std::string &newName);

    void setTriangles(const std::vector<Triangle> &tris);
    void setTriangles(std::vector<Triangle> &&tris);

    void setBVHNodes(const std::vector<BVHNode> &newNodes);
    void setBVHNodes(std::vector<BVHNode> &&newNodes);

    void setRootIndex(int index);

    void addTriangle(const Triangle &tri);
    void addTriangle(const Triangle &tri, int materialId);
    void addTriangle(const Triangle &tri, const std::string &materialName);

    void addBVHNode(const BVHNode &node);
    void computeTriangleAABBs();

    void setInstanceTransforms(const std::vector<SceneTransformMatrix> &transforms);
    void setInstanceTransforms(std::vector<SceneTransformMatrix> &&transforms);
    void addInstanceTransform(const SceneTransformMatrix &transform);
    void addInstanceTransform(const SceneTransformMatrix &transform,
                              const SceneObjectInstanceMeta &meta);
    void setInstanceMetadata(const std::vector<SceneObjectInstanceMeta> &metadata);
    void setInstanceMetadata(std::vector<SceneObjectInstanceMeta> &&metadata);

    void setMeshAssetId(const std::string &id);
    void setMeshSpace(const std::string &space);

    // ---- Вспомогательные методы ----
    bool isEmpty() const;
    std::size_t getTriangleCount() const;

    bool hasBVH() const;

    const BVHNode &getBVHNode(int index) const;
    const BVHNode &getRootNode() const;

    const Vec3 &getBaseColor() const;
    void setBaseColor(const Vec3 &color);

    void applyMaterial(const Vec3 &baseColor,
                       const Vec3 &emission,
                       float metallic,
                       float roughness);

    void clear();

private:
    friend class BVHBuilder;

    std::string name;
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;
    int rootIndex = -1;

    Vec3 baseColor_{1.0f, 1.0f, 1.0f};

    std::vector<std::string> materialNames_;
    std::vector<int> triangleMaterialIds_;
    std::unordered_map<std::string, int> materialNameToId_;

    std::string meshAssetId_;
    std::string meshSpace_;
    std::vector<SceneTransformMatrix> instanceTransforms_;
    std::vector<SceneObjectInstanceMeta> instanceMetadata_;
};
