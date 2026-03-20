#include "../Headers/Classes/SceneObject.h"
#include "../Headers/MathUtils.h"

#include <algorithm>
#include <stdexcept>

SceneObject::SceneObject(const std::string &name_)
    : name(name_)
{
}

const std::string &SceneObject::getName() const
{
    return name;
}

const std::vector<Triangle> &SceneObject::getTriangles() const
{
    return triangles;
}

std::vector<Triangle> &SceneObject::getTrianglesMutable()
{
    return triangles;
}

const std::vector<BVHNode> &SceneObject::getBVHNodes() const
{
    return bvhNodes;
}

int SceneObject::getRootIndex() const
{
    return rootIndex;
}

const std::string *SceneObject::getTriangleMaterialName(std::size_t triIndex) const
{
    if (triIndex >= triangleMaterialIds_.size())
        return nullptr;

    const int id = triangleMaterialIds_[triIndex];
    if (id < 0)
        return nullptr;

    const std::size_t idx = static_cast<std::size_t>(id);
    if (idx >= materialNames_.size())
        return nullptr;

    return &materialNames_[idx];
}

int SceneObject::registerMaterialName(const std::string &matName)
{
    if (matName.empty())
        return -1;

    auto it = materialNameToId_.find(matName);
    if (it != materialNameToId_.end())
        return it->second;

    const int id = static_cast<int>(materialNames_.size());
    materialNames_.push_back(matName);
    materialNameToId_.emplace(materialNames_.back(), id);
    return id;
}

const std::vector<SceneTransformMatrix> &SceneObject::getInstanceTransforms() const
{
    return instanceTransforms_;
}

std::vector<SceneTransformMatrix> &SceneObject::getInstanceTransformsMutable()
{
    return instanceTransforms_;
}

std::size_t SceneObject::getInstanceCount() const
{
    return instanceTransforms_.empty() ? 1u : instanceTransforms_.size();
}

bool SceneObject::hasInstances() const
{
    return !instanceTransforms_.empty();
}

const std::string &SceneObject::getMeshAssetId() const
{
    return meshAssetId_;
}

const std::string &SceneObject::getMeshSpace() const
{
    return meshSpace_;
}

void SceneObject::setName(const std::string &newName)
{
    name = newName;
}

void SceneObject::setTriangles(const std::vector<Triangle> &tris)
{
    triangles = tris;
    triangleMaterialIds_.assign(triangles.size(), -1);
}

void SceneObject::setTriangles(std::vector<Triangle> &&tris)
{
    triangles = std::move(tris);
    triangleMaterialIds_.assign(triangles.size(), -1);
}

void SceneObject::setBVHNodes(const std::vector<BVHNode> &newNodes)
{
    bvhNodes = newNodes;
}

void SceneObject::setBVHNodes(std::vector<BVHNode> &&newNodes)
{
    bvhNodes = std::move(newNodes);
}

void SceneObject::setRootIndex(int index)
{
    rootIndex = index;
}

void SceneObject::addTriangle(const Triangle &tri)
{
    triangles.push_back(tri);
    triangleMaterialIds_.push_back(-1);
}

void SceneObject::addTriangle(const Triangle &tri, int materialId)
{
    triangles.push_back(tri);
    triangleMaterialIds_.push_back(materialId);
}

void SceneObject::addTriangle(const Triangle &tri, const std::string &materialName)
{
    const int id = registerMaterialName(materialName);
    addTriangle(tri, id);
}

void SceneObject::addBVHNode(const BVHNode &node)
{
    bvhNodes.push_back(node);
}

void SceneObject::setInstanceTransforms(const std::vector<SceneTransformMatrix> &transforms)
{
    instanceTransforms_ = transforms;
}

void SceneObject::setInstanceTransforms(std::vector<SceneTransformMatrix> &&transforms)
{
    instanceTransforms_ = std::move(transforms);
}

void SceneObject::addInstanceTransform(const SceneTransformMatrix &transform)
{
    instanceTransforms_.push_back(transform);
}

void SceneObject::setMeshAssetId(const std::string &id)
{
    meshAssetId_ = id;
}

void SceneObject::setMeshSpace(const std::string &space)
{
    meshSpace_ = space;
}

bool SceneObject::isEmpty() const
{
    return triangles.empty();
}

std::size_t SceneObject::getTriangleCount() const
{
    return triangles.size();
}

bool SceneObject::hasBVH() const
{
    return !bvhNodes.empty() &&
           rootIndex >= 0 &&
           static_cast<std::size_t>(rootIndex) < bvhNodes.size();
}

const BVHNode &SceneObject::getBVHNode(int index) const
{
    if (index < 0 || static_cast<std::size_t>(index) >= bvhNodes.size())
        throw std::out_of_range("SceneObject::getBVHNode: index out of range");

    return bvhNodes[static_cast<std::size_t>(index)];
}

const BVHNode &SceneObject::getRootNode() const
{
    return getBVHNode(rootIndex);
}

const Vec3 &SceneObject::getBaseColor() const
{
    return baseColor_;
}

void SceneObject::setBaseColor(const Vec3 &color)
{
    baseColor_ = color;
}

void SceneObject::applyMaterial(const Vec3 &baseColor,
                                const Vec3 &emission,
                                float metallic,
                                float roughness)
{
    baseColor_ = baseColor;

    for (auto &tri : triangles)
    {
        tri.color = baseColor;
        tri.emission = emission;
        tri.metallic = metallic;
        tri.roughness = roughness;
    }
}

void SceneObject::computeTriangleAABBs()
{
    for (auto &tri : triangles)
    {
        const float minX = std::min(tri.v0.x, std::min(tri.v1.x, tri.v2.x));
        const float minY = std::min(tri.v0.y, std::min(tri.v1.y, tri.v2.y));
        const float minZ = std::min(tri.v0.z, std::min(tri.v1.z, tri.v2.z));

        const float maxX = std::max(tri.v0.x, std::max(tri.v1.x, tri.v2.x));
        const float maxY = std::max(tri.v0.y, std::max(tri.v1.y, tri.v2.y));
        const float maxZ = std::max(tri.v0.z, std::max(tri.v1.z, tri.v2.z));

        tri.ABoBa.v0 = Vec3{minX, minY, minZ};
        tri.ABoBa.v1 = Vec3{maxX, maxY, maxZ};
    }
}

void SceneObject::clear()
{
    name.clear();
    triangles.clear();
    bvhNodes.clear();
    triangleMaterialIds_.clear();
    materialNames_.clear();
    materialNameToId_.clear();
    rootIndex = -1;
    baseColor_ = Vec3{1.0f, 1.0f, 1.0f};
    meshAssetId_.clear();
    meshSpace_.clear();
    instanceTransforms_.clear();
}
