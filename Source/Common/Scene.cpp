#include "../Headers/Classes/Scene.h"
#include "../Headers/MathUtils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
    static std::uint32_t HashStableString32(const std::string &s)
    {
        constexpr std::uint32_t kOffset = 2166136261u;
        constexpr std::uint32_t kPrime  = 16777619u;

        std::uint32_t hash = kOffset;
        for (unsigned char ch : s)
        {
            hash ^= static_cast<std::uint32_t>(ch);
            hash *= kPrime;
        }
        return hash;
    }

    static SceneTransformMatrix IdentityMatrix()
    {
        return SceneTransformMatrix{
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
    }

    static Vec3 TransformPoint(const SceneTransformMatrix &m, const Vec3 &p)
    {
        return Vec3{
            m[0] * p.x + m[1] * p.y + m[2]  * p.z + m[3],
            m[4] * p.x + m[5] * p.y + m[6]  * p.z + m[7],
            m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]
        };
    }

    static bool InvertAffine3x4(const SceneTransformMatrix &m, float outInv[12], float outNormalToWorld[12])
    {
        const float a00 = m[0], a01 = m[1], a02 = m[2];
        const float a10 = m[4], a11 = m[5], a12 = m[6];
        const float a20 = m[8], a21 = m[9], a22 = m[10];

        const float c00 =  a11 * a22 - a12 * a21;
        const float c01 = -(a10 * a22 - a12 * a20);
        const float c02 =  a10 * a21 - a11 * a20;

        const float c10 = -(a01 * a22 - a02 * a21);
        const float c11 =  a00 * a22 - a02 * a20;
        const float c12 = -(a00 * a21 - a01 * a20);

        const float c20 =  a01 * a12 - a02 * a11;
        const float c21 = -(a00 * a12 - a02 * a10);
        const float c22 =  a00 * a11 - a01 * a10;

        const float det = a00 * c00 + a01 * c01 + a02 * c02;
        if (!(std::abs(det) > 1.0e-20f) || !std::isfinite(det))
            return false;

        const float invDet = 1.0f / det;

        const float i00 = c00 * invDet;
        const float i01 = c10 * invDet;
        const float i02 = c20 * invDet;

        const float i10 = c01 * invDet;
        const float i11 = c11 * invDet;
        const float i12 = c21 * invDet;

        const float i20 = c02 * invDet;
        const float i21 = c12 * invDet;
        const float i22 = c22 * invDet;

        const float tx = m[3];
        const float ty = m[7];
        const float tz = m[11];

        outInv[0] = i00; outInv[1] = i01; outInv[2]  = i02; outInv[3]  = -(i00 * tx + i01 * ty + i02 * tz);
        outInv[4] = i10; outInv[5] = i11; outInv[6]  = i12; outInv[7]  = -(i10 * tx + i11 * ty + i12 * tz);
        outInv[8] = i20; outInv[9] = i21; outInv[10] = i22; outInv[11] = -(i20 * tx + i21 * ty + i22 * tz);

        // local normal -> world = transpose(inverse(objectToWorld linear))
        outNormalToWorld[0]  = i00; outNormalToWorld[1]  = i10; outNormalToWorld[2]  = i20; outNormalToWorld[3]  = 0.0f;
        outNormalToWorld[4]  = i01; outNormalToWorld[5]  = i11; outNormalToWorld[6]  = i21; outNormalToWorld[7]  = 0.0f;
        outNormalToWorld[8]  = i02; outNormalToWorld[9]  = i12; outNormalToWorld[10] = i22; outNormalToWorld[11] = 0.0f;
        return true;
    }

    static AABB ComputeTriangleBounds(const std::vector<Triangle> &tris)
    {
        if (tris.empty())
            return AABB{Vec3{0, 0, 0}, Vec3{0, 0, 0}};

        float minX =  std::numeric_limits<float>::infinity();
        float minY =  std::numeric_limits<float>::infinity();
        float minZ =  std::numeric_limits<float>::infinity();
        float maxX = -std::numeric_limits<float>::infinity();
        float maxY = -std::numeric_limits<float>::infinity();
        float maxZ = -std::numeric_limits<float>::infinity();

        for (const Triangle &tri : tris)
        {
            minX = std::min(minX, std::min(tri.v0.x, std::min(tri.v1.x, tri.v2.x)));
            minY = std::min(minY, std::min(tri.v0.y, std::min(tri.v1.y, tri.v2.y)));
            minZ = std::min(minZ, std::min(tri.v0.z, std::min(tri.v1.z, tri.v2.z)));

            maxX = std::max(maxX, std::max(tri.v0.x, std::max(tri.v1.x, tri.v2.x)));
            maxY = std::max(maxY, std::max(tri.v0.y, std::max(tri.v1.y, tri.v2.y)));
            maxZ = std::max(maxZ, std::max(tri.v0.z, std::max(tri.v1.z, tri.v2.z)));
        }

        return normalizeBox(AABB{Vec3{minX, minY, minZ}, Vec3{maxX, maxY, maxZ}});
    }

    static AABB TransformBounds(const AABB &box, const SceneTransformMatrix &m)
    {
        const Vec3 c000 = TransformPoint(m, Vec3{box.v0.x, box.v0.y, box.v0.z});
        const Vec3 c001 = TransformPoint(m, Vec3{box.v0.x, box.v0.y, box.v1.z});
        const Vec3 c010 = TransformPoint(m, Vec3{box.v0.x, box.v1.y, box.v0.z});
        const Vec3 c011 = TransformPoint(m, Vec3{box.v0.x, box.v1.y, box.v1.z});
        const Vec3 c100 = TransformPoint(m, Vec3{box.v1.x, box.v0.y, box.v0.z});
        const Vec3 c101 = TransformPoint(m, Vec3{box.v1.x, box.v0.y, box.v1.z});
        const Vec3 c110 = TransformPoint(m, Vec3{box.v1.x, box.v1.y, box.v0.z});
        const Vec3 c111 = TransformPoint(m, Vec3{box.v1.x, box.v1.y, box.v1.z});

        Vec3 mn = c000;
        Vec3 mx = c000;
        auto grow = [&](const Vec3 &p)
        {
            mn = vmin(mn, p);
            mx = vmax(mx, p);
        };

        grow(c001); grow(c010); grow(c011);
        grow(c100); grow(c101); grow(c110); grow(c111);

        return AABB{mn, mx};
    }

    static inline bool containsName(const std::unordered_map<std::string, std::size_t>& m, const std::string& name)
    {
        return m.find(name) != m.end();
    }

    static int ComputeBVHDepthLeaf0(const std::vector<BVHNode>& nodes, int root)
    {
        if (root < 0 || static_cast<std::size_t>(root) >= nodes.size())
            return 0;

        int maxDepth = 0;
        std::vector<std::pair<int,int>> stack;
        stack.reserve(128);
        stack.emplace_back(root, 0);

        while (!stack.empty())
        {
            const auto [idx, depth] = stack.back();
            stack.pop_back();

            if (depth > maxDepth)
                maxDepth = depth;

            const BVHNode& n = nodes[static_cast<std::size_t>(idx)];
            if (n.tri >= 0)
                continue;

            if (n.left >= 0 && static_cast<std::size_t>(n.left) < nodes.size())
                stack.emplace_back(n.left, depth + 1);
            if (n.right >= 0 && static_cast<std::size_t>(n.right) < nodes.size())
                stack.emplace_back(n.right, depth + 1);
        }
        return maxDepth;
    }
}

Scene::Scene() : bvhBuilder_(BVHBuilder::Strategy::BottomUp)
{
}

void Scene::invalidateGlobalBVH_()
{
    ++sceneRevision_;
    globalBVHBuilt_ = false;
    globalNodes_.clear();
    globalMeshNodes_.clear();
    globalTriangles_.clear();
    globalInstances_.clear();
    globalRootIndex_ = -1;
}

void Scene::rebuildNameIndex_()
{
    objectIndexByName_.clear();
    objectIndexByName_.reserve(objects_.size());

    for (std::size_t i = 0; i < objects_.size(); ++i)
        objectIndexByName_[objects_[i].getName()] = i;
}

void Scene::addObject(const SceneObject &object)
{
    if (object.isEmpty())
        return;

    SceneObject copy = object;
    addObject(std::move(copy));
}

void Scene::addObject(SceneObject &&object)
{
    if (object.isEmpty())
        return;

    std::string baseName = object.getName();
    if (baseName.empty())
        baseName = "Object";

    std::string unique = baseName;
    int counter = 1;
    while (containsName(objectIndexByName_, unique))
        unique = baseName + "_" + std::to_string(counter++);

    object.setName(unique);

    const std::size_t newIndex = objects_.size();
    objects_.push_back(std::move(object));
    objectIndexByName_[unique] = newIndex;

    invalidateGlobalBVH_();
    invalidateBBox_();
}

void Scene::removeObject(const std::string &name)
{
    auto it = objectIndexByName_.find(name);
    if (it == objectIndexByName_.end())
        return;

    const std::size_t idx = it->second;
    if (idx >= objects_.size())
        return;

    objects_.erase(objects_.begin() + static_cast<std::ptrdiff_t>(idx));
    rebuildNameIndex_();

    invalidateGlobalBVH_();
    invalidateBBox_();
}

void Scene::clearObjects()
{
    objects_.clear();
    objectIndexByName_.clear();

    invalidateGlobalBVH_();
    invalidateBBox_();
}

const SceneObject *Scene::getObject(const std::string &name) const
{
    auto it = objectIndexByName_.find(name);
    if (it == objectIndexByName_.end())
        return nullptr;

    const std::size_t idx = it->second;
    return (idx < objects_.size()) ? &objects_[idx] : nullptr;
}

SceneObject *Scene::getObject(const std::string &name)
{
    auto it = objectIndexByName_.find(name);
    if (it == objectIndexByName_.end())
        return nullptr;

    const std::size_t idx = it->second;
    return (idx < objects_.size()) ? &objects_[idx] : nullptr;
}

void Scene::buildObjectBVHs(BVHBuilder::Strategy strategy)
{
    bvhBuilder_.setStrategy(strategy);
    for (auto &obj : objects_)
    {
        if (!obj.isEmpty())
            bvhBuilder_.buildForObject(obj);
    }
}

void Scene::buildGlobalBVH(BVHBuilder::Strategy strategy)
{
    bvhBuilder_.setStrategy(strategy);
    invalidateGlobalBVH_();

    std::vector<AABB> instanceBounds;
    instanceBounds.reserve(objects_.size());

    for (auto &obj : objects_)
    {
        if (obj.isEmpty())
            continue;

        if (!obj.hasBVH())
            bvhBuilder_.buildForObject(obj);
        if (!obj.hasBVH())
            continue;

        const std::size_t triOffset  = globalTriangles_.size();
        const std::size_t nodeOffset = globalMeshNodes_.size();

        const auto &srcTris = obj.getTriangles();
        const auto &srcNodes = obj.getBVHNodes();

        globalTriangles_.insert(globalTriangles_.end(), srcTris.begin(), srcTris.end());
        globalMeshNodes_.reserve(globalMeshNodes_.size() + srcNodes.size());

        for (const BVHNode &src : srcNodes)
        {
            BVHNode dst = src;
            if (dst.left   >= 0) dst.left   += static_cast<int>(nodeOffset);
            if (dst.right  >= 0) dst.right  += static_cast<int>(nodeOffset);
            if (dst.parent >= 0) dst.parent += static_cast<int>(nodeOffset);
            if (dst.tri    >= 0) dst.tri    += static_cast<int>(triOffset);
            globalMeshNodes_.push_back(dst);
        }

        const int globalBlasRoot = obj.getRootIndex() + static_cast<int>(nodeOffset);
        AABB localBounds = (globalBlasRoot >= 0 && static_cast<std::size_t>(globalBlasRoot) < globalMeshNodes_.size())
                         ? globalMeshNodes_[static_cast<std::size_t>(globalBlasRoot)].box
                         : ComputeTriangleBounds(srcTris);

        const auto &instanceTransforms = obj.getInstanceTransforms();
        const auto &instanceMetadata = obj.getInstanceMetadata();
        const bool useIdentity = instanceTransforms.empty();
        const std::size_t instanceCount = useIdentity ? 1u : instanceTransforms.size();

        for (std::size_t i = 0; i < instanceCount; ++i)
        {
            const SceneTransformMatrix matrix = useIdentity ? IdentityMatrix() : instanceTransforms[i];
            const SceneObjectInstanceMeta *meta =
                (i < instanceMetadata.size()) ? &instanceMetadata[i] : nullptr;

            SceneInstanceGPU inst{};
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 4; ++c)
                {
                    const int idx = r * 4 + c;
                    inst.objectToWorld[idx] = matrix[idx];
                }
            }

            float inv[12];
            float normalToWorld[12];
            if (!InvertAffine3x4(matrix, inv, normalToWorld))
            {
                const SceneTransformMatrix identity = IdentityMatrix();
                for (int r = 0; r < 3; ++r)
                {
                    for (int c = 0; c < 4; ++c)
                    {
                        const int idx = r * 4 + c;
                        inst.objectToWorld[idx] = identity[idx];
                        inv[idx] = identity[idx];
                        normalToWorld[idx] = identity[idx];
                    }
                }
            }

            for (int k = 0; k < 12; ++k)
            {
                inst.worldToObject[k] = inv[k];
                inst.normalToWorld[k] = normalToWorld[k];
            }

            inst.worldBounds = TransformBounds(localBounds, matrix);
            inst.blasRootIndex = globalBlasRoot;
            inst.flags = (meta == nullptr || meta->castsShadow) ? 1u : 0u;
            inst.ownerId = (meta != nullptr && !meta->actorPath.empty())
                         ? HashStableString32(meta->actorPath)
                         : 0u;
            globalInstances_.push_back(inst);
            instanceBounds.push_back(inst.worldBounds);
        }
    }

    if (instanceBounds.empty())
    {
        globalBVHBuilt_ = false;
        globalRootIndex_ = -1;
        return;
    }

    globalRootIndex_ = bvhBuilder_.buildFromAABBs(instanceBounds, globalNodes_);
    globalBVHBuilt_ = (globalRootIndex_ >= 0);
}

AABB Scene::getBoundingBox() const
{
    if (!bboxDirty_)
        return bboxCache_;

    bboxDirty_ = false;

    if (objects_.empty())
    {
        bboxCache_ = AABB{Vec3{0, 0, 0}, Vec3{0, 0, 0}};
        return bboxCache_;
    }

    bool hasAny = false;
    Vec3 minV{ std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity() };
    Vec3 maxV{-std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity() };

    for (const auto &obj : objects_)
    {
        if (obj.isEmpty())
            continue;

        const AABB localBounds = ComputeTriangleBounds(obj.getTriangles());
        const auto &instanceTransforms = obj.getInstanceTransforms();
        const bool useIdentity = instanceTransforms.empty();
        const std::size_t instanceCount = useIdentity ? 1u : instanceTransforms.size();

        for (std::size_t i = 0; i < instanceCount; ++i)
        {
            const SceneTransformMatrix identity = IdentityMatrix();
            const SceneTransformMatrix &m = useIdentity ? identity : instanceTransforms[i];
            const AABB worldBounds = TransformBounds(localBounds, m);

            if (!hasAny)
            {
                minV = worldBounds.v0;
                maxV = worldBounds.v1;
                hasAny = true;
            }
            else
            {
                minV = vmin(minV, worldBounds.v0);
                maxV = vmax(maxV, worldBounds.v1);
            }
        }
    }

    if (!hasAny)
    {
        bboxCache_ = AABB{Vec3{0, 0, 0}, Vec3{0, 0, 0}};
        return bboxCache_;
    }

    bboxCache_ = normalizeBox(AABB{minV, maxV});
    return bboxCache_;
}

Vec3 Scene::getCenter() const
{
    const AABB bbox = getBoundingBox();
    return Vec3{
        (bbox.v0.x + bbox.v1.x) * 0.5f,
        (bbox.v0.y + bbox.v1.y) * 0.5f,
        (bbox.v0.z + bbox.v1.z) * 0.5f
    };
}

Scene::SceneStats Scene::getStats() const
{
    SceneStats stats;
    stats.prototypeCount = objects_.size();

    for (const auto &obj : objects_)
    {
        const std::size_t triCount = obj.getTriangleCount();
        const std::size_t instCount = obj.getInstanceCount();
        stats.uniqueTriangles += triCount;
        stats.totalTriangles += triCount * instCount;
        stats.instanceCount += instCount;
    }

    stats.objectCount = stats.instanceCount;
    stats.globalBVHNodes = globalNodes_.size();
    stats.meshBVHNodes = globalMeshNodes_.size();
    stats.boundingBox = getBoundingBox();

    if (globalBVHBuilt_)
        stats.globalBVHDepth = ComputeBVHDepthLeaf0(globalNodes_, globalRootIndex_);

    return stats;
}

void Scene::addLight(const Light &light)
{
    lights_.push_back(light);
    ++sceneRevision_;
}

void Scene::clearLights()
{
    lights_.clear();
    ++sceneRevision_;
}

const Light &Scene::getMainLight() const
{
    if (lights_.empty())
        throw std::runtime_error("Scene::getMainLight: в сцене нет источников света");
    return lights_.front();
}
