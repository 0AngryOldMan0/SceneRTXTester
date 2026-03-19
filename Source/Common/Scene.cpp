// Scene.cpp
#include "../Headers/Classes/Scene.h"

#include <algorithm>
#include <stdexcept>
#include <limits>

Scene::Scene() : bvhBuilder_(BVHBuilder::Strategy::BottomUp)
{
}

void Scene::invalidateGlobalBVH_()
{
    globalBVHBuilt_ = false;
    globalNodes_.clear();
    globalTriangles_.clear();
    globalRootIndex_ = -1;
}

void Scene::rebuildNameIndex_()
{
    objectIndexByName_.clear();
    objectIndexByName_.reserve(objects_.size());

    for (std::size_t i = 0; i < objects_.size(); ++i)
    {
        objectIndexByName_[objects_[i].getName()] = i;
    }
}

static inline bool containsName(const std::unordered_map<std::string, std::size_t>& m, const std::string& name)
{
    return m.find(name) != m.end();
}

void Scene::addObject(const SceneObject &object)
{
    if (object.isEmpty())
        return;

    SceneObject objCopy = object;
    addObject(std::move(objCopy));
}

void Scene::addObject(SceneObject &&object)
{
    if (object.isEmpty())
        return;

    // Уникализируем имя без линейного поиска по vector
    std::string baseName = object.getName();
    if (baseName.empty())
        baseName = "Object";

    std::string unique = baseName;
    int counter = 1;
    while (containsName(objectIndexByName_, unique))
    {
        unique = baseName + "_" + std::to_string(counter++);
    }

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

    // Сохраняем порядок объектов (erase), но затем перестраиваем индекс
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
        bvhBuilder_.buildForObject(obj);
    }
}

void Scene::collectGlobalTriangles()
{
    globalTriangles_.clear();

    std::size_t total = 0;
    for (const auto &obj : objects_)
        total += obj.getTriangles().size();

    globalTriangles_.reserve(total);

    for (const auto &obj : objects_)
    {
        const auto &tris = obj.getTriangles();
        globalTriangles_.insert(globalTriangles_.end(), tris.begin(), tris.end());
    }
}

void Scene::buildGlobalBVH(BVHBuilder::Strategy strategy)
{
    bvhBuilder_.setStrategy(strategy);

    // Собираем все треугольники
    collectGlobalTriangles();

    // Вычисляем AABB для всех треугольников
    const std::size_t triCount = globalTriangles_.size();
    std::vector<AABB> globalAABBs;
    globalAABBs.resize(triCount);

    for (std::size_t i = 0; i < triCount; ++i)
    {
        const Triangle &tri = globalTriangles_[i];

        const float minX = std::min(tri.v0.x, std::min(tri.v1.x, tri.v2.x));
        const float minY = std::min(tri.v0.y, std::min(tri.v1.y, tri.v2.y));
        const float minZ = std::min(tri.v0.z, std::min(tri.v1.z, tri.v2.z));

        const float maxX = std::max(tri.v0.x, std::max(tri.v1.x, tri.v2.x));
        const float maxY = std::max(tri.v0.y, std::max(tri.v1.y, tri.v2.y));
        const float maxZ = std::max(tri.v0.z, std::max(tri.v1.z, tri.v2.z));

        globalAABBs[i] = AABB{Vec3{minX, minY, minZ}, Vec3{maxX, maxY, maxZ}};
    }

    // Строим глобальный BVH (не трогаем globalTriangles_)
    globalNodes_.clear();
    globalRootIndex_ = bvhBuilder_.buildFromAABBs(globalAABBs, globalNodes_);
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

    float minX =  std::numeric_limits<float>::infinity();
    float minY =  std::numeric_limits<float>::infinity();
    float minZ =  std::numeric_limits<float>::infinity();
    float maxX = -std::numeric_limits<float>::infinity();
    float maxY = -std::numeric_limits<float>::infinity();
    float maxZ = -std::numeric_limits<float>::infinity();

    for (const auto &obj : objects_)
    {
        const auto &tris = obj.getTriangles();
        for (const auto &tri : tris)
        {
            // Используем вершины (не зависим от tri.ABoBa)
            const float tminX = std::min(tri.v0.x, std::min(tri.v1.x, tri.v2.x));
            const float tminY = std::min(tri.v0.y, std::min(tri.v1.y, tri.v2.y));
            const float tminZ = std::min(tri.v0.z, std::min(tri.v1.z, tri.v2.z));

            const float tmaxX = std::max(tri.v0.x, std::max(tri.v1.x, tri.v2.x));
            const float tmaxY = std::max(tri.v0.y, std::max(tri.v1.y, tri.v2.y));
            const float tmaxZ = std::max(tri.v0.z, std::max(tri.v1.z, tri.v2.z));

            minX = std::min(minX, tminX);
            minY = std::min(minY, tminY);
            minZ = std::min(minZ, tminZ);
            maxX = std::max(maxX, tmaxX);
            maxY = std::max(maxY, tmaxY);
            maxZ = std::max(maxZ, tmaxZ);
        }
    }

    if (!std::isfinite(minX))
    {
        bboxCache_ = AABB{Vec3{0, 0, 0}, Vec3{0, 0, 0}};
        return bboxCache_;
    }

    bboxCache_ = normalizeBox(AABB{Vec3{minX, minY, minZ}, Vec3{maxX, maxY, maxZ}});
    return bboxCache_;
}

Vec3 Scene::getCenter() const
{
    const AABB bbox = getBoundingBox();
    return Vec3{
        (bbox.v0.x + bbox.v1.x) * 0.5f,
        (bbox.v0.y + bbox.v1.y) * 0.5f,
        (bbox.v0.z + bbox.v1.z) * 0.5f};
}

static int ComputeBVHDepthLeaf0(const std::vector<BVHNode>& nodes, int root)
{
    if (root < 0 || static_cast<std::size_t>(root) >= nodes.size())
        return 0;

    int maxDepth = 0;
    // stack of (node, depth) where leaf depth = 0
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

Scene::SceneStats Scene::getStats() const
{
    SceneStats stats;
    stats.objectCount = objects_.size();

    for (const auto &obj : objects_)
        stats.totalTriangles += obj.getTriangleCount();

    stats.globalBVHNodes = globalNodes_.size();
    stats.boundingBox = getBoundingBox();

    if (globalBVHBuilt_)
        stats.globalBVHDepth = ComputeBVHDepthLeaf0(globalNodes_, globalRootIndex_);

    return stats;
}

void Scene::addLight(const Light &light)
{
    lights_.push_back(light);
}

void Scene::clearLights()
{
    lights_.clear();
}

const Light &Scene::getMainLight() const
{
    if (lights_.empty())
        throw std::runtime_error("Scene::getMainLight: в сцене нет источников света");
    return lights_.front();
}
