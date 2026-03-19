#pragma once

#include "SceneObject.h"
#include "BVHBuilder.h"
#include "MathUtils.h"
#include "Structures/CameraData.h"
#include "Structures/Light.h"

#include <vector>
#include <string>
#include <unordered_map>

class Scene
{
public:
    Scene();
    ~Scene() = default;

    // Управление объектами
    void addObject(const SceneObject &object);
    void addObject(SceneObject &&object); // ускоряет загрузчики (move)
    void removeObject(const std::string &name);
    void clearObjects();

    // Построение BVH
    void buildObjectBVHs(BVHBuilder::Strategy strategy = BVHBuilder::Strategy::BottomUp);
    void buildGlobalBVH(BVHBuilder::Strategy strategy = BVHBuilder::Strategy::BottomUp);

    // Доступ к объектам и BVH
    const std::vector<SceneObject> &getObjects() const { return objects_; }
    const SceneObject *getObject(const std::string &name) const;
    SceneObject *getObject(const std::string &name);

    bool hasGlobalBVH() const { return globalBVHBuilt_; }
    int getGlobalRootIndex() const { return globalRootIndex_; }
    const std::vector<BVHNode> &getGlobalNodes() const { return globalNodes_; }
    const std::vector<Triangle> &getGlobalTriangles() const { return globalTriangles_; }

    // Работа с AABB сцены
    AABB getBoundingBox() const;
    Vec3 getCenter() const;

    // Статистика
    struct SceneStats
    {
        size_t objectCount = 0;
        size_t totalTriangles = 0;
        size_t globalBVHNodes = 0;
        int globalBVHDepth = 0;
        AABB boundingBox;
    };

    SceneStats getStats() const;

    // --- Свет ---
    void addLight(const Light &light);
    void clearLights();
    const std::vector<Light> &getLights() const { return lights_; }
    bool hasLights() const { return !lights_.empty(); }

    // Удобный хелпер: “главный” свет (первый)
    const Light &getMainLight() const;

private:
    std::vector<SceneObject> objects_;

    // Быстрый доступ по имени (ускоряет getObject/removeObject и уникализацию имён)
    std::unordered_map<std::string, std::size_t> objectIndexByName_;

    std::vector<BVHNode> globalNodes_;
    std::vector<Triangle> globalTriangles_;
    int globalRootIndex_ = -1;
    bool globalBVHBuilt_ = false;

    BVHBuilder bvhBuilder_;
    std::vector<Light> lights_;

    // Кэш bbox сцены (bbox редко меняется, но часто читается)
    mutable bool bboxDirty_ = true;
    mutable AABB bboxCache_{Vec3{0,0,0}, Vec3{0,0,0}};

    void collectGlobalTriangles();
    void rebuildNameIndex_();

    void invalidateGlobalBVH_();
    void invalidateBBox_() const { bboxDirty_ = true; }
};
