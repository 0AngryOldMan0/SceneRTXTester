#include "TestFramework.h"

#include "Scene.h"

namespace
{
    Triangle makeTriangle(const Vec3 &origin)
    {
        Triangle t{};
        t.v0 = origin;
        t.v1 = Vec3{origin.x + 1.0f, origin.y, origin.z};
        t.v2 = Vec3{origin.x, origin.y + 1.0f, origin.z};
        t.normal = Vec3{0.0f, 0.0f, 1.0f};
        return t;
    }
}

TEST_CASE(SystemSmoke, BuildSceneBVHFromTwoObjects)
{
    Scene scene;

    SceneObject a("Mesh");
    a.addTriangle(makeTriangle(Vec3{0.0f, 0.0f, 0.0f}), "MatA");

    SceneObject b("Mesh");
    b.addTriangle(makeTriangle(Vec3{2.0f, 0.0f, 0.0f}), "MatB");

    scene.addObject(std::move(a));
    scene.addObject(std::move(b));

    const auto &objectsAfterInsert = scene.getObjects();
    CHECK_EQ(objectsAfterInsert.size(), static_cast<std::size_t>(2));
    CHECK_EQ(objectsAfterInsert[0].getName(), std::string("Mesh"));
    CHECK_EQ(objectsAfterInsert[1].getName(), std::string("Mesh_1"));

    scene.buildObjectBVHs(BVHBuilder::Strategy::TopDown);
    scene.buildGlobalBVH(BVHBuilder::Strategy::SAH);

    CHECK(scene.hasGlobalBVH());
    CHECK(scene.getGlobalRootIndex() >= 0);
    CHECK(!scene.getGlobalNodes().empty());
    CHECK(!scene.getGlobalTriangles().empty());
    CHECK(!scene.getGlobalInstances().empty());

    const Scene::SceneStats stats = scene.getStats();
    CHECK_EQ(stats.prototypeCount, static_cast<std::size_t>(2));
    CHECK_EQ(stats.uniqueTriangles, static_cast<std::size_t>(2));
    CHECK(stats.globalBVHNodes >= 1);
}
