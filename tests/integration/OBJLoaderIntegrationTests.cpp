#include "TestFramework.h"

#include "OBJLoader.h"

#include "TestFsUtils.h"

#include <filesystem>

namespace
{
    bool isFinite(float v)
    {
        return std::isfinite(static_cast<double>(v));
    }
}

TEST_CASE(IntegrationOBJLoader, LoadsSingleTriangleWithMaterial)
{
    testfs::TempDir tmp;
    const std::filesystem::path objPath = tmp.path() / "sample.obj";

    const std::string objText =
        "o tri\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "vn 0 0 1\n"
        "usemtl Mat_Track\n"
        "f 1/1/1 2/2/1 3/3/1\n";

    testfs::writeTextFile(objPath, objText);

    OBJLoader loader;
    const auto objects = loader.load(objPath.string());

    CHECK_EQ(objects.size(), static_cast<std::size_t>(1));
    const SceneObject &obj = objects[0];

    CHECK_EQ(obj.getName(), std::string("tri"));
    CHECK_EQ(obj.getTriangleCount(), static_cast<std::size_t>(1));

    const std::string *matName = obj.getTriangleMaterialName(0);
    CHECK(matName != nullptr);
    CHECK_EQ(*matName, std::string("Mat_Track"));

    const Triangle &t = obj.getTriangles()[0];
    CHECK_NEAR(t.normal.z, 1.0f, 1e-5f);
    CHECK_NEAR(TriangleUV(t, 0, 1).x, 1.0f, 1e-6f);
}

TEST_CASE(IntegrationOBJLoader, ComputesFallbackNormalWithoutVN)
{
    testfs::TempDir tmp;
    const std::filesystem::path objPath = tmp.path() / "fallback.obj";

    const std::string objText =
        "o plane\n"
        "v 0 0 0\n"
        "v 2 0 0\n"
        "v 0 2 0\n"
        "f 1 2 3\n";

    testfs::writeTextFile(objPath, objText);

    OBJLoader loader;
    const auto objects = loader.load(objPath.string());

    CHECK_EQ(objects.size(), static_cast<std::size_t>(1));
    CHECK_EQ(objects[0].getTriangleCount(), static_cast<std::size_t>(1));

    const Triangle &t = objects[0].getTriangles()[0];
    CHECK(isFinite(t.normal.x));
    CHECK(isFinite(t.normal.y));
    CHECK(isFinite(t.normal.z));
    CHECK_NEAR(t.normal.z, 1.0f, 1e-5f);
}
