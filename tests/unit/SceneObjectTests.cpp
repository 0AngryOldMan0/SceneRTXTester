#include "TestFramework.h"

#include "SceneObject.h"

namespace
{
    Triangle makeTriangle(float offset)
    {
        Triangle t{};
        t.v0 = Vec3{offset + 0.0f, 0.0f, 0.0f};
        t.v1 = Vec3{offset + 1.0f, 0.0f, 0.0f};
        t.v2 = Vec3{offset + 0.0f, 1.0f, 0.0f};
        return t;
    }

    SceneTransformMatrix identityTransform()
    {
        return SceneTransformMatrix{
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f};
    }
}

TEST_CASE(UnitSceneObject, MaterialNameMapping)
{
    SceneObject object("mesh");
    object.addTriangle(makeTriangle(0.0f), "Mat_A");
    object.addTriangle(makeTriangle(2.0f), "Mat_A");
    object.addTriangle(makeTriangle(4.0f), "Mat_B");

    CHECK_EQ(object.getTriangleCount(), static_cast<std::size_t>(3));

    const std::string *mat0 = object.getTriangleMaterialName(0);
    const std::string *mat1 = object.getTriangleMaterialName(1);
    const std::string *mat2 = object.getTriangleMaterialName(2);

    CHECK(mat0 != nullptr);
    CHECK(mat1 != nullptr);
    CHECK(mat2 != nullptr);

    CHECK_EQ(*mat0, std::string("Mat_A"));
    CHECK_EQ(*mat1, std::string("Mat_A"));
    CHECK_EQ(*mat2, std::string("Mat_B"));
}

TEST_CASE(UnitSceneObject, InstanceAccounting)
{
    SceneObject object("instanced");
    object.addTriangle(makeTriangle(0.0f));

    CHECK(!object.hasInstances());
    CHECK_EQ(object.getInstanceCount(), static_cast<std::size_t>(1));

    object.addInstanceTransform(identityTransform());
    CHECK(object.hasInstances());
    CHECK_EQ(object.getInstanceCount(), static_cast<std::size_t>(1));

    object.addInstanceTransform(identityTransform());
    CHECK_EQ(object.getInstanceCount(), static_cast<std::size_t>(2));
}

TEST_CASE(UnitSceneObject, ClearResetsState)
{
    SceneObject object("to_clear");
    object.addTriangle(makeTriangle(0.0f), "AnyMat");
    object.addInstanceTransform(identityTransform());

    object.clear();

    CHECK(object.isEmpty());
    CHECK_EQ(object.getName(), std::string(""));
    CHECK_EQ(object.getTriangleCount(), static_cast<std::size_t>(0));
    CHECK_EQ(object.getInstanceCount(), static_cast<std::size_t>(1));
}
