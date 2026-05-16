#include "TestFramework.h"

#include "MathUtils.h"

namespace
{
    void checkVec3Near(const Vec3 &a, const Vec3 &b, float eps)
    {
        CHECK_NEAR(a.x, b.x, eps);
        CHECK_NEAR(a.y, b.y, eps);
        CHECK_NEAR(a.z, b.z, eps);
    }
}

TEST_CASE(UnitMathUtils, DotAndCross)
{
    const Vec3 a{1.0f, 2.0f, 3.0f};
    const Vec3 b{-4.0f, 5.0f, -6.0f};

    CHECK_NEAR(dot(a, b), -12.0f, 1e-6f);

    const Vec3 c = cross(a, b);
    checkVec3Near(c, Vec3{-27.0f, -6.0f, 13.0f}, 1e-6f);
}

TEST_CASE(UnitMathUtils, NormalizeAndClamp)
{
    const Vec3 n = normalize(Vec3{0.0f, 3.0f, 4.0f});
    checkVec3Near(n, Vec3{0.0f, 0.6f, 0.8f}, 1e-5f);

    const Vec3 zero = normalize(Vec3{0.0f, 0.0f, 0.0f});
    checkVec3Near(zero, Vec3{0.0f, 0.0f, 0.0f}, 1e-6f);

    CHECK_NEAR(clamp01(-0.4f), 0.0f, 1e-6f);
    CHECK_NEAR(clamp01(0.3f), 0.3f, 1e-6f);
    CHECK_NEAR(clamp01(2.2f), 1.0f, 1e-6f);
}

TEST_CASE(UnitMathUtils, AABBHelpers)
{
    const AABB a{Vec3{2.0f, 0.0f, 1.0f}, Vec3{0.0f, 2.0f, 3.0f}};
    const AABB b{Vec3{-2.0f, -1.0f, 0.0f}, Vec3{-1.0f, 1.0f, 2.0f}};

    const AABB na = normalizeBox(a);
    checkVec3Near(na.v0, Vec3{0.0f, 0.0f, 1.0f}, 1e-6f);
    checkVec3Near(na.v1, Vec3{2.0f, 2.0f, 3.0f}, 1e-6f);

    const AABB u = unite(a, b);
    checkVec3Near(u.v0, Vec3{-2.0f, -1.0f, 0.0f}, 1e-6f);
    checkVec3Near(u.v1, Vec3{2.0f, 2.0f, 3.0f}, 1e-6f);

    CHECK_NEAR(surfaceArea(na), 24.0f, 1e-6f);
}

TEST_CASE(UnitMathUtils, RayTriangleIntersection)
{
    Triangle tri{};
    tri.v0 = Vec3{0.0f, 0.0f, 0.0f};
    tri.v1 = Vec3{1.0f, 0.0f, 0.0f};
    tri.v2 = Vec3{0.0f, 1.0f, 0.0f};

    Ray hitRay{};
    hitRay.origin = Vec3{0.2f, 0.2f, -1.0f};
    hitRay.direction = Vec3{0.0f, 0.0f, 1.0f};

    float tHit = -1.0f;
    CHECK(IntersectTriangle(hitRay, tri, tHit));
    CHECK_NEAR(tHit, 1.0f, 1e-6f);

    Ray missRay{};
    missRay.origin = Vec3{1.2f, 1.2f, -1.0f};
    missRay.direction = Vec3{0.0f, 0.0f, 1.0f};

    float tMiss = -1.0f;
    CHECK(!IntersectTriangle(missRay, tri, tMiss));
}
