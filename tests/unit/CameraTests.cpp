#include "TestFramework.h"

#include "Camera.h"

#include <cmath>

namespace
{
    void checkVec3Near(const Vec3 &a, const Vec3 &b, float eps)
    {
        CHECK_NEAR(a.x, b.x, eps);
        CHECK_NEAR(a.y, b.y, eps);
        CHECK_NEAR(a.z, b.z, eps);
    }
}

TEST_CASE(UnitCamera, GenerateRayThroughViewportCenter)
{
    Camera camera;
    camera.lookAt(Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f});
    camera.setViewport(3, 3);
    camera.setPerspective(60.0f, 1.0f);

    const Ray ray = camera.generateRay(1.0f, 1.0f);
    checkVec3Near(ray.origin, Vec3{0.0f, 0.0f, 0.0f}, 1e-6f);
    checkVec3Near(ray.direction, camera.getForward(), 1e-5f);
}

TEST_CASE(UnitCamera, HandlesDegenerateUpVector)
{
    Camera camera;
    camera.lookAt(Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f}, Vec3{0.0f, 0.0f, 1.0f});

    const Vec3 right = camera.getRight();
    const Vec3 up = camera.getUp();
    const Vec3 forward = camera.getForward();

    CHECK(std::isfinite(right.x) && std::isfinite(right.y) && std::isfinite(right.z));
    CHECK(std::isfinite(up.x) && std::isfinite(up.y) && std::isfinite(up.z));

    CHECK_NEAR(dot(right, forward), 0.0f, 1e-4f);
    CHECK_NEAR(dot(up, forward), 0.0f, 1e-4f);
}

TEST_CASE(UnitCamera, MoveUsesCameraBasis)
{
    Camera camera;
    camera.lookAt(Vec3{1.0f, 2.0f, 3.0f}, Vec3{1.0f, 3.0f, 3.0f});

    camera.move(Vec3{0.0f, 0.0f, 2.0f});
    const Vec3 pos = camera.getPosition();

    CHECK_NEAR(pos.x, 1.0f, 1e-5f);
    CHECK_NEAR(pos.y, 4.0f, 1e-5f);
    CHECK_NEAR(pos.z, 3.0f, 1e-5f);
}
