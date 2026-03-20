#include "../Headers/Classes/Camera.h"
#include "../Headers/MathUtils.h"
#include <cmath>
#include <algorithm>

namespace
{
    inline Vec3 safeNormalize(const Vec3 &v, const Vec3 &fallback)
    {
        const float l2 = v.x * v.x + v.y * v.y + v.z * v.z;
        if (l2 <= 1e-20f)
            return fallback;
        const float invL = 1.0f / std::sqrt(l2);
        return Vec3{v.x * invL, v.y * invL, v.z * invL};
    }

    inline Vec3 pickNonParallelUp(const Vec3 &forward)
    {
        // Если forward почти параллелен +Z/-Z, берём Y как "up", иначе Z.
        return (std::fabs(forward.z) < 0.999f) ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{0.0f, 1.0f, 0.0f};
    }
}

Camera::Camera()
{
    lookAt(Vec3{0, 0, 0}, Vec3{0, 1, 0});
    updateProjectionCache();
}

Camera::Camera(const Vec3 &position, const Vec3 &target, const Vec3 &up)
{
    lookAt(position, target, up);
    updateProjectionCache();
}

void Camera::updateProjectionCache()
{
    // Защита от некорректных значений
    const float fov = std::max(1e-6f, fovYRadians_);
    tanHalfFov_ = std::tan(0.5f * fov);

    const int w = std::max(1, viewportWidth_);
    const int h = std::max(1, viewportHeight_);
    invViewportWidth_ = 1.0f / static_cast<float>(w);
    invViewportHeight_ = 1.0f / static_cast<float>(h);
}

void Camera::lookAt(const Vec3 &position, const Vec3 &target, const Vec3 &up)
{
    position_ = position;

    // forward
    const Vec3 f = Vec3{target.x - position.x, target.y - position.y, target.z - position.z};
    forward_ = safeNormalize(f, Vec3{0.0f, 1.0f, 0.0f});

    // up / right (робастно при вырожденном up)
    const Vec3 worldUp = safeNormalize(up, pickNonParallelUp(forward_));
    Vec3 r = cross(forward_, worldUp);

    // Если forward и up оказались почти параллельны — берём альтернативный up
    const float rl2 = r.x * r.x + r.y * r.y + r.z * r.z;
    if (rl2 <= 1e-20f)
    {
        const Vec3 altUp = pickNonParallelUp(forward_);
        r = cross(forward_, altUp);
    }

    right_ = safeNormalize(r, Vec3{1.0f, 0.0f, 0.0f});
    up_ = cross(right_, forward_);
}

void Camera::setPerspective(float fovYDegrees, float aspectRatio, float nearPlane, float farPlane)
{
    fovYRadians_ = fovYDegrees * DEG2RAD;
    aspectRatio_ = aspectRatio;
    nearPlane_ = nearPlane;
    farPlane_ = farPlane;
    updateProjectionCache();
}

void Camera::setViewport(int width, int height)
{
    viewportWidth_ = std::max(1, width);
    viewportHeight_ = std::max(1, height);
    updateProjectionCache();
}

Ray Camera::generateRay(float screenX, float screenY, float jitterX, float jitterY) const
{
    // Применяем джиттер (+0.5 = центр пикселя)
    const float fx = (screenX + 0.5f + jitterX);
    const float fy = (screenY + 0.5f + jitterY);

    // NDC [-1, 1]
    const float ndcX = (fx * invViewportWidth_) * 2.0f - 1.0f;
    float ndcY = (fy * invViewportHeight_) * 2.0f - 1.0f;
    ndcY = -ndcY; // Инвертируем Y

    // Направление в мировых координатах:
    // dir = right*(ndcX*aspect*tanHalfFov) + up*(ndcY*tanHalfFov) + forward
    const float dx = ndcX * (aspectRatio_ * tanHalfFov_);
    const float dy = ndcY * tanHalfFov_;

    Vec3 dirWorld{
        right_.x * dx + up_.x * dy + forward_.x,
        right_.y * dx + up_.y * dy + forward_.y,
        right_.z * dx + up_.z * dy + forward_.z};

    dirWorld = normalize(dirWorld);
    return Ray{position_, dirWorld};
}

void Camera::move(const Vec3 &delta)
{
    // delta.x вдоль right, delta.y вдоль up, delta.z вдоль forward
    position_.x += delta.x * right_.x + delta.y * up_.x + delta.z * forward_.x;
    position_.y += delta.x * right_.y + delta.y * up_.y + delta.z * forward_.y;
    position_.z += delta.x * right_.z + delta.y * up_.z + delta.z * forward_.z;
}

void Camera::updateBasis()
{
    forward_ = safeNormalize(forward_, Vec3{0.0f, 1.0f, 0.0f});

    Vec3 r = cross(forward_, up_);
    const float rl2 = r.x * r.x + r.y * r.y + r.z * r.z;
    if (rl2 <= 1e-20f)
    {
        const Vec3 altUp = pickNonParallelUp(forward_);
        r = cross(forward_, altUp);
    }

    right_ = safeNormalize(r, Vec3{1.0f, 0.0f, 0.0f});
    up_ = cross(right_, forward_);
}
