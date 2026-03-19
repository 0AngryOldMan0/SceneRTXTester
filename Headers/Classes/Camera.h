#pragma once

#include "Point3D.h"
#include "Ray.h"
#include "MathUtils.h"
#include <cmath>

class Camera
{
public:
    Camera();
    Camera(const Vec3 &position, const Vec3 &target, const Vec3 &up = {0, 0, 1});

    // Установка параметров
    void lookAt(const Vec3 &position, const Vec3 &target, const Vec3 &up = {0, 0, 1});
    void setPerspective(float fovYDegrees, float aspectRatio, float nearPlane = 0.1f, float farPlane = 1000.0f);
    void setViewport(int width, int height);

    // Генерация лучей
    Ray generateRay(float screenX, float screenY, float jitterX = 0.0f, float jitterY = 0.0f) const;

    // Геттеры
    const Vec3 &getPosition() const { return position_; }
    const Vec3 &getForward() const { return forward_; }
    const Vec3 &getUp() const { return up_; }
    const Vec3 &getRight() const { return right_; }
    float getFovY() const { return fovYRadians_; }
    float getAspectRatio() const { return aspectRatio_; }
    int getViewportWidth() const { return viewportWidth_; }
    int getViewportHeight() const { return viewportHeight_; }

    // Фокальная плоскость (для будущего DOF)
    void setFocusDistance(float dist) { focusDistance_ = dist; }
    float getFocusDistance() const { return focusDistance_; }

    // Near/Far clip (будет полезно, если захочешь использовать)
    void setClipPlanes(float nearP, float farP)
    {
        nearPlane_ = nearP;
        farPlane_ = farP;
    }

    float getNearPlane() const { return nearPlane_; }
    float getFarPlane() const { return farPlane_; }

    // Перемещение камеры
    void move(const Vec3 &delta);
    void rotate(float yaw, float pitch); // в радианах (реализация может быть в другом .cpp)

private:
    Vec3 position_{};
    Vec3 forward_{0.0f, 1.0f, 0.0f};
    Vec3 up_{0.0f, 0.0f, 1.0f};
    Vec3 right_{1.0f, 0.0f, 0.0f};

    float fovYRadians_ = 60.0f * (DEG2RAD);
    float aspectRatio_ = 16.0f / 9.0f;
    float nearPlane_ = 0.1f;
    float farPlane_ = 1000.0f;

    int viewportWidth_ = 1920;
    int viewportHeight_ = 1080;

    // Кэши для generateRay(): меньше делений и tan() на каждый пиксель
    float tanHalfFov_ = 0.0f;
    float invViewportWidth_ = 1.0f / 1920.0f;
    float invViewportHeight_ = 1.0f / 1080.0f;

    void updateBasis();
    void updateProjectionCache();

    float focusDistance_ = 10.0f; // расстояние до фокусной плоскости (для будущего DOF)
};
