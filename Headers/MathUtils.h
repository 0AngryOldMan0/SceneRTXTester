#pragma once

#include "Point3D.h"
#include "Ray.h"
#include "AABB.h"
#include "Triangles.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

static constexpr float PI = 3.14159265358979323846f;
static constexpr float DEG2RAD = PI / 180.0f;
static constexpr float INV_4PI = 1.0f / (4.0f * PI);
static constexpr float LIGHT_EXPOSURE = 2.5f;

// -----------------------------
// Vec3 math (CPU)
// -----------------------------

inline Vec3 cross(const Vec3 &a, const Vec3 &b) noexcept
{
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

inline float dot(const Vec3 &a, const Vec3 &b) noexcept
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Backward-compat: старое имя (используется в проекте)
inline float ScalarMUL(const Vec3 &a, const Vec3 &b) noexcept
{
    return dot(a, b);
}

inline float length2(const Vec3 &v) noexcept
{
    return dot(v, v);
}

inline Vec3 normalize(const Vec3 &v) noexcept
{
    const float lenSq = length2(v);
    // Если lenSq == 0 или NaN, нормализовать нельзя → возвращаем ноль (как и раньше)
    if (!(lenSq > 0.0f))
        return Vec3{0.f, 0.f, 0.f};

    const float invLen = 1.0f / std::sqrt(lenSq);
    return Vec3{v.x * invLen, v.y * invLen, v.z * invLen};
}

inline bool is_finite_vec(const Vec3 &v) noexcept
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// Быстрее, чем std::min/std::max (внутри — один compare + select)
inline Vec3 vmin(const Vec3 &a, const Vec3 &b) noexcept
{
    return Vec3{
        (a.x < b.x) ? a.x : b.x,
        (a.y < b.y) ? a.y : b.y,
        (a.z < b.z) ? a.z : b.z};
}

inline Vec3 vmax(const Vec3 &a, const Vec3 &b) noexcept
{
    return Vec3{
        (a.x > b.x) ? a.x : b.x,
        (a.y > b.y) ? a.y : b.y,
        (a.z > b.z) ? a.z : b.z};
}

// -----------------------------
// AABB helpers
// -----------------------------

inline AABB normalizeBox(const AABB &b) noexcept
{
    return AABB{vmin(b.v0, b.v1), vmax(b.v0, b.v1)};
}

// Быстрый unite, если AABB уже нормализованы (v0 <= v1 по всем осям)
inline AABB uniteNormalized(const AABB &a, const AABB &b) noexcept
{
    return AABB{vmin(a.v0, b.v0), vmax(a.v1, b.v1)};
}

// Безопасный unite (с нормализацией входов)
inline AABB unite(const AABB &a, const AABB &b) noexcept
{
    const AABB an = normalizeBox(a);
    const AABB bn = normalizeBox(b);
    return uniteNormalized(an, bn);
}

// Площадь поверхности для нормализованного AABB
inline float surfaceAreaNormalized(const AABB &b) noexcept
{
    // На всякий случай оставляем clamp по нулю — защищает от накопления отрицательных размеров
    const float dx = (b.v1.x - b.v0.x);
    const float dy = (b.v1.y - b.v0.y);
    const float dz = (b.v1.z - b.v0.z);

    const float adx = (dx > 0.0f) ? dx : 0.0f;
    const float ady = (dy > 0.0f) ? dy : 0.0f;
    const float adz = (dz > 0.0f) ? dz : 0.0f;

    return 2.0f * (adx * ady + adx * adz + ady * adz);
}

// Безопасная версия (нормализует вход один раз)
inline float surfaceArea(const AABB &b) noexcept
{
    return surfaceAreaNormalized(normalizeBox(b));
}

// Стоимость объединения (SA = SA(union) - SA(a) - SA(b))
// Оптимизация: нормализуем входы ОДИН раз, избегая тройной normalize в старом коде.
inline float mergeCostNormalized(const AABB &a, const AABB &b) noexcept
{
    const AABB u = uniteNormalized(a, b);
    return surfaceAreaNormalized(u) - surfaceAreaNormalized(a) - surfaceAreaNormalized(b);
}

inline float mergeCost(const AABB &a, const AABB &b) noexcept
{
    const AABB an = normalizeBox(a);
    const AABB bn = normalizeBox(b);
    return mergeCostNormalized(an, bn);
}

// -----------------------------
// RNG (CPU), согласовано с Metal
// -----------------------------

inline float Rand01(std::uint32_t seed) noexcept
{
    // xorshift32
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);

    // Нормируем в [0,1]. Деление на (2^32-1) даёт 1.0 при seed=0xFFFFFFFF.
    constexpr float kInv = 1.0f / 4294967295.0f; // (2^32-1)
    return static_cast<float>(seed) * kInv;
}

// -----------------------------
// Ray / Triangle intersection (CPU)
// Möller–Trumbore, double-sided
// -----------------------------

inline bool IntersectTriangle(const Ray &ray, const Triangle &tri, float &tHitOut) noexcept
{
    // Чуть более «плотный» EPS уменьшает ложные пропуски на больших сценах,
    // но оставим по смыслу как было.
    constexpr float EPS = 1e-6f;

    // Загружаем в локалы (компилятору проще держать в регистрах)
    const float ox = ray.origin.x, oy = ray.origin.y, oz = ray.origin.z;
    const float dx = ray.direction.x, dy = ray.direction.y, dz = ray.direction.z;

    const float v0x = tri.v0.x, v0y = tri.v0.y, v0z = tri.v0.z;
    const float v1x = tri.v1.x, v1y = tri.v1.y, v1z = tri.v1.z;
    const float v2x = tri.v2.x, v2y = tri.v2.y, v2z = tri.v2.z;

    // edge1 = v1 - v0
    const float e1x = v1x - v0x;
    const float e1y = v1y - v0y;
    const float e1z = v1z - v0z;

    // edge2 = v2 - v0
    const float e2x = v2x - v0x;
    const float e2y = v2y - v0y;
    const float e2z = v2z - v0z;

    // pvec = dir x edge2
    const float px = dy * e2z - dz * e2y;
    const float py = dz * e2x - dx * e2z;
    const float pz = dx * e2y - dy * e2x;

    // det = edge1 · pvec
    const float det = e1x * px + e1y * py + e1z * pz;
    if (std::fabs(det) < EPS)
        return false;

    const float invDet = 1.0f / det;

    // tvec = origin - v0
    const float tx = ox - v0x;
    const float ty = oy - v0y;
    const float tz = oz - v0z;

    // u = tvec · pvec * invDet
    const float u = (tx * px + ty * py + tz * pz) * invDet;
    if (u < 0.0f || u > 1.0f)
        return false;

    // qvec = tvec x edge1
    const float qx = ty * e1z - tz * e1y;
    const float qy = tz * e1x - tx * e1z;
    const float qz = tx * e1y - ty * e1x;

    // v = dir · qvec * invDet
    const float v = (dx * qx + dy * qy + dz * qz) * invDet;
    if (v < 0.0f || (u + v) > 1.0f)
        return false;

    // t = edge2 · qvec * invDet
    const float t = (e2x * qx + e2y * qy + e2z * qz) * invDet;
    if (t <= EPS)
        return false;

    tHitOut = t;
    return true;
}

// -----------------------------
// Misc
// -----------------------------

inline float clamp01(float x) noexcept
{
    // std::clamp может быть не быстрее на всех компиляторах
    if (x < 0.0f)
        return 0.0f;
    if (x > 1.0f)
        return 1.0f;
    return x;
}
