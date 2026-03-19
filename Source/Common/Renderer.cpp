#include "Renderer.h"
#include "MathUtils.h"
#include "HitInfo.h"
#include "Light.h"
#include <cmath>
#include <algorithm>

Vec3 Renderer::calculateLighting(const HitInfo &hit,
                                 const Light &light) const
{
    if (!hit.hit || hit.visibility <= 0.0f)
        return Vec3{0.0f, 0.0f, 0.0f};

    // Normal (may be non-unit after interpolation)
    const Vec3 N = normalize(hit.normal);

    Vec3 L{0.0f, 0.0f, 0.0f};   // direction from hit -> light (unit)
    float distance = 1.0f;     // for finite lights
    bool finiteLight = true;

    switch (light.type)
    {
    case LightType::Directional:
    {
        // Need direction from hit -> light, so -light.direction (light.direction is "where it shines")
        const Vec3 d{-light.direction.x, -light.direction.y, -light.direction.z};
        const float l2 = d.x * d.x + d.y * d.y + d.z * d.z;
        if (l2 <= 1e-20f)
            return Vec3{0.0f, 0.0f, 0.0f};

        const float invL = 1.0f / std::sqrt(l2);
        L = Vec3{d.x * invL, d.y * invL, d.z * invL};
        finiteLight = false;
        break;
    }

    case LightType::Point:
    case LightType::Spot:
    case LightType::Area:
    default:
    {
        const Vec3 toLight{
            light.position.x - hit.position.x,
            light.position.y - hit.position.y,
            light.position.z - hit.position.z};

        const float dist2 = toLight.x * toLight.x + toLight.y * toLight.y + toLight.z * toLight.z;
        if (dist2 <= 1e-20f)
            return Vec3{0.0f, 0.0f, 0.0f};

        distance = std::sqrt(dist2);
        const float invDist = 1.0f / distance;
        L = Vec3{toLight.x * invDist, toLight.y * invDist, toLight.z * invDist};
        finiteLight = true;
        break;
    }
    }

    // Lambert term
    const float nl = ScalarMUL(N, L);
    if (nl <= 0.0f)
        return Vec3{0.0f, 0.0f, 0.0f};

    // Distance attenuation
    float attenuation = 1.0f;
    if (finiteLight)
    {
        const float r2 = distance * distance;
        if (r2 <= 1e-20f)
            return Vec3{0.0f, 0.0f, 0.0f};

        attenuation = INV_4PI / r2;

        // Optional attenuation radius
        if (light.attenuationRadius > 0.0f)
        {
            const float R = light.attenuationRadius;
            if (distance >= R)
                return Vec3{0.0f, 0.0f, 0.0f};

            const float s = distance / R; // 0..1
            const float s2 = s * s;

            // (1 - s^2)^2 / (1 + F*s^2)
            constexpr float F = 1.0f;
            float num = 1.0f - s2;
            num *= num;
            const float den = 1.0f + F * s2;
            const float extra = (den > 0.0f) ? (num / den) : 0.0f;

            attenuation *= extra;
        }
    }

    float intensity = light.intensity * attenuation * LIGHT_EXPOSURE;

    // Spot cone
    if (light.type == LightType::Spot)
    {
        const Vec3 spotDir = normalize(light.direction);

        // Direction from light -> hit is the opposite of (hit -> light)
        const Vec3 dirLightToHit{-L.x, -L.y, -L.z};

        const float cosTheta = ScalarMUL(spotDir, dirLightToHit);
        if (cosTheta <= 0.0f)
            return Vec3{0.0f, 0.0f, 0.0f};

        const float spotSize = std::max(light.spotSize, 1e-4f);
        const float spotBlend = clamp01(light.spotBlend);

        const float outerAngle = 0.5f * spotSize;
        const float innerAngle = outerAngle * (1.0f - spotBlend);

        const float cosOuter = std::cos(outerAngle);
        const float cosInner = std::cos(innerAngle);

        float spotFactor = 0.0f;

        if (spotBlend <= 0.0f || cosInner <= cosOuter)
        {
            spotFactor = (cosTheta >= cosOuter) ? 1.0f : 0.0f;
        }
        else
        {
            if (cosTheta <= cosOuter)
            {
                spotFactor = 0.0f;
            }
            else if (cosTheta >= cosInner)
            {
                spotFactor = 1.0f;
            }
            else
            {
                float t = (cosTheta - cosOuter) / (cosInner - cosOuter);
                t = clamp01(t);
                spotFactor = t * t;
            }
        }

        intensity *= spotFactor;
        if (intensity <= 0.0f)
            return Vec3{0.0f, 0.0f, 0.0f};
    }

    // Area light: emits only along its normal, with cosine falloff
    if (light.type == LightType::Area)
    {
        const Vec3 areaNormal = normalize(light.direction);

        // Direction from light -> hit is opposite to L
        const Vec3 dirLightToHit{-L.x, -L.y, -L.z};

        const float cosEmit = ScalarMUL(areaNormal, dirLightToHit);
        if (cosEmit <= 0.0f)
            return Vec3{0.0f, 0.0f, 0.0f};

        intensity *= cosEmit;
        if (intensity <= 0.0f)
            return Vec3{0.0f, 0.0f, 0.0f};
    }

    const float scale = nl * hit.visibility * intensity;

    const Vec3 base{
        hit.color.x * light.color.x,
        hit.color.y * light.color.y,
        hit.color.z * light.color.z};

    return Vec3{base.x * scale, base.y * scale, base.z * scale};
}
