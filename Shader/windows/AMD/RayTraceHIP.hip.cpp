#include <hip/hip_runtime.h>

#ifndef __HIP_DEVICE_COMPILE__
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "HIPRendererInternals.h"

using uint = unsigned int;

static inline __host__ __device__ float make_float(float v) { return v; }
static inline __host__ __device__ float2 ctor_float2(float x, float y) { return ::make_float2(x, y); }
static inline __host__ __device__ float2 ctor_float2(float v) { return ::make_float2(v, v); }
static inline __host__ __device__ float2 ctor_float2(const Vec2 &v) { return ::make_float2(v.x, v.y); }
static inline __host__ __device__ float3 ctor_float3(float x, float y, float z) { return ::make_float3(x, y, z); }
static inline __host__ __device__ float3 ctor_float3(float v) { return ::make_float3(v, v, v); }
static inline __host__ __device__ float3 ctor_float3(const Vec3 &v) { return ::make_float3(v.x, v.y, v.z); }
static inline __host__ __device__ float4 ctor_float4(float x, float y, float z, float w) { return ::make_float4(x, y, z, w); }
static inline __host__ __device__ float4 ctor_float4(float v) { return ::make_float4(v, v, v, v); }
static inline __host__ __device__ float4 ctor_float4(float3 xyz, float w) { return ::make_float4(xyz.x, xyz.y, xyz.z, w); }
static inline __host__ __device__ uint2 ctor_uint2(uint x, uint y) { return ::make_uint2(x, y); }

#define float2(...) ctor_float2(__VA_ARGS__)
#define float3(...) ctor_float3(__VA_ARGS__)
#define float4(...) ctor_float4(__VA_ARGS__)
#define uint2(...) ctor_uint2(__VA_ARGS__)

static inline __host__ __device__ float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
static inline __host__ __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

static inline __host__ __device__ float length(float3 v) { return sqrtf(dot(v, v)); }
static inline __host__ __device__ float length(float2 v) { return sqrtf(dot(v, v)); }
static inline __host__ __device__ float3 normalize(float3 v)
{
    const float len2 = dot(v, v);
    if (len2 <= 1.0e-20f)
        return float3(0.0f, 0.0f, 1.0f);
    return v * rsqrtf(len2);
}

static inline __host__ __device__ float2 normalize(float2 v)
{
    const float len2 = dot(v, v);
    if (len2 <= 1.0e-20f)
        return float2(0.0f, 0.0f);
    return v * rsqrtf(len2);
}

static inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * dot(n, i) * n;
}

static inline __host__ __device__ float clamp(float x, float a, float b) { return fminf(fmaxf(x, a), b); }
static inline __host__ __device__ float3 clamp(float3 v, float a, float b) { return float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)); }
static inline __host__ __device__ float3 clamp(float3 v, float3 a, float3 b)
{
    return float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

static inline __host__ __device__ float2 clamp(float2 v, float2 a, float2 b)
{
    return float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

static inline __host__ __device__ float3 max(float3 a, float3 b) { return float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
static inline __host__ __device__ float3 min(float3 a, float3 b) { return float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }

static inline __host__ __device__ float mix(float a, float b, float t) { return a + (b - a) * t; }
static inline __host__ __device__ float3 mix(float3 a, float3 b, float t) { return a + (b - a) * t; }
static inline __host__ __device__ float fract(float x) { return x - floorf(x); }
static inline __host__ __device__ float2 fract(float2 v) { return float2(fract(v.x), fract(v.y)); }
static inline __host__ __device__ float3 pow(float3 v, float3 p) { return float3(powf(v.x, p.x), powf(v.y, p.y), powf(v.z, p.z)); }
static inline __host__ __device__ float smoothstep(float a, float b, float x)
{
    const float t = clamp((x - a) / max(b - a, 1.0e-6f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static inline __host__ __device__ float saturate(float x) { return clamp(x, 0.0f, 1.0f); }

constexpr float PI = 3.14159265358979323846f;
constexpr float INV_PI = 1.0f / PI;
constexpr float INV_4PI = 1.0f / (4.0f * PI);
constexpr float SHADOW_EPS = 1.0e-3f;
constexpr float EMISSION_VISIBLE_EXPOSURE_GPU = 1.0f;
constexpr float EMISSION_LIGHT_EXPOSURE_GPU = 1.0f;
constexpr float ENV_INTENSITY_GPU = 0.10f;
constexpr float LIGHT_FALLOFF_DISTANCE_SCALE = 1.0f;
constexpr float LIGHT_ATTENUATION_RADIUS_SCALE = 1.15f;
constexpr float LOCAL_LIGHT_EXPOSURE_GPU = 0.10f;
constexpr float VOLUME_FOG_DENSITY_SCALE = 0.035f;
constexpr float VOLUME_FOG_START_FADE_METERS = 30.0f;
constexpr float VOLUME_AMBIENT_SCATTER_SCALE = 0.010f;
constexpr float VOLUME_LIGHT_SCATTER_SCALE = 0.10f;
constexpr bool DENOISE_ENABLE = true;
constexpr float DENOISE_SIGMA_SPACE = 1.0f;
constexpr float DENOISE_SIGMA_COLOR = 0.20f;
constexpr float DENOISE_BLEND_FACTOR = 0.40f;
constexpr bool NORMALMAP_FLIP_Y = false;
constexpr int UV_DEBUG_MODE = 0;
constexpr int LIGHTING_CALIBRATION_MODE = 0;
constexpr int BLOCK_SIZE = 8;
constexpr int LIGHT_TYPE_POINT = 0;
constexpr int LIGHT_TYPE_DIRECTIONAL = 1;
constexpr int LIGHT_TYPE_SPOT = 2;
constexpr int LIGHT_TYPE_AREA = 3;
constexpr int MATERIAL_FLAG_EMISSION_USE_ALPHA_MASK = HIP_MATERIAL_FLAG_EMISSION_USE_ALPHA_MASK;
constexpr int MATERIAL_FLAG_THIN_EMISSIVE_SURFACE = HIP_MATERIAL_FLAG_THIN_EMISSIVE_SURFACE;
constexpr int SPECIAL_MATERIAL_UE_HEADLIGHT = HIP_SPECIAL_MATERIAL_UE_HEADLIGHT;
constexpr int SPECIAL_MATERIAL_UE_TRAFFIC_LIGHT = HIP_SPECIAL_MATERIAL_UE_TRAFFIC_LIGHT;
constexpr std::uint32_t INSTANCE_FLAG_CASTS_SHADOW = HIP_INSTANCE_FLAG_CASTS_SHADOW;
constexpr std::uint32_t LIGHT_FLAG_CASTS_SHADOW = HIP_LIGHT_FLAG_CASTS_SHADOW;

struct SampledTexel
{
    float x;
    float y;
    float z;
    float w;

    __host__ __device__ SampledTexel() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}
    __host__ __device__ SampledTexel(float xx, float yy, float zz, float ww) : x(xx), y(yy), z(zz), w(ww) {}
};

struct SamplerTag {};
constexpr SamplerTag g_texSampler{};
constexpr SamplerTag g_postSampler{};

struct SceneTextureProxy
{
    const HIPTextureDescGPU *descs = nullptr;
    const std::uint8_t *texels = nullptr;
    std::uint32_t textureCount = 0u;
    std::uint32_t index = 0u;

    __device__ SampledTexel sample(SamplerTag, float2 uv) const;
};

struct SceneTextureArray
{
    const HIPTextureDescGPU *descs = nullptr;
    const std::uint8_t *texels = nullptr;
    std::uint32_t textureCount = 0u;

    __device__ SceneTextureProxy operator[](std::uint32_t idx) const
    {
        return SceneTextureProxy{descs, texels, textureCount, idx};
    }
};

using CameraData = CameraDataCPU;
using SceneInstanceGPU = HIPSceneInstanceGPU;
using LightGPU = HIPLightGPU;
using MaterialGPU = HIPMaterialGPU;
using MaterialGPU_PBR = HIPMaterialPBRGPU;
using DecalGPU = HIPDecalGPU;
using EmissiveTriangleGPU = HIPEmissiveTriangleGPU;
using AirDustVolumeGPU = HIPAirDustVolumeGPU;
using PrimaryVolumeParams = HIPPostProcessParams;

struct Ray
{
    float3 origin;
    float3 direction;
};

struct HitInfo
{
    bool hit = false;
    int triIndex = -1;
    int instanceIndex = -1;
    int materialIndex = -1;
    float t = 0.0f;
    float3 position = float3(0.0f);
    float3 normal = float3(0.0f, 0.0f, 1.0f);
    float3 color = float3(1.0f);
    float3 emission = float3(0.0f);
    float metallic = 0.0f;
    float roughness = 0.5f;
    float2 uv0 = float2(0.0f);
    float2 uv1 = float2(0.0f);
    float2 uv2 = float2(0.0f);
};

struct BSDFSample
{
    uint valid = 0u;
    float3 direction = float3(0.0f);
    float3 weight = float3(0.0f);
    float pdf = 0.0f;
};

struct EmissiveLightSample
{
    uint valid = 0u;
    uint triIndex = 0u;
    uint instanceIndex = 0u;
    float area = 0.0f;
    float selectionPdf = 0.0f;
    float3 position = float3(0.0f);
    float3 normal = float3(0.0f, 0.0f, 1.0f);
    float2 uv0 = float2(0.0f);
    float2 uv1 = float2(0.0f);
    float2 uv2 = float2(0.0f);
};

struct PrimaryMediumSample
{
    float sigmaT = 0.0f;
    float3 albedo = float3(0.0f);
    float anisotropy = 0.0f;
    float3 ambientTint = float3(0.0f);
};

struct PrimaryVolumetricResult
{
    float3 inscattering = float3(0.0f);
    float3 transmittance = float3(1.0f);
};

struct PathTraceResult
{
    float3 color = float3(0.0f);
    float depth = 1.0e30f;
};

struct PathTraceTextureResult
{
    float3 color = float3(0.0f);
    float depth = 1.0e30f;
    float3 albedo = float3(0.0f);
    float hitMask = 0.0f;
    float3 normal = float3(0.0f);
    float _pad0 = 0.0f;
};

static_assert(sizeof(float3) == sizeof(Vec3), "float3 and Vec3 must stay byte-compatible");

__device__ static inline float srgbToLinear(float c)
{
    if (c <= 0.04045f)
        return c / 12.92f;
    return powf((c + 0.055f) / 1.055f, 2.4f);
}

__device__ static inline SampledTexel loadTexturePoint(const HIPTextureDescGPU &desc,
                                                       const std::uint8_t *texels,
                                                       int x,
                                                       int y)
{
    if (!texels || desc.width == 0u || desc.height == 0u)
        return SampledTexel(1.0f, 0.0f, 1.0f, 1.0f);

    const int w = static_cast<int>(desc.width);
    const int h = static_cast<int>(desc.height);
    x = ((x % w) + w) % w;
    y = ((y % h) + h) % h;

    const std::size_t idx = static_cast<std::size_t>(desc.texelOffset) +
                            (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4u;

    const float r8 = texels[idx + 0u] / 255.0f;
    const float g8 = texels[idx + 1u] / 255.0f;
    const float b8 = texels[idx + 2u] / 255.0f;
    const float a8 = texels[idx + 3u] / 255.0f;

    if ((desc.flags & HIP_TEXTURE_FLAG_SRGB) != 0u)
        return SampledTexel(srgbToLinear(r8), srgbToLinear(g8), srgbToLinear(b8), a8);

    return SampledTexel(r8, g8, b8, a8);
}

__device__ SampledTexel SceneTextureProxy::sample(SamplerTag, float2 uv) const
{
    if (!descs || index >= textureCount)
        return SampledTexel(1.0f, 0.0f, 1.0f, 1.0f);

    const HIPTextureDescGPU &desc = descs[index];
    const float2 uvWrap = fract(uv);
    const float fx = uvWrap.x * static_cast<float>(desc.width) - 0.5f;
    const float fy = uvWrap.y * static_cast<float>(desc.height) - 0.5f;
    const int x0 = static_cast<int>(floorf(fx));
    const int y0 = static_cast<int>(floorf(fy));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);

    const SampledTexel c00 = loadTexturePoint(desc, texels, x0, y0);
    const SampledTexel c10 = loadTexturePoint(desc, texels, x1, y0);
    const SampledTexel c01 = loadTexturePoint(desc, texels, x0, y1);
    const SampledTexel c11 = loadTexturePoint(desc, texels, x1, y1);

    const float4 cx0 = float4(c00.x, c00.y, c00.z, c00.w) * (1.0f - tx) + float4(c10.x, c10.y, c10.z, c10.w) * tx;
    const float4 cx1 = float4(c01.x, c01.y, c01.z, c01.w) * (1.0f - tx) + float4(c11.x, c11.y, c11.z, c11.w) * tx;
    const float4 c = cx0 * (1.0f - ty) + cx1 * ty;
    return SampledTexel(c.x, c.y, c.z, c.w);
}

__device__ static inline float rand01(uint seed)
{
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return static_cast<float>(seed) / 4294967295.0f;
}

__device__ static inline float clamp01(float x)
{
    return clamp(x, 0.0f, 1.0f);
}

__device__ static inline float3 lerp3(float3 a, float3 b, float t)
{
    return mix(a, b, t);
}

__device__ static inline float3 safeNormalize(float3 v)
{
    return normalize(v);
}

__device__ static inline float3 filmicTonemap(float3 c, float exposure)
{
    c *= exposure;
    const float a = 2.51f;
    const float b = 0.03f;
    const float c1 = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    float3 x = (c * (a * c + b)) / (c * (c1 * c + d) + e);
    x = clamp(x, 0.0f, 1.0f);
    return pow(x, float3(1.0f / 2.2f));
}

__device__ static inline float3 cosineSampleHemisphere(float u1, float u2)
{
    const float r = sqrtf(u1);
    const float phi = 2.0f * PI * u2;
    return float3(r * cosf(phi), r * sinf(phi), sqrtf(max(0.0f, 1.0f - u1)));
}

__device__ static inline void buildOrthonormalBasis(float3 n, float3 &tangent, float3 &bitangent)
{
    const float3 N = safeNormalize(n);
    const float3 helper = (fabsf(N.z) < 0.999f) ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    tangent = safeNormalize(cross(helper, N));
    bitangent = cross(N, tangent);
}

__device__ static inline float3 transformPoint3x4(const float m[12], float3 p)
{
    return float3(m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
                  m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
                  m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
}

__device__ static inline float3 transformDirection3x4(const float m[12], float3 v)
{
    return float3(m[0] * v.x + m[1] * v.y + m[2] * v.z,
                  m[4] * v.x + m[5] * v.y + m[6] * v.z,
                  m[8] * v.x + m[9] * v.y + m[10] * v.z);
}

__device__ static inline float3 transformPointObjectToWorld(const SceneInstanceGPU &inst, float3 p) { return transformPoint3x4(inst.objectToWorld, p); }
__device__ static inline float3 transformPointWorldToObject(const SceneInstanceGPU &inst, float3 p) { return transformPoint3x4(inst.worldToObject, p); }
__device__ static inline float3 transformDirectionObjectToWorld(const SceneInstanceGPU &inst, float3 v) { return transformDirection3x4(inst.objectToWorld, v); }
__device__ static inline float3 transformDirectionWorldToObject(const SceneInstanceGPU &inst, float3 v) { return transformDirection3x4(inst.worldToObject, v); }
__device__ static inline float3 transformNormalObjectToWorld(const SceneInstanceGPU &inst, float3 n) { return transformDirection3x4(inst.normalToWorld, n); }

__device__ static inline float2 triangleUvVertex(const Triangle &tri, int uvSet, int vertexIndex)
{
    const int setIndex = max(0, min(uvSet, 2));
    const int vertex = max(0, min(vertexIndex, 2));
    return float2(tri.uv[setIndex * 3 + vertex]);
}

__device__ static inline float2 triangleSampleUvSet(const Triangle &tri, float b0, float b1, float b2, int uvSet)
{
    const float2 uv0 = triangleUvVertex(tri, uvSet, 0);
    const float2 uv1 = triangleUvVertex(tri, uvSet, 1);
    const float2 uv2 = triangleUvVertex(tri, uvSet, 2);
    return uv0 * b0 + uv1 * b1 + uv2 * b2;
}

__device__ static inline float2 selectUvSet(float2 uv0, float2 uv1, float2 uv2, int uvSet)
{
    if (uvSet == 1)
        return uv1;
    if (uvSet == 2)
        return uv2;
    return uv0;
}

__device__ static inline Ray makePrimaryRayJittered(int px,
                                                    int py,
                                                    int width,
                                                    int height,
                                                    float2 jitter,
                                                    float3 camPos,
                                                    float3 camForward,
                                                    float3 camUp,
                                                    float3 camRight,
                                                    float fovY,
                                                    float aspectRatio)
{
    const float fx = static_cast<float>(px) + 0.5f + jitter.x;
    const float fy = static_cast<float>(py) + 0.5f + jitter.y;
    float ndcX = (fx / static_cast<float>(width)) * 2.0f - 1.0f;
    float ndcY = (fy / static_cast<float>(height)) * 2.0f - 1.0f;
    ndcY = -ndcY;

    const float aspect = (aspectRatio > 0.0f) ? aspectRatio : (static_cast<float>(width) / static_cast<float>(height));
    const float tanHalfFov = tanf(0.5f * fovY);
    const float3 dirCam = float3(ndcX * aspect * tanHalfFov, ndcY * tanHalfFov, -1.0f);
    const float3 dirWorld = normalize(dirCam.x * camRight + dirCam.y * camUp - dirCam.z * camForward);
    return Ray{camPos, dirWorld};
}

__device__ static inline bool intersectTriangle(const Ray &ray, const Triangle &tri, float *tHitOut, float *uOut, float *vOut)
{
    const float3 v0 = float3(tri.v0);
    const float3 v1 = float3(tri.v1);
    const float3 v2 = float3(tri.v2);
    const float3 edge1 = v1 - v0;
    const float3 edge2 = v2 - v0;
    const float3 pvec = cross(ray.direction, edge2);
    const float det = dot(edge1, pvec);
    if (fabsf(det) < 1.0e-6f)
        return false;

    const float invDet = 1.0f / det;
    const float3 tvec = ray.origin - v0;
    const float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f)
        return false;

    const float3 qvec = cross(tvec, edge1);
    const float v = dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f)
        return false;

    const float t = dot(edge2, qvec) * invDet;
    if (t <= 1.0e-6f)
        return false;

    *tHitOut = t;
    *uOut = u;
    *vOut = v;
    return true;
}

__device__ static inline bool intersectAABBWithRayFast(const AABB &box,
                                                       float3 origin,
                                                       float3 dir,
                                                       float3 invDir,
                                                       float tMaxClip,
                                                       float *tEnterOut)
{
    const float3 bmin = float3(box.v0);
    const float3 bmax = float3(box.v1);
    const float eps = 1.0e-8f;
    if (fabsf(dir.x) < eps && (origin.x < bmin.x || origin.x > bmax.x)) return false;
    if (fabsf(dir.y) < eps && (origin.y < bmin.y || origin.y > bmax.y)) return false;
    if (fabsf(dir.z) < eps && (origin.z < bmin.z || origin.z > bmax.z)) return false;

    const float3 t0 = (bmin - origin) * invDir;
    const float3 t1 = (bmax - origin) * invDir;
    const float3 tmin3 = min(t0, t1);
    const float3 tmax3 = max(t0, t1);
    float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);
    float tExit = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (tExit < 0.0f)
        return false;
    tEnter = max(tEnter, 0.0f);
    tExit = min(tExit, tMaxClip);
    if (tExit < tEnter)
        return false;
    *tEnterOut = tEnter;
    return true;
}

__device__ static inline HitInfo traceRayBVH(const BVHNode *nodes, const Triangle *tris, uint nodeCount, int rootIndex, const Ray &ray)
{
    HitInfo result{};
    result.t = INFINITY;
    if (!nodes || !tris || nodeCount == 0u || rootIndex < 0 || rootIndex >= static_cast<int>(nodeCount))
        return result;

    struct StackEntry { int index; float tEnter; };
    StackEntry stack[96];
    int sp = 0;

    const float3 o = ray.origin;
    const float3 d = ray.direction;
    const float3 invD = float3(1.0f) / d;

    float tEnterRoot = 0.0f;
    if (!intersectAABBWithRayFast(nodes[rootIndex].box, o, d, invD, INFINITY, &tEnterRoot))
        return result;

    int nodeIndex = rootIndex;
    float bestT = INFINITY;
    int deferredIndex = -1;
    float deferredEnter = 0.0f;

    while (true)
    {
        if (nodeIndex < 0 || nodeIndex >= static_cast<int>(nodeCount))
        {
            bool found = false;
            while (sp > 0)
            {
                const StackEntry e = stack[--sp];
                if (e.tEnter <= bestT) { nodeIndex = e.index; found = true; break; }
            }
            if (!found && deferredIndex != -1 && deferredEnter <= bestT)
            {
                nodeIndex = deferredIndex;
                deferredIndex = -1;
                found = true;
            }
            if (!found)
                break;
            continue;
        }

        const BVHNode &node = nodes[nodeIndex];
        if (node.tri >= 0)
        {
            float tHit = 0.0f;
            float uHit = 0.0f;
            float vHit = 0.0f;
            const Triangle &tri = tris[node.tri];
            if (intersectTriangle(ray, tri, &tHit, &uHit, &vHit) && tHit < bestT)
            {
                const float wHit = 1.0f - uHit - vHit;
                bestT = tHit;
                result.hit = true;
                result.triIndex = node.tri;
                result.materialIndex = tri.materialIndex;
                result.t = tHit;
                result.position = o + d * tHit;
                result.normal = safeNormalize(float3(tri.normal));
                result.color = float3(tri.color);
                result.emission = float3(tri.emission);
                result.metallic = tri.metallic;
                result.roughness = tri.roughness;
                result.uv0 = triangleSampleUvSet(tri, wHit, uHit, vHit, 0);
                result.uv1 = triangleSampleUvSet(tri, wHit, uHit, vHit, 1);
                result.uv2 = triangleSampleUvSet(tri, wHit, uHit, vHit, 2);
            }

            bool found = false;
            while (sp > 0)
            {
                const StackEntry e = stack[--sp];
                if (e.tEnter <= bestT) { nodeIndex = e.index; found = true; break; }
            }
            if (!found && deferredIndex != -1 && deferredEnter <= bestT)
            {
                nodeIndex = deferredIndex;
                deferredIndex = -1;
                found = true;
            }
            if (!found)
                break;
            continue;
        }

        const int left = node.left;
        const int right = node.right;
        bool hitL = false;
        bool hitR = false;
        float tL = 0.0f;
        float tR = 0.0f;
        if (left >= 0) hitL = intersectAABBWithRayFast(nodes[left].box, o, d, invD, bestT, &tL);
        if (right >= 0) hitR = intersectAABBWithRayFast(nodes[right].box, o, d, invD, bestT, &tR);

        if (hitL && hitR)
        {
            const bool leftNear = tL < tR;
            const int nearIdx = leftNear ? left : right;
            const int farIdx = leftNear ? right : left;
            const float farT = leftNear ? tR : tL;
            if (sp < 96) stack[sp++] = StackEntry{farIdx, farT};
            else { deferredIndex = farIdx; deferredEnter = farT; }
            nodeIndex = nearIdx;
            continue;
        }

        if (hitL) { nodeIndex = left; continue; }
        if (hitR) { nodeIndex = right; continue; }

        bool found = false;
        while (sp > 0)
        {
            const StackEntry e = stack[--sp];
            if (e.tEnter <= bestT) { nodeIndex = e.index; found = true; break; }
        }
        if (!found && deferredIndex != -1 && deferredEnter <= bestT)
        {
            nodeIndex = deferredIndex;
            deferredIndex = -1;
            found = true;
        }
        if (!found)
            break;
    }

    return result;
}

__device__ static inline bool traceShadowBVH(const BVHNode *nodes, const Triangle *tris, uint nodeCount, int rootIndex, const Ray &ray, float maxDist)
{
    const HitInfo hit = traceRayBVH(nodes, tris, nodeCount, rootIndex, ray);
    return hit.hit && hit.t > 1.0e-4f && hit.t < maxDist;
}

__device__ static inline HitInfo traceRaySceneBVH(const BVHNode *tlasNodes,
                                                  const BVHNode *meshNodes,
                                                  const Triangle *tris,
                                                  const SceneInstanceGPU *instances,
                                                  uint tlasNodeCount,
                                                  uint meshNodeCount,
                                                  uint instanceCount,
                                                  int tlasRootIndex,
                                                  const Ray &worldRay)
{
    HitInfo result{};
    result.t = INFINITY;
    if (!instances || instanceCount == 0u || !tlasNodes || !meshNodes || !tris)
        return result;
    if (tlasRootIndex < 0 || tlasRootIndex >= static_cast<int>(tlasNodeCount))
        return result;

    struct StackEntry { int index; float tEnter; };
    StackEntry stack[96];
    int sp = 0;

    const float3 o = worldRay.origin;
    const float3 d = worldRay.direction;
    const float3 invD = float3(1.0f) / d;

    float tEnterRoot = 0.0f;
    if (!intersectAABBWithRayFast(tlasNodes[tlasRootIndex].box, o, d, invD, INFINITY, &tEnterRoot))
        return result;

    int nodeIndex = tlasRootIndex;
    float bestT = INFINITY;
    while (true)
    {
        if (nodeIndex < 0 || nodeIndex >= static_cast<int>(tlasNodeCount))
        {
            if (sp == 0)
                break;
            nodeIndex = stack[--sp].index;
            continue;
        }

        const BVHNode &node = tlasNodes[nodeIndex];
        if (node.tri >= 0)
        {
            const int instIndex = node.tri;
            if (instIndex >= 0 && static_cast<uint>(instIndex) < instanceCount)
            {
                const SceneInstanceGPU &inst = instances[instIndex];
                Ray localRay{};
                localRay.origin = transformPointWorldToObject(inst, worldRay.origin);
                localRay.direction = transformDirectionWorldToObject(inst, worldRay.direction);
                HitInfo localHit = traceRayBVH(meshNodes, tris, meshNodeCount, inst.blasRootIndex, localRay);
                if (localHit.hit)
                {
                    const float3 worldPos = transformPointObjectToWorld(inst, localHit.position);
                    const float worldT = length(worldPos - worldRay.origin);
                    if (worldT < bestT)
                    {
                        bestT = worldT;
                        result = localHit;
                        result.hit = true;
                        result.instanceIndex = instIndex;
                        result.t = worldT;
                        result.position = worldPos;
                        result.normal = safeNormalize(transformNormalObjectToWorld(inst, localHit.normal));
                    }
                }
            }

            if (sp == 0)
                break;
            nodeIndex = stack[--sp].index;
            continue;
        }

        const int left = node.left;
        const int right = node.right;
        bool hitL = false;
        bool hitR = false;
        float tL = 0.0f;
        float tR = 0.0f;
        if (left >= 0) hitL = intersectAABBWithRayFast(tlasNodes[left].box, o, d, invD, bestT, &tL);
        if (right >= 0) hitR = intersectAABBWithRayFast(tlasNodes[right].box, o, d, invD, bestT, &tR);

        if (hitL && hitR)
        {
            const bool leftNear = tL < tR;
            if (sp < 96)
                stack[sp++] = StackEntry{leftNear ? right : left, leftNear ? tR : tL};
            nodeIndex = leftNear ? left : right;
            continue;
        }

        if (hitL) { nodeIndex = left; continue; }
        if (hitR) { nodeIndex = right; continue; }
        if (sp == 0)
            break;
        nodeIndex = stack[--sp].index;
    }

    return result;
}

__device__ static inline float3 sampledTexelRgb(const SampledTexel &t)
{
    return float3(t.x, t.y, t.z);
}

__device__ static inline float sampledTexelLuminance(const SampledTexel &t)
{
    return t.x * 0.2126f + t.y * 0.7152f + t.z * 0.0722f;
}

__device__ static inline float luminance3(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

__device__ static inline float powerHeuristic(float pdfA, float pdfB)
{
    const float a2 = pdfA * pdfA;
    const float b2 = pdfB * pdfB;
    return a2 / max(a2 + b2, 1.0e-8f);
}

__device__ static inline float3 sampleDiskUniform(float u1, float u2)
{
    const float r = sqrtf(clamp01(u1));
    const float phi = 2.0f * PI * u2;
    return float3(cosf(phi) * r, sinf(phi) * r, 0.0f);
}

__device__ static inline float3 sampleSphereUniform(float u1, float u2)
{
    const float z = 1.0f - 2.0f * u1;
    const float phi = 2.0f * PI * u2;
    const float rxy = sqrtf(max(0.0f, 1.0f - z * z));
    return float3(rxy * cosf(phi), rxy * sinf(phi), z);
}

__device__ static inline bool materialHasPBRFlag(int matId,
                                                 int flag,
                                                 const MaterialGPU_PBR *materialsPBR,
                                                 uint materialPBRCount)
{
    return materialsPBR != nullptr &&
           matId >= 0 &&
           static_cast<uint>(matId) < materialPBRCount &&
           ((materialsPBR[matId].flags & flag) != 0);
}

__device__ static inline SampledTexel sampleTexture4(int texIndex,
                                                     float2 uv,
                                                     const SceneTextureArray &sceneTextures,
                                                     SampledTexel fallback = SampledTexel(1.0f, 1.0f, 1.0f, 1.0f))
{
    if (texIndex < 0 || static_cast<uint>(texIndex) >= sceneTextures.textureCount)
        return fallback;
    return sceneTextures[static_cast<uint>(texIndex)].sample(g_texSampler, fract(uv));
}

__device__ static inline float3 sampleTextureRGB(int texIndex,
                                                 float2 uv,
                                                 const SceneTextureArray &sceneTextures,
                                                 float3 fallback = float3(1.0f))
{
    const SampledTexel s = sampleTexture4(texIndex, uv, sceneTextures, SampledTexel(fallback.x, fallback.y, fallback.z, 1.0f));
    return sampledTexelRgb(s);
}

__device__ static inline float sampleTextureR(int texIndex,
                                              float2 uv,
                                              const SceneTextureArray &sceneTextures,
                                              float fallback = 1.0f)
{
    const SampledTexel s = sampleTexture4(texIndex, uv, sceneTextures, SampledTexel(fallback, fallback, fallback, 1.0f));
    return s.x;
}

__device__ static inline float lightEffectiveRadius(const LightGPU &light)
{
    return max(light.radius, light.softSourceRadius);
}

__device__ static inline float3 sampleFiniteLightPosition(const LightGPU &light,
                                                          uint seedA,
                                                          uint seedB,
                                                          uint seedC)
{
    const float3 lightPos = float3(light.position);
    const float radius = max(0.0f, lightEffectiveRadius(light));
    const float sourceLength = max(light.sourceLength, 0.0f);

    if (!(radius > 0.0f || sourceLength > 0.0f))
        return lightPos;

    if (sourceLength <= 1.0e-6f)
        return lightPos + sampleSphereUniform(rand01(seedA), rand01(seedB)) * radius;

    const float3 axis = safeNormalize(float3(light.direction));
    float3 samplePos = lightPos + axis * ((rand01(seedA) - 0.5f) * sourceLength);
    if (radius <= 1.0e-6f)
        return samplePos;

    float3 tangent;
    float3 bitangent;
    buildOrthonormalBasis(axis, tangent, bitangent);
    const float3 disk = sampleDiskUniform(rand01(seedB), rand01(seedC)) * radius;
    return samplePos + tangent * disk.x + bitangent * disk.y;
}

__device__ static inline float computeFiniteLightAttenuation(const LightGPU &light,
                                                             float dist,
                                                             float worldUnitToMeters)
{
    const float distMeters = max(dist * max(worldUnitToMeters, 1.0e-4f), 1.0e-4f);
    const float falloffDist = max(distMeters * LIGHT_FALLOFF_DISTANCE_SCALE, 1.0e-4f);
    const float r2 = falloffDist * falloffDist;
    if (r2 <= 0.0f)
        return 0.0f;

    float attenuation = ((light.type == LIGHT_TYPE_AREA) ? INV_PI : INV_4PI) / r2;
    attenuation *= LOCAL_LIGHT_EXPOSURE_GPU;

    float attenuationRadiusScale = LIGHT_ATTENUATION_RADIUS_SCALE;
    if (light.type == LIGHT_TYPE_POINT)
        attenuationRadiusScale *= 1.55f;
    else if (light.type == LIGHT_TYPE_SPOT)
        attenuationRadiusScale *= 0.90f;

    if (light.attenuationRadius > 0.0f)
    {
        const float radius = light.attenuationRadius * attenuationRadiusScale;
        if (dist >= radius)
            return 0.0f;

        const float s = dist / radius;
        const float s2 = s * s;
        const float s4 = s2 * s2;
        float radiusFade = clamp(1.0f - s4, 0.0f, 1.0f);
        radiusFade *= radiusFade;

        if (light.type == LIGHT_TYPE_POINT)
        {
            const float edgeBlend = smoothstep(0.82f, 1.0f, s);
            radiusFade = mix(1.0f, radiusFade, edgeBlend);
        }
        else if (light.type == LIGHT_TYPE_SPOT)
        {
            radiusFade *= (1.0f - 0.18f * smoothstep(0.55f, 1.0f, s));
        }

        attenuation *= radiusFade;
    }

    return attenuation;
}

__device__ static inline float computeSpotFactor(const LightGPU &light,
                                                 float3 samplePos,
                                                 float3 hitPos)
{
    const float spotSize = max(light.spotSize, 1.0e-4f);
    const float spotBlend = clamp01(light.spotBlend);
    const float outerAngle = 0.5f * spotSize;
    const float innerAngle = outerAngle * (1.0f - spotBlend);
    const float cosOuter = cosf(outerAngle);
    const float cosInner = cosf(innerAngle);

    const float3 lightDir = safeNormalize(float3(light.direction));
    const float3 dirToPoint = safeNormalize(hitPos - samplePos);
    const float cosTheta = dot(lightDir, dirToPoint);
    if (cosTheta <= 0.0f)
        return 0.0f;

    if (spotBlend <= 0.0f || cosInner <= cosOuter)
        return (cosTheta >= cosOuter) ? 1.0f : 0.0f;

    if (cosTheta <= cosOuter)
        return 0.0f;
    if (cosTheta >= cosInner)
        return 1.0f;

    float t = (cosTheta - cosOuter) / max(cosInner - cosOuter, 1.0e-6f);
    t = clamp01(t);
    return t * t;
}

__device__ static inline bool shouldIgnoreOwnerShadowForLight(const LightGPU &light)
{
    if (light.ownerId == 0u)
        return false;

    const float effectiveRadius = max(light.radius, light.softSourceRadius);
    const float halfLength = max(light.sourceLength, 0.0f) * 0.5f;
    const float sourceExtent = max(effectiveRadius, halfLength);
    if (sourceExtent <= 0.0f || light.attenuationRadius <= 0.0f)
        return false;

    return (sourceExtent / max(light.attenuationRadius, 1.0e-4f)) >= 0.06f;
}

__device__ static inline int sanitizeUvSet(int uvSet)
{
    return max(0, min(uvSet, 2));
}

__device__ static inline void computeTangentBasis(const Triangle *triangles,
                                                  int triIndex,
                                                  int uvSet,
                                                  float3 N,
                                                  float3 &T,
                                                  float3 &B)
{
    const Triangle &tri = triangles[triIndex];
    const float3 p0 = float3(tri.v0);
    const float3 p1 = float3(tri.v1);
    const float3 p2 = float3(tri.v2);
    const float2 uv0 = triangleUvVertex(tri, uvSet, 0);
    const float2 uv1 = triangleUvVertex(tri, uvSet, 1);
    const float2 uv2 = triangleUvVertex(tri, uvSet, 2);

    const float3 e1 = p1 - p0;
    const float3 e2 = p2 - p0;
    const float2 dUV1 = uv1 - uv0;
    const float2 dUV2 = uv2 - uv0;

    const float det = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
    if (fabsf(det) < 1.0e-8f)
    {
        buildOrthonormalBasis(N, T, B);
        return;
    }

    const float invDet = 1.0f / det;
    float3 tangent = (e1 * dUV2.y - e2 * dUV1.y) * invDet;
    tangent = tangent - N * dot(N, tangent);
    T = safeNormalize(tangent);
    const float handedness = (det < 0.0f) ? -1.0f : 1.0f;
    B = safeNormalize(cross(N, T) * handedness);
}

__device__ static inline void computeTangentBasisInstanced(const Triangle *triangles,
                                                           const SceneInstanceGPU *instances,
                                                           int instanceIndex,
                                                           int triIndex,
                                                           int uvSet,
                                                           float3 N,
                                                           float3 &T,
                                                           float3 &B)
{
    const float3 localN = safeNormalize(float3(triangles[triIndex].normal));
    computeTangentBasis(triangles, triIndex, sanitizeUvSet(uvSet), localN, T, B);

    if (instances != nullptr && instanceIndex >= 0)
    {
        const SceneInstanceGPU &inst = instances[instanceIndex];
        T = safeNormalize(transformDirectionObjectToWorld(inst, T));
        B = safeNormalize(transformDirectionObjectToWorld(inst, B));
        T = safeNormalize(T - N * dot(N, T));
        B = safeNormalize(B - N * dot(N, B));
        if (dot(cross(N, T), B) < 0.0f)
            B = -B;
    }
}

__device__ static inline float3 sampleNormalMapInstanced(int normalTexIndex,
                                                         int normalUvSet,
                                                         float2 uv,
                                                         float3 Ng,
                                                         const Triangle *triangles,
                                                         const SceneInstanceGPU *instances,
                                                         int instanceIndex,
                                                         int triIndex,
                                                         const SceneTextureArray &sceneTextures)
{
    if (normalTexIndex < 0 || static_cast<uint>(normalTexIndex) >= sceneTextures.textureCount)
        return Ng;

    float3 nTex = sampleTextureRGB(normalTexIndex, fract(uv), sceneTextures, float3(0.5f, 0.5f, 1.0f)) * 2.0f - 1.0f;
    if (NORMALMAP_FLIP_Y)
        nTex.y = -nTex.y;
    nTex = safeNormalize(nTex);

    float3 T;
    float3 B;
    computeTangentBasisInstanced(triangles, instances, instanceIndex, triIndex, normalUvSet, Ng, T, B);
    return safeNormalize(nTex.x * T + nTex.y * B + nTex.z * Ng);
}

__device__ static inline float sampleEmissiveMask(const SampledTexel &texel)
{
    const float alphaMask = saturate(texel.w);
    const float lumaMask = saturate(sampledTexelLuminance(texel));
    if (alphaMask > 1.0e-4f)
        return smoothstep(0.22f, 0.82f, alphaMask);
    return smoothstep(0.16f, 0.42f, lumaMask);
}

__device__ static inline float3 environmentColor(float3 dir)
{
    if (LIGHTING_CALIBRATION_MODE != 0)
        return float3(0.0f);

    const float t = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    const float3 ground = float3(0.05f, 0.05f, 0.06f);
    const float3 skyBot = float3(0.08f, 0.08f, 0.10f);
    const float3 skyTop = float3(0.14f, 0.13f, 0.17f);
    const float3 sky = lerp3(skyBot, skyTop, t);
    const float3 base = (dir.y >= 0.0f) ? sky : lerp3(ground, sky, 0.5f);
    return base * ENV_INTENSITY_GPU;
}

__device__ static inline float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (float3(1.0f) - F0) * pow(float3(1.0f - cosTheta), float3(5.0f));
}

__device__ static inline float distributionGGX(float NdH, float alpha)
{
    const float a2 = alpha * alpha;
    float denom = NdH * NdH * (a2 - 1.0f) + 1.0f;
    denom = PI * denom * denom;
    return a2 / max(denom, 1.0e-7f);
}

__device__ static inline float geometrySchlickGGX(float NdV, float k)
{
    return NdV / max(NdV * (1.0f - k) + k, 1.0e-6f);
}

__device__ static inline float geometrySmith(float NdV, float NdL, float k)
{
    return geometrySchlickGGX(NdV, k) * geometrySchlickGGX(NdL, k);
}

__device__ static inline float3 triangleFaceNormal(const Triangle &tri)
{
    const float3 e1 = float3(tri.v1) - float3(tri.v0);
    const float3 e2 = float3(tri.v2) - float3(tri.v0);
    const float3 n = cross(e1, e2);
    const float len2 = dot(n, n);
    if (len2 > 1.0e-16f)
        return safeNormalize(n);
    return safeNormalize(float3(tri.normal));
}

__device__ static inline bool traceShadowSceneBVH(const BVHNode *tlasNodes,
                                                  const BVHNode *meshNodes,
                                                  const Triangle *tris,
                                                  const SceneInstanceGPU *instances,
                                                  uint tlasNodeCount,
                                                  uint meshNodeCount,
                                                  uint instanceCount,
                                                  int tlasRootIndex,
                                                  const Ray &worldRay,
                                                  float maxDist)
{
    const HitInfo hit = traceRaySceneBVH(tlasNodes, meshNodes, tris, instances,
                                         tlasNodeCount, meshNodeCount, instanceCount,
                                         tlasRootIndex, worldRay);
    return hit.hit && hit.t > 1.0e-4f && hit.t < maxDist;
}

__device__ static inline bool traceShadowSceneBVHSkippingThin(const BVHNode *tlasNodes,
                                                              const BVHNode *meshNodes,
                                                              const Triangle *tris,
                                                              const SceneInstanceGPU *instances,
                                                              uint tlasNodeCount,
                                                              uint meshNodeCount,
                                                              uint instanceCount,
                                                              int tlasRootIndex,
                                                              const MaterialGPU_PBR *materialsPBR,
                                                              uint materialPBRCount,
                                                              uint ignoreOwnerId,
                                                              const Ray &worldRay,
                                                              float maxDist)
{
    Ray ray = worldRay;
    float remainingDist = (maxDist > 0.0f) ? maxDist : INFINITY;

    for (int skip = 0; skip < 8; ++skip)
    {
        const HitInfo hit = traceRaySceneBVH(tlasNodes, meshNodes, tris, instances,
                                             tlasNodeCount, meshNodeCount, instanceCount,
                                             tlasRootIndex, ray);
        if (!(hit.hit && hit.t > 1.0e-4f && hit.t < remainingDist))
            return false;

        bool skipHit = materialHasPBRFlag(hit.materialIndex,
                                          MATERIAL_FLAG_THIN_EMISSIVE_SURFACE,
                                          materialsPBR,
                                          materialPBRCount);

        if (!skipHit && instances != nullptr && hit.instanceIndex >= 0 &&
            static_cast<uint>(hit.instanceIndex) < instanceCount)
        {
            const SceneInstanceGPU &inst = instances[static_cast<uint>(hit.instanceIndex)];
            if ((inst.flags & INSTANCE_FLAG_CASTS_SHADOW) == 0u)
                skipHit = true;
            else if (ignoreOwnerId != 0u && inst.ownerId == ignoreOwnerId)
                skipHit = true;
        }

        if (!skipHit)
            return true;

        const float advance = hit.t + SHADOW_EPS;
        remainingDist -= advance;
        if (remainingDist <= 1.0e-4f)
            return false;

        ray.origin = hit.position + ray.direction * SHADOW_EPS;
    }

    return false;
}

__device__ static inline float computeTrafficLightDirtFactor(MaterialGPU_PBR mp,
                                                             float2 uvBase,
                                                             const SceneTextureArray &sceneTextures)
{
    if (mp.specialTex0Index < 0 || mp.specialTex1Index < 0)
        return 1.0f;

    const float tiling = max(mp.specialScalar0, 1.0e-3f);
    const float2 dirtUv = fract(uvBase * tiling);
    const float dirt = luminance3(sampleTextureRGB(mp.specialTex0Index, dirtUv, sceneTextures, float3(1.0f)));
    const float opacityMask = luminance3(sampleTextureRGB(mp.specialTex1Index, fract(uvBase), sceneTextures, float3(1.0f)));
    const float sharpness = max(0.25f, 1.0f + mp.specialScalar1 * 3.0f);
    const float stainStrength = max(mp.specialScalar3, 0.0f);
    const float dirtMask = powf(clamp01((1.0f - opacityMask) * dirt), sharpness);
    return clamp(1.0f - dirtMask * 0.22f * stainStrength, 0.55f, 1.0f);
}

__device__ static inline float3 evaluateMaterialEmissionPBR(MaterialGPU_PBR mp,
                                                            float3 emissiveBase,
                                                            float3 baseColorTex,
                                                            float2 uvBase,
                                                            float2 uvEmission,
                                                            float NdV,
                                                            const SceneTextureArray &sceneTextures)
{
    if (mp.specialModel == SPECIAL_MATERIAL_UE_TRAFFIC_LIGHT)
    {
        float3 emissive = emissiveBase;
        if (mp.emissionTexIndex >= 0)
            emissive *= sampleTextureRGB(mp.emissionTexIndex, uvEmission, sceneTextures, float3(1.0f));
        return emissive;
    }

    float3 emissive = emissiveBase;
    if (mp.emissionTexIndex >= 0)
    {
        const SampledTexel eTex = sampleTexture4(mp.emissionTexIndex, uvEmission, sceneTextures);
        if ((mp.flags & MATERIAL_FLAG_EMISSION_USE_ALPHA_MASK) != 0)
            emissive *= sampleEmissiveMask(eTex);
        else
            emissive *= sampledTexelRgb(eTex);
    }

    if (mp.specialModel == SPECIAL_MATERIAL_UE_HEADLIGHT)
    {
        const float baseLuma = clamp01(luminance3(baseColorTex));
        const float innerGlow = clamp01(mp.specialScalar0);
        const float transparency = clamp01(mp.specialScalar4);
        const float rim = powf(clamp01(1.0f - NdV), max(mp.specialScalar1, 1.0f));
        const float dirtTransmission =
            clamp(mp.specialScalar2 + baseLuma * max(mp.specialScalar5, 0.5f) + mp.specialScalar3,
                  0.20f,
                  1.40f);
        const float glowGain = (0.95f + innerGlow * 0.30f) * (0.75f + baseLuma * 0.35f);
        const float rimGain = 1.0f + rim * 0.10f * transparency;
        emissive *= max(dirtTransmission * glowGain * rimGain, 0.25f);
    }

    return emissive;
}

__device__ static inline float3 resolveTriangleEmissionTextured(const Triangle *triangles,
                                                                uint triIndex,
                                                                float2 uv0,
                                                                float2 uv1,
                                                                float2 uv2,
                                                                const MaterialGPU *materials,
                                                                uint materialCount,
                                                                const MaterialGPU_PBR *materialsPBR,
                                                                uint materialPBRCount,
                                                                const SceneTextureArray &sceneTextures)
{
    const Triangle &tri = triangles[triIndex];
    float3 emissive = float3(tri.emission);
    const int matId = tri.materialIndex;

    if (materialsPBR != nullptr && matId >= 0 && static_cast<uint>(matId) < materialPBRCount)
    {
        const MaterialGPU_PBR mp = materialsPBR[matId];
        const float2 uvBase = fract(selectUvSet(uv0, uv1, uv2, mp.baseColorUvSet));
        const float2 uvEmission = fract(selectUvSet(uv0, uv1, uv2, mp.emissionUvSet));
        float3 baseColorTex = float3(1.0f);
        if (mp.baseColorTexIndex >= 0)
            baseColorTex = sampleTextureRGB(mp.baseColorTexIndex, uvBase, sceneTextures, float3(1.0f));
        emissive = evaluateMaterialEmissionPBR(mp, emissive, baseColorTex, uvBase, uvEmission, 1.0f, sceneTextures);
    }
    else if (materials != nullptr && matId >= 0 && static_cast<uint>(matId) < materialCount)
    {
        const MaterialGPU m = materials[matId];
        if (m.emissionTexIndex >= 0)
            emissive *= sampleTextureRGB(m.emissionTexIndex,
                                         fract(selectUvSet(uv0, uv1, uv2, m.emissionUvSet)),
                                         sceneTextures,
                                         float3(1.0f));
    }

    return emissive * EMISSION_LIGHT_EXPOSURE_GPU;
}

__device__ static inline bool findEmissiveEntry(uint triIndex,
                                                const EmissiveTriangleGPU *emissiveTriangles,
                                                uint emissiveTriangleCount,
                                                EmissiveTriangleGPU &outEntry)
{
    for (uint i = 0u; i < emissiveTriangleCount; ++i)
    {
        const EmissiveTriangleGPU entry = emissiveTriangles[i];
        if (entry.triIndex == triIndex)
        {
            outEntry = entry;
            return true;
        }
    }
    return false;
}

__device__ static inline EmissiveLightSample sampleOneEmissiveTriangle(const Triangle *triangles,
                                                                       const SceneInstanceGPU *instances,
                                                                       const EmissiveTriangleGPU *emissiveTriangles,
                                                                       uint emissiveTriangleCount,
                                                                       uint seedSelect,
                                                                       uint seedBaryU,
                                                                       uint seedBaryV)
{
    EmissiveLightSample sample{};
    sample.valid = 0u;

    if (emissiveTriangles == nullptr || emissiveTriangleCount == 0u)
        return sample;

    const float target = rand01(seedSelect);
    uint lo = 0u;
    uint hi = emissiveTriangleCount - 1u;
    while (lo < hi)
    {
        const uint mid = (lo + hi) >> 1u;
        if (target <= emissiveTriangles[mid].cdf)
            hi = mid;
        else
            lo = mid + 1u;
    }

    const EmissiveTriangleGPU entry = emissiveTriangles[lo];
    const Triangle &tri = triangles[entry.triIndex];

    const float r1 = rand01(seedBaryU);
    const float r2 = rand01(seedBaryV);
    const float su = sqrtf(r1);
    const float b0 = 1.0f - su;
    const float b1 = r2 * su;
    const float b2 = 1.0f - b0 - b1;

    float3 position = float3(tri.v0) * b0 + float3(tri.v1) * b1 + float3(tri.v2) * b2;
    float3 normal = triangleFaceNormal(tri);

    if (instances != nullptr)
    {
        const SceneInstanceGPU &inst = instances[entry.instanceIndex];
        position = transformPointObjectToWorld(inst, position);
        normal = safeNormalize(transformNormalObjectToWorld(inst, normal));
    }

    sample.valid = 1u;
    sample.triIndex = entry.triIndex;
    sample.instanceIndex = entry.instanceIndex;
    sample.area = max(entry.area, 1.0e-8f);
    sample.selectionPdf = max(entry.selectionPdf, 1.0e-8f);
    sample.position = position;
    sample.normal = normal;
    sample.uv0 = triangleSampleUvSet(tri, b0, b1, b2, 0);
    sample.uv1 = triangleSampleUvSet(tri, b0, b1, b2, 1);
    sample.uv2 = triangleSampleUvSet(tri, b0, b1, b2, 2);
    return sample;
}

__device__ static inline float emissiveLightPdfForHit(float3 prevSurfacePos,
                                                      float3 lightPos,
                                                      float3 lightNormal,
                                                      uint triIndex,
                                                      const EmissiveTriangleGPU *emissiveTriangles,
                                                      uint emissiveTriangleCount)
{
    EmissiveTriangleGPU entry{};
    if (!findEmissiveEntry(triIndex, emissiveTriangles, emissiveTriangleCount, entry))
        return 0.0f;

    const float3 toLight = lightPos - prevSurfacePos;
    const float dist2 = dot(toLight, toLight);
    if (dist2 <= 1.0e-10f)
        return 0.0f;

    const float dist = sqrtf(dist2);
    const float3 wi = toLight / dist;
    const float cosLight = fabsf(dot(safeNormalize(lightNormal), -wi));
    if (cosLight <= 1.0e-6f)
        return 0.0f;

    return entry.selectionPdf * dist2 / max(cosLight * entry.area, 1.0e-6f);
}

__device__ static inline float3 evalSurfaceBRDF(float3 N,
                                                float3 V,
                                                float3 L,
                                                float3 baseColor,
                                                float metallic,
                                                float roughness)
{
    const float NdV = max(dot(N, V), 0.0f);
    const float NdL = max(dot(N, L), 0.0f);
    if (NdV <= 0.0f || NdL <= 0.0f)
        return float3(0.0f);

    const float r = clamp(roughness, 0.02f, 0.98f);
    const float3 F0 = lerp3(float3(0.04f), baseColor, metallic);
    const float3 H = safeNormalize(V + L);
    const float NdH = max(dot(N, H), 0.0f);
    const float HdV = max(dot(H, V), 0.0f);
    const float alpha = r * r;

    float k = r + 1.0f;
    k = (k * k) * 0.125f;

    const float D = distributionGGX(NdH, alpha);
    const float G = geometrySmith(NdV, NdL, k);
    const float3 F = fresnelSchlick(HdV, F0);
    const float3 spec = (D * G * F) / max(4.0f * NdV * NdL, 1.0e-6f);
    const float3 kd = (float3(1.0f) - F) * (1.0f - metallic);
    const float3 diff = kd * baseColor * INV_PI;
    return diff + spec;
}

__device__ static inline float pdfSurfaceBSDF(float3 N,
                                              float3 V,
                                              float3 L,
                                              float3 baseColor,
                                              float metallic,
                                              float roughness)
{
    const float NdL = max(dot(N, L), 0.0f);
    const float NdV = max(dot(N, V), 0.0f);
    if (NdL <= 0.0f || NdV <= 0.0f)
        return 0.0f;

    const float r = clamp(roughness, 0.02f, 0.98f);
    const float3 F0 = lerp3(float3(0.04f), baseColor, metallic);
    const float specProb = clamp(max(F0.x, max(F0.y, F0.z)), 0.05f, 0.95f);
    const float diffProb = 1.0f - specProb;
    const float diffPdf = NdL * INV_PI;

    const float3 H = safeNormalize(V + L);
    const float NdH = max(dot(N, H), 0.0f);
    const float HdV = max(dot(H, V), 0.0f);
    const float alpha = r * r;
    const float D = distributionGGX(NdH, alpha);
    const float specPdf = (D * NdH) / max(4.0f * HdV, 1.0e-6f);

    return diffProb * diffPdf + specProb * specPdf;
}

__device__ static inline BSDFSample sampleSurfaceBSDF(float3 N,
                                                      float3 V,
                                                      float3 baseColor,
                                                      float metallic,
                                                      float roughness,
                                                      float u1,
                                                      float u2,
                                                      float uLobe)
{
    BSDFSample sample{};
    sample.valid = 0u;

    const float r = clamp(roughness, 0.02f, 0.98f);
    const float3 F0 = lerp3(float3(0.04f), baseColor, metallic);
    const float specProb = clamp(max(F0.x, max(F0.y, F0.z)), 0.05f, 0.95f);

    float3 L;
    if (uLobe < specProb)
    {
        const float alpha = r * r;
        const float alpha2 = alpha * alpha;
        const float phi = 2.0f * PI * u2;
        const float cosTheta = sqrtf((1.0f - u1) / max(1.0f + (alpha2 - 1.0f) * u1, 1.0e-6f));
        const float sinTheta = sqrtf(max(0.0f, 1.0f - cosTheta * cosTheta));

        float3 tangent;
        float3 bitangent;
        buildOrthonormalBasis(N, tangent, bitangent);
        const float3 H = safeNormalize(tangent * (sinTheta * cosf(phi)) +
                                       bitangent * (sinTheta * sinf(phi)) +
                                       N * cosTheta);
        L = safeNormalize(reflect(-V, H));
    }
    else
    {
        const float3 localDir = cosineSampleHemisphere(u1, u2);
        float3 tangent;
        float3 bitangent;
        buildOrthonormalBasis(N, tangent, bitangent);
        L = safeNormalize(localDir.x * tangent + localDir.y * bitangent + localDir.z * N);
    }

    const float NdL = max(dot(N, L), 0.0f);
    if (NdL <= 0.0f)
        return sample;

    const float pdf = pdfSurfaceBSDF(N, V, L, baseColor, metallic, roughness);
    if (pdf <= 1.0e-8f)
        return sample;

    sample.valid = 1u;
    sample.direction = L;
    sample.pdf = pdf;
    sample.weight = evalSurfaceBRDF(N, V, L, baseColor, metallic, roughness) * (NdL / pdf);
    return sample;
}

__device__ static inline float computeDecalDerivedMask(const SampledTexel &albedo)
{
    const float luma = sampledTexelLuminance(albedo);
    const float maxC = max(max(albedo.x, albedo.y), albedo.z);
    const float minC = min(min(albedo.x, albedo.y), albedo.z);
    const float chroma = maxC - minC;
    const float contentMask = saturate(chroma * 2.10f + (1.0f - luma) * 1.05f - 0.03f);
    return saturate(max(albedo.w, contentMask));
}

__device__ static inline float sampleDecalMask(float2 uv,
                                               float2 detailUV,
                                               const SampledTexel &decalAlbedo,
                                               DecalGPU decal,
                                               const SceneTextureArray &sceneTextures)
{
    float mask = computeDecalDerivedMask(decalAlbedo);
    const bool hasOpacityTex = (decal.opacityTexIndex >= 0);
    const bool hasDetailTex = (decal.detailTexIndex >= 0);

    if (hasOpacityTex)
    {
        const SampledTexel m = sampleTexture4(decal.opacityTexIndex, uv, sceneTextures);
        const float texMask = saturate(max(m.w, sampledTexelLuminance(m)));
        mask *= texMask;
    }

    if (hasDetailTex)
    {
        const SampledTexel det = sampleTexture4(decal.detailTexIndex, detailUV, sceneTextures);
        const float detailMask = saturate(max(det.w, sampledTexelLuminance(det)));
        mask *= mix(1.0f, detailMask, hasOpacityTex ? 0.25f : 0.12f);
    }

    float opacityPower = max(decal.opacityPower, 0.25f);
    if (!hasOpacityTex)
    {
        mask = smoothstep(0.05f, 0.48f, saturate(mask));
        opacityPower = min(opacityPower, 1.25f);
    }
    else
    {
        opacityPower = min(opacityPower, 4.0f);
    }

    mask = powf(max(mask, 1.0e-4f), opacityPower);
    return saturate(mask * decal.opacity);
}

__device__ static inline void applyProjectedDecalsPrimary(float3 hitPos,
                                                          float3 Ng,
                                                          float3 &Ns,
                                                          float3 &baseColor,
                                                          float &roughness,
                                                          const DecalGPU *decals,
                                                          uint decalCount,
                                                          const SceneTextureArray &sceneTextures)
{
    if (decals == nullptr || decalCount == 0u)
        return;

    const float3 NgNorm = safeNormalize(Ng);

    for (uint i = 0u; i < decalCount; ++i)
    {
        const DecalGPU decal = decals[i];
        if (decal.baseColorTexIndex < 0)
            continue;

        const float3 pos = float3(decal.posX, decal.posY, decal.posZ);
        const float3 projAxis = safeNormalize(float3(decal.axisXx, decal.axisXy, decal.axisXz));
        const float3 uAxis = safeNormalize(float3(decal.axisZx, decal.axisZy, decal.axisZz));
        const float3 vAxis = safeNormalize(-float3(decal.axisYx, decal.axisYy, decal.axisYz));

        const float3 rel = hitPos - pos;
        const float depth = dot(rel, projAxis);
        const float lx = dot(rel, uAxis);
        const float ly = dot(rel, vAxis);
        if (fabsf(depth) > decal.sizeX || fabsf(lx) > decal.sizeZ || fabsf(ly) > decal.sizeY)
            continue;

        const float facing = dot(NgNorm, -projAxis);
        if (facing < 0.06f)
            continue;

        const float2 uv = float2(0.5f + 0.5f * (lx / max(decal.sizeZ, 1.0e-4f)),
                                 0.5f - 0.5f * (ly / max(decal.sizeY, 1.0e-4f)));
        if (uv.x < 0.0f || uv.y < 0.0f || uv.x > 1.0f || uv.y > 1.0f)
            continue;

        const float2 detailScale = float2(max(fabsf(decal.tilingU), 1.0f), max(fabsf(decal.tilingV), 1.0f));
        const float2 detailUV = fract((uv - 0.5f) * detailScale + 0.5f);

        const SampledTexel decalTex = sampleTexture4(decal.baseColorTexIndex, uv, sceneTextures);
        float alpha = sampleDecalMask(uv, detailUV, decalTex, decal, sceneTextures);
        if (alpha <= 1.0e-4f)
            continue;

        const float depthFade = 1.0f - smoothstep(decal.sizeX * 0.82f, decal.sizeX, fabsf(depth));
        const float edgeFadeU = 1.0f - smoothstep(decal.sizeZ * 0.90f, decal.sizeZ, fabsf(lx));
        const float edgeFadeV = 1.0f - smoothstep(decal.sizeY * 0.90f, decal.sizeY, fabsf(ly));
        alpha *= depthFade * edgeFadeU * edgeFadeV;
        if (alpha <= 1.0e-4f)
            continue;

        const float3 decalColor = sampledTexelRgb(decalTex) * float3(decal.baseColorX, decal.baseColorY, decal.baseColorZ);
        baseColor = mix(baseColor, decalColor, clamp01(alpha));

        float decalRoughness = clamp(roughness + decal.roughnessBias, 0.02f, 0.98f);
        if (decal.ormTexIndex >= 0)
            decalRoughness = clamp(sampleTextureRGB(decal.ormTexIndex, uv, sceneTextures, float3(1.0f)).y + decal.roughnessBias, 0.02f, 0.98f);
        else if (decal.roughnessTexIndex >= 0)
            decalRoughness = clamp(sampleTextureR(decal.roughnessTexIndex, uv, sceneTextures, roughness) + decal.roughnessBias, 0.02f, 0.98f);
        roughness = mix(roughness, decalRoughness, clamp01(alpha * 0.65f));

        if (decal.normalTexIndex >= 0)
        {
            float3 decalNt = sampleTextureRGB(decal.normalTexIndex, uv, sceneTextures, float3(0.5f, 0.5f, 1.0f)) * 2.0f - 1.0f;
            decalNt.x *= decal.normalIntensity;
            decalNt.y *= decal.normalIntensity;
            decalNt = safeNormalize(decalNt);
            const float3 decalNw = safeNormalize(decalNt.x * uAxis + decalNt.y * vAxis - decalNt.z * projAxis);
            Ns = safeNormalize(mix(Ns, decalNw, clamp01(alpha * 0.5f)));
        }
    }
}

__device__ static inline float3 sampleDirectEmissiveLightingTexturedInstanced(float3 hitPos,
                                                                               float3 N,
                                                                               float3 Ng,
                                                                               float3 V,
                                                                               uint hitTriIndex,
                                                                               uint hitInstanceIndex,
                                                                               float3 baseColor,
                                                                               float metallic,
                                                                               float roughness,
                                                                               const BVHNode *tlasNodes,
                                                                               const BVHNode *meshNodes,
                                                                               const Triangle *triangles,
                                                                               const SceneInstanceGPU *instances,
                                                                               uint tlasNodeCount,
                                                                               uint meshNodeCount,
                                                                               uint instanceCount,
                                                                               int rootIndex,
                                                                               const MaterialGPU *materials,
                                                                               uint materialCount,
                                                                               const MaterialGPU_PBR *materialsPBR,
                                                                               uint materialPBRCount,
                                                                               const SceneTextureArray &sceneTextures,
                                                                               const EmissiveTriangleGPU *emissiveTriangles,
                                                                               uint emissiveTriangleCount,
                                                                               uint seedSelect,
                                                                               uint seedBaryU,
                                                                               uint seedBaryV)
{
    if (emissiveTriangles == nullptr || emissiveTriangleCount == 0u)
        return float3(0.0f);

    const EmissiveLightSample ls =
        sampleOneEmissiveTriangle(triangles, instances, emissiveTriangles, emissiveTriangleCount,
                                  seedSelect, seedBaryU, seedBaryV);
    if (ls.valid == 0u)
        return float3(0.0f);
    if (ls.triIndex == hitTriIndex && ls.instanceIndex == hitInstanceIndex)
        return float3(0.0f);

    const float3 toLight = ls.position - hitPos;
    const float dist2 = dot(toLight, toLight);
    if (dist2 <= 1.0e-10f)
        return float3(0.0f);

    const float dist = sqrtf(dist2);
    const float3 L = toLight / dist;
    const float NdL = max(dot(N, L), 0.0f);
    if (NdL <= 0.0f)
        return float3(0.0f);

    const float cosLight = fabsf(dot(safeNormalize(ls.normal), -L));
    if (cosLight <= 1.0e-6f)
        return float3(0.0f);

    const float maxDist = dist - 2.0f * SHADOW_EPS;
    if (maxDist <= 0.0f)
        return float3(0.0f);

    Ray shadowRay{};
    shadowRay.origin = hitPos + Ng * SHADOW_EPS;
    shadowRay.direction = L;
    if (traceShadowSceneBVHSkippingThin(tlasNodes, meshNodes, triangles, instances,
                                        tlasNodeCount, meshNodeCount, instanceCount,
                                        rootIndex, materialsPBR, materialPBRCount, 0u,
                                        shadowRay, maxDist))
    {
        return float3(0.0f);
    }

    const float3 Le = resolveTriangleEmissionTextured(triangles,
                                                      ls.triIndex,
                                                      ls.uv0,
                                                      ls.uv1,
                                                      ls.uv2,
                                                      materials,
                                                      materialCount,
                                                      materialsPBR,
                                                      materialPBRCount,
                                                      sceneTextures);
    if (luminance3(Le) <= 1.0e-6f)
        return float3(0.0f);

    const float pdfLight = ls.selectionPdf * dist2 / max(cosLight * ls.area, 1.0e-6f);
    if (pdfLight <= 1.0e-8f)
        return float3(0.0f);

    const float pdfBsdf = pdfSurfaceBSDF(N, V, L, baseColor, metallic, roughness);
    const float mis = powerHeuristic(pdfLight, pdfBsdf);
    const float3 f = evalSurfaceBRDF(N, V, L, baseColor, metallic, roughness);
    return Le * f * (NdL * mis / pdfLight);
}

__device__ static inline float3 computeLightingAtPointInstanced(float3 hitPos,
                                                                float3 N,
                                                                float3 Ng,
                                                                float3 baseColor,
                                                                float metallic,
                                                                float roughness,
                                                                float3 V,
                                                                const BVHNode *tlasNodes,
                                                                const BVHNode *meshNodes,
                                                                const Triangle *triangles,
                                                                const SceneInstanceGPU *instances,
                                                                uint tlasNodeCount,
                                                                uint meshNodeCount,
                                                                uint instanceCount,
                                                                int rootIndex,
                                                                const LightGPU *lights,
                                                                uint lightCount,
                                                                float worldUnitToMeters,
                                                                const MaterialGPU_PBR *materialsPBR,
                                                                uint materialPBRCount,
                                                                uint baseSeed)
{
    float3 result = float3(0.0f);
    if (lights == nullptr || lightCount == 0u)
        return result;

    const float3 shadowOrigin = hitPos + Ng * SHADOW_EPS;
    const float r = clamp(roughness, 0.05f, 0.95f);
    const float3 F0 = lerp3(float3(0.04f), baseColor, clamp(metallic, 0.0f, 1.0f));
    const float alpha = r * r;
    const float NdV = max(dot(N, V), 0.0f);

    float k = r + 1.0f;
    k = (k * k) * 0.125f;

    for (uint i = 0u; i < lightCount; ++i)
    {
        const LightGPU &light = lights[i];
        float3 lightPos = float3(light.position);
        const float3 lightDir = safeNormalize(float3(light.direction));

        float3 L;
        float dist = 1.0f;
        bool hasFinite = true;
        if (light.type == LIGHT_TYPE_DIRECTIONAL)
        {
            L = -lightDir;
            hasFinite = false;
        }
        else
        {
            const uint seed1 = baseSeed ^ (0x9E3779B9u * (i * 3u + 1u));
            const uint seed2 = baseSeed ^ (0x9E3779B9u * (i * 3u + 2u));
            const uint seed3 = baseSeed ^ (0x9E3779B9u * (i * 3u + 3u));
            if (light.type == LIGHT_TYPE_POINT || light.type == LIGHT_TYPE_SPOT)
                lightPos = sampleFiniteLightPosition(light, seed1, seed2, seed3);

            const float3 toLight = lightPos - hitPos;
            dist = length(toLight);
            if (dist <= 0.0f)
                continue;
            L = toLight / dist;
        }

        float NdL = dot(N, L);
        if (NdL <= 0.0f)
            continue;

        float attenuation = 1.0f;
        if (hasFinite)
        {
            attenuation = computeFiniteLightAttenuation(light, dist, worldUnitToMeters);
            if (attenuation <= 0.0f)
                continue;
        }

        float intensity = light.intensity * attenuation;
        if (light.type == LIGHT_TYPE_SPOT)
        {
            intensity *= computeSpotFactor(light, lightPos, hitPos);
            if (intensity <= 0.0f)
                continue;
        }

        if (light.type == LIGHT_TYPE_AREA)
        {
            const float cosEmit = dot(lightDir, safeNormalize(hitPos - lightPos));
            if (cosEmit <= 0.0f)
                continue;
            intensity *= cosEmit;
            if (intensity <= 0.0f)
                continue;
        }

        const float maxDist = hasFinite ? (dist - SHADOW_EPS) : -1.0f;
        if ((light.flags & LIGHT_FLAG_CASTS_SHADOW) != 0u)
        {
            Ray shadowRay{};
            shadowRay.origin = shadowOrigin;
            shadowRay.direction = L;
            const uint ignoreOwnerId = shouldIgnoreOwnerShadowForLight(light) ? light.ownerId : 0u;
            if (traceShadowSceneBVHSkippingThin(tlasNodes, meshNodes, triangles, instances,
                                                tlasNodeCount, meshNodeCount, instanceCount,
                                                rootIndex, materialsPBR, materialPBRCount,
                                                ignoreOwnerId, shadowRay, maxDist))
            {
                continue;
            }
        }

        const float3 H = safeNormalize(L + V);
        const float NdH = max(dot(N, H), 0.0f);
        const float HdV = max(dot(V, H), 0.0f);
        NdL = max(NdL, 0.0f);
        if (NdL <= 0.0f || NdH <= 0.0f || NdV <= 0.0f)
            continue;

        const float3 F = fresnelSchlick(HdV, F0);
        const float D = distributionGGX(NdH, alpha);
        const float G = geometrySmith(NdV, NdL, k);
        const float3 spec = (D * G * F) / max(4.0f * NdV * NdL, 1.0e-4f);
        const float3 kd = (float3(1.0f) - F) * (1.0f - clamp(metallic, 0.0f, 1.0f));
        const float3 diffuse = kd * baseColor * INV_PI;
        result += float3(light.color) * intensity * (diffuse + spec) * NdL;
    }

    return result;
}

__device__ static inline float computeAirDustDensityWeightAtPoint(float3 p,
                                                                  const AirDustVolumeGPU &volume)
{
    const float3 extent = max(float3(volume.extentX, volume.extentY, volume.extentZ), float3(1.0f));
    const float3 q = (p - float3(volume.centerX, volume.centerY, volume.centerZ)) / extent;
    const float r2 = dot(q, q);
    const float core = clamp(1.0f - r2, 0.0f, 1.0f);
    return core * core;
}

__device__ static inline float henyeyGreensteinPhase(float cosTheta, float g)
{
    g = clamp(g, -0.95f, 0.95f);
    const float gg = g * g;
    const float denom = powf(max(1.0f + gg - 2.0f * g * cosTheta, 1.0e-4f), 1.5f);
    return (1.0f - gg) / max(4.0f * PI * denom, 1.0e-4f);
}

__device__ static inline PrimaryMediumSample evaluatePrimaryMediumAtPoint(float3 p,
                                                                          float distanceAlongRay,
                                                                          PrimaryVolumeParams pp,
                                                                          const AirDustVolumeGPU *airDustVolumes,
                                                                          uint airDustVolumeCount)
{
    PrimaryMediumSample sample{};
    sample.sigmaT = 0.0f;
    sample.albedo = float3(0.0f);
    sample.anisotropy = 0.0f;
    sample.ambientTint = float3(0.0f);

    const float worldToMeters = max(pp.worldUnitToMeters, 1.0e-4f);
    float globalSigmaT = 0.0f;
    const float distanceMeters = max(distanceAlongRay, 0.0f) * worldToMeters;
    const float startMeters = max(pp.fogStartDistance, 0.0f) * worldToMeters;
    const float startFade = smoothstep(startMeters,
                                       startMeters + VOLUME_FOG_START_FADE_METERS,
                                       distanceMeters);

    if (pp.fogDensity > 1.0e-6f && startFade > 1.0e-4f)
    {
        const float heightMeters = (p.z - pp.fogHeightZ) * worldToMeters;
        const float exponent = clamp(-pp.fogHeightFalloff * heightMeters, -10.0f, 10.0f);
        const float heightFactor = expf(exponent);
        globalSigmaT = max(pp.fogDensity, 0.0f) *
                       max(pp.fogExtinctionScale, 0.0f) *
                       VOLUME_FOG_DENSITY_SCALE *
                       heightFactor *
                       startFade;
        globalSigmaT = min(globalSigmaT, 0.08f);
    }

    float localSigmaT = 0.0f;
    float localAnisotropyWeighted = 0.0f;
    for (uint i = 0u; i < airDustVolumeCount; ++i)
    {
        const AirDustVolumeGPU &volume = airDustVolumes[i];
        const float shape = computeAirDustDensityWeightAtPoint(p, volume);
        if (shape <= 1.0e-4f)
            continue;

        const float sigma = max(volume.density, 0.0f) * shape;
        localSigmaT += sigma;
        localAnisotropyWeighted += sigma * clamp(volume.anisotropy, -0.95f, 0.95f);
    }

    sample.sigmaT = globalSigmaT + localSigmaT;
    if (sample.sigmaT <= 1.0e-6f)
        return sample;

    float3 globalAlbedo = clamp(float3(pp.fogAlbedoX, pp.fogAlbedoY, pp.fogAlbedoZ), 0.0f, 1.0f);
    if (luminance3(globalAlbedo) <= 1.0e-5f)
        globalAlbedo = clamp(float3(pp.fogColorX, pp.fogColorY, pp.fogColorZ), 0.0f, 1.0f);

    const float3 localAlbedo = float3(0.96f);
    sample.albedo = (globalAlbedo * globalSigmaT + localAlbedo * localSigmaT) / max(sample.sigmaT, 1.0e-6f);
    sample.ambientTint = float3(pp.fogColorX, pp.fogColorY, pp.fogColorZ) *
                         (globalSigmaT / max(sample.sigmaT, 1.0e-6f));
    sample.anisotropy = clamp((clamp(pp.fogScatteringG, -0.95f, 0.95f) * globalSigmaT +
                               localAnisotropyWeighted) / max(sample.sigmaT, 1.0e-6f),
                              -0.95f,
                              0.95f);
    return sample;
}

__device__ static inline float computeVolumeLightImportance(const LightGPU &light,
                                                            float3 samplePos,
                                                            float worldUnitToMeters)
{
    const float colorWeight = max(luminance3(float3(light.color)), 1.0e-3f);
    const float baseWeight = max(light.intensity, 0.0f) * colorWeight;
    if (baseWeight <= 1.0e-6f)
        return 0.0f;

    if (light.type == LIGHT_TYPE_DIRECTIONAL)
        return baseWeight;

    const float3 toLight = float3(light.position) - samplePos;
    const float dist = length(toLight);
    if (dist <= 1.0e-4f)
        return 0.0f;

    float weight = baseWeight * computeFiniteLightAttenuation(light, dist, worldUnitToMeters);
    if (weight <= 1.0e-8f)
        return 0.0f;

    if (light.type == LIGHT_TYPE_SPOT)
        weight *= computeSpotFactor(light, float3(light.position), samplePos);

    if (light.type == LIGHT_TYPE_AREA)
    {
        const float cosEmit = dot(safeNormalize(float3(light.direction)), safeNormalize(samplePos - float3(light.position)));
        if (cosEmit <= 0.0f)
            return 0.0f;
        weight *= cosEmit;
    }

    return max(weight, 0.0f);
}

__device__ static inline bool sampleVolumeLightIndex(float3 samplePos,
                                                     const LightGPU *lights,
                                                     uint lightCount,
                                                     float worldUnitToMeters,
                                                     uint seedSelect,
                                                     uint &outLightIndex,
                                                     float &outPmf)
{
    outLightIndex = 0u;
    outPmf = 0.0f;
    if (lights == nullptr || lightCount == 0u)
        return false;

    float totalWeight = 0.0f;
    for (uint i = 0u; i < lightCount; ++i)
        totalWeight += computeVolumeLightImportance(lights[i], samplePos, worldUnitToMeters);

    if (totalWeight <= 1.0e-8f)
        return false;

    const float target = rand01(seedSelect) * totalWeight;
    float accum = 0.0f;
    uint fallbackIndex = 0u;
    float fallbackWeight = 0.0f;
    for (uint i = 0u; i < lightCount; ++i)
    {
        const float weight = computeVolumeLightImportance(lights[i], samplePos, worldUnitToMeters);
        if (weight <= 0.0f)
            continue;

        fallbackIndex = i;
        fallbackWeight = weight;
        accum += weight;
        if (target <= accum)
        {
            outLightIndex = i;
            outPmf = weight / totalWeight;
            return true;
        }
    }

    if (fallbackWeight > 0.0f)
    {
        outLightIndex = fallbackIndex;
        outPmf = fallbackWeight / totalWeight;
        return true;
    }

    return false;
}

__device__ static inline float3 computePrimaryVolumeAmbientSource(const PrimaryMediumSample &medium)
{
    if (luminance3(medium.ambientTint) <= 1.0e-6f)
        return float3(0.0f);
    return medium.ambientTint * VOLUME_AMBIENT_SCATTER_SCALE;
}

__device__ static inline float3 samplePrimaryVolumeExplicitLight(float3 samplePos,
                                                                 float3 rayDir,
                                                                 const PrimaryMediumSample &medium,
                                                                 PrimaryVolumeParams pp,
                                                                 const LightGPU *lights,
                                                                 uint lightCount,
                                                                 const BVHNode *tlasNodes,
                                                                 const BVHNode *meshNodes,
                                                                 const Triangle *triangles,
                                                                 const SceneInstanceGPU *instances,
                                                                 uint tlasNodeCount,
                                                                 uint meshNodeCount,
                                                                 uint instanceCount,
                                                                 int rootIndex,
                                                                 const MaterialGPU_PBR *materialsPBR,
                                                                 uint materialPBRCount,
                                                                 uint seedSelect,
                                                                 uint seedA,
                                                                 uint seedB,
                                                                 uint seedC)
{
    (void)medium;

    uint lightIndex = 0u;
    float lightPmf = 0.0f;
    if (!sampleVolumeLightIndex(samplePos, lights, lightCount, pp.worldUnitToMeters,
                                seedSelect, lightIndex, lightPmf))
    {
        return float3(0.0f);
    }

    const LightGPU &light = lights[lightIndex];
    float3 lightPos = float3(light.position);
    const float3 lightDir = safeNormalize(float3(light.direction));

    float3 L = float3(0.0f);
    float dist = 1.0f;
    bool hasFinite = true;

    if (light.type == LIGHT_TYPE_DIRECTIONAL)
    {
        L = -lightDir;
        hasFinite = false;
    }
    else
    {
        if (light.type == LIGHT_TYPE_POINT || light.type == LIGHT_TYPE_SPOT)
            lightPos = sampleFiniteLightPosition(light, seedA, seedB, seedC);

        const float3 toLight = lightPos - samplePos;
        dist = length(toLight);
        if (dist <= 1.0e-4f)
            return float3(0.0f);
        L = toLight / dist;
    }

    float attenuation = 1.0f;
    if (hasFinite)
    {
        attenuation = computeFiniteLightAttenuation(light, dist, pp.worldUnitToMeters);
        if (attenuation <= 0.0f)
            return float3(0.0f);
    }

    float intensity = light.intensity * attenuation;
    if (light.type == LIGHT_TYPE_SPOT)
    {
        intensity *= computeSpotFactor(light, lightPos, samplePos);
        if (intensity <= 0.0f)
            return float3(0.0f);
    }

    if (light.type == LIGHT_TYPE_AREA)
    {
        const float cosEmit = dot(lightDir, safeNormalize(samplePos - lightPos));
        if (cosEmit <= 0.0f)
            return float3(0.0f);
        intensity *= cosEmit;
        if (intensity <= 0.0f)
            return float3(0.0f);
    }

    const float maxDist = hasFinite ? (dist - SHADOW_EPS) : -1.0f;
    if ((light.flags & LIGHT_FLAG_CASTS_SHADOW) != 0u)
    {
        Ray shadowRay{};
        shadowRay.origin = samplePos + L * SHADOW_EPS;
        shadowRay.direction = L;
        const uint ignoreOwnerId = shouldIgnoreOwnerShadowForLight(light) ? light.ownerId : 0u;
        if (traceShadowSceneBVHSkippingThin(tlasNodes, meshNodes, triangles, instances,
                                            tlasNodeCount, meshNodeCount, instanceCount,
                                            rootIndex, materialsPBR, materialPBRCount,
                                            ignoreOwnerId, shadowRay, maxDist))
        {
            return float3(0.0f);
        }
    }

    const float phase = henyeyGreensteinPhase(dot(rayDir, L), medium.anisotropy);
    const float3 Li = float3(light.color) * intensity;
    return Li * (phase / max(lightPmf, 1.0e-6f)) * VOLUME_LIGHT_SCATTER_SCALE;
}

__device__ static inline PrimaryVolumetricResult integratePrimaryVolumetrics(const Ray &ray,
                                                                             float maxDistance,
                                                                             PrimaryVolumeParams pp,
                                                                             const AirDustVolumeGPU *airDustVolumes,
                                                                             uint airDustVolumeCount,
                                                                             const LightGPU *lights,
                                                                             uint lightCount,
                                                                             const BVHNode *tlasNodes,
                                                                             const BVHNode *meshNodes,
                                                                             const Triangle *triangles,
                                                                             const SceneInstanceGPU *instances,
                                                                             uint tlasNodeCount,
                                                                             uint meshNodeCount,
                                                                             uint instanceCount,
                                                                             int rootIndex,
                                                                             const MaterialGPU_PBR *materialsPBR,
                                                                             uint materialPBRCount,
                                                                             uint seedBase)
{
    PrimaryVolumetricResult result{};
    result.inscattering = float3(0.0f);
    result.transmittance = float3(1.0f);

    if (maxDistance <= 1.0e-4f)
        return result;
    if (pp.fogDensity <= 1.0e-6f && airDustVolumeCount == 0u)
        return result;

    const float worldToMeters = max(pp.worldUnitToMeters, 1.0e-4f);
    const float marchStart = max(pp.nearPlane, 0.0f);
    const float marchEnd = min(maxDistance, pp.farPlane);
    if (marchEnd <= marchStart + 1.0e-4f)
        return result;

    const float marchDistance = marchEnd - marchStart;
    const float marchDistanceMeters = marchDistance * worldToMeters;
    int stepCount = max(8, min(static_cast<int>(ceilf(max(marchDistanceMeters, 1.0f) * 0.45f)), 28));
    if (airDustVolumeCount > 0u)
        stepCount = min(stepCount + 4, 32);

    const float stepSize = marchDistance / static_cast<float>(stepCount);
    const float stepSizeMeters = stepSize * worldToMeters;
    const float jitter = rand01(seedBase ^ 0x51C3u);
    const float minTransmittance = max(0.0f, 1.0f - clamp01(pp.fogMaxOpacity));

    for (int step = 0; step < stepCount; ++step)
    {
        float tSample = marchStart + (static_cast<float>(step) + jitter) * stepSize;
        tSample = clamp(tSample, marchStart, marchEnd - 0.5f * stepSize);

        const float3 samplePos = ray.origin + ray.direction * tSample;
        const PrimaryMediumSample medium =
            evaluatePrimaryMediumAtPoint(samplePos, tSample, pp, airDustVolumes, airDustVolumeCount);
        if (medium.sigmaT <= 1.0e-6f)
            continue;

        const float stepTransmittance = expf(-medium.sigmaT * stepSizeMeters);
        const float stepScatter = 1.0f - stepTransmittance;
        if (stepScatter <= 1.0e-6f)
            continue;

        float3 source = computePrimaryVolumeAmbientSource(medium);
        if (lights != nullptr && lightCount > 0u)
        {
            const uint stepSeed = seedBase ^ (0x9E3779B9u * static_cast<uint>(step + 1));
            source += samplePrimaryVolumeExplicitLight(samplePos,
                                                       ray.direction,
                                                       medium,
                                                       pp,
                                                       lights,
                                                       lightCount,
                                                       tlasNodes,
                                                       meshNodes,
                                                       triangles,
                                                       instances,
                                                       tlasNodeCount,
                                                       meshNodeCount,
                                                       instanceCount,
                                                       rootIndex,
                                                       materialsPBR,
                                                       materialPBRCount,
                                                       stepSeed ^ 0x11u,
                                                       stepSeed ^ 0x23u,
                                                       stepSeed ^ 0x47u,
                                                       stepSeed ^ 0x83u);
        }

        result.inscattering += result.transmittance * (medium.albedo * stepScatter) * source;
        result.transmittance *= stepTransmittance;
        result.transmittance = max(result.transmittance, float3(minTransmittance));

        if (max(result.transmittance.x, max(result.transmittance.y, result.transmittance.z)) <= 1.0e-3f)
            break;
    }

    return result;
}

__device__ static inline PathTraceTextureResult tracePathPixelTextured(uint2 gid,
                                                                       uint2 imgSize,
                                                                       const BVHNode *tlasNodes,
                                                                       const BVHNode *meshNodes,
                                                                       const Triangle *triangles,
                                                                       const SceneInstanceGPU *instances,
                                                                       uint tlasNodeCount,
                                                                       uint meshNodeCount,
                                                                       uint instanceCount,
                                                                       int rootIndex,
                                                                       float3 camPos,
                                                                       float3 camForward,
                                                                       float3 camUp,
                                                                       float3 camRight,
                                                                       float fovY,
                                                                       float aspectRatio,
                                                                       int samplesPerPixel,
                                                                       const LightGPU *lights,
                                                                       uint lightCount,
                                                                       uint sampleBaseIndex,
                                                                       const MaterialGPU *materials,
                                                                       uint materialCount,
                                                                       const MaterialGPU_PBR *materialsPBR,
                                                                       uint materialPBRCount,
                                                                       const EmissiveTriangleGPU *emissiveTriangles,
                                                                       uint emissiveTriangleCount,
                                                                       const DecalGPU *decals,
                                                                       uint decalCount,
                                                                       PrimaryVolumeParams volumeParams,
                                                                       const AirDustVolumeGPU *airDustVolumes,
                                                                       uint airDustVolumeCount,
                                                                       const SceneTextureArray &sceneTextures)
{
    PathTraceTextureResult result{};
    result.color = float3(0.0f);
    result.depth = 1.0e30f;
    result.albedo = float3(0.0f);
    result.hitMask = 0.0f;
    result.normal = float3(0.0f);

    if (gid.x >= imgSize.x || gid.y >= imgSize.y)
        return result;

    int spp = max(samplesPerPixel, 1);
    volumeParams.worldUnitToMeters = max(volumeParams.worldUnitToMeters, 1.0e-4f);

    int strataDim = static_cast<int>(floorf(sqrtf(static_cast<float>(spp))));
    if (strataDim < 1)
        strataDim = 1;
    const int strataCount = strataDim * strataDim;

    float3 pixelColor = float3(0.0f);
    const int maxBounces = 4;

    for (int s = 0; s < spp; ++s)
    {
        const uint sampleIndex = sampleBaseIndex + static_cast<uint>(s);
        const uint seedBase = gid.x * 73856093u ^
                              gid.y * 19349663u ^
                              static_cast<uint>(s) * 83492791u ^
                              sampleIndex * 2654435761u;

        float jx = 0.0f;
        float jy = 0.0f;
        if (s < strataCount)
        {
            const int sx = s % strataDim;
            const int sy = s / strataDim;
            jx = (static_cast<float>(sx) + rand01(seedBase ^ 0x1234u)) / static_cast<float>(strataDim) - 0.5f;
            jy = (static_cast<float>(sy) + rand01(seedBase ^ 0x5678u)) / static_cast<float>(strataDim) - 0.5f;
        }
        else
        {
            jx = rand01(seedBase ^ 0xABCDEFu) - 0.5f;
            jy = rand01(seedBase ^ 0x13579Bu) - 0.5f;
        }

        const float2 jitter = (s == 0) ? float2(0.0f) : float2(jx, jy);
        Ray ray = makePrimaryRayJittered(static_cast<int>(gid.x),
                                         static_cast<int>(gid.y),
                                         static_cast<int>(imgSize.x),
                                         static_cast<int>(imgSize.y),
                                         jitter,
                                         camPos,
                                         camForward,
                                         camUp,
                                         camRight,
                                         fovY,
                                         aspectRatio);

        float3 throughput = float3(1.0f);
        float3 radiance = float3(0.0f);
        float prevBsdfPdf = 0.0f;
        float3 prevSurfacePos = float3(0.0f);
        uint prevWasBsdfSample = 0u;

        for (int bounce = 0; bounce < maxBounces; ++bounce)
        {
            const HitInfo hit = traceRaySceneBVH(tlasNodes,
                                                 meshNodes,
                                                 triangles,
                                                 instances,
                                                 tlasNodeCount,
                                                 meshNodeCount,
                                                 instanceCount,
                                                 rootIndex,
                                                 ray);

            if (prevWasBsdfSample == 0u)
            {
                const float segmentMaxDistance = (hit.hit && hit.triIndex >= 0) ? hit.t : volumeParams.farPlane;
                const PrimaryVolumetricResult volumeResult =
                    integratePrimaryVolumetrics(ray,
                                                segmentMaxDistance,
                                                volumeParams,
                                                airDustVolumes,
                                                airDustVolumeCount,
                                                lights,
                                                lightCount,
                                                tlasNodes,
                                                meshNodes,
                                                triangles,
                                                instances,
                                                tlasNodeCount,
                                                meshNodeCount,
                                                instanceCount,
                                                rootIndex,
                                                materialsPBR,
                                                materialPBRCount,
                                                seedBase ^ (0x6000u + static_cast<uint>(bounce) * 131u));
                radiance += throughput * volumeResult.inscattering;
                throughput *= volumeResult.transmittance;
            }

            if (!(hit.hit && hit.triIndex >= 0))
            {
                radiance += throughput * environmentColor(safeNormalize(ray.direction));
                break;
            }

            float3 hitPos = hit.position;
            const float3 NgRaw = safeNormalize(hit.normal);
            float3 N = NgRaw;
            float3 Ng = NgRaw;
            float3 Ns = NgRaw;

            float3 baseColor = hit.color;
            const float3 baseColorFactor = baseColor;
            float3 baseColorTexOnly = float3(1.0f);
            float3 emissive = hit.emission;

            const int matId = hit.materialIndex;
            const float2 uv0 = hit.uv0;
            const float2 uv1 = hit.uv1;
            const float2 uv2 = hit.uv2;
            float metallic = clamp(hit.metallic, 0.0f, 1.0f);
            float roughness = clamp(hit.roughness, 0.02f, 0.98f);
            float ao = 1.0f;

            if (materialsPBR != nullptr && matId >= 0 && static_cast<uint>(matId) < materialPBRCount)
            {
                const MaterialGPU_PBR mp = materialsPBR[matId];
                const float2 uvBase = fract(selectUvSet(uv0, uv1, uv2, mp.baseColorUvSet));
                const float2 uvEmission = fract(selectUvSet(uv0, uv1, uv2, mp.emissionUvSet));
                const float2 uvNormal = fract(selectUvSet(uv0, uv1, uv2, mp.normalUvSet));
                const float2 uvOrm = fract(selectUvSet(uv0, uv1, uv2, mp.ormUvSet));
                const float2 uvRoughness = fract(selectUvSet(uv0, uv1, uv2, mp.roughnessUvSet));
                const float2 uvMetallic = fract(selectUvSet(uv0, uv1, uv2, mp.metallicUvSet));
                const float2 uvOcclusion = fract(selectUvSet(uv0, uv1, uv2, mp.occlusionUvSet));

                if (mp.baseColorTexIndex >= 0)
                {
                    baseColorTexOnly = sampleTextureRGB(mp.baseColorTexIndex, uvBase, sceneTextures, float3(1.0f));
                    baseColor = baseColorFactor * baseColorTexOnly;
                }
                if (mp.ormTexIndex >= 0)
                {
                    const float3 orm = sampleTextureRGB(mp.ormTexIndex, uvOrm, sceneTextures, float3(1.0f));
                    ao = clamp01(orm.x);
                    roughness = clamp(orm.y, 0.02f, 0.98f);
                    metallic = clamp01(orm.z);
                }
                if (mp.roughnessTexIndex >= 0)
                    roughness = clamp(sampleTextureR(mp.roughnessTexIndex, uvRoughness, sceneTextures, roughness), 0.02f, 0.98f);
                if (mp.metallicTexIndex >= 0)
                    metallic = clamp01(sampleTextureR(mp.metallicTexIndex, uvMetallic, sceneTextures, metallic));
                if (mp.occlusionTexIndex >= 0)
                    ao = clamp01(sampleTextureR(mp.occlusionTexIndex, uvOcclusion, sceneTextures, ao));
                if (mp.normalTexIndex >= 0)
                    Ns = sampleNormalMapInstanced(mp.normalTexIndex, mp.normalUvSet, uvNormal, Ng, triangles, instances, hit.instanceIndex, hit.triIndex, sceneTextures);

                if (mp.specialModel == SPECIAL_MATERIAL_UE_TRAFFIC_LIGHT)
                {
                    const float dirtFactor = computeTrafficLightDirtFactor(mp, uvBase, sceneTextures);
                    const float dirtBlend = mix(1.0f, dirtFactor, 0.35f);
                    baseColor *= dirtBlend;
                    roughness = clamp(mix(roughness, max(mp.specialScalar2, roughness), (1.0f - dirtFactor) * 0.5f), 0.02f, 0.98f);
                }

                const float rawNdV = max(dot(NgRaw, safeNormalize(-ray.direction)), 0.0f);
                emissive = evaluateMaterialEmissionPBR(mp, emissive, baseColorTexOnly, uvBase, uvEmission, rawNdV, sceneTextures);
            }
            else if (materials != nullptr && matId >= 0 && static_cast<uint>(matId) < materialCount)
            {
                const MaterialGPU m = materials[matId];
                const float2 uvBase = fract(selectUvSet(uv0, uv1, uv2, m.baseColorUvSet));
                const float2 uvEmission = fract(selectUvSet(uv0, uv1, uv2, m.emissionUvSet));
                if (m.baseColorTexIndex >= 0)
                {
                    baseColorTexOnly = sampleTextureRGB(m.baseColorTexIndex, uvBase, sceneTextures, float3(1.0f));
                    baseColor = baseColorFactor * baseColorTexOnly;
                }
                if (m.emissionTexIndex >= 0)
                    emissive *= sampleTextureRGB(m.emissionTexIndex, uvEmission, sceneTextures, float3(1.0f));
            }

            (void)ao;
            emissive *= EMISSION_VISIBLE_EXPOSURE_GPU;

            const bool isThinEmissiveSurface =
                materialHasPBRFlag(matId, MATERIAL_FLAG_THIN_EMISSIVE_SURFACE, materialsPBR, materialPBRCount);

            float3 emitted = emissive;
            if (bounce > 0 && prevWasBsdfSample != 0u && luminance3(emitted) > 1.0e-6f)
            {
                const float pdfLight = emissiveLightPdfForHit(prevSurfacePos,
                                                              hitPos,
                                                              NgRaw,
                                                              static_cast<uint>(hit.triIndex),
                                                              emissiveTriangles,
                                                              emissiveTriangleCount);
                if (pdfLight > 1.0e-8f)
                    emitted *= powerHeuristic(prevBsdfPdf, pdfLight);
            }

            if (isThinEmissiveSurface)
            {
                radiance += throughput * emitted;
                ray.origin = hitPos + ray.direction * SHADOW_EPS;
                continue;
            }

            if (bounce == 0)
            {
                applyProjectedDecalsPrimary(hitPos, Ng, Ns, baseColor, roughness, decals, decalCount, sceneTextures);
                if (s == 0)
                {
                    result.depth = hit.t;
                    result.albedo = max(baseColor, float3(0.0f));
                    result.normal = safeNormalize(Ns);
                    result.hitMask = 1.0f;
                }
            }

            if (dot(Ng, ray.direction) > 0.0f)
                Ng = -Ng;
            if (dot(Ns, ray.direction) > 0.0f)
                Ns = -Ns;
            N = Ns;

            if (bounce == 0 && s == 0)
            {
                result.depth = hit.t;
                result.albedo = max(baseColor, float3(0.0f));
                result.normal = safeNormalize(N);
                result.hitMask = 1.0f;
            }

            const float3 V = safeNormalize(-ray.direction);
            const uint lightSeed = seedBase ^ (0x9E3779B9u * (static_cast<uint>(bounce) + 1u));
            const float3 directExplicit =
                computeLightingAtPointInstanced(hitPos,
                                                N,
                                                Ng,
                                                baseColor,
                                                metallic,
                                                roughness,
                                                V,
                                                tlasNodes,
                                                meshNodes,
                                                triangles,
                                                instances,
                                                tlasNodeCount,
                                                meshNodeCount,
                                                instanceCount,
                                                rootIndex,
                                                lights,
                                                lightCount,
                                                volumeParams.worldUnitToMeters,
                                                materialsPBR,
                                                materialPBRCount,
                                                lightSeed);
            const float3 directEmissive =
                sampleDirectEmissiveLightingTexturedInstanced(hitPos,
                                                              N,
                                                              Ng,
                                                              V,
                                                              static_cast<uint>(hit.triIndex),
                                                              static_cast<uint>(max(hit.instanceIndex, 0)),
                                                              baseColor,
                                                              metallic,
                                                              roughness,
                                                              tlasNodes,
                                                              meshNodes,
                                                              triangles,
                                                              instances,
                                                              tlasNodeCount,
                                                              meshNodeCount,
                                                              instanceCount,
                                                              rootIndex,
                                                              materials,
                                                              materialCount,
                                                              materialsPBR,
                                                              materialPBRCount,
                                                              sceneTextures,
                                                              emissiveTriangles,
                                                              emissiveTriangleCount,
                                                              seedBase ^ (0x7000u + static_cast<uint>(bounce) * 17u),
                                                              seedBase ^ (0x7001u + static_cast<uint>(bounce) * 17u),
                                                              seedBase ^ (0x7002u + static_cast<uint>(bounce) * 17u));

            radiance += throughput * (directExplicit + directEmissive + emitted);

            if (bounce >= 1)
            {
                float maxChannel = max(throughput.x, max(throughput.y, throughput.z));
                maxChannel = clamp(maxChannel, 0.1f, 0.95f);
                if (rand01(seedBase ^ (0x10000u * static_cast<uint>(bounce) ^ 0x10u)) > maxChannel)
                    break;
                throughput /= maxChannel;
            }

            const BSDFSample bsdfSample =
                sampleSurfaceBSDF(N,
                                  V,
                                  baseColor,
                                  metallic,
                                  roughness,
                                  rand01(seedBase ^ (0x10000u * static_cast<uint>(bounce) ^ 0x21u)),
                                  rand01(seedBase ^ (0x10000u * static_cast<uint>(bounce) ^ 0x43u)),
                                  rand01(seedBase ^ (0x10000u * static_cast<uint>(bounce) ^ 0x77u)));
            if (bsdfSample.valid == 0u)
                break;

            throughput *= bsdfSample.weight;
            prevWasBsdfSample = 1u;
            prevBsdfPdf = bsdfSample.pdf;
            prevSurfacePos = hitPos;
            ray.origin = hitPos + bsdfSample.direction * SHADOW_EPS;
            ray.direction = bsdfSample.direction;
        }

        pixelColor += radiance;
    }

    result.color = pixelColor / static_cast<float>(spp);
    return result;
}

__global__ void RayTraceTextureKernel(const BVHNode *tlasNodes,
                                      uint tlasNodeCount,
                                      const BVHNode *meshNodes,
                                      uint meshNodeCount,
                                      const Triangle *triangles,
                                      const SceneInstanceGPU *instances,
                                      uint instanceCount,
                                      int rootIndex,
                                      CameraData camera,
                                      const LightGPU *lights,
                                      uint lightCount,
                                      uint sampleBaseIndex,
                                      const MaterialGPU *materials,
                                      uint materialCount,
                                      const MaterialGPU_PBR *materialsPBR,
                                      uint materialPBRCount,
                                      const EmissiveTriangleGPU *emissiveTriangles,
                                      uint emissiveTriangleCount,
                                      const DecalGPU *decals,
                                      uint decalCount,
                                      PrimaryVolumeParams volumeParams,
                                      const AirDustVolumeGPU *airDustVolumes,
                                      uint airDustVolumeCount,
                                      SceneTextureArray sceneTextures,
                                      Vec3 *outFramebuffer)
{
    const uint2 gid = uint2(blockIdx.x * blockDim.x + threadIdx.x,
                            blockIdx.y * blockDim.y + threadIdx.y);
    const uint2 imgSize = uint2(static_cast<uint>(max(camera.width, 0)),
                                static_cast<uint>(max(camera.height, 0)));
    if (gid.x >= imgSize.x || gid.y >= imgSize.y)
        return;
    if (!outFramebuffer || !tlasNodes || !meshNodes || !triangles || !instances)
        return;

    const PathTraceTextureResult pathResult =
        tracePathPixelTextured(gid,
                               imgSize,
                               tlasNodes,
                               meshNodes,
                               triangles,
                               instances,
                               tlasNodeCount,
                               meshNodeCount,
                               instanceCount,
                               rootIndex,
                               float3(camera.position),
                               safeNormalize(float3(camera.forward)),
                               safeNormalize(float3(camera.up)),
                               safeNormalize(float3(camera.right)),
                               camera.fovY,
                               camera.aspectRatio,
                               camera.samplesPerPixel,
                               lights,
                               lightCount,
                               sampleBaseIndex,
                               materials,
                               materialCount,
                               materialsPBR,
                               materialPBRCount,
                               emissiveTriangles,
                               emissiveTriangleCount,
                               decals,
                               decalCount,
                               volumeParams,
                               airDustVolumes,
                               airDustVolumeCount,
                               sceneTextures);

    const std::size_t pixelIndex =
        static_cast<std::size_t>(gid.y) * static_cast<std::size_t>(imgSize.x) +
        static_cast<std::size_t>(gid.x);
    outFramebuffer[pixelIndex] = Vec3{pathResult.color.x, pathResult.color.y, pathResult.color.z};
}

#ifndef __HIP_DEVICE_COMPILE__
namespace
{
    template <typename T>
    struct DeviceBuffer
    {
        T *ptr = nullptr;
        std::size_t capacity = 0u;
    };

    struct HIPRuntimeState
    {
        DeviceBuffer<BVHNode> tlasNodes;
        DeviceBuffer<BVHNode> meshNodes;
        DeviceBuffer<Triangle> triangles;
        DeviceBuffer<SceneInstanceGPU> instances;
        DeviceBuffer<LightGPU> lights;
        DeviceBuffer<MaterialGPU> materials;
        DeviceBuffer<MaterialGPU_PBR> materialsPBR;
        DeviceBuffer<EmissiveTriangleGPU> emissiveTriangles;
        DeviceBuffer<DecalGPU> decals;
        DeviceBuffer<AirDustVolumeGPU> airDustVolumes;
        DeviceBuffer<HIPTextureDescGPU> textureDescs;
        DeviceBuffer<std::uint8_t> textureTexels;
        DeviceBuffer<Vec3> outputPixels;

        std::vector<float3> currentFrameHdr;
        std::vector<float3> accumulatedHdr;
        std::vector<float3> denoisedHdr;
        std::vector<float3> bloomA;
        std::vector<float3> bloomB;
        int accumWidth = 0;
        int accumHeight = 0;
    };

    HIPRuntimeState g_runtime;

    template <typename T>
    void ReleaseDeviceBuffer(DeviceBuffer<T> &buffer)
    {
        if (buffer.ptr != nullptr)
        {
            (void)hipFree(buffer.ptr);
            buffer.ptr = nullptr;
            buffer.capacity = 0u;
        }
    }

    template <typename T>
    bool EnsureDeviceCapacity(DeviceBuffer<T> &buffer,
                              std::size_t requiredCount,
                              const char *label)
    {
        if (requiredCount == 0u)
            return true;
        if (buffer.capacity >= requiredCount && buffer.ptr != nullptr)
            return true;

        ReleaseDeviceBuffer(buffer);
        hipError_t err = hipMalloc(reinterpret_cast<void **>(&buffer.ptr), requiredCount * sizeof(T));
        if (err != hipSuccess)
        {
            std::fprintf(stderr, "HIP: hipMalloc failed for %s (%s)\n", label, hipGetErrorString(err));
            return false;
        }

        buffer.capacity = requiredCount;
        return true;
    }

    template <typename T>
    bool UploadDeviceArray(DeviceBuffer<T> &buffer,
                           const T *src,
                           std::size_t count,
                           const char *label)
    {
        if (count == 0u || src == nullptr)
            return true;
        if (!EnsureDeviceCapacity(buffer, count, label))
            return false;

        hipError_t err = hipMemcpy(buffer.ptr, src, count * sizeof(T), hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            std::fprintf(stderr, "HIP: hipMemcpy H2D failed for %s (%s)\n", label, hipGetErrorString(err));
            return false;
        }

        return true;
    }

    static inline Vec3 ToVec3(float3 v)
    {
        return Vec3{v.x, v.y, v.z};
    }

    static inline float Clamp01Host(float v)
    {
        return std::clamp(v, 0.0f, 1.0f);
    }

    static inline float3 Clamp01Host(float3 v)
    {
        return float3(Clamp01Host(v.x), Clamp01Host(v.y), Clamp01Host(v.z));
    }

    static inline float LuminanceHost(float3 c)
    {
        return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f;
    }

    static inline float3 Lerp3Host(float3 a, float3 b, float t)
    {
        return a + (b - a) * t;
    }

    static inline float Noise2DHost(int x, int y, float time)
    {
        const float n = static_cast<float>(x) * 12.9898f + static_cast<float>(y) * 78.233f + time * 37.719f;
        return std::fmod(std::sin(n) * 43758.5453123f, 1.0f) + 0.5f;
    }

    static float3 SampleImageClampHost(const std::vector<float3> &image,
                                       int width,
                                       int height,
                                       float2 uv)
    {
        if (image.empty() || width <= 0 || height <= 0)
            return float3(0.0f);

        uv = clamp(uv, float2(0.0f), float2(1.0f));
        const float fx = uv.x * static_cast<float>(width) - 0.5f;
        const float fy = uv.y * static_cast<float>(height) - 0.5f;
        const int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, width - 1);
        const int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, height - 1);
        const int x1 = std::clamp(x0 + 1, 0, width - 1);
        const int y1 = std::clamp(y0 + 1, 0, height - 1);
        const float tx = fx - static_cast<float>(x0);
        const float ty = fy - static_cast<float>(y0);

        const float3 c00 = image[static_cast<std::size_t>(y0) * width + x0];
        const float3 c10 = image[static_cast<std::size_t>(y0) * width + x1];
        const float3 c01 = image[static_cast<std::size_t>(y1) * width + x0];
        const float3 c11 = image[static_cast<std::size_t>(y1) * width + x1];
        const float3 cx0 = Lerp3Host(c00, c10, tx);
        const float3 cx1 = Lerp3Host(c01, c11, tx);
        return Lerp3Host(cx0, cx1, ty);
    }

    static void DenoiseImageHost(const std::vector<float3> &src,
                                 int width,
                                 int height,
                                 std::vector<float3> &dst)
    {
        dst = src;
        if (!DENOISE_ENABLE || src.empty() || width <= 0 || height <= 0)
            return;

        const float denomSpace = 2.0f * DENOISE_SIGMA_SPACE * DENOISE_SIGMA_SPACE;
        const float wS1 = (denomSpace > 0.0f) ? std::exp(-1.0f / denomSpace) : 1.0f;
        const float wS2 = (denomSpace > 0.0f) ? std::exp(-2.0f / denomSpace) : 1.0f;
        const float denomColor = 2.0f * DENOISE_SIGMA_COLOR * DENOISE_SIGMA_COLOR;
        const bool useColorExp = denomColor > 0.0f;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const std::size_t idx = static_cast<std::size_t>(y) * width + x;
                const float3 center = src[idx];
                float3 sum = float3(0.0f);
                float wsum = 0.0f;

                for (int dy = -1; dy <= 1; ++dy)
                {
                    const int ny = y + dy;
                    if (ny < 0 || ny >= height)
                        continue;

                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        const int nx = x + dx;
                        if (nx < 0 || nx >= width)
                            continue;

                        const float3 c = src[static_cast<std::size_t>(ny) * width + nx];
                        const int dist2 = dx * dx + dy * dy;
                        const float wSpatial = (dist2 == 0) ? 1.0f : ((dist2 == 1) ? wS1 : wS2);
                        const float3 diff = c - center;
                        const float colorDist2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                        const float wColor = useColorExp ? std::exp(-colorDist2 / denomColor) : 1.0f;
                        const float weight = wSpatial * wColor;
                        sum += c * weight;
                        wsum += weight;
                    }
                }

                if (wsum > 0.0f)
                {
                    const float3 blurred = sum / wsum;
                    dst[idx] = Lerp3Host(blurred, center, DENOISE_BLEND_FACTOR);
                }
            }
        }
    }

    static void ExtractBloomHost(const std::vector<float3> &src,
                                 const HIPPostProcessParams &pp,
                                 std::vector<float3> &dst)
    {
        dst.resize(src.size(), float3(0.0f));
        if (src.empty())
            return;

        const float threshold = std::max(0.18f, 0.78f + 0.42f * pp.bloomThreshold);
        const float knee = std::max(0.12f, threshold * 0.40f);
        for (std::size_t i = 0; i < src.size(); ++i)
        {
            const float3 hdr = src[i];
            const float lum = LuminanceHost(hdr);
            float soft = std::clamp((lum - threshold + knee) / (2.0f * knee), 0.0f, 1.0f);
            soft = soft * soft * (3.0f - 2.0f * soft);
            dst[i] = hdr * soft;
        }
    }

    static void BlurImageHost(const std::vector<float3> &src,
                              int width,
                              int height,
                              bool horizontal,
                              std::vector<float3> &dst)
    {
        dst.resize(src.size(), float3(0.0f));
        if (src.empty())
            return;

        const float weights[5] = {0.227027f, 0.1945946f, 0.1216216f, 0.054054f, 0.016216f};
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float3 c = src[static_cast<std::size_t>(y) * width + x] * weights[0];
                for (int tap = 1; tap < 5; ++tap)
                {
                    const int xPos = horizontal ? std::clamp(x + tap, 0, width - 1) : x;
                    const int xNeg = horizontal ? std::clamp(x - tap, 0, width - 1) : x;
                    const int yPos = horizontal ? y : std::clamp(y + tap, 0, height - 1);
                    const int yNeg = horizontal ? y : std::clamp(y - tap, 0, height - 1);
                    c += src[static_cast<std::size_t>(yPos) * width + xPos] * weights[tap];
                    c += src[static_cast<std::size_t>(yNeg) * width + xNeg] * weights[tap];
                }

                dst[static_cast<std::size_t>(y) * width + x] = c;
            }
        }
    }

    static float3 ApplyColorSaturationHost(float3 c, float3 sat)
    {
        const float l = LuminanceHost(c);
        return max(float3(0.0f), float3(l) + (c - float3(l)) * sat);
    }

    static float3 FilmicTonemapCustomHost(float3 c, const HIPPostProcessParams &pp)
    {
        c *= std::max(pp.exposure, 0.001f);

        const float l0 = LuminanceHost(max(c, float3(0.0f)));
        const float shadowMask = 1.0f - smoothstep(0.02f, 0.40f, l0);
        c += float3(pp.shadowLift * shadowMask);

        c = max(c - pp.filmBlackClip, float3(0.0f));
        c *= std::max(pp.filmSlope, 0.05f);

        const float a = 2.35f + pp.filmShoulder * 0.22f;
        const float b = 0.028f + pp.filmBlackClip * 0.05f;
        const float c1 = 2.43f;
        const float d = 0.64f + pp.filmToe * 0.12f;
        const float e = std::max(0.025f, 0.16f - pp.filmWhiteClip * 0.42f);

        float3 x = (c * (a * c + b)) / (c * (c1 * c + d) + e);
        x = clamp(x, 0.0f, 1.0f);
        x = ApplyColorSaturationHost(x, float3(pp.colorSaturationX, pp.colorSaturationY, pp.colorSaturationZ));
        return pow(x, float3(1.0f / 2.18f));
    }

    static void PostProcessHost(const std::vector<float3> &hdrInput,
                                int width,
                                int height,
                                const HIPPostProcessParams &pp,
                                Vec3 *framebuffer)
    {
        if (framebuffer == nullptr || hdrInput.empty())
            return;

        DenoiseImageHost(hdrInput, width, height, g_runtime.denoisedHdr);
        ExtractBloomHost(g_runtime.denoisedHdr, pp, g_runtime.bloomA);
        BlurImageHost(g_runtime.bloomA, width, height, true, g_runtime.bloomB);
        BlurImageHost(g_runtime.bloomB, width, height, false, g_runtime.bloomA);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const std::size_t idx = static_cast<std::size_t>(y) * width + x;
                const float2 uv = float2((static_cast<float>(x) + 0.5f) / std::max(width, 1),
                                         (static_cast<float>(y) + 0.5f) / std::max(height, 1));
                const float2 centered = uv - 0.5f;
                const float2 caOffset = centered * (pp.chromaticAberration * 0.0025f);

                const float3 cR = SampleImageClampHost(g_runtime.denoisedHdr, width, height, uv + caOffset);
                const float3 cG = SampleImageClampHost(g_runtime.denoisedHdr, width, height, uv);
                const float3 cB = SampleImageClampHost(g_runtime.denoisedHdr, width, height, uv - caOffset);
                float3 color = float3(cR.x, cG.y, cB.z);
                color += SampleImageClampHost(g_runtime.bloomA, width, height, uv) * std::max(pp.bloomIntensity, 0.0f);

                float3 outColor = FilmicTonemapCustomHost(color, pp);
                const float dist = length(centered) * 1.41421356f;
                const float vignette = 1.0f - std::clamp(pp.vignetteIntensity, 0.0f, 1.0f) * smoothstep(0.2f, 1.0f, dist);
                outColor *= std::clamp(vignette, 0.0f, 1.0f);

                if (pp.filmGrainIntensity > 0.0f)
                {
                    const float grain = (Noise2DHost(x, y, pp.time) - 0.5f) * pp.filmGrainIntensity * 0.06f;
                    outColor = clamp(outColor + grain, 0.0f, 1.0f);
                }

                framebuffer[idx] = ToVec3(Clamp01Host(outColor));
            }
        }
    }

    static void ResetHostAccumulation()
    {
        g_runtime.currentFrameHdr.clear();
        g_runtime.accumulatedHdr.clear();
        g_runtime.denoisedHdr.clear();
        g_runtime.bloomA.clear();
        g_runtime.bloomB.clear();
        g_runtime.accumWidth = 0;
        g_runtime.accumHeight = 0;
    }
}

extern "C" bool HIP_RenderFrameTexture_C(const BVHNode *tlasNodes,
                                         std::uint32_t tlasNodeCount,
                                         const BVHNode *meshNodes,
                                         std::uint32_t meshNodeCount,
                                         const Triangle *triangles,
                                         std::uint32_t triCount,
                                         const HIPSceneInstanceGPU *instances,
                                         std::uint32_t instanceCount,
                                         const HIPLightGPU *lights,
                                         std::uint32_t lightCount,
                                         int rootIndex,
                                         const CameraDataCPU *camera,
                                         const HIPMaterialGPU *materials,
                                         std::uint32_t materialCount,
                                         const HIPMaterialPBRGPU *materialsPBR,
                                         std::uint32_t materialPBRCount,
                                         const HIPEmissiveTriangleGPU *emissiveTriangles,
                                         std::uint32_t emissiveTriangleCount,
                                         const HIPDecalGPU *decals,
                                         std::uint32_t decalCount,
                                         const HIPAirDustVolumeGPU *airDustVolumes,
                                         std::uint32_t airDustVolumeCount,
                                         const HIPPostProcessParams *postParams,
                                         const HIPTextureDescGPU *textureDescs,
                                         std::uint32_t textureCount,
                                         const std::uint8_t *textureTexels,
                                         std::size_t textureTexelBytes,
                                         std::uint64_t sceneRevision,
                                         std::uint64_t metaGeneration,
                                         std::uint32_t accumulatedSampleCountBefore,
                                         Vec3 *framebuffer)
{
    (void)sceneRevision;
    (void)metaGeneration;

    if (camera == nullptr || framebuffer == nullptr)
    {
        std::fprintf(stderr, "HIP_RenderFrameTexture_C: camera/framebuffer is null\n");
        return false;
    }
    if (camera->width <= 0 || camera->height <= 0)
    {
        std::fprintf(stderr, "HIP_RenderFrameTexture_C: invalid frame size\n");
        return false;
    }
    if (tlasNodes == nullptr || meshNodes == nullptr || triangles == nullptr || instances == nullptr)
    {
        std::fprintf(stderr, "HIP_RenderFrameTexture_C: scene geometry buffers are null\n");
        return false;
    }
    if (rootIndex < 0 || rootIndex >= static_cast<int>(tlasNodeCount))
    {
        std::fprintf(stderr, "HIP_RenderFrameTexture_C: invalid TLAS root index\n");
        return false;
    }

    const std::size_t pixelCount =
        static_cast<std::size_t>(camera->width) * static_cast<std::size_t>(camera->height);
    if (!EnsureDeviceCapacity(g_runtime.outputPixels, pixelCount, "outputPixels"))
        return false;

    if (!UploadDeviceArray(g_runtime.tlasNodes, tlasNodes, tlasNodeCount, "tlasNodes") ||
        !UploadDeviceArray(g_runtime.meshNodes, meshNodes, meshNodeCount, "meshNodes") ||
        !UploadDeviceArray(g_runtime.triangles, triangles, triCount, "triangles") ||
        !UploadDeviceArray(g_runtime.instances,
                           reinterpret_cast<const SceneInstanceGPU *>(instances),
                           instanceCount,
                           "instances") ||
        !UploadDeviceArray(g_runtime.lights,
                           reinterpret_cast<const LightGPU *>(lights),
                           lightCount,
                           "lights") ||
        !UploadDeviceArray(g_runtime.materials,
                           reinterpret_cast<const MaterialGPU *>(materials),
                           materialCount,
                           "materials") ||
        !UploadDeviceArray(g_runtime.materialsPBR,
                           reinterpret_cast<const MaterialGPU_PBR *>(materialsPBR),
                           materialPBRCount,
                           "materialsPBR") ||
        !UploadDeviceArray(g_runtime.emissiveTriangles,
                           reinterpret_cast<const EmissiveTriangleGPU *>(emissiveTriangles),
                           emissiveTriangleCount,
                           "emissiveTriangles") ||
        !UploadDeviceArray(g_runtime.decals,
                           reinterpret_cast<const DecalGPU *>(decals),
                           decalCount,
                           "decals") ||
        !UploadDeviceArray(g_runtime.airDustVolumes,
                           reinterpret_cast<const AirDustVolumeGPU *>(airDustVolumes),
                           airDustVolumeCount,
                           "airDustVolumes") ||
        !UploadDeviceArray(g_runtime.textureDescs, textureDescs, textureCount, "textureDescs") ||
        !UploadDeviceArray(g_runtime.textureTexels, textureTexels, textureTexelBytes, "textureTexels"))
    {
        return false;
    }

    const CameraData cameraGpu = *camera;
    PrimaryVolumeParams volumeParams{};
    if (postParams != nullptr)
        std::memcpy(&volumeParams, postParams, sizeof(PrimaryVolumeParams));
    else
    {
        volumeParams.width = static_cast<float>(camera->width);
        volumeParams.height = static_cast<float>(camera->height);
        volumeParams.nearPlane = camera->nearPlane;
        volumeParams.farPlane = camera->farPlane;
        volumeParams.worldUnitToMeters = 1.0f;
    }

    const SceneTextureArray sceneTextures{
        g_runtime.textureDescs.ptr,
        g_runtime.textureTexels.ptr,
        textureCount
    };

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 grid((static_cast<unsigned int>(camera->width) + BLOCK_SIZE - 1u) / BLOCK_SIZE,
                    (static_cast<unsigned int>(camera->height) + BLOCK_SIZE - 1u) / BLOCK_SIZE,
                    1u);

    hipLaunchKernelGGL(RayTraceTextureKernel,
                       grid,
                       block,
                       0,
                       0,
                       g_runtime.tlasNodes.ptr,
                       tlasNodeCount,
                       g_runtime.meshNodes.ptr,
                       meshNodeCount,
                       g_runtime.triangles.ptr,
                       g_runtime.instances.ptr,
                       instanceCount,
                       rootIndex,
                       cameraGpu,
                       g_runtime.lights.ptr,
                       lightCount,
                       accumulatedSampleCountBefore,
                       g_runtime.materials.ptr,
                       materialCount,
                       g_runtime.materialsPBR.ptr,
                       materialPBRCount,
                       g_runtime.emissiveTriangles.ptr,
                       emissiveTriangleCount,
                       g_runtime.decals.ptr,
                       decalCount,
                       volumeParams,
                       g_runtime.airDustVolumes.ptr,
                       airDustVolumeCount,
                       sceneTextures,
                       g_runtime.outputPixels.ptr);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        std::fprintf(stderr, "HIP: kernel launch failed (%s)\n", hipGetErrorString(err));
        return false;
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess)
    {
        std::fprintf(stderr, "HIP: kernel execution failed (%s)\n", hipGetErrorString(err));
        return false;
    }

    g_runtime.currentFrameHdr.resize(pixelCount);
    err = hipMemcpy(g_runtime.currentFrameHdr.data(),
                    g_runtime.outputPixels.ptr,
                    pixelCount * sizeof(Vec3),
                    hipMemcpyDeviceToHost);
    if (err != hipSuccess)
    {
        std::fprintf(stderr, "HIP: hipMemcpy D2H failed for outputPixels (%s)\n", hipGetErrorString(err));
        return false;
    }

    if (g_runtime.accumWidth != camera->width ||
        g_runtime.accumHeight != camera->height ||
        accumulatedSampleCountBefore == 0u ||
        g_runtime.accumulatedHdr.size() != pixelCount)
    {
        g_runtime.accumWidth = camera->width;
        g_runtime.accumHeight = camera->height;
        g_runtime.accumulatedHdr = g_runtime.currentFrameHdr;
    }
    else
    {
        const float prevCount = static_cast<float>(accumulatedSampleCountBefore);
        const float dispatchCount = static_cast<float>(std::max(camera->samplesPerPixel, 1));
        const float newCount = prevCount + dispatchCount;
        for (std::size_t i = 0; i < pixelCount; ++i)
        {
            g_runtime.accumulatedHdr[i] =
                (g_runtime.accumulatedHdr[i] * prevCount + g_runtime.currentFrameHdr[i] * dispatchCount) /
                std::max(newCount, 1.0f);
        }
    }

    const HIPPostProcessParams fallbackPostParams = (postParams != nullptr) ? *postParams : HIPPostProcessParams{};
    PostProcessHost(g_runtime.accumulatedHdr,
                    camera->width,
                    camera->height,
                    fallbackPostParams,
                    framebuffer);
    return true;
}

extern "C" void HIP_ResetAccumulation_C()
{
    ResetHostAccumulation();
}
#endif
