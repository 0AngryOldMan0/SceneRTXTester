#include <metal_stdlib>
using namespace metal;

constant float PI               = 3.14159265358979323846f;
constant float INV_PI           = 1.0f / PI;
constant float INV_4PI          = 1.0f / (4.0f * PI);
constant float AMBIENT_STRENGTH = 0.11f;
constant float SHADOW_EPS       = 1e-3f;
constant float IMAGE_EXPOSURE   = 0.6f;

constant float LIGHT_EXPOSURE_GPU             = 265.0f; // прямые UE-источники: заметно усиливаем, чтобы расширить освещённые зоны
constant float EMISSION_VISIBLE_EXPOSURE_GPU   = 0.065f;  // видимая яркость emissive-поверхностей: сильнее приглушаем, чтобы текстура стекла читалась
constant float EMISSION_LIGHT_EXPOSURE_GPU     = 1.12f;  // вклад emissive-поверхностей как источников света держим умеренным
constant float ENV_INTENSITY_GPU               = 0.9f;   // яркость environment
constant float LIGHT_FALLOFF_DISTANCE_SCALE    = 0.30f;  // < 1 => свет спадает мягче и покрывает заметно большую площадь
constant float LIGHT_ATTENUATION_RADIUS_SCALE  = 2.20f;  // сильнее расширяем effective attenuation radius UE

// --- параметры денойзера ---
constant bool  DENOISE_ENABLE        = true;
constant float DENOISE_SIGMA_SPACE   = 1.0f;   // радиус влияния по экрану (пиксели)
constant float DENOISE_SIGMA_COLOR   = 0.20f;  // чувствительность к разнице цветов
constant float DENOISE_BLEND_FACTOR  = 0.40f;  // 0 = полностью размыть, 1 = без фильтра

constant int UV_DEBUG_MODE = 0;

/*
0 — обычный рендер
1 — UV checker без света (чистая проверка UV)
2 — UV checker как albedo со светом
3 — UV gradient (u,v)
4 — materialIndex “хеш-цвет” (проверка, что материалы назначаются и materialIndex меняется)
5 — итоговый baseColor после семплинга (factor * texture) без света
6 — только baseColor texture (игнорирует per-tri factor)
7 — AO (grayscale)
8 — roughness (grayscale)
9 — metallic (grayscale)
10 — geometric normal (RGB)
11 — shading normal (RGB) после normal map
12 — |Ns-Ng| (сила normal map; чем светлее — тем сильнее отклонение)
13 — emissive (texture/factor, до умножения на EMISSION_VISIBLE_EXPOSURE_GPU)
14 — ORM decoded как RGB = (AO, Rough, Metal)
15 — presence bits RGB = (hasPBR, hasBaseColorTex, hasNormalTex)
16 — presence bits RGB = (hasORMTex, hasRoughnessTex, hasMetallicTex)
*/

// Normal map conventions (toggle if green channel looks inverted)
constant bool NORMALMAP_FLIP_Y = false;


constexpr sampler g_texSampler(address::repeat,
                               filter::linear,
                               mip_filter::none);


constexpr sampler g_postSampler(address::clamp_to_edge,
                                filter::linear,
                                mip_filter::none);



// --- Accumulation stamp/count packing into accumTex.w (RGBA32Float safe) ---
// packed = [stamp:8bits | sampleCount:24bits]
inline uint packStampCount(uint stamp, uint count)
{
    return ((stamp & 0xFFu) << 24) | (count & 0x00FFFFFFu);
}
inline void unpackStampCount(uint packed, thread uint &stamp, thread uint &count)
{
    stamp = (packed >> 24) & 0xFFu;
    count = packed & 0x00FFFFFFu;
}

// ======================
// Структуры (C++ совместимы)
// ======================

struct AABB
{
    packed_float3 v0; // min
    packed_float3 v1; // max
};

struct Triangle
{
    packed_float3 v0;
    packed_float3 v1;
    packed_float3 v2;

    packed_float2 uv0;
    packed_float2 uv1;
    packed_float2 uv2;

    packed_float3 normal;
    AABB          ABoBa;

    packed_float3 color;
    packed_float3 emission;
    float metallic;
    float roughness;

    int   materialIndex;
    int   _padMat;
};

struct BVHNode
{
    AABB box;   // 24 байта
    int  left;  // 4
    int  right; // 4
    int  parent;// 4
    int  tri;   // 4  -> всего 40 байт
};

struct SceneInstanceGPU
{
    float objectToWorld[12];
    float worldToObject[12];
    float normalToWorld[12];
    AABB  worldBounds;
    int   blasRootIndex;
    int   _pad0;
    int   _pad1;
    int   _pad2;
};

struct CameraData
{
    packed_float3 position;        //  0
    packed_float3 forward;         // 12
    packed_float3 up;              // 24
    packed_float3 right;           // 36

    float fovY;                    // 48
    int   width;                   // 52
    int   height;                  // 56
    int   samplesPerPixel;         // 60

    float nearPlane;               // 64
    float farPlane;                // 68
    float focusDistance;           // 72
    float _pad;                    // 76 -> 80 байт
};

struct Ray
{
    float3 origin;
    float3 direction;
};

struct HitInfo
{
    bool   hit;
    int    triIndex;
    int    instanceIndex;
    int    materialIndex;
    float  t;
    float3 position;
    float3 normal;
    float3 color;
    float3 emission;

    float  metallic;
    float  roughness;

    float2 uv;
};

// Типы источников света (должны соответствовать C++ LightType)
constant int LIGHT_TYPE_POINT       = 0;
constant int LIGHT_TYPE_DIRECTIONAL = 1;
constant int LIGHT_TYPE_SPOT        = 2;
constant int LIGHT_TYPE_AREA        = 3;

// GPU-структура света. Совместима по байтам с C++ LightGPU.
struct LightGPU
{
    int   type;
    int   _pad0;          // выравнивание до 8 байт

    packed_float3 position;
    packed_float3 direction;
    packed_float3 color;

    float intensity;
    float radius;             // геометрический размер источника (для мягких теней)
    float spotSize;           // в радианах (для Spot)
    float spotBlend;          // 0..1

    // Радиус затухания (UE AttenuationRadius, в тех же единицах, что и позиция)
    float attenuationRadius;
};

constant uint MAX_BASE_TEX = 124; // unified scene texture pool (sRGB + linear), bound from texture(4)

struct MaterialGPU
{
    int baseColorTexIndex;  // индекс в массиве baseColorTextures, -1 если нет
    int emissionTexIndex;   // на будущее
    int _pad0;
    int _pad1;
};

// Expanded PBR material layout (matches CPU MaterialGPU_PBR, buffer(14)).
struct MaterialGPU_PBR
{
    int baseColorTexIndex;   // sRGB baseColor in unified pool, -1 if none
    int emissionTexIndex;    // sRGB emission in unified pool, -1 if none
    int normalTexIndex;      // linear normal map in unified pool, -1 if none
    int ormTexIndex;         // linear ORM map in unified pool, -1 if none (AO/Rough/Metal)
    int roughnessTexIndex;   // linear roughness map in unified pool, -1 if none
    int metallicTexIndex;    // linear metallic map in unified pool, -1 if none
    int occlusionTexIndex;   // linear occlusion map in unified pool, -1 if none
    int flags;               // bit0 = use alpha/derived mask for emissive modulation
};

constant int MATERIAL_FLAG_EMISSION_USE_ALPHA_MASK = 1;

struct DecalGPU
{
    float posX,   posY,   posZ,   sizeX;   // half depth along projection axis (UE local X)
    float axisXx, axisXy, axisXz, sizeY;   // half width  along UE local Y
    float axisYx, axisYy, axisYz, sizeZ;   // half height along UE local Z
    float axisZx, axisZy, axisZz, opacity;

    int baseColorTexIndex;
    int ormTexIndex;
    int roughnessTexIndex;
    int normalTexIndex;

    int opacityTexIndex;
    int detailTexIndex;
    int flags;
    int _pad0;

    float baseColorX, baseColorY, baseColorZ, roughnessBias;
    float tilingU, tilingV, opacityPower, normalIntensity;
};

struct EmissiveTriangleGPU
{
    uint  triIndex;
    uint  instanceIndex;
    float area;
    float selectionPdf;
    float cdf;
};

struct EmissiveLightSample
{
    uint   valid;
    uint   triIndex;
    uint   instanceIndex;
    float  area;
    float  selectionPdf;
    float3 position;
    float3 normal;
    float2 uv;
};

struct BSDFSample
{
    uint   valid;
    float3 direction;
    float3 weight;
    float  pdf;
};

struct PathTraceTextureResult
{
    float3 color;
    float  depth;
    float3 albedo;
    float  hitMask;
    float3 normal;
    float  _pad0;
};

// ======================
// Простейший псевдо-рандом
// ======================

inline float rand01(uint seed)
{
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return (float(seed) / 4294967295.0f);
}

// ======================
// Вспомогательные функции
// ======================

inline float clamp01(float x)
{
    return clamp(x, 0.0f, 1.0f);
}

inline float3 filmicTonemap(float3 c, float exposure)
{
    c *= exposure;

    const float a  = 2.51f;
    const float b  = 0.03f;
    const float c1 = 2.43f;
    const float d  = 0.59f;
    const float e  = 0.14f;

    float3 x = (c * (a * c + b)) / (c * (c1 * c + d) + e);
    x = clamp(x, 0.0f, 1.0f);

    x = pow(x, float3(1.0f / 2.2f));
    return x;
}

inline float3 cosineSampleHemisphere(float u1, float u2)
{
    float r   = sqrt(u1);
    float phi = 2.0f * PI * u2;

    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0f, 1.0f - u1));

    return float3(x, y, z);
}

inline void buildOrthonormalBasis(float3 n,
                                  thread float3 &tangent,
                                  thread float3 &bitangent)
{
    float3 N = normalize(n);

    float3 helper = (fabs(N.z) < 0.999f)
                    ? float3(0.0f, 0.0f, 1.0f)
                    : float3(1.0f, 0.0f, 0.0f);

    tangent   = normalize(cross(helper, N));
    bitangent = cross(N, tangent);
}

inline float3 transformPoint3x4(const float m[12], float3 p)
{
    return float3(m[0] * p.x + m[1] * p.y + m[2]  * p.z + m[3],
                  m[4] * p.x + m[5] * p.y + m[6]  * p.z + m[7],
                  m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
}

inline float3 transformDirection3x4(const float m[12], float3 v)
{
    return float3(m[0] * v.x + m[1] * v.y + m[2]  * v.z,
                  m[4] * v.x + m[5] * v.y + m[6]  * v.z,
                  m[8] * v.x + m[9] * v.y + m[10] * v.z);
}

inline float3 transformPoint3x4(const device float *m, float3 p)
{
    return float3(m[0] * p.x + m[1] * p.y + m[2]  * p.z + m[3],
                  m[4] * p.x + m[5] * p.y + m[6]  * p.z + m[7],
                  m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
}

inline float3 transformDirection3x4(const device float *m, float3 v)
{
    return float3(m[0] * v.x + m[1] * v.y + m[2]  * v.z,
                  m[4] * v.x + m[5] * v.y + m[6]  * v.z,
                  m[8] * v.x + m[9] * v.y + m[10] * v.z);
}

inline float3 transformPointObjectToWorld(const device SceneInstanceGPU &inst, float3 p)
{
    return transformPoint3x4(inst.objectToWorld, p);
}

inline float3 transformPointWorldToObject(const device SceneInstanceGPU &inst, float3 p)
{
    return transformPoint3x4(inst.worldToObject, p);
}

inline float3 transformDirectionObjectToWorld(const device SceneInstanceGPU &inst, float3 v)
{
    return transformDirection3x4(inst.objectToWorld, v);
}

inline float3 transformDirectionWorldToObject(const device SceneInstanceGPU &inst, float3 v)
{
    return transformDirection3x4(inst.worldToObject, v);
}

inline float3 transformNormalObjectToWorld(const device SceneInstanceGPU &inst, float3 n)
{
    return transformDirection3x4(inst.normalToWorld, n);
}

// ======================
// Tangent basis + PBR texture sampling helpers
// ======================

inline float3 safeNormalize(float3 v)
{
    float len2 = dot(v, v);
    if (len2 > 1e-20f) return v * rsqrt(len2);
    return float3(0.0f, 0.0f, 1.0f);
}

inline void computeTangentBasis(const device Triangle* triangles,
                                int                   triIndex,
                                float3                N,
                                thread float3&        T,
                                thread float3&        B)
{
    const device Triangle& tri = triangles[triIndex];

    float3 p0  = float3(tri.v0);
    float3 p1  = float3(tri.v1);
    float3 p2  = float3(tri.v2);

    float2 uv0 = float2(tri.uv0);
    float2 uv1 = float2(tri.uv1);
    float2 uv2 = float2(tri.uv2);

    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;

    float2 dUV1 = uv1 - uv0;
    float2 dUV2 = uv2 - uv0;

    float det = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
    if (fabs(det) < 1e-8f)
    {
        buildOrthonormalBasis(N, T, B);
        return;
    }

    float invDet = 1.0f / det;

    float3 t = (e1 * dUV2.y - e2 * dUV1.y) * invDet;
    t = t - N * dot(N, t);
    T = safeNormalize(t);

    // handedness from UV orientation
    float handedness = (det < 0.0f) ? -1.0f : 1.0f;
    B = safeNormalize(cross(N, T) * handedness);
}

inline void computeTangentBasisInstanced(const device Triangle* triangles,
                                         const device SceneInstanceGPU* instances,
                                         int instanceIndex,
                                         int triIndex,
                                         float3 N,
                                         thread float3& T,
                                         thread float3& B)
{
    float3 localN = safeNormalize(float3(triangles[triIndex].normal));
    computeTangentBasis(triangles, triIndex, localN, T, B);
    if (instances != nullptr && instanceIndex >= 0)
    {
        const device SceneInstanceGPU &inst = instances[instanceIndex];
        T = safeNormalize(transformDirectionObjectToWorld(inst, T));
        T = safeNormalize(T - N * dot(N, T));
        float handedness = (dot(cross(N, T), B) < 0.0f) ? -1.0f : 1.0f;
        B = safeNormalize(cross(N, T) * handedness);
    }
}

inline float3 sampleNormalMapInstanced(int normalTexIndex,
                              float2 uv,
                              float3 Ng,
                              const device Triangle* triangles,
                              const device SceneInstanceGPU* instances,
                              int instanceIndex,
                              int triIndex,
                              const array<texture2d<float, access::sample>, MAX_BASE_TEX> sceneTextures)
{
    if (normalTexIndex < 0 || uint(normalTexIndex) >= MAX_BASE_TEX)
        return Ng;

    float2 uvW = fract(uv);

    float3 nTex = sceneTextures[uint(normalTexIndex)].sample(g_texSampler, uvW).xyz * 2.0f - 1.0f;
    if (NORMALMAP_FLIP_Y) nTex.y = -nTex.y;
    nTex = safeNormalize(nTex);

    float3 T, B;
    computeTangentBasisInstanced(triangles, instances, instanceIndex, triIndex, Ng, T, B);

    float3 Ns = safeNormalize(nTex.x * T + nTex.y * B + nTex.z * Ng);
    return Ns;
}

inline float3 sampleRGB(int texIndex,
                        float2 uv,
                        const array<texture2d<float, access::sample>, MAX_BASE_TEX> sceneTextures,
                        float3 fallback)
{
    if (texIndex < 0 || uint(texIndex) >= MAX_BASE_TEX)
        return fallback;
    return sceneTextures[uint(texIndex)].sample(g_texSampler, fract(uv)).rgb;
}

inline float sampleR(int texIndex,
                     float2 uv,
                     const array<texture2d<float, access::sample>, MAX_BASE_TEX> sceneTextures,
                     float fallback)
{
    if (texIndex < 0 || uint(texIndex) >= MAX_BASE_TEX)
        return fallback;
    return sceneTextures[uint(texIndex)].sample(g_texSampler, fract(uv)).r;
}

inline float sampleEmissiveMask(float4 texel)
{
    const float alphaMask = saturate(texel.a);
    const float lumaMask  = saturate(dot(texel.rgb, float3(0.2126f, 0.7152f, 0.0722f)));

    if (alphaMask > 1.0e-4f)
        return smoothstep(0.22f, 0.82f, alphaMask);

    return smoothstep(0.16f, 0.42f, lumaMask);
}

inline float3 lerp3(float3 a, float3 b, float t)
{
    return a + t * (b - a);
}

inline float3 checkerFromUV(float2 uv, float scale)
{
    // wrap (повторение)
    uv = fract(uv);

    int x = int(floor(uv.x * scale));
    int y = int(floor(uv.y * scale));

    // XOR шахматка
    int c = (x ^ y) & 1;

    float v = (c == 0) ? 0.15f : 0.85f;
    return float3(v, v, v);
}

inline float3 uvGradient(float2 uv)
{
    uv = fract(uv);
    return float3(uv.x, uv.y, 0.0f);
}

inline float3 uvChecker(float2 uv, float scale)
{
    uv = fract(uv);
    int x = int(floor(uv.x * scale));
    int y = int(floor(uv.y * scale));
    int c = (x ^ y) & 1;
    float v = (c == 0) ? 0.15f : 0.85f;
    return float3(v, v, v);
}


inline float3 normalToRGB(float3 n)
{
    n = normalize(n);
    return clamp(n * 0.5f + 0.5f, 0.0f, 1.0f);
}

inline float3 hashColorFromInt(int v)
{
    // simple integer hash -> RGB in 0..1
    uint x = uint(v);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;

    float r = float((x >>  0) & 0xFFu) / 255.0f;
    float g = float((x >>  8) & 0xFFu) / 255.0f;
    float b = float((x >> 16) & 0xFFu) / 255.0f;
    // avoid too-dark colors
    return clamp(float3(r, g, b) * 0.85f + 0.15f, 0.0f, 1.0f);
}

inline float3 debugGamma(float3 c)
{
    c = clamp(c, 0.0f, 1.0f);
    return pow(c, float3(1.0f / 2.2f));
}


inline float3 environmentColor(float3 dir)
{
    float t = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);

    float3 ground = float3(0.05f, 0.05f, 0.06f);
    float3 skyBot = float3(0.08f, 0.08f, 0.10f);
    float3 skyTop = float3(0.14f, 0.13f, 0.17f);

    float3 sky  = lerp3(skyBot, skyTop, t);
    float3 base = (dir.y >= 0.0f)
                ? sky
                : lerp3(ground, sky, 0.5f);

    return base * ENV_INTENSITY_GPU;
}

// ======================
// PBR-вспомогательные функции (GGX + Schlick)
// ======================

inline float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (float3(1.0f) - F0) * pow(1.0f - cosTheta, 5.0f);
}

inline float distributionGGX(float NdH, float alpha)
{
    float a2    = alpha * alpha;
    float denom = NdH * NdH * (a2 - 1.0f) + 1.0f;
    denom       = PI * denom * denom;
    return a2 / max(denom, 1e-7f);
}

inline float geometrySchlickGGX(float NdV, float k)
{
    return NdV / (NdV * (1.0f - k) + k);
}

inline float geometrySmith(float NdV, float NdL, float k)
{
    float g1 = geometrySchlickGGX(NdV, k);
    float g2 = geometrySchlickGGX(NdL, k);
    return g1 * g2;
}

// Гауссов вес по квадрату расстояния
inline float gaussianWeight(float dist2, float sigma)
{
    float denom = 2.0f * sigma * sigma;
    if (denom <= 0.0f) return 1.0f;
    return exp(-dist2 / denom);
}

// ======================
// Построение первичных лучей
// ======================

inline Ray makePrimaryRayJittered(int px, int py,
                                  int width, int height,
                                  float2 jitter,
                                  float3 camPos,
                                  float3 camForward,
                                  float3 camUp,
                                  float3 camRight,
                                  float  fovY)
{
    float fx = (float(px) + 0.5f + jitter.x);
    float fy = (float(py) + 0.5f + jitter.y);

    float ndcX = (fx / float(width))  * 2.0f - 1.0f;
    float ndcY = (fy / float(height)) * 2.0f - 1.0f;
    ndcY = -ndcY;

    float aspect     = float(width) / float(height);
    float tanHalfFov = tan(0.5f * fovY);

    float3 dirCam = float3(
        ndcX * aspect * tanHalfFov,
        ndcY * tanHalfFov,
        -1.0f
    );

    float3 dirWorld =
          dirCam.x * camRight
        + dirCam.y * camUp
        - dirCam.z * camForward;

    dirWorld = normalize(dirWorld);

    Ray r;
    r.origin    = camPos;
    r.direction = dirWorld;
    return r;
}

// ======================
// Пересечение с AABB
// ======================

inline bool processAxis(float origin, float dir,
                        float minB, float maxB,
                        thread float *tEnter,
                        thread float *tExit)
{
    const float eps = 1e-8f;
    if (fabs(dir) < eps)
    {
        if (origin < minB || origin > maxB)
            return false;
        return true;
    }

    float invD = 1.0f / dir;
    float t0   = (minB - origin) * invD;
    float t1   = (maxB - origin) * invD;
    if (t0 > t1)
    {
        float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }

    if (t0 > *tEnter) *tEnter = t0;
    if (t1 < *tExit)  *tExit  = t1;

    return *tEnter <= *tExit;
}

// ======================
// Пересечение с треугольником
// ======================

inline bool intersectTriangle(const Ray ray,
                              const device Triangle &tri,
                              thread float *tHitOut,
                              thread float *uOut,
                              thread float *vOut)

{
    const float EPS = 1e-6f;

    float3 v0 = float3(tri.v0);
    float3 v1 = float3(tri.v1);
    float3 v2 = float3(tri.v2);

    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;

    float3 pvec = cross(ray.direction, edge2);
    float  det  = dot(edge1, pvec);

    if (fabs(det) < EPS)
        return false;

    float invDet = 1.0f / det;

    float3 tvec = ray.origin - v0;

    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f)
        return false;

    float3 qvec = cross(tvec, edge1);
    float  v    = dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f)
        return false;

    float t = dot(edge2, qvec) * invDet;
    if (t <= EPS)
        return false;

    *tHitOut = t;
    *uOut = u;
    *vOut = v;
    return true;
}

inline bool intersectAABBWithRay(const AABB box,
                                 const Ray  ray,
                                 thread float *tEnter,
                                 thread float *tExit)
{
    const float tMinInit = 0.0f;
    const float tMaxInit = INFINITY;

    *tEnter = tMinInit;
    *tExit  = tMaxInit;

    float3 bmin = float3(box.v0);
    float3 bmax = float3(box.v1);

    float3 o = ray.origin;
    float3 d = ray.direction;

    if (!processAxis(o.x, d.x, bmin.x, bmax.x, tEnter, tExit)) return false;
    if (!processAxis(o.y, d.y, bmin.y, bmax.y, tEnter, tExit)) return false;
    if (!processAxis(o.z, d.z, bmin.z, bmax.z, tEnter, tExit)) return false;

    if (*tExit < 0.0f)
        return false;

    return true;
}

// Быстрый slab-тест AABB с предвычисленным invDir + отсечением по tMaxClip.
// Возвращает tEnter (tExit не нужен для обхода, но учитывается для отсечений).
inline bool intersectAABBWithRayFast(const AABB  box,
                                    float3      origin,
                                    float3      dir,
                                    float3      invDir,
                                    float       tMaxClip,
                                    thread float *tEnterOut)
{
    float3 bmin = float3(box.v0);
    float3 bmax = float3(box.v1);

    // Для компонент направления ~0: если луч параллелен и старт вне среза — нет пересечения.
    const float eps = 1e-8f;
    if (fabs(dir.x) < eps && (origin.x < bmin.x || origin.x > bmax.x)) return false;
    if (fabs(dir.y) < eps && (origin.y < bmin.y || origin.y > bmax.y)) return false;
    if (fabs(dir.z) < eps && (origin.z < bmin.z || origin.z > bmax.z)) return false;

    float3 t0 = (bmin - origin) * invDir;
    float3 t1 = (bmax - origin) * invDir;

    float3 tmin3 = min(t0, t1);
    float3 tmax3 = max(t0, t1);

    float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);
    float tExit  = min(min(tmax3.x, tmax3.y), tmax3.z);

    // Полностью позади камеры
    if (tExit < 0.0f) return false;

    // Вперёд от камеры + отсечение по текущему лучшему t
    tEnter = max(tEnter, 0.0f);
    tExit  = min(tExit, tMaxClip);

    if (tExit < tEnter) return false;

    *tEnterOut = tEnter;
    return true;
}


// ======================
// Обход BVH
// ======================

inline HitInfo traceRayBVH(const device BVHNode   *nodes,
                           const device Triangle  *tris,
                           uint                    nodeCount,
                           int                     rootIndex,
                           const Ray               ray)
{
    HitInfo result;
    result.hit           = false;
    result.triIndex      = -1;
    result.instanceIndex = -1;
    result.materialIndex = -1;
    result.t             = INFINITY;
    result.position      = float3(0.0f);
    result.normal        = float3(0.0f);
    result.color         = float3(1.0f);
    result.uv            = float2(0.0f);

    result.emission  = float3(0.0f);
    result.metallic  = 0.0f;
    result.roughness = 0.5f;

    if (nodeCount == 0u ||
        rootIndex < 0  ||
        rootIndex >= int(nodeCount))
    {
        return result;
    }

    // Предвычисления для AABB-тестов
    const float3 o = ray.origin;
    const float3 d = ray.direction;
    const float3 invD = 1.0f / d;

    struct StackEntry
    {
        int   index;
        float tEnter;
    };

    // В этом варианте пушим максимум один узел на уровень (far child),
    // поэтому 64 обычно достаточно; увеличивать сильно не стоит из-за регистров/локальной памяти.
    const int MAX_STACK = 96;
    StackEntry stack[MAX_STACK];
    int sp = 0;

    float tEnterRoot = 0.0f;
    if (!intersectAABBWithRayFast(nodes[rootIndex].box, o, d, invD, INFINITY, &tEnterRoot))
        return result;

    int   nodeIndex = rootIndex;
    float bestT     = INFINITY;

    // Одно дополнительное "deferred" место на случай переполнения стека (очень редко)
    int   deferredIndex = -1;
    float deferredEnter = 0.0f;

    while (true)
    {
        // Если узел невалиден — просто достаём следующий из стека
        if (nodeIndex < 0 || nodeIndex >= int(nodeCount))
        {
            bool found = false;
            while (sp > 0)
            {
                StackEntry e = stack[--sp];
                if (e.tEnter <= bestT)
                {
                    nodeIndex = e.index;
                    found = true;
                    break;
                }
            }
            if (!found && deferredIndex != -1 && deferredEnter <= bestT)
            {
                nodeIndex = deferredIndex;
                deferredIndex = -1;
                found = true;
            }
            if (!found) break;
            continue;
        }

        const device BVHNode &node = nodes[nodeIndex];

        // Leaf: один треугольник
        if (node.tri >= 0)
        {
            float tHit = 0.0f;
            float uHit = 0.0f;
            float vHit = 0.0f;

            const device Triangle &tri = tris[node.tri];

            if (intersectTriangle(ray, tri, &tHit, &uHit, &vHit) &&
                tHit >= 0.0f && tHit < bestT)
            {
                bestT                = tHit;
                result.hit           = true;
                result.triIndex      = node.tri;
                result.materialIndex = tri.materialIndex;
                result.t             = tHit;

                result.position = o + d * tHit;

                // tri.normal должен быть близок к нормализованному, но на всякий случай нормализуем.
                result.normal   = fast::normalize((float3)tri.normal);

                result.color    = (float3)tri.color;
                result.emission = (float3)tri.emission;
                result.metallic  = tri.metallic;
                result.roughness = tri.roughness;

                // Интерполяция UV по барицентрикам
                float wHit = 1.0f - uHit - vHit;
                float2 uv0 = float2(tri.uv0);
                float2 uv1 = float2(tri.uv1);
                float2 uv2 = float2(tri.uv2);
                result.uv  = uv0 * wHit + uv1 * uHit + uv2 * vHit;
            }

            // Возврат к следующему узлу из стека
            bool found = false;
            while (sp > 0)
            {
                StackEntry e = stack[--sp];
                if (e.tEnter <= bestT)
                {
                    nodeIndex = e.index;
                    found = true;
                    break;
                }
            }
            if (!found && deferredIndex != -1 && deferredEnter <= bestT)
            {
                nodeIndex = deferredIndex;
                deferredIndex = -1;
                found = true;
            }
            if (!found) break;
            continue;
        }

        // Internal: тестим детей с отсечением по bestT
        int left  = node.left;
        int right = node.right;

        bool  hitL = false;
        bool  hitR = false;
        float tL   = 0.0f;
        float tR   = 0.0f;

        const float tClip = bestT;

        if (left >= 0)
            hitL = intersectAABBWithRayFast(nodes[left].box, o, d, invD, tClip, &tL);

        if (right >= 0)
            hitR = intersectAABBWithRayFast(nodes[right].box, o, d, invD, tClip, &tR);

        if (hitL && hitR)
        {
            // Near/Far порядок: far кладём в стек, near идём сразу (меньше пуш/поп).
            int   nearIdx, farIdx;
            float nearT,   farT;

            if (tL < tR)
            {
                nearIdx = left;  nearT = tL;
                farIdx  = right; farT  = tR;
            }
            else
            {
                nearIdx = right; nearT = tR;
                farIdx  = left;  farT  = tL;
            }

            if (farT <= bestT)
            {
                if (sp < MAX_STACK)
                {
                    stack[sp++] = StackEntry{ farIdx, farT };
                }
                else
                {
                    // стек полон: запоминаем один узел во временный слот
                    deferredIndex = farIdx;
                    deferredEnter = farT;
                }
            }

            nodeIndex = nearIdx;
            continue;
        }
        else if (hitL)
        {
            nodeIndex = left;
            continue;
        }
        else if (hitR)
        {
            nodeIndex = right;
            continue;
        }
        else
        {
            // Нет пересечений — достаём из стека
            bool found = false;
            while (sp > 0)
            {
                StackEntry e = stack[--sp];
                if (e.tEnter <= bestT)
                {
                    nodeIndex = e.index;
                    found = true;
                    break;
                }
            }
            if (!found && deferredIndex != -1 && deferredEnter <= bestT)
            {
                nodeIndex = deferredIndex;
                deferredIndex = -1;
                found = true;
            }
            if (!found) break;
            continue;
        }
    }

    return result;
}


// ======================
// Теневые лучи
// ======================


// Быстрый тест "есть ли любой хит" для теней (early-out).
// maxDist > 0 => искать только до этой дистанции (например до источника света).
inline bool traceShadowBVH(const device BVHNode   *nodes,
                          const device Triangle  *tris,
                          uint                    nodeCount,
                          int                     rootIndex,
                          const Ray               ray,
                          float                   maxDist)
{
    if (nodeCount == 0u ||
        rootIndex < 0  ||
        rootIndex >= int(nodeCount))
        return false;

    const float3 o = ray.origin;
    const float3 d = ray.direction;
    const float3 invD = 1.0f / d;

    struct StackEntry { int index; float tEnter; };
    const int MAX_STACK = 96;
    StackEntry stack[MAX_STACK];
    int sp = 0;

    float bestT = (maxDist > 0.0f) ? maxDist : INFINITY;

    int   deferredIndex = -1;
    float deferredEnter = 0.0f;

    float tEnterRoot = 0.0f;
    if (!intersectAABBWithRayFast(nodes[rootIndex].box, o, d, invD, bestT, &tEnterRoot))
        return false;

    int nodeIndex = rootIndex;

    while (true)
    {
        if (nodeIndex < 0 || nodeIndex >= int(nodeCount))
        {
            bool found = false;
            while (sp > 0)
            {
                StackEntry e = stack[--sp];
                if (e.tEnter <= bestT)
                {
                    nodeIndex = e.index;
                    found = true;
                    break;
                }
            }
            if (!found && deferredIndex != -1 && deferredEnter <= bestT)
            {
                nodeIndex = deferredIndex;
                deferredIndex = -1;
                found = true;
            }
            if (!found) break;
            continue;
        }

        const device BVHNode &node = nodes[nodeIndex];

        if (node.tri >= 0)
        {
            float tHit = 0.0f, uHit = 0.0f, vHit = 0.0f;
            if (intersectTriangle(ray, tris[node.tri], &tHit, &uHit, &vHit))
            {
                // небольшой отсев самопересечения
                if (tHit > 1e-4f && tHit < bestT)
                    return true;
            }

            bool found = false;
            while (sp > 0)
            {
                StackEntry e = stack[--sp];
                if (e.tEnter <= bestT)
                {
                    nodeIndex = e.index;
                    found = true;
                    break;
                }
            }
            if (!found && deferredIndex != -1 && deferredEnter <= bestT)
            {
                nodeIndex = deferredIndex;
                deferredIndex = -1;
                found = true;
            }
            if (!found) break;
            continue;
        }

        int left  = node.left;
        int right = node.right;

        bool  hitL = false;
        bool  hitR = false;
        float tL = 0.0f;
        float tR = 0.0f;

        if (left >= 0)  hitL = intersectAABBWithRayFast(nodes[left].box,  o, d, invD, bestT, &tL);
        if (right >= 0) hitR = intersectAABBWithRayFast(nodes[right].box, o, d, invD, bestT, &tR);

        if (hitL && hitR)
        {
            int   nearIdx, farIdx;
            float nearT, farT;

            if (tL < tR)
            {
                nearIdx = left;  nearT = tL;
                farIdx  = right; farT  = tR;
            }
            else
            {
                nearIdx = right; nearT = tR;
                farIdx  = left;  farT  = tL;
            }

            if (sp < MAX_STACK)
                stack[sp++] = StackEntry{ farIdx, farT };
            else
            {
                deferredIndex = farIdx;
                deferredEnter = farT;
            }

            nodeIndex = nearIdx;
            continue;
        }
        else if (hitL)
        {
            nodeIndex = left;
            continue;
        }
        else if (hitR)
        {
            nodeIndex = right;
            continue;
        }
        else
        {
            bool found = false;
            while (sp > 0)
            {
                StackEntry e = stack[--sp];
                if (e.tEnter <= bestT)
                {
                    nodeIndex = e.index;
                    found = true;
                    break;
                }
            }
            if (!found && deferredIndex != -1 && deferredEnter <= bestT)
            {
                nodeIndex = deferredIndex;
                deferredIndex = -1;
                found = true;
            }
            if (!found) break;
            continue;
        }
    }

    return false;
}
inline HitInfo traceRaySceneBVH(const device BVHNode *tlasNodes,
                                 const device BVHNode *meshNodes,
                                 const device Triangle *tris,
                                 const device SceneInstanceGPU *instances,
                                 uint tlasNodeCount,
                                 uint meshNodeCount,
                                 uint instanceCount,
                                 int tlasRootIndex,
                                 const Ray worldRay)
{
    HitInfo result;
    result.hit = false;
    result.triIndex = -1;
    result.instanceIndex = -1;
    result.materialIndex = -1;
    result.t = INFINITY;
    result.position = float3(0.0f);
    result.normal = float3(0.0f, 0.0f, 1.0f);
    result.color = float3(1.0f);
    result.emission = float3(0.0f);
    result.metallic = 0.0f;
    result.roughness = 0.5f;
    result.uv = float2(0.0f);

    if (instances == nullptr || instanceCount == 0u)
        return result;

    const float3 o = worldRay.origin;
    const float3 d = worldRay.direction;
    const float3 invD = 1.0f / d;

    struct StackEntry { int index; float tEnter; };
    const int MAX_STACK = 96;
    StackEntry stack[MAX_STACK];
    int sp = 0;

    float tEnterRoot = 0.0f;
    if (!intersectAABBWithRayFast(tlasNodes[tlasRootIndex].box, o, d, invD, INFINITY, &tEnterRoot))
        return result;

    int nodeIndex = tlasRootIndex;
    float bestT = INFINITY;
    int deferredIndex = -1;
    float deferredEnter = 0.0f;

    while (true)
    {
        if (nodeIndex < 0 || nodeIndex >= int(tlasNodeCount))
        {
            bool found = false;
            while (sp > 0)
            {
                StackEntry e = stack[--sp];
                if (e.tEnter <= bestT) { nodeIndex = e.index; found = true; break; }
            }
            if (!found && deferredIndex != -1 && deferredEnter <= bestT)
            { nodeIndex = deferredIndex; deferredIndex = -1; found = true; }
            if (!found) break;
            continue;
        }

        const device BVHNode &node = tlasNodes[nodeIndex];
        if (node.tri >= 0)
        {
            const int instIndex = node.tri;
            if (instIndex >= 0 && uint(instIndex) < instanceCount)
            {
                const device SceneInstanceGPU &inst = instances[instIndex];
                Ray localRay;
                localRay.origin = transformPointWorldToObject(inst, worldRay.origin);
                localRay.direction = transformDirectionWorldToObject(inst, worldRay.direction);
                HitInfo localHit = traceRayBVH(meshNodes, tris, meshNodeCount, inst.blasRootIndex, localRay);
                if (localHit.hit)
                {
                    float3 worldPos = transformPointObjectToWorld(inst, localHit.position);
                    float worldT = length(worldPos - worldRay.origin);
                    if (worldT < bestT)
                    {
                        bestT = worldT;
                        result = localHit;
                        result.hit = true;
                        result.instanceIndex = instIndex;
                        result.t = worldT;
                        result.position = worldPos;
                        result.normal = normalize(transformNormalObjectToWorld(inst, localHit.normal));
                    }
                }
            }

            bool found = false;
            while (sp > 0)
            {
                StackEntry e = stack[--sp];
                if (e.tEnter <= bestT) { nodeIndex = e.index; found = true; break; }
            }
            if (!found && deferredIndex != -1 && deferredEnter <= bestT)
            { nodeIndex = deferredIndex; deferredIndex = -1; found = true; }
            if (!found) break;
            continue;
        }

        int left = node.left;
        int right = node.right;
        bool hitL = false, hitR = false;
        float tL = 0.0f, tR = 0.0f;
        if (left >= 0) hitL = intersectAABBWithRayFast(tlasNodes[left].box, o, d, invD, bestT, &tL);
        if (right >= 0) hitR = intersectAABBWithRayFast(tlasNodes[right].box, o, d, invD, bestT, &tR);

        if (hitL && hitR)
        {
            int nearIdx, farIdx; float nearT, farT;
            if (tL < tR) { nearIdx = left; nearT = tL; farIdx = right; farT = tR; }
            else { nearIdx = right; nearT = tR; farIdx = left; farT = tL; }
            if (sp < MAX_STACK) stack[sp++] = StackEntry{farIdx, farT};
            else { deferredIndex = farIdx; deferredEnter = farT; }
            nodeIndex = nearIdx;
            continue;
        }
        else if (hitL) { nodeIndex = left; continue; }
        else if (hitR) { nodeIndex = right; continue; }
        else
        {
            bool found = false;
            while (sp > 0)
            {
                StackEntry e = stack[--sp];
                if (e.tEnter <= bestT) { nodeIndex = e.index; found = true; break; }
            }
            if (!found && deferredIndex != -1 && deferredEnter <= bestT)
            { nodeIndex = deferredIndex; deferredIndex = -1; found = true; }
            if (!found) break;
        }
    }

    return result;
}

inline bool traceShadowSceneBVH(const device BVHNode *tlasNodes,
                                const device BVHNode *meshNodes,
                                const device Triangle *tris,
                                const device SceneInstanceGPU *instances,
                                uint tlasNodeCount,
                                uint meshNodeCount,
                                uint instanceCount,
                                int tlasRootIndex,
                                const Ray worldRay,
                                float maxDist)
{
    HitInfo hit = traceRaySceneBVH(tlasNodes, meshNodes, tris, instances, tlasNodeCount, meshNodeCount, instanceCount, tlasRootIndex, worldRay);
    return hit.hit && hit.t > 1.0e-4f && hit.t < maxDist;
}

inline bool isShadowed(const device BVHNode  *bvhNodes,
                       const device Triangle *triangles,
                       uint                   nodeCount,
                       int                    rootIndex,
                       float3                 origin,
                       float3                 dir,
                       float                  maxDist)
{
    Ray shadowRay;
    shadowRay.origin    = origin;
    shadowRay.direction = dir;

    return traceShadowBVH(bvhNodes, triangles, nodeCount, rootIndex, shadowRay, maxDist);
}

inline float luminance3(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

inline float powerHeuristic(float pdfA, float pdfB)
{
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / max(a2 + b2, 1e-8f);
}

inline float3 triangleFaceNormal(const device Triangle &tri)
{
    float3 e1 = float3(tri.v1) - float3(tri.v0);
    float3 e2 = float3(tri.v2) - float3(tri.v0);
    float3 n = cross(e1, e2);
    float len2 = dot(n, n);
    if (len2 > 1e-16f) return normalize(n);
    return normalize(float3(tri.normal));
}

inline float2 triangleSampleUV(const device Triangle &tri, float b0, float b1, float b2)
{
    return float2(tri.uv0) * b0 + float2(tri.uv1) * b1 + float2(tri.uv2) * b2;
}

inline float3 resolveTriangleEmissionTextured(const device Triangle *triangles,
                                              uint triIndex,
                                              float2 uv,
                                              const device MaterialGPU *materials,
                                              uint materialCount,
                                              const device MaterialGPU_PBR *materialsPBR,
                                              uint materialPBRCount,
                                              const array<texture2d<float, access::sample>, MAX_BASE_TEX> sceneTextures)
{
    const device Triangle &tri = triangles[triIndex];
    float3 emissive = float3(tri.emission) * 3.0f;
    int matId = tri.materialIndex;

    if (materialsPBR != nullptr && materialPBRCount > 0u && matId >= 0 && uint(matId) < materialPBRCount)
    {
        MaterialGPU_PBR mp = materialsPBR[matId];
        if (mp.emissionTexIndex >= 0 && uint(mp.emissionTexIndex) < MAX_BASE_TEX)
        {
            float4 eTex = sceneTextures[uint(mp.emissionTexIndex)].sample(g_texSampler, fract(uv));
            if ((mp.flags & MATERIAL_FLAG_EMISSION_USE_ALPHA_MASK) != 0)
                emissive *= sampleEmissiveMask(eTex);
            else
                emissive *= eTex.rgb;
        }
    }
    else if (materials != nullptr && materialCount > 0u && matId >= 0 && uint(matId) < materialCount)
    {
        int eTexId = materials[matId].emissionTexIndex;
        if (eTexId >= 0 && uint(eTexId) < MAX_BASE_TEX)
        {
            emissive *= sceneTextures[uint(eTexId)].sample(g_texSampler, fract(uv)).rgb;
        }
    }

    return emissive * EMISSION_LIGHT_EXPOSURE_GPU;
}

inline bool findEmissiveEntry(uint triIndex,
                              const device EmissiveTriangleGPU *emissiveTriangles,
                              uint emissiveTriangleCount,
                              thread EmissiveTriangleGPU &outEntry)
{
    for (uint i = 0u; i < emissiveTriangleCount; ++i)
    {
        EmissiveTriangleGPU e = emissiveTriangles[i];
        if (e.triIndex == triIndex)
        {
            outEntry = e;
            return true;
        }
    }
    return false;
}

inline EmissiveLightSample sampleOneEmissiveTriangle(const device Triangle *triangles,
                                                     const device SceneInstanceGPU *instances,
                                                     const device EmissiveTriangleGPU *emissiveTriangles,
                                                     uint emissiveTriangleCount,
                                                     uint seedSelect,
                                                     uint seedBaryU,
                                                     uint seedBaryV)
{
    EmissiveLightSample s{};
    s.valid = 0u;

    if (emissiveTriangles == nullptr || emissiveTriangleCount == 0u)
        return s;

    float target = rand01(seedSelect);
    uint lo = 0u;
    uint hi = emissiveTriangleCount - 1u;
    while (lo < hi)
    {
        uint mid = (lo + hi) >> 1u;
        if (target <= emissiveTriangles[mid].cdf) hi = mid;
        else lo = mid + 1u;
    }

    EmissiveTriangleGPU e = emissiveTriangles[lo];
    const device Triangle &tri = triangles[e.triIndex];

    float r1 = rand01(seedBaryU);
    float r2 = rand01(seedBaryV);
    float su = sqrt(r1);
    float b0 = 1.0f - su;
    float b1 = r2 * su;
    float b2 = 1.0f - b0 - b1;

    float3 pos = float3(tri.v0) * b0 + float3(tri.v1) * b1 + float3(tri.v2) * b2;
    float3 nrm = triangleFaceNormal(tri);

    if (instances != nullptr)
    {
        const device SceneInstanceGPU &inst = instances[e.instanceIndex];
        pos = transformPointObjectToWorld(inst, pos);
        nrm = normalize(transformNormalObjectToWorld(inst, nrm));
    }

    s.valid = 1u;
    s.triIndex = e.triIndex;
    s.instanceIndex = e.instanceIndex;
    s.area = max(e.area, 1e-8f);
    s.selectionPdf = max(e.selectionPdf, 1e-8f);
    s.position = pos;
    s.normal = nrm;
    s.uv = triangleSampleUV(tri, b0, b1, b2);
    return s;
}

inline EmissiveLightSample sampleOneEmissiveTriangle(const device Triangle *triangles,
                                                     const device EmissiveTriangleGPU *emissiveTriangles,
                                                     uint emissiveTriangleCount,
                                                     uint seedSelect,
                                                     uint seedBaryU,
                                                     uint seedBaryV)
{
    return sampleOneEmissiveTriangle(triangles,
                                     nullptr,
                                     emissiveTriangles,
                                     emissiveTriangleCount,
                                     seedSelect,
                                     seedBaryU,
                                     seedBaryV);
}

inline float emissiveLightPdfForHit(float3 prevSurfacePos,
                                    float3 lightPos,
                                    float3 lightNormal,
                                    uint triIndex,
                                    const device EmissiveTriangleGPU *emissiveTriangles,
                                    uint emissiveTriangleCount)
{
    EmissiveTriangleGPU entry{};
    if (!findEmissiveEntry(triIndex, emissiveTriangles, emissiveTriangleCount, entry))
        return 0.0f;

    float3 toLight = lightPos - prevSurfacePos;
    float dist2 = dot(toLight, toLight);
    if (dist2 <= 1e-10f)
        return 0.0f;

    float dist = sqrt(dist2);
    float3 wi = toLight / dist;
    float cosLight = fabs(dot(normalize(lightNormal), -wi));
    if (cosLight <= 1e-6f)
        return 0.0f;

    return entry.selectionPdf * dist2 / max(cosLight * entry.area, 1e-6f);
}

inline float3 evalSurfaceBRDF(float3 N,
                              float3 V,
                              float3 L,
                              float3 baseColor,
                              float metallic,
                              float roughness)
{
    float NdV = max(dot(N, V), 0.0f);
    float NdL = max(dot(N, L), 0.0f);
    if (NdV <= 0.0f || NdL <= 0.0f)
        return float3(0.0f);

    float r = clamp(roughness, 0.02f, 0.98f);
    float3 F0 = lerp3(float3(0.04f), baseColor, metallic);
    float3 H = normalize(V + L);
    float NdH = max(dot(N, H), 0.0f);
    float HdV = max(dot(H, V), 0.0f);

    float alpha = r * r;
    float D = distributionGGX(NdH, alpha);
    float k = (r + 1.0f);
    k = (k * k) * 0.125f;
    float G = geometrySmith(NdV, NdL, k);
    float3 F = fresnelSchlick(HdV, F0);

    float3 spec = (D * G * F) / max(4.0f * NdV * NdL, 1e-6f);
    float3 kd = (float3(1.0f) - F) * (1.0f - metallic);
    float3 diff = kd * baseColor * INV_PI;
    return diff + spec;
}

inline float pdfSurfaceBSDF(float3 N,
                            float3 V,
                            float3 L,
                            float3 baseColor,
                            float metallic,
                            float roughness)
{
    float NdL = max(dot(N, L), 0.0f);
    float NdV = max(dot(N, V), 0.0f);
    if (NdL <= 0.0f || NdV <= 0.0f)
        return 0.0f;

    float r = clamp(roughness, 0.02f, 0.98f);
    float3 F0 = lerp3(float3(0.04f), baseColor, metallic);
    float specProb = clamp(max(F0.x, max(F0.y, F0.z)), 0.05f, 0.95f);
    float diffProb = 1.0f - specProb;

    float diffPdf = NdL * INV_PI;

    float3 H = normalize(V + L);
    float NdH = max(dot(N, H), 0.0f);
    float HdV = max(dot(H, V), 0.0f);
    float alpha = r * r;
    float D = distributionGGX(NdH, alpha);
    float specPdf = (D * NdH) / max(4.0f * HdV, 1e-6f);

    return diffProb * diffPdf + specProb * specPdf;
}

inline BSDFSample sampleSurfaceBSDF(float3 N,
                                    float3 V,
                                    float3 baseColor,
                                    float metallic,
                                    float roughness,
                                    float u1,
                                    float u2,
                                    float uLobe)
{
    BSDFSample s{};
    s.valid = 0u;
    s.direction = float3(0.0f);
    s.weight = float3(0.0f);
    s.pdf = 0.0f;

    float r = clamp(roughness, 0.02f, 0.98f);
    float3 F0 = lerp3(float3(0.04f), baseColor, metallic);
    float specProb = clamp(max(F0.x, max(F0.y, F0.z)), 0.05f, 0.95f);

    float3 L;
    if (uLobe < specProb)
    {
        float alpha = r * r;
        float alpha2 = alpha * alpha;
        float phi = 2.0f * PI * u2;
        float cosTheta = sqrt((1.0f - u1) / (1.0f + (alpha2 - 1.0f) * u1));
        float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));

        float3 tangent, bitangent;
        buildOrthonormalBasis(N, tangent, bitangent);
        float3 H = normalize(tangent * (sinTheta * cos(phi)) +
                             bitangent * (sinTheta * sin(phi)) +
                             N * cosTheta);
        L = normalize(reflect(-V, H));
    }
    else
    {
        float3 localDir = cosineSampleHemisphere(u1, u2);
        float3 tangent, bitangent;
        buildOrthonormalBasis(N, tangent, bitangent);
        L = normalize(localDir.x * tangent + localDir.y * bitangent + localDir.z * N);
    }

    float NdL = max(dot(N, L), 0.0f);
    if (NdL <= 0.0f)
        return s;

    float pdf = pdfSurfaceBSDF(N, V, L, baseColor, metallic, roughness);
    if (pdf <= 1e-8f)
        return s;

    float3 f = evalSurfaceBRDF(N, V, L, baseColor, metallic, roughness);
    s.valid = 1u;
    s.direction = L;
    s.pdf = pdf;
    s.weight = f * (NdL / pdf);
    return s;
}

inline float3 sampleDirectEmissiveLightingTextured(float3 hitPos,
                                                   float3 N,
                                                   float3 Ng,
                                                   float3 V,
                                                   uint hitTriIndex,
                                                   float3 baseColor,
                                                   float metallic,
                                                   float roughness,
                                                   const device BVHNode *bvhNodes,
                                                   const device Triangle *triangles,
                                                   uint nodeCount,
                                                   int rootIndex,
                                                   const device MaterialGPU *materials,
                                                   uint materialCount,
                                                   const device MaterialGPU_PBR *materialsPBR,
                                                   uint materialPBRCount,
                                                   const array<texture2d<float, access::sample>, MAX_BASE_TEX> sceneTextures,
                                                   const device EmissiveTriangleGPU *emissiveTriangles,
                                                   uint emissiveTriangleCount,
                                                   uint seedSelect,
                                                   uint seedBaryU,
                                                   uint seedBaryV)
{
    if (emissiveTriangles == nullptr || emissiveTriangleCount == 0u)
        return float3(0.0f);

    EmissiveLightSample ls = sampleOneEmissiveTriangle(triangles, emissiveTriangles, emissiveTriangleCount,
                                                       seedSelect, seedBaryU, seedBaryV);
    if (ls.valid == 0u)
        return float3(0.0f);
    if (ls.triIndex == hitTriIndex)
        return float3(0.0f);

    float3 toLight = ls.position - hitPos;
    float dist2 = dot(toLight, toLight);
    if (dist2 <= 1e-10f)
        return float3(0.0f);

    float dist = sqrt(dist2);
    float3 L = toLight / dist;
    float NdL = max(dot(N, L), 0.0f);
    if (NdL <= 0.0f)
        return float3(0.0f);

    float cosLight = fabs(dot(normalize(ls.normal), -L));
    if (cosLight <= 1e-6f)
        return float3(0.0f);

    float maxDist = dist - 2.0f * SHADOW_EPS;
    if (maxDist <= 0.0f)
        return float3(0.0f);

    Ray shadowRay;
    shadowRay.origin = hitPos + Ng * SHADOW_EPS;
    shadowRay.direction = L;
    if (traceShadowBVH(bvhNodes, triangles, nodeCount, rootIndex, shadowRay, maxDist))
        return float3(0.0f);

    float3 Le = resolveTriangleEmissionTextured(triangles, ls.triIndex, ls.uv,
                                                materials, materialCount,
                                                materialsPBR, materialPBRCount,
                                                sceneTextures);
    if (luminance3(Le) <= 1e-6f)
        return float3(0.0f);

    float pdfLight = ls.selectionPdf * dist2 / max(cosLight * ls.area, 1e-6f);
    if (pdfLight <= 1e-8f)
        return float3(0.0f);

    float pdfBsdf = pdfSurfaceBSDF(N, V, L, baseColor, metallic, roughness);
    float mis = powerHeuristic(pdfLight, pdfBsdf);
    float3 f = evalSurfaceBRDF(N, V, L, baseColor, metallic, roughness);
    return Le * f * (NdL * mis / pdfLight);
}


inline float computeDecalDerivedMask(float4 albedo)
{
    float luma = luminance3(albedo.rgb);
    float maxC = max(max(albedo.r, albedo.g), albedo.b);
    float minC = min(min(albedo.r, albedo.g), albedo.b);
    float chroma = maxC - minC;

    // Fallback when the exported decal material graph does not expose its dedicated opacity
    // texture explicitly. White backgrounds should disappear, while painted/dirty regions survive.
    float contentMask = saturate(chroma * 2.10f + (1.0f - luma) * 1.05f - 0.03f);
    return saturate(max(albedo.a, contentMask));
}

inline float sampleDecalMask(float2 uv,
                             float2 detailUV,
                             float4 decalAlbedo,
                             DecalGPU d,
                             const array<texture2d<float, access::sample>, MAX_BASE_TEX> sceneTextures)
{
    float mask = computeDecalDerivedMask(decalAlbedo);
    const bool hasOpacityTex = (d.opacityTexIndex >= 0 && uint(d.opacityTexIndex) < MAX_BASE_TEX);
    const bool hasDetailTex  = (d.detailTexIndex  >= 0 && uint(d.detailTexIndex)  < MAX_BASE_TEX);

    if (hasOpacityTex)
    {
        // Opacity/mask textures are authored in the same projected UV space as albedo.
        float4 m = sceneTextures[uint(d.opacityTexIndex)].sample(g_texSampler, uv);
        float texMask = saturate(max(m.a, luminance3(m.rgb)));
        mask *= texMask;
    }

    if (hasDetailTex)
    {
        float4 det = sceneTextures[uint(d.detailTexIndex)].sample(g_texSampler, detailUV);
        float detailMask = saturate(max(det.a, luminance3(det.rgb)));
        // Treat detail as breakup/noise, not as a hard opacity gate.
        mask *= mix(1.0f, detailMask, hasOpacityTex ? 0.25f : 0.12f);
    }

    float opacityPower = max(d.opacityPower, 0.25f);
    if (!hasOpacityTex)
    {
        // Decals exported from UE often have only albedo + optional detail, while
        // opacity is implicit in the albedo alpha/chroma. Keep that path soft.
        mask = smoothstep(0.05f, 0.48f, saturate(mask));
        opacityPower = min(opacityPower, 1.25f);
    }
    else
    {
        opacityPower = min(opacityPower, 4.0f);
    }

    mask = pow(max(mask, 1.0e-4f), opacityPower);
    return saturate(mask * d.opacity);
}

inline void applyProjectedDecalsPrimary(float3 hitPos,
                                        float3 Ng,
                                        thread float3 &Ns,
                                        thread float3 &baseColor,
                                        thread float  &roughness,
                                        const device DecalGPU *decals,
                                        uint decalCount,
                                        const array<texture2d<float, access::sample>, MAX_BASE_TEX> sceneTextures)
{
    if (decals == nullptr || decalCount == 0u)
        return;

    const float3 NgNorm = normalize(Ng);

    for (uint i = 0u; i < decalCount; ++i)
    {
        DecalGPU d = decals[i];
        if (d.baseColorTexIndex < 0 || uint(d.baseColorTexIndex) >= MAX_BASE_TEX)
            continue;

        float3 pos      = float3(d.posX, d.posY, d.posZ);
        float3 projAxis = normalize(float3(d.axisXx, d.axisXy, d.axisXz)); // decal projection axis

        // Exported UE decal bases are already correct in scene.json/LevelMeta20, but the runtime was
        // interpreting the in-plane axes in the wrong order. For wall graffiti this rotates/offsets the
        // artwork and makes the projected box outline visible.
        //
        // Use Z as the horizontal/U axis and -Y as the vertical/V axis.
        float3 uAxis    = normalize(float3(d.axisZx, d.axisZy, d.axisZz));
        float3 vAxis    = normalize(-float3(d.axisYx, d.axisYy, d.axisYz));

        float3 rel = hitPos - pos;
        float depth = dot(rel, projAxis);
        float lx    = dot(rel, uAxis);
        float ly    = dot(rel, vAxis);

        if (fabs(depth) > d.sizeX || fabs(lx) > d.sizeZ || fabs(ly) > d.sizeY)
            continue;

        // Decals should project onto the receiver facing the projection direction, not onto both sides
        // of the box. Using an unsigned facing test was one of the reasons why the decal frame became visible.
        float facing = dot(NgNorm, -projAxis);
        if (facing < 0.06f)
            continue;

        float2 uv = float2(0.5f + 0.5f * (lx / max(d.sizeZ, 1.0e-4f)),
                           0.5f - 0.5f * (ly / max(d.sizeY, 1.0e-4f)));
        if (any(uv < 0.0f) || any(uv > 1.0f))
            continue;

        float2 detailScale = float2(max(fabs(d.tilingU), 1.0f), max(fabs(d.tilingV), 1.0f));
        float2 detailUV = fract((uv - 0.5f) * detailScale + 0.5f);

        float4 decalTex = sceneTextures[uint(d.baseColorTexIndex)].sample(g_texSampler, uv);
        float alpha = sampleDecalMask(uv, detailUV, decalTex, d, sceneTextures);
        if (alpha <= 1.0e-4f)
            continue;

        float depthFade = 1.0f - smoothstep(d.sizeX * 0.82f, d.sizeX, fabs(depth));
        float edgeFadeU = 1.0f - smoothstep(d.sizeZ * 0.90f, d.sizeZ, fabs(lx));
        float edgeFadeV = 1.0f - smoothstep(d.sizeY * 0.90f, d.sizeY, fabs(ly));
        alpha *= depthFade * edgeFadeU * edgeFadeV;
        if (alpha <= 1.0e-4f)
            continue;

        float3 decalColor = decalTex.rgb * float3(d.baseColorX, d.baseColorY, d.baseColorZ);
        baseColor = mix(baseColor, decalColor, clamp01(alpha));

        float decalRoughness = clamp(roughness + d.roughnessBias, 0.02f, 0.98f);
        if (d.ormTexIndex >= 0 && uint(d.ormTexIndex) < MAX_BASE_TEX)
        {
            float3 orm = sceneTextures[uint(d.ormTexIndex)].sample(g_texSampler, uv).rgb;
            decalRoughness = clamp(orm.g + d.roughnessBias, 0.02f, 0.98f);
        }
        else if (d.roughnessTexIndex >= 0 && uint(d.roughnessTexIndex) < MAX_BASE_TEX)
        {
            decalRoughness = clamp(sceneTextures[uint(d.roughnessTexIndex)].sample(g_texSampler, uv).r + d.roughnessBias,
                                   0.02f, 0.98f);
        }
        roughness = mix(roughness, decalRoughness, clamp01(alpha * 0.65f));

        if (d.normalTexIndex >= 0 && uint(d.normalTexIndex) < MAX_BASE_TEX)
        {
            float3 decalNt = sceneTextures[uint(d.normalTexIndex)].sample(g_texSampler, uv).xyz * 2.0f - 1.0f;
            decalNt.xy *= d.normalIntensity;
            decalNt = normalize(decalNt);
            float3 decalNw = normalize(decalNt.x * uAxis + decalNt.y * vAxis - decalNt.z * projAxis);
            Ns = normalize(mix(Ns, decalNw, clamp01(alpha * 0.5f)));
        }
    }
}

// ======================
// Просчет теней/освещения от источников
// ======================

inline float3 computeLightingAtPoint(
    float3                        hitPos,
    float3                        N,
    float3                        Ng,
    float3                        baseColor,
    float                         metallic,
    float                         roughness,
    float3                        V,
    const device BVHNode         *bvhNodes,
    const device Triangle        *triangles,
    uint                          nodeCount,
    int                           rootIndex,
    const device LightGPU        *lights,
    uint                          lightCount,
    uint                          baseSeed)
{
    float3 result = float3(0.0f);
    if (lights == nullptr || lightCount == 0u)
        return result;

    // Use geometric normal for shadow ray offset to avoid self-shadowing with normal maps
    float3 shadowOrigin = hitPos + Ng * SHADOW_EPS;

    float r = clamp(roughness, 0.05f, 0.95f);
    metallic = clamp(metallic, 0.0f, 1.0f);

    float3 F0 = lerp3(float3(0.04f), baseColor, metallic);

    float alpha = r * r;
    float NdV   = max(dot(N, V), 0.0f);

    float k = (r + 1.0f);
    k = (k * k) * 0.125f; // (r+1)^2 / 8

    for (uint i = 0u; i < lightCount; ++i)
    {
        const device LightGPU &light = lights[i];

        int   type    = light.type;
        float3 lp     = float3(light.position);
        float3 ld     = normalize(float3(light.direction));
        float  radius = light.radius;

        float3 L;
        float  dist      = 1.0f;
        bool   hasFinite = true;

        if (type == LIGHT_TYPE_DIRECTIONAL)
        {
            L         = -ld;
            hasFinite = false;
        }
        else
        {
            float3 toLightCenter = lp - hitPos;

            if (radius > 0.0f)
            {
                uint seed1 = baseSeed ^ (0x9E3779B9u * (i * 2u + 1u));
                uint seed2 = baseSeed ^ (0x9E3779B9u * (i * 2u + 2u));

                float u1 = rand01(seed1);
                float u2 = rand01(seed2);

                float z   = 1.0f - 2.0f * u1;
                float phi = 2.0f * PI * u2;
                float rxy = sqrt(max(0.0f, 1.0f - z * z));

                float3 offset = float3(
                    rxy * cos(phi),
                    rxy * sin(phi),
                    z
                ) * radius;

                float3 samplePos = lp + offset;
                float3 toLight   = samplePos - hitPos;
                dist = length(toLight);
                if (dist <= 0.0f)
                    continue;

                L  = toLight / dist;
                lp = samplePos;
            }
            else
            {
                dist = length(toLightCenter);
                if (dist <= 0.0f)
                    continue;
                L = toLightCenter / dist;
            }
        }

        float NdL = dot(N, L);
        if (NdL <= 0.0f)
            continue;

                float attenuation = 1.0f;
        if (hasFinite)
        {
            float falloffDist = max(dist * LIGHT_FALLOFF_DISTANCE_SCALE, 1e-3f);
            float r2 = falloffDist * falloffDist;
            if (r2 <= 0.0f)
                continue;

            // Базовый inverse-square
            attenuation = INV_4PI / r2;

            // Дополнительная форма затухания по AttenuationRadius
            if (light.attenuationRadius > 0.0f)
            {
                float R = light.attenuationRadius * LIGHT_ATTENUATION_RADIUS_SCALE;

                if (dist >= R)
                    continue;

                float s  = dist / R;
                float s2 = s * s;

                const float F = 1.0f; // тот же F, что и на CPU
                float num = 1.0f - s2;
                num *= num;           // (1 - s^2)^2
                float den = 1.0f + F * s2;

                float extra = (den > 0.0f) ? (num / den) : 0.0f;

                attenuation *= extra;
            }
        }

        float intensity = light.intensity * attenuation * LIGHT_EXPOSURE_GPU;

        if (type == LIGHT_TYPE_SPOT)
        {
            float  spotSize  = max(light.spotSize, 1e-4f);
            float  spotBlend = clamp01(light.spotBlend);

            float outerAngle = 0.5f * spotSize;
            float innerAngle = outerAngle * (1.0f - spotBlend);

            float cosOuter = cos(outerAngle);
            float cosInner = cos(innerAngle);

            float3 dirToPoint = normalize(hitPos - lp);
            float  cosTheta   = dot(ld, dirToPoint);

            if (cosTheta <= 0.0f)
                continue;

            float spotFactor = 0.0f;
            if (spotBlend <= 0.0f || cosInner <= cosOuter)
            {
                spotFactor = (cosTheta >= cosOuter) ? 1.0f : 0.0f;
            }
            else
            {
                if (cosTheta <= cosOuter)
                    spotFactor = 0.0f;
                else if (cosTheta >= cosInner)
                    spotFactor = 1.0f;
                else
                {
                    float t = (cosTheta - cosOuter) / (cosInner - cosOuter);
                    t = clamp01(t);
                    spotFactor = t * t;
                }
            }

            intensity *= spotFactor;
            if (intensity <= 0.0f)
                continue;
        }

        if (type == LIGHT_TYPE_AREA)
        {
            float3 dirFromLight = normalize(hitPos - lp);
            float  cosEmit      = dot(ld, dirFromLight);
            if (cosEmit <= 0.0f)
                continue;
            intensity *= cosEmit;
            if (intensity <= 0.0f)
                continue;
        }

        float maxDist = hasFinite ? (dist - SHADOW_EPS) : -1.0f;
        if (isShadowed(bvhNodes, triangles, nodeCount, rootIndex,
                       shadowOrigin, L, maxDist))
        {
            continue;
        }

        float3 lightColor = float3(light.color) * intensity;

        float3 H    = normalize(L + V);
        float  NdH  = max(dot(N, H), 0.0f);
        float  HdV  = max(dot(V, H), 0.0f);
        NdL        = max(NdL, 0.0f);

        if (NdL <= 0.0f || NdH <= 0.0f || NdV <= 0.0f)
            continue;

        float3 F = fresnelSchlick(HdV, F0);
        float  D = distributionGGX(NdH, alpha);
        float  G = geometrySmith(NdV, NdL, k);

        float3 numerator = D * G * F;
        float  denom     = max(4.0f * NdV * NdL, 1e-4f);
        float3 spec      = numerator / denom;

        float3 kd      = (float3(1.0f) - F) * (1.0f - metallic);
        float3 diffuse = kd * baseColor * INV_PI;

        float3 brdf = diffuse + spec;

        result += lightColor * brdf * NdL;
    }

    return result;
}


inline float3 sampleDirectEmissiveLightingTexturedInstanced(float3 hitPos,
                                                   float3 N,
                                                   float3 Ng,
                                                   float3 V,
                                                   uint hitTriIndex,
                                                   uint hitInstanceIndex,
                                                   float3 baseColor,
                                                   float metallic,
                                                   float roughness,
                                                   const device BVHNode *tlasNodes,
                                                   const device BVHNode *meshNodes,
                                                   const device Triangle *triangles,
                                                   const device SceneInstanceGPU *instances,
                                                   uint tlasNodeCount,
                                                   uint meshNodeCount,
                                                   uint instanceCount,
                                                   int rootIndex,
                                                   const device MaterialGPU *materials,
                                                   uint materialCount,
                                                   const device MaterialGPU_PBR *materialsPBR,
                                                   uint materialPBRCount,
                                                   const array<texture2d<float, access::sample>, MAX_BASE_TEX> sceneTextures,
                                                   const device EmissiveTriangleGPU *emissiveTriangles,
                                                   uint emissiveTriangleCount,
                                                   uint seedSelect,
                                                   uint seedBaryU,
                                                   uint seedBaryV)
{
    if (emissiveTriangles == nullptr || emissiveTriangleCount == 0u)
        return float3(0.0f);

    EmissiveLightSample ls = sampleOneEmissiveTriangle(triangles, instances, emissiveTriangles, emissiveTriangleCount,
                                                       seedSelect, seedBaryU, seedBaryV);
    if (ls.valid == 0u)
        return float3(0.0f);
    if (ls.triIndex == hitTriIndex && ls.instanceIndex == hitInstanceIndex)
        return float3(0.0f);

    float3 toLight = ls.position - hitPos;
    float dist2 = dot(toLight, toLight);
    if (dist2 <= 1e-10f)
        return float3(0.0f);

    float dist = sqrt(dist2);
    float3 L = toLight / dist;
    float NdL = max(dot(N, L), 0.0f);
    if (NdL <= 0.0f)
        return float3(0.0f);

    float cosLight = fabs(dot(normalize(ls.normal), -L));
    if (cosLight <= 1e-6f)
        return float3(0.0f);

    float maxDist = dist - 2.0f * SHADOW_EPS;
    if (maxDist <= 0.0f)
        return float3(0.0f);

    Ray shadowRay;
    shadowRay.origin = hitPos + Ng * SHADOW_EPS;
    shadowRay.direction = L;
    if (traceShadowSceneBVH(tlasNodes, meshNodes, triangles, instances, tlasNodeCount, meshNodeCount, instanceCount, rootIndex, shadowRay, maxDist))
        return float3(0.0f);

    float3 Le = resolveTriangleEmissionTextured(triangles, ls.triIndex, ls.uv,
                                                materials, materialCount,
                                                materialsPBR, materialPBRCount,
                                                sceneTextures);
    if (luminance3(Le) <= 1e-6f)
        return float3(0.0f);

    float pdfLight = ls.selectionPdf * dist2 / max(cosLight * ls.area, 1e-6f);
    if (pdfLight <= 1e-8f)
        return float3(0.0f);

    float pdfBsdf = pdfSurfaceBSDF(N, V, L, baseColor, metallic, roughness);
    float mis = powerHeuristic(pdfLight, pdfBsdf);
    float3 f = evalSurfaceBRDF(N, V, L, baseColor, metallic, roughness);
    return Le * f * (NdL * mis / pdfLight);
}


// ======================
// Просчет теней/освещения от источников
// ======================

inline float3 computeLightingAtPointInstanced(
    float3                        hitPos,
    float3                        N,
    float3                        Ng,
    float3                        baseColor,
    float                         metallic,
    float                         roughness,
    float3                        V,
    const device BVHNode         *tlasNodes,
    const device BVHNode         *meshNodes,
    const device Triangle        *triangles,
    const device SceneInstanceGPU *instances,
    uint                          tlasNodeCount,
    uint                          meshNodeCount,
    uint                          instanceCount,
    int                           rootIndex,
    const device LightGPU        *lights,
    uint                          lightCount,
    uint                          baseSeed)
{
    float3 result = float3(0.0f);
    if (lights == nullptr || lightCount == 0u)
        return result;

    // Use geometric normal for shadow ray offset to avoid self-shadowing with normal maps
    float3 shadowOrigin = hitPos + Ng * SHADOW_EPS;

    float r = clamp(roughness, 0.05f, 0.95f);
    metallic = clamp(metallic, 0.0f, 1.0f);

    float3 F0 = lerp3(float3(0.04f), baseColor, metallic);

    float alpha = r * r;
    float NdV   = max(dot(N, V), 0.0f);

    float k = (r + 1.0f);
    k = (k * k) * 0.125f; // (r+1)^2 / 8

    for (uint i = 0u; i < lightCount; ++i)
    {
        const device LightGPU &light = lights[i];

        int   type    = light.type;
        float3 lp     = float3(light.position);
        float3 ld     = normalize(float3(light.direction));
        float  radius = light.radius;

        float3 L;
        float  dist      = 1.0f;
        bool   hasFinite = true;

        if (type == LIGHT_TYPE_DIRECTIONAL)
        {
            L         = -ld;
            hasFinite = false;
        }
        else
        {
            float3 toLightCenter = lp - hitPos;

            if (radius > 0.0f)
            {
                uint seed1 = baseSeed ^ (0x9E3779B9u * (i * 2u + 1u));
                uint seed2 = baseSeed ^ (0x9E3779B9u * (i * 2u + 2u));

                float u1 = rand01(seed1);
                float u2 = rand01(seed2);

                float z   = 1.0f - 2.0f * u1;
                float phi = 2.0f * PI * u2;
                float rxy = sqrt(max(0.0f, 1.0f - z * z));

                float3 offset = float3(
                    rxy * cos(phi),
                    rxy * sin(phi),
                    z
                ) * radius;

                float3 samplePos = lp + offset;
                float3 toLight   = samplePos - hitPos;
                dist = length(toLight);
                if (dist <= 0.0f)
                    continue;

                L  = toLight / dist;
                lp = samplePos;
            }
            else
            {
                dist = length(toLightCenter);
                if (dist <= 0.0f)
                    continue;
                L = toLightCenter / dist;
            }
        }

        float NdL = dot(N, L);
        if (NdL <= 0.0f)
            continue;

                float attenuation = 1.0f;
        if (hasFinite)
        {
            float falloffDist = max(dist * LIGHT_FALLOFF_DISTANCE_SCALE, 1e-3f);
            float r2 = falloffDist * falloffDist;
            if (r2 <= 0.0f)
                continue;

            // Базовый inverse-square
            attenuation = INV_4PI / r2;

            // Дополнительная форма затухания по AttenuationRadius
            if (light.attenuationRadius > 0.0f)
            {
                float R = light.attenuationRadius * LIGHT_ATTENUATION_RADIUS_SCALE;

                if (dist >= R)
                    continue;

                float s  = dist / R;
                float s2 = s * s;

                const float F = 1.0f; // тот же F, что и на CPU
                float num = 1.0f - s2;
                num *= num;           // (1 - s^2)^2
                float den = 1.0f + F * s2;

                float extra = (den > 0.0f) ? (num / den) : 0.0f;

                attenuation *= extra;
            }
        }

        float intensity = light.intensity * attenuation * LIGHT_EXPOSURE_GPU;

        if (type == LIGHT_TYPE_SPOT)
        {
            float  spotSize  = max(light.spotSize, 1e-4f);
            float  spotBlend = clamp01(light.spotBlend);

            float outerAngle = 0.5f * spotSize;
            float innerAngle = outerAngle * (1.0f - spotBlend);

            float cosOuter = cos(outerAngle);
            float cosInner = cos(innerAngle);

            float3 dirToPoint = normalize(hitPos - lp);
            float  cosTheta   = dot(ld, dirToPoint);

            if (cosTheta <= 0.0f)
                continue;

            float spotFactor = 0.0f;
            if (spotBlend <= 0.0f || cosInner <= cosOuter)
            {
                spotFactor = (cosTheta >= cosOuter) ? 1.0f : 0.0f;
            }
            else
            {
                if (cosTheta <= cosOuter)
                    spotFactor = 0.0f;
                else if (cosTheta >= cosInner)
                    spotFactor = 1.0f;
                else
                {
                    float t = (cosTheta - cosOuter) / (cosInner - cosOuter);
                    t = clamp01(t);
                    spotFactor = t * t;
                }
            }

            intensity *= spotFactor;
            if (intensity <= 0.0f)
                continue;
        }

        if (type == LIGHT_TYPE_AREA)
        {
            float3 dirFromLight = normalize(hitPos - lp);
            float  cosEmit      = dot(ld, dirFromLight);
            if (cosEmit <= 0.0f)
                continue;
            intensity *= cosEmit;
            if (intensity <= 0.0f)
                continue;
        }

        float maxDist = hasFinite ? (dist - SHADOW_EPS) : -1.0f;
        Ray shadowRay;
        shadowRay.origin = shadowOrigin;
        shadowRay.direction  = L;
        if (traceShadowSceneBVH(tlasNodes, meshNodes, triangles, instances,
                                tlasNodeCount, meshNodeCount, instanceCount,
                                rootIndex, shadowRay, maxDist))
        {
            continue;
        }

        float3 lightColor = float3(light.color) * intensity;

        float3 H    = normalize(L + V);
        float  NdH  = max(dot(N, H), 0.0f);
        float  HdV  = max(dot(V, H), 0.0f);
        NdL        = max(NdL, 0.0f);

        if (NdL <= 0.0f || NdH <= 0.0f || NdV <= 0.0f)
            continue;

        float3 F = fresnelSchlick(HdV, F0);
        float  D = distributionGGX(NdH, alpha);
        float  G = geometrySmith(NdV, NdL, k);

        float3 numerator = D * G * F;
        float  denom     = max(4.0f * NdV * NdL, 1e-4f);
        float3 spec      = numerator / denom;

        float3 kd      = (float3(1.0f) - F) * (1.0f - metallic);
        float3 diffuse = kd * baseColor * INV_PI;

        float3 brdf = diffuse + spec;

        result += lightColor * brdf * NdL;
    }

    return result;
}

// ======================
// Просчет пикселя: базовый (без материалов/текстур)
// ======================
inline float3 tracePathPixel(uint2                  gid,
                             uint2                  imgSize,
                             const device BVHNode  *bvhNodes,
                             const device Triangle *triangles,
                             uint                   nodeCount,
                             int                    rootIndex,
                             float3                 camPos,
                             float3                 camForward,
                             float3                 camUp,
                             float3                 camRight,
                             float                  fovY,
                             int                    samplesPerPixel,
                             const device LightGPU *lights,
                             uint                   lightCount,
                             uint                   sampleBaseIndex)
{
    const int   MAX_BOUNCES = 4;
    const float EPSILON_POS = SHADOW_EPS;

    uint w = imgSize.x;
    uint h = imgSize.y;
    if (gid.x >= w || gid.y >= h)
        return float3(0.0f);

    int spp = samplesPerPixel;
    if (spp <= 0) spp = 1;
    if (UV_DEBUG_MODE != 0) spp = 1; // stable debug


    int strataDim   = (int)floor(sqrt((float)spp));
    if (strataDim < 1) strataDim = 1;
    int strataCount = strataDim * strataDim;

    float3 pixelColor = float3(0.0f);

    for (int s = 0; s < spp; ++s)
    {
        uint sampleIndex = sampleBaseIndex + uint(s);
        uint seedBase = uint(gid.x) * 73856093u
                      ^ uint(gid.y) * 19349663u
                      ^ uint(s)     * 83492791u
                      ^ sampleIndex * 2654435761u;

        float jx = 0.0f;
        float jy = 0.0f;

        if (s < strataCount)
        {
            int sx = s % strataDim;
            int sy = s / strataDim;

            uint seedJx = seedBase ^ 0x1234u;
            uint seedJy = seedBase ^ 0x5678u;
            float u1 = rand01(seedJx);
            float u2 = rand01(seedJy);

            float subX = (float(sx) + u1) / float(strataDim);
            float subY = (float(sy) + u2) / float(strataDim);

            jx = subX - 0.5f;
            jy = subY - 0.5f;
        }
        else
        {
            uint seedJx = seedBase ^ 0xABCDEFu;
            uint seedJy = seedBase ^ 0x13579Bu;
            jx = rand01(seedJx) - 0.5f;
            jy = rand01(seedJy) - 0.5f;
        }

        float2 jitter = (UV_DEBUG_MODE != 0) ? float2(0.0f, 0.0f) : float2(jx, jy);

        Ray ray = makePrimaryRayJittered(
            int(gid.x), int(gid.y),
            int(w), int(h),
            jitter,
            camPos, camForward, camUp, camRight,
            fovY
        );

        float3 throughput = float3(1.0f);
        float3 radiance   = float3(0.0f);

        for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce)
        {
            HitInfo hit = traceRayBVH(bvhNodes,
                                      triangles,
                                      nodeCount,
                                      rootIndex,
                                      ray);
            if (!(hit.hit && hit.triIndex >= 0))
            {
                float3 env = environmentColor(normalize(ray.direction));
                radiance += throughput * env;
                break;
            }

            float3 hitPos    = hit.position;
            float3 N         = normalize(hit.normal);
            float3 Ng        = N;
            float3 Ns        = N;
            float  ao        = 1.0f;
            float  metallic  = clamp(hit.metallic, 0.0f, 1.0f);
            float  roughness = clamp(hit.roughness, 0.02f, 0.98f);

            float3 baseColor = hit.color;
            float3 emissive  = hit.emission * 3.0f;

            // --- UV DEBUG (modes 0..16) ---
            if (UV_DEBUG_MODE != 0 && bounce == 0)
            {
                float2 uvW = fract(hit.uv);
                float3 checker = uvChecker(uvW, 12.0f);

                switch (UV_DEBUG_MODE)
                {
                    case 1:  return checker;
                    case 2:
                        // UV checker as albedo WITH lighting (no emissive / no metal).
                        baseColor = checker;
                        emissive  = float3(0.0f);
                        metallic  = 0.0f;
                        roughness = 0.5f;
                        ao        = 1.0f;
                        break;
                    case 3:  return uvGradient(uvW);
                    case 4:  return hashColorFromInt(hit.materialIndex);
                    case 5:  return baseColor;
                    case 6:  return baseColor; // no textures in this kernel
                    case 7:  return float3(1.0f);
                    case 8:  return float3(roughness);
                    case 9:  return float3(metallic);
                    case 10: return normalToRGB(Ng);
                    case 11: return normalToRGB(Ns);
                    case 12: return float3(0.0f);
                    case 13: return emissive;
                    case 14: return float3(1.0f, roughness, metallic);
                    case 15: return float3(0.0f);
                    case 16: return float3(0.0f);
                    default: break;
                }
            }
            // --- UV DEBUG ---

            // Ensure normals face the incoming ray
            if (dot(Ng, ray.direction) > 0.0f) { Ng = -Ng; }
            if (dot(Ns, ray.direction) > 0.0f) { Ns = -Ns; }

            N = Ns;
            float3 V = normalize(-ray.direction);

            uint lightSeed = seedBase ^ (0x9E3779B9u * (uint(bounce) + 1u));

            float3 ambient = baseColor * AMBIENT_STRENGTH * ao;
            float3 direct  = computeLightingAtPoint(
                                hitPos, N, Ng, baseColor,
                                metallic, roughness, V,
                                bvhNodes, triangles,
                                nodeCount, rootIndex,
                                lights, lightCount,
                                lightSeed);

            if (bounce == 0)
            {
                radiance += throughput * (ambient + direct + emissive);
            }
            else
            {
                radiance += throughput * (direct + emissive);
            }

            if (bounce >= 1)
            {
                float maxChannel = max(throughput.x, max(throughput.y, throughput.z));
                maxChannel = clamp(maxChannel, 0.1f, 0.95f);

                uint  seedRR = seedBase ^ (0x10000u * uint(bounce) ^ 0x10u);
                float rr     = rand01(seedRR);

                if (rr > maxChannel)
                    break;

                throughput /= maxChannel;
            }

            uint  seedU1 = seedBase ^ (0x10000u * uint(bounce) ^ 0x21u);
            uint  seedU2 = seedBase ^ (0x10000u * uint(bounce) ^ 0x43u);
            float u1 = rand01(seedU1);
            float u2 = rand01(seedU2);

            float3 newDir;

            // --- BSDF sampling (diffuse + GGX specular) ---
            float r = clamp(roughness, 0.02f, 0.98f);
            float3 F0 = lerp3(float3(0.04f), baseColor, metallic);
            float NdV = max(dot(N, V), 0.0f);
            float3 Fv = fresnelSchlick(NdV, F0);

            // Specular lobe probability (avoid 0/1 extremes for stability)
            float specProb = clamp(max(F0.x, max(F0.y, F0.z)), 0.05f, 0.95f);
            float diffProb = 1.0f - specProb;

            uint  seedPick = seedBase ^ (0x10000u * uint(bounce) ^ 0x77u);
            float pick = rand01(seedPick);

            if (pick < specProb)
            {
                // GGX half-vector sampling
                float alpha  = r * r;
                float alpha2 = alpha * alpha;

                float phi       = 2.0f * PI * u2;
                float cosTheta  = sqrt((1.0f - u1) / (1.0f + (alpha2 - 1.0f) * u1));
                float sinTheta  = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));

                float3 tangent, bitangent;
                buildOrthonormalBasis(N, tangent, bitangent);

                float3 H =
                    tangent   * (sinTheta * cos(phi)) +
                    bitangent * (sinTheta * sin(phi)) +
                    N         * cosTheta;
                H = normalize(H);

                newDir = normalize(reflect(-V, H));

                float NdL = dot(N, newDir);
                if (NdL <= 0.0f)
                    break;

                float NdH = max(dot(N, H), 0.0f);
                float HdV = max(dot(H, V), 0.0f);

                // Geometry term (Schlick-Smith)
                float k = (r + 1.0f);
                k = (k * k) * 0.125f;

                float G = geometrySmith(max(NdV, 0.0f), max(NdL, 0.0f), k);
                float3 Fh = fresnelSchlick(HdV, F0);

                // Weight for specular component sampling:
                // w = (spec * NdL) / (specProb * pdf)
                // With GGX VNDF approx, cancellation yields:
                // w ≈ (G * F * HdV) / (specProb * NdV * NdH)
                float denom = max(NdV * NdH, 1e-6f);
                float3 wSpec = (G * Fh * HdV) / (specProb * denom);

                throughput *= wSpec;
            }
            else
            {
                // Cosine-weighted diffuse sampling
                float3 localDir = cosineSampleHemisphere(u1, u2);

                float3 tangent, bitangent;
                buildOrthonormalBasis(N, tangent, bitangent);

                newDir =
                    localDir.x * tangent +
                    localDir.y * bitangent +
                    localDir.z * N;
                newDir = normalize(newDir);

                float NdL = max(dot(N, newDir), 0.0f);
                if (NdL <= 0.0f)
                    break;

                // Energy-conserving diffuse term (Lambert * kd)
                float3 kd = (float3(1.0f) - Fv) * (1.0f - metallic);

                // Component-sampling weight simplifies to: kd * baseColor / diffProb
                throughput *= (kd * baseColor) / max(diffProb, 1e-6f);
            }

            ray.origin    = hitPos + newDir * EPSILON_POS;
            ray.direction = newDir;
        }

        pixelColor += radiance;
    }

    pixelColor /= float(spp);
    return pixelColor;
}

inline PathTraceTextureResult tracePathPixelTextured(uint2                  gid,
                                                     uint2                  imgSize,
                                                     const device BVHNode  *tlasNodes,
                                                     const device BVHNode  *meshNodes,
                                                     const device Triangle *triangles,
                                                     const device SceneInstanceGPU *instances,
                                                     uint                   tlasNodeCount,
                                                     uint                   meshNodeCount,
                                                     uint                   instanceCount,
                                                     int                    rootIndex,
                                                     float3                 camPos,
                                                     float3                 camForward,
                                                     float3                 camUp,
                                                     float3                 camRight,
                                                     float                  fovY,
                                                     int                    samplesPerPixel,
                                                     const device LightGPU *lights,
                                                     uint                   lightCount,
                                                     uint                   sampleBaseIndex,
                                                     const device MaterialGPU *materials,
                                                     uint                    materialCount,
                                                     const device MaterialGPU_PBR *materialsPBR,
                                                     uint                    materialPBRCount,
                                                     const device EmissiveTriangleGPU *emissiveTriangles,
                                                     uint                    emissiveTriangleCount,
                                                     const device DecalGPU   *decals,
                                                     uint                    decalCount,
                                                     const array<texture2d<float, access::sample>, MAX_BASE_TEX> sceneTextures)
{
    PathTraceTextureResult result;
    result.color = float3(0.0f);
    result.depth = 1.0e30f;
    result.albedo = float3(0.0f);
    result.hitMask = 0.0f;
    result.normal = float3(0.0f);
    result._pad0 = 0.0f;

    const int   MAX_BOUNCES = 4;
    const float EPSILON_POS = SHADOW_EPS;

    uint w = imgSize.x;
    uint h = imgSize.y;
    if (gid.x >= w || gid.y >= h)
        return result;

    int spp = samplesPerPixel;
    if (spp <= 0) spp = 1;
    if (UV_DEBUG_MODE != 0) spp = 1;

    int strataDim   = (int)floor(sqrt((float)spp));
    if (strataDim < 1) strataDim = 1;
    int strataCount = strataDim * strataDim;

    float3 pixelColor = float3(0.0f);

    for (int s = 0; s < spp; ++s)
    {
        uint sampleIndex = sampleBaseIndex + uint(s);
        uint seedBase = uint(gid.x) * 73856093u
                      ^ uint(gid.y) * 19349663u
                      ^ uint(s)     * 83492791u
                      ^ sampleIndex * 2654435761u;

        float jx = 0.0f;
        float jy = 0.0f;

        if (s < strataCount)
        {
            int sx = s % strataDim;
            int sy = s / strataDim;
            jx = (float(sx) + rand01(seedBase ^ 0x1234u)) / float(strataDim) - 0.5f;
            jy = (float(sy) + rand01(seedBase ^ 0x5678u)) / float(strataDim) - 0.5f;
        }
        else
        {
            jx = rand01(seedBase ^ 0xABCDEFu) - 0.5f;
            jy = rand01(seedBase ^ 0x13579Bu) - 0.5f;
        }

        float2 jitter = (UV_DEBUG_MODE != 0 || s == 0) ? float2(0.0f) : float2(jx, jy);
        Ray ray = makePrimaryRayJittered(int(gid.x), int(gid.y), int(w), int(h), jitter,
                                         camPos, camForward, camUp, camRight, fovY);

        float3 throughput = float3(1.0f);
        float3 radiance   = float3(0.0f);

        float  prevBsdfPdf = 0.0f;
        float3 prevSurfacePos = float3(0.0f);
        uint   prevWasBSDFSample = 0u;

        for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce)
        {
            HitInfo hit = traceRaySceneBVH(tlasNodes, meshNodes, triangles, instances, tlasNodeCount, meshNodeCount, instanceCount, rootIndex, ray);
            if (!(hit.hit && hit.triIndex >= 0))
            {
                radiance += throughput * environmentColor(normalize(ray.direction));
                break;
            }

            float3 hitPos = hit.position;
            float3 NgRaw = normalize(hit.normal);
            float3 N = NgRaw;

            float3 baseColor = hit.color;
            float3 emissive  = hit.emission * 3.0f;

            int    matId = hit.materialIndex;
            float2 uv    = fract(hit.uv);

            float  metallic  = clamp(hit.metallic, 0.0f, 1.0f);
            float  roughness = clamp(hit.roughness, 0.02f, 0.98f);
            float  ao        = 1.0f;

            float3 baseColorFactor     = baseColor;
            float3 baseColorTexOnly    = float3(1.0f);
            float3 emissivePreExposure = emissive;

            bool hasBaseColorTex = false;
            bool hasNormalTex    = false;
            bool hasORMTex       = false;
            bool hasRoughnessTex = false;
            bool hasMetallicTex  = false;

            float3 Ng = N;
            float3 Ns = N;

            bool hasMatPBR = (materialsPBR != nullptr) && (materialPBRCount > 0u) && (matId >= 0) && (uint(matId) < materialPBRCount);
            if (hasMatPBR)
            {
                MaterialGPU_PBR mp = materialsPBR[matId];
                if (mp.baseColorTexIndex >= 0 && uint(mp.baseColorTexIndex) < MAX_BASE_TEX)
                {
                    float3 texBC = sceneTextures[uint(mp.baseColorTexIndex)].sample(g_texSampler, uv).rgb;
                    baseColorTexOnly = texBC;
                    hasBaseColorTex  = true;
                    baseColor = baseColorFactor * texBC;
                }
                if (mp.emissionTexIndex >= 0 && uint(mp.emissionTexIndex) < MAX_BASE_TEX)
                {
                    float4 eTex = sceneTextures[uint(mp.emissionTexIndex)].sample(g_texSampler, uv);
                    if ((mp.flags & MATERIAL_FLAG_EMISSION_USE_ALPHA_MASK) != 0)
                        emissive *= sampleEmissiveMask(eTex);
                    else
                        emissive *= eTex.rgb;
                }
                if (mp.ormTexIndex >= 0 && uint(mp.ormTexIndex) < MAX_BASE_TEX)
                {
                    float3 orm = sceneTextures[uint(mp.ormTexIndex)].sample(g_texSampler, uv).rgb;
                    hasORMTex = true;
                    ao = clamp01(orm.r);
                    roughness = clamp(orm.g, 0.02f, 0.98f);
                    metallic = clamp01(orm.b);
                }
                if (mp.roughnessTexIndex >= 0 && uint(mp.roughnessTexIndex) < MAX_BASE_TEX)
                {
                    hasRoughnessTex = true;
                    roughness = clamp(sceneTextures[uint(mp.roughnessTexIndex)].sample(g_texSampler, uv).r, 0.02f, 0.98f);
                }
                if (mp.metallicTexIndex >= 0 && uint(mp.metallicTexIndex) < MAX_BASE_TEX)
                {
                    hasMetallicTex = true;
                    metallic = clamp01(sceneTextures[uint(mp.metallicTexIndex)].sample(g_texSampler, uv).r);
                }
                if (mp.occlusionTexIndex >= 0 && uint(mp.occlusionTexIndex) < MAX_BASE_TEX)
                {
                    ao = clamp01(sceneTextures[uint(mp.occlusionTexIndex)].sample(g_texSampler, uv).r);
                }
                Ng = N;
                if (mp.normalTexIndex >= 0 && uint(mp.normalTexIndex) < MAX_BASE_TEX)
                {
                    hasNormalTex = true;
                    Ns = sampleNormalMapInstanced(mp.normalTexIndex, uv, Ng, triangles, instances, hit.instanceIndex, hit.triIndex, sceneTextures);
                }
                else
                {
                    Ns = Ng;
                }
            }
            else
            {
                if (materials != nullptr && materialCount > 0u && matId >= 0 && uint(matId) < materialCount)
                {
                    int texId = materials[matId].baseColorTexIndex;
                    if (texId >= 0 && uint(texId) < MAX_BASE_TEX)
                    {
                        float3 texBC = sceneTextures[uint(texId)].sample(g_texSampler, uv).rgb;
                        baseColorTexOnly = texBC;
                        hasBaseColorTex = true;
                        baseColor = baseColorFactor * texBC;
                    }
                    int eTexId = materials[matId].emissionTexIndex;
                    if (eTexId >= 0 && uint(eTexId) < MAX_BASE_TEX)
                    {
                        emissive *= sceneTextures[uint(eTexId)].sample(g_texSampler, uv).rgb;
                    }
                }
                Ng = N;
                Ns = Ng;
            }

            emissivePreExposure = emissive;

            if (bounce == 0)
            {
                applyProjectedDecalsPrimary(hitPos, Ng, Ns, baseColor, roughness,
                                           decals, decalCount, sceneTextures);
                if (s == 0)
                {
                    result.depth = hit.t;
                    result.albedo = max(baseColor, float3(0.0f));
                    result.normal = normalize(Ns);
                    result.hitMask = 1.0f;
                }
            }

            if (UV_DEBUG_MODE != 0 && bounce == 0)
            {
                float2 uvW = uv;
                float3 checker = uvChecker(uvW, 12.0f);
                float3 bits15 = float3(hasMatPBR ? 1.0f : 0.0f, hasBaseColorTex ? 1.0f : 0.0f, hasNormalTex ? 1.0f : 0.0f);
                float3 bits16 = float3(hasORMTex ? 1.0f : 0.0f, hasRoughnessTex ? 1.0f : 0.0f, hasMetallicTex ? 1.0f : 0.0f);
                switch (UV_DEBUG_MODE)
                {
                    case 1: result.color = checker; return result;
                    case 2: baseColor = checker; emissive = float3(0.0f); emissivePreExposure = float3(0.0f); metallic = 0.0f; roughness = 0.5f; ao = 1.0f; break;
                    case 3: result.color = uvGradient(uvW); return result;
                    case 4: result.color = hashColorFromInt(matId); return result;
                    case 5: result.color = baseColor; return result;
                    case 6: result.color = baseColorTexOnly; return result;
                    case 7: result.color = float3(ao); return result;
                    case 8: result.color = float3(roughness); return result;
                    case 9: result.color = float3(metallic); return result;
                    case 10: result.color = normalToRGB(Ng); return result;
                    case 11: result.color = normalToRGB(Ns); return result;
                    case 12: { float d = clamp01(0.5f * length(Ns - Ng)); result.color = float3(d); return result; }
                    case 13: result.color = emissivePreExposure; return result;
                    case 14: result.color = float3(ao, roughness, metallic); return result;
                    case 15: result.color = bits15; return result;
                    case 16: result.color = bits16; return result;
                    default: break;
                }
            }

            emissive *= EMISSION_VISIBLE_EXPOSURE_GPU;

            if (dot(Ng, ray.direction) > 0.0f) Ng = -Ng;
            if (dot(Ns, ray.direction) > 0.0f) Ns = -Ns;
            N = Ns;
            if (bounce == 0 && s == 0)
            {
                result.depth = hit.t;
                result.albedo = max(baseColor, float3(0.0f));
                result.normal = normalize(N);
                result.hitMask = 1.0f;
            }
            float3 V = normalize(-ray.direction);

            uint lightSeed = seedBase ^ (0x9E3779B9u * (uint(bounce) + 1u));
            float3 directExplicit = computeLightingAtPointInstanced(hitPos, N, Ng, baseColor, metallic, roughness, V,
                                                           tlasNodes, meshNodes, triangles, instances, tlasNodeCount, meshNodeCount, instanceCount, rootIndex,
                                                           lights, lightCount, lightSeed);
            float3 directEmissive = sampleDirectEmissiveLightingTexturedInstanced(hitPos, N, Ng, V, uint(hit.triIndex), uint(hit.instanceIndex),
                                                                         baseColor, metallic, roughness,
                                                                         tlasNodes, meshNodes, triangles, instances, tlasNodeCount, meshNodeCount, instanceCount, rootIndex,
                                                                         materials, materialCount,
                                                                         materialsPBR, materialPBRCount,
                                                                         sceneTextures,
                                                                         emissiveTriangles, emissiveTriangleCount,
                                                                         seedBase ^ (0x7000u + uint(bounce) * 17u),
                                                                         seedBase ^ (0x7001u + uint(bounce) * 17u),
                                                                         seedBase ^ (0x7002u + uint(bounce) * 17u));

            float3 emitted = emissive;
            if (bounce > 0 && prevWasBSDFSample != 0u && luminance3(emitted) > 1e-6f)
            {
                float pdfLight = emissiveLightPdfForHit(prevSurfacePos, hitPos, NgRaw, uint(hit.triIndex),
                                                        emissiveTriangles, emissiveTriangleCount);
                if (pdfLight > 1e-8f)
                {
                    float mis = powerHeuristic(prevBsdfPdf, pdfLight);
                    emitted *= mis;
                }
            }

            radiance += throughput * (directExplicit + directEmissive + emitted);

            if (bounce >= 1)
            {
                float maxChannel = max(throughput.x, max(throughput.y, throughput.z));
                maxChannel = clamp(maxChannel, 0.1f, 0.95f);
                if (rand01(seedBase ^ (0x10000u * uint(bounce) ^ 0x10u)) > maxChannel)
                    break;
                throughput /= maxChannel;
            }

            BSDFSample bsdfSample = sampleSurfaceBSDF(N, V, baseColor, metallic, roughness,
                                                      rand01(seedBase ^ (0x10000u * uint(bounce) ^ 0x21u)),
                                                      rand01(seedBase ^ (0x10000u * uint(bounce) ^ 0x43u)),
                                                      rand01(seedBase ^ (0x10000u * uint(bounce) ^ 0x77u)));
            if (bsdfSample.valid == 0u)
                break;

            throughput *= bsdfSample.weight;
            prevWasBSDFSample = 1u;
            prevBsdfPdf = bsdfSample.pdf;
            prevSurfacePos = hitPos;

            ray.origin = hitPos + bsdfSample.direction * EPSILON_POS;
            ray.direction = bsdfSample.direction;
        }

        pixelColor += radiance;
    }

    pixelColor /= float(spp);
    result.color = pixelColor;
    return result;
}



// ==================================================
// KERNEL 1: вывод в буфер framebuffer
// ==================================================
kernel void RayTraceKernel(
    const device BVHNode   *bvhNodes      [[buffer(0)]],
    const device Triangle  *triangles     [[buffer(1)]],
    constant uint          *nodeCountPtr  [[buffer(2)]],
    constant int           *rootIndexPtr  [[buffer(3)]],
    constant CameraData    *camPtr        [[buffer(4)]],
    constant uint2         *imageSizePtr  [[buffer(5)]],
    constant uint          *triCountPtr   [[buffer(6)]],
    device packed_float3   *framebuffer   [[buffer(7)]],
    const device LightGPU  *lights        [[buffer(8)]],
    constant uint          *lightCountPtr [[buffer(9)]],
    uint2                   gid           [[thread_position_in_grid]])
{
    uint2 imgSize = *imageSizePtr;
    uint  w = imgSize.x;
    uint  h = imgSize.y;

    if (gid.x >= w || gid.y >= h)
        return;

    const uint pixelIndex = gid.y * w + gid.x;

    uint nodeCount = *nodeCountPtr;
    int  rootIndex = *rootIndexPtr;
    CameraData cam = *camPtr;

    if (nodeCount == 0u || rootIndex < 0 || rootIndex >= int(nodeCount))
    {
        framebuffer[pixelIndex] = packed_float3(0.0f, 0.0f, 0.0f);
        return;
    }

    float3 camPos     = float3(cam.position);
    float3 camForward = normalize(float3(cam.forward));
    float3 camUp      = normalize(float3(cam.up));
    float3 camRight   = normalize(float3(cam.right));

    int spp = cam.samplesPerPixel;
    if (spp <= 0) spp = 1;

    uint lightCount = (lightCountPtr != nullptr) ? *lightCountPtr : 0u;

    float3 linearColor = tracePathPixel(
        gid, imgSize,
        bvhNodes, triangles,
        nodeCount, rootIndex,
        camPos, camForward, camUp, camRight,
        cam.fovY,
        spp,
        lights, lightCount,
        0u
    );

    float3 c = filmicTonemap(linearColor, IMAGE_EXPOSURE);
    framebuffer[pixelIndex] = packed_float3(c);
}


// ==================================================
// Post-process params + helpers
// ==================================================
struct PostProcessParams
{
    float exposure;
    float bloomIntensity;
    float bloomThreshold;
    float vignetteIntensity;

    float chromaticAberration;
    float filmGrainIntensity;
    float filmSlope;
    float filmToe;

    float filmShoulder;
    float filmBlackClip;
    float filmWhiteClip;
    float fogDensity;

    float fogHeightFalloff;
    float fogScatteringG;
    float fogColorX;
    float fogColorY;

    float fogColorZ;
    float fogExtinctionScale;
    float fogAlbedoX;
    float fogAlbedoY;

    float fogAlbedoZ;
    float volumetricFog;
    float nearPlane;
    float farPlane;

    float time;
    float width;
    float height;
    float _pad0;

    float colorSaturationX;
    float colorSaturationY;
    float colorSaturationZ;
    float shadowLift;
};

inline float hash11(float n)
{
    return fract(sin(n) * 43758.5453123f);
}

inline float noise2d(uint2 p, float time)
{
    float n = float(p.x) * 12.9898f + float(p.y) * 78.233f + time * 37.719f;
    return hash11(n);
}

inline float3 sampleCombined(texture2d<float, access::sample> hdrTex,
                             texture2d<float, access::sample> bloomTex,
                             float2 uv)
{
    (void)bloomTex;
    uv = clamp(uv, float2(0.0f), float2(1.0f));
    return hdrTex.sample(g_postSampler, uv).rgb;
}

inline float computeFogAmount(float depth,
                              float2 uv,
                              PostProcessParams pp)
{
    float nearP = max(pp.nearPlane, 1e-4f);
    float farP  = max(pp.farPlane, nearP + 1e-3f);
    float depthN = clamp((depth - nearP) / (farP - nearP), 0.0f, 1.0f);

    float density = max(pp.fogDensity, 0.0f) * max(pp.fogExtinctionScale, 0.0f);
    float distFog = 1.0f - exp(-depthN * density * 3.0f);

    float heightT = clamp(1.0f - uv.y, 0.0f, 1.0f);
    float heightFog = exp(-max(pp.fogHeightFalloff, 0.0f) * heightT * 1.1f);

    float fog = distFog * mix(1.0f, heightFog, clamp(pp.fogHeightFalloff, 0.0f, 1.0f));
    return clamp(fog, 0.0f, 0.68f);
}

inline float3 applyColorSaturation(float3 c, float3 sat)
{
    float l = luminance3(c);
    return max(float3(0.0f), float3(l) + (c - float3(l)) * sat);
}

inline float3 filmicTonemapCustom(float3 c,
                                  PostProcessParams pp)
{
    c *= max(pp.exposure, 0.001f);

    float l0 = luminance3(max(c, float3(0.0f)));
    float shadowMask = 1.0f - smoothstep(0.02f, 0.40f, l0);
    c += float3(pp.shadowLift * shadowMask);

    c = max(c - pp.filmBlackClip, float3(0.0f));
    c *= max(pp.filmSlope, 0.05f);

    const float a  = 2.35f + pp.filmShoulder * 0.22f;
    const float b  = 0.028f + pp.filmBlackClip * 0.05f;
    const float c1 = 2.43f;
    const float d  = 0.64f + pp.filmToe * 0.12f;
    const float e  = max(0.025f, 0.16f - pp.filmWhiteClip * 0.42f);

    float3 x = (c * (a * c + b)) / (c * (c1 * c + d) + e);
    x = clamp(x, 0.0f, 1.0f);
    x = applyColorSaturation(x, float3(pp.colorSaturationX, pp.colorSaturationY, pp.colorSaturationZ));
    return pow(x, float3(1.0f / 2.18f));
}

inline float tracePrimaryDepth(uint2 gid,
                               uint2 imgSize,
                               const device BVHNode  *tlasNodes,
                               const device BVHNode  *meshNodes,
                               const device Triangle *triangles,
                               const device SceneInstanceGPU *instances,
                               uint tlasNodeCount,
                               uint meshNodeCount,
                               uint instanceCount,
                               int rootIndex,
                               float3 camPos,
                               float3 camForward,
                               float3 camUp,
                               float3 camRight,
                               float fovY,
                               float fallbackDepth)
{
    Ray primary = makePrimaryRayJittered(int(gid.x), int(gid.y),
                                         int(imgSize.x), int(imgSize.y),
                                         float2(0.0f),
                                         camPos, camForward, camUp, camRight,
                                         fovY);
    HitInfo hit = traceRaySceneBVH(tlasNodes, meshNodes, triangles, instances, tlasNodeCount, meshNodeCount, instanceCount, rootIndex, primary);
    return hit.hit ? hit.t : fallbackDepth;
}

// ==================================================
// KERNEL 2: path tracing -> HDR accumulation + first-hit guides
// depth remains packed into hdr.a; albedo/normal are written to dedicated guide textures
// ==================================================
kernel void RayTraceTextureKernel(
    const device BVHNode   *bvhNodes      [[buffer(0)]],
    const device Triangle  *triangles     [[buffer(1)]],
    constant uint          *nodeCountPtr  [[buffer(2)]],
    constant int           *rootIndexPtr  [[buffer(3)]],
    constant CameraData    *camPtr        [[buffer(4)]],
    constant uint2         *imageSizePtr  [[buffer(5)]],
    constant uint          *triCountPtr   [[buffer(6)]],
    const device LightGPU  *lights        [[buffer(7)]],
    constant uint          *lightCountPtr [[buffer(8)]],
    constant uint          *sampleCountPtr [[buffer(9)]],
    texture2d<float, access::read_write> accumTex [[texture(0)]],
    texture2d<float, access::write>      hdrTex   [[texture(1)]],
    texture2d<float, access::write>      albedoTex [[texture(2)]],
    texture2d<float, access::write>      normalTex [[texture(3)]],
    const device MaterialGPU         *materials                  [[buffer(10)]],
    constant uint                    *materialCountPtr           [[buffer(11)]],
    const device BVHNode             *meshNodes                  [[buffer(12)]],
    const device SceneInstanceGPU    *instances                  [[buffer(13)]],
    const device MaterialGPU_PBR     *materialsPBR               [[buffer(14)]],
    constant uint                    *materialPBRCountPtr        [[buffer(15)]],
    const device EmissiveTriangleGPU *emissiveTriangles          [[buffer(16)]],
    constant uint                    *emissiveTriangleCountPtr   [[buffer(17)]],
    const device DecalGPU            *decals                     [[buffer(18)]],
    constant uint                    *decalCountPtr              [[buffer(19)]],
    constant uint                    *meshNodeCountPtr           [[buffer(20)]],
    constant uint                    *instanceCountPtr           [[buffer(21)]],
    array<texture2d<float, access::sample>, MAX_BASE_TEX> sceneTextures [[texture(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint2 imgSize = *imageSizePtr;
    uint  w = imgSize.x;
    uint  h = imgSize.y;

    if (gid.x >= w || gid.y >= h)
        return;

    uint nodeCount = *nodeCountPtr;
    uint meshNodeCount = (meshNodeCountPtr != nullptr) ? *meshNodeCountPtr : 0u;
    uint instanceCount = (instanceCountPtr != nullptr) ? *instanceCountPtr : 0u;
    int  rootIndex = *rootIndexPtr;
    CameraData cam = *camPtr;

    if (nodeCount == 0 || meshNodeCount == 0 || instanceCount == 0 || rootIndex < 0 || rootIndex >= int(nodeCount))
    {
        hdrTex.write(float4(0.0, 0.0, 0.0, 1.0e30f), gid);
        return;
    }

    float3 camPos     = float3(cam.position);
    float3 camForward = normalize(float3(cam.forward));
    float3 camUp      = normalize(float3(cam.up));
    float3 camRight   = normalize(float3(cam.right));

    int spp = cam.samplesPerPixel;
    if (spp <= 0) spp = 1;

    uint lightCount = (lightCountPtr != nullptr) ? *lightCountPtr : 0u;
    uint sampleBaseIndex = (sampleCountPtr != nullptr) ? *sampleCountPtr : 0u;
    uint materialCount = (materialCountPtr != nullptr) ? *materialCountPtr : 0u;
    uint materialPBRCount = (materialPBRCountPtr != nullptr) ? *materialPBRCountPtr : 0u;
    uint emissiveTriangleCount = (emissiveTriangleCountPtr != nullptr) ? *emissiveTriangleCountPtr : 0u;
    uint decalCount = (decalCountPtr != nullptr) ? *decalCountPtr : 0u;

    PathTraceTextureResult pathResult = tracePathPixelTextured(
        gid, imgSize,
        bvhNodes, meshNodes, triangles, instances,
        nodeCount, meshNodeCount, instanceCount, rootIndex,
        camPos, camForward, camUp, camRight,
        cam.fovY,
        spp,
        lights, lightCount,
        sampleBaseIndex,
        materials,
        materialCount,
        materialsPBR,
        materialPBRCount,
        emissiveTriangles,
        emissiveTriangleCount,
        decals,
        decalCount,
        sceneTextures);
    float3 frameColor = pathResult.color;
    float depthValue = pathResult.depth;
    uint samplesThisDispatch = (UV_DEBUG_MODE != 0) ? 0u : uint(max(spp, 1));

    frameColor = min(frameColor, float3(32.0f));
    albedoTex.write(float4(pathResult.albedo, pathResult.hitMask), gid);
    normalTex.write(float4(pathResult.normal, pathResult.hitMask), gid);

    const uint stampNow = uint(UV_DEBUG_MODE) & 0xFFu;
    if (UV_DEBUG_MODE != 0)
    {
        accumTex.write(float4(0.0f, 0.0f, 0.0f, as_type<float>(packStampCount(stampNow, 0u))), gid);
        hdrTex.write(float4(debugGamma(frameColor), depthValue), gid);
        return;
    }

    float4 prev = accumTex.read(gid);
    uint  prevPacked = as_type<uint>(prev.w);
    uint  stampPrev  = 0u;
    uint  countPrev  = 0u;
    unpackStampCount(prevPacked, stampPrev, countPrev);

    float3 prevAvg = prev.xyz;
    if (stampPrev != stampNow)
    {
        countPrev = 0u;
        prevAvg   = float3(0.0f);
    }

    float3 newAvg = prevAvg;
    uint   countNew = countPrev;
    if (countPrev == 0u)
    {
        newAvg   = frameColor;
        countNew = min(samplesThisDispatch, 0x00FFFFFFu);
    }
    else
    {
        uint samplesToAccumulate = min(samplesThisDispatch, 0x00FFFFFFu - countPrev);
        if (samplesToAccumulate > 0u)
        {
            newAvg   = (prevAvg * float(countPrev) + frameColor * float(samplesToAccumulate))
                     / float(countPrev + samplesToAccumulate);
            countNew = countPrev + samplesToAccumulate;
        }
    }

    accumTex.write(float4(newAvg, as_type<float>(packStampCount(stampNow, countNew))), gid);

    float3 colorForOutput = newAvg;
    if (DENOISE_ENABLE)
    {
        float3 center = newAvg;
        float3 sum    = float3(0.0f);
        float  wsum   = 0.0f;

        float denomSpace = 2.0f * DENOISE_SIGMA_SPACE * DENOISE_SIGMA_SPACE;
        float wS1 = (denomSpace > 0.0f) ? exp(-1.0f / denomSpace) : 1.0f;
        float wS2 = (denomSpace > 0.0f) ? exp(-2.0f / denomSpace) : 1.0f;

        float denomColor = 2.0f * DENOISE_SIGMA_COLOR * DENOISE_SIGMA_COLOR;
        bool  useColorExp = (denomColor > 0.0f);

        for (int dy = -1; dy <= 1; ++dy)
        {
            int ny = int(gid.y) + dy;
            if (ny < 0 || ny >= int(h)) continue;

            for (int dx = -1; dx <= 1; ++dx)
            {
                int nx = int(gid.x) + dx;
                if (nx < 0 || nx >= int(w)) continue;

                uint2  ng = uint2(nx, ny);
                float3 c  = (dx == 0 && dy == 0) ? center : accumTex.read(ng).xyz;

                int dist2i = dx * dx + dy * dy;
                float wSpatial = (dist2i == 0) ? 1.0f : ((dist2i == 1) ? wS1 : wS2);

                float3 diff       = c - center;
                float  colorDist2 = dot(diff, diff);
                float  wColor     = useColorExp ? exp(-colorDist2 / denomColor) : 1.0f;

                float  w = wSpatial * wColor;
                sum  += c * w;
                wsum += w;
            }
        }

        if (wsum > 0.0f)
        {
            float3 blurred = sum / wsum;
            colorForOutput = lerp3(blurred, center, DENOISE_BLEND_FACTOR);
        }
    }

    hdrTex.write(float4(max(colorForOutput, float3(0.0f)), depthValue), gid);
}

// ==================================================
// KERNEL 3: bloom extract
// ==================================================
kernel void BloomExtractKernel(
    constant PostProcessParams       *ppPtr     [[buffer(0)]],
    texture2d<float, access::sample>  hdrTex    [[texture(0)]],
    texture2d<float, access::write>   bloomOut  [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    const PostProcessParams pp = *ppPtr;
    if (gid.x >= uint(pp.width) || gid.y >= uint(pp.height))
        return;

    float2 uv = (float2(gid) + 0.5f) / float2(pp.width, pp.height);
    float3 hdr = hdrTex.sample(g_postSampler, uv).rgb;
    float lum = luminance3(hdr);

    float threshold = max(0.18f, 0.78f + 0.42f * pp.bloomThreshold);
    float knee = max(0.12f, threshold * 0.40f);
    float soft = clamp((lum - threshold + knee) / (2.0f * knee), 0.0f, 1.0f);
    soft = soft * soft * (3.0f - 2.0f * soft);

    float3 bright = hdr * soft;
    bloomOut.write(float4(bright, 1.0f), gid);
}

// ==================================================
// KERNEL 4: bloom blur horizontal
// ==================================================
kernel void BloomBlurKernelH(
    constant PostProcessParams       *ppPtr   [[buffer(0)]],
    texture2d<float, access::sample>  srcTex  [[texture(0)]],
    texture2d<float, access::write>   dstTex  [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    const PostProcessParams pp = *ppPtr;
    if (gid.x >= uint(pp.width) || gid.y >= uint(pp.height))
        return;

    float2 uv = (float2(gid) + 0.5f) / float2(pp.width, pp.height);
    float2 texel = float2(1.0f / max(pp.width, 1.0f), 0.0f);
    const float w0 = 0.227027f;
    const float w1 = 0.1945946f;
    const float w2 = 0.1216216f;
    const float w3 = 0.054054f;
    const float w4 = 0.016216f;

    float3 c = srcTex.sample(g_postSampler, uv).rgb * w0;
    c += srcTex.sample(g_postSampler, uv + texel * 1.0f).rgb * w1;
    c += srcTex.sample(g_postSampler, uv - texel * 1.0f).rgb * w1;
    c += srcTex.sample(g_postSampler, uv + texel * 2.0f).rgb * w2;
    c += srcTex.sample(g_postSampler, uv - texel * 2.0f).rgb * w2;
    c += srcTex.sample(g_postSampler, uv + texel * 3.0f).rgb * w3;
    c += srcTex.sample(g_postSampler, uv - texel * 3.0f).rgb * w3;
    c += srcTex.sample(g_postSampler, uv + texel * 4.0f).rgb * w4;
    c += srcTex.sample(g_postSampler, uv - texel * 4.0f).rgb * w4;

    dstTex.write(float4(c, 1.0f), gid);
}

// ==================================================
// KERNEL 5: bloom blur vertical
// ==================================================
kernel void BloomBlurKernelV(
    constant PostProcessParams       *ppPtr   [[buffer(0)]],
    texture2d<float, access::sample>  srcTex  [[texture(0)]],
    texture2d<float, access::write>   dstTex  [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    const PostProcessParams pp = *ppPtr;
    if (gid.x >= uint(pp.width) || gid.y >= uint(pp.height))
        return;

    float2 uv = (float2(gid) + 0.5f) / float2(pp.width, pp.height);
    float2 texel = float2(0.0f, 1.0f / max(pp.height, 1.0f));
    const float w0 = 0.227027f;
    const float w1 = 0.1945946f;
    const float w2 = 0.1216216f;
    const float w3 = 0.054054f;
    const float w4 = 0.016216f;

    float3 c = srcTex.sample(g_postSampler, uv).rgb * w0;
    c += srcTex.sample(g_postSampler, uv + texel * 1.0f).rgb * w1;
    c += srcTex.sample(g_postSampler, uv - texel * 1.0f).rgb * w1;
    c += srcTex.sample(g_postSampler, uv + texel * 2.0f).rgb * w2;
    c += srcTex.sample(g_postSampler, uv - texel * 2.0f).rgb * w2;
    c += srcTex.sample(g_postSampler, uv + texel * 3.0f).rgb * w3;
    c += srcTex.sample(g_postSampler, uv - texel * 3.0f).rgb * w3;
    c += srcTex.sample(g_postSampler, uv + texel * 4.0f).rgb * w4;
    c += srcTex.sample(g_postSampler, uv - texel * 4.0f).rgb * w4;

    dstTex.write(float4(c, 1.0f), gid);
}

// ==================================================
// KERNEL 6: final post-process / fog / tonemap (reads depth from hdr.a)
// ==================================================
kernel void PostProcessKernel(
    constant PostProcessParams       *ppPtr   [[buffer(0)]],
    texture2d<float, access::sample>  hdrTex   [[texture(0)]],
    texture2d<float, access::sample>  bloomTex [[texture(1)]],
    texture2d<float, access::write>   outTex   [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const PostProcessParams pp = *ppPtr;
    if (gid.x >= uint(pp.width) || gid.y >= uint(pp.height))
        return;

    float2 uv = (float2(gid) + 0.5f) / float2(pp.width, pp.height);

    if (UV_DEBUG_MODE != 0)
    {
        outTex.write(float4(hdrTex.sample(g_postSampler, uv).rgb, 1.0f), gid);
        return;
    }

    float2 centered = uv - 0.5f;
    float2 caOffset = centered * (pp.chromaticAberration * 0.0025f);

    float3 cR = sampleCombined(hdrTex, bloomTex, uv + caOffset);
    float3 cG = sampleCombined(hdrTex, bloomTex, uv);
    float3 cB = sampleCombined(hdrTex, bloomTex, uv - caOffset);
    float3 color = float3(cR.r, cG.g, cB.b);

    float depth = hdrTex.sample(g_postSampler, uv).a;
    bool hasSceneHit = (depth < min(pp.farPlane * 0.999f, 1.0e29f));
    float fogAmount = hasSceneHit ? computeFogAmount(depth, uv, pp) : 0.0f;
    float3 fogColor = float3(pp.fogColorX, pp.fogColorY, pp.fogColorZ);
    float3 fogAlbedo = float3(pp.fogAlbedoX, pp.fogAlbedoY, pp.fogAlbedoZ);

    if (pp.fogDensity > 0.0f && hasSceneHit)
    {
        float bloomL = luminance3(bloomTex.sample(g_postSampler, uv).rgb);
        float volumetricBoost = (pp.volumetricFog > 0.5f)
                              ? min(0.12f, 0.045f * bloomL * (0.65f + max(pp.fogScatteringG, 0.0f)))
                              : 0.0f;
        float3 fogLit = fogColor * fogAlbedo * (1.0f + volumetricBoost);
        color = mix(color, fogLit, fogAmount);
    }

    float3 bloom = bloomTex.sample(g_postSampler, uv).rgb;
    color += bloom * max(pp.bloomIntensity, 0.0f);

    float3 outColor = filmicTonemapCustom(color, pp);

    float dist = length(centered) * 1.41421356f;
    float vignette = 1.0f - clamp(pp.vignetteIntensity, 0.0f, 1.0f) * smoothstep(0.2f, 1.0f, dist);
    outColor *= clamp(vignette, 0.0f, 1.0f);

    if (pp.filmGrainIntensity > 0.0f)
    {
        float grain = (noise2d(gid, pp.time) - 0.5f) * pp.filmGrainIntensity * 0.06f;
        outColor = clamp(outColor + grain, 0.0f, 1.0f);
    }

    outTex.write(float4(outColor, 1.0f), gid);
}
