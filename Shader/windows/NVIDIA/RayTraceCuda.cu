#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>
#include <cuda_fp16.h>

#define PI               3.14159265358979323846f
#define INV_PI           0.31830988618379067154f
#define INV_4PI          0.07957747154594766788f
#define AMBIENT_STRENGTH 0.11f
#define SHADOW_EPS       1e-3f
#define IMAGE_EXPOSURE   0.75f

#define LIGHT_EXPOSURE_GPU    3.8f
#define EMISSION_EXPOSURE_GPU 0.95f
#define ENV_INTENSITY_GPU     0.70f

#define DENOISE_ENABLE       true
#define DENOISE_SIGMA_SPACE  1.0f
#define DENOISE_SIGMA_COLOR  0.20f
#define DENOISE_BLEND_FACTOR 0.40f

// ======================
// Структуры (аналогичные Metal)
// ======================

struct AABB
{
    float3 v0; // min
    float3 v1; // max
};

struct Triangle
{
    float3 v0;
    float3 v1;
    float3 v2;
    float3 normal;
    AABB   ABoBa;

    float3 color;
    float3 emission;
    float metallic;
    float roughness;
    float _pad0;
    float _pad1;
};

struct BVHNode
{
    AABB box;
    int  left;
    int  right;
    int  parent;
    int  tri;
};

struct CameraData
{
    float3 position;
    float3 forward;
    float3 up;
    float3 right;

    float fovY;
    int   width;
    int   height;
    int   samplesPerPixel;

    float nearPlane;
    float farPlane;
    float focusDistance;
    float aspectRatio;
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
    float  t;
    float3 position;
    float3 normal;
    float3 color;
    float3 emission;

    float  metallic;
    float  roughness;
};

// Типы источников света
#define LIGHT_TYPE_POINT       0
#define LIGHT_TYPE_DIRECTIONAL 1
#define LIGHT_TYPE_SPOT        2
#define LIGHT_TYPE_AREA        3

struct LightGPU
{
    int   type;
    int   _pad0;

    float3 position;
    float3 direction;
    float3 color;

    float intensity;
    float radius;
    float spotSize;
    float spotBlend;
};

static_assert(sizeof(float3) == 12, "float3 must be 12 bytes (3 floats)");
static_assert(sizeof(AABB) == 24, "AABB size mismatch between CPU and CUDA");
static_assert(sizeof(Triangle) == 112, "Triangle size mismatch between CPU and CUDA");
static_assert(sizeof(CameraData) == 80, "CameraData size mismatch between CPU and CUDA");
static_assert(sizeof(LightGPU) == 60, "LightGPU size mismatch between CPU and CUDA");

// ======================
// Вспомогательные функции CUDA
// ======================

__device__ __inline__ float clamp01(float x)
{
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ __inline__ float3 filmicTonemap(float3 c, float exposure)
{
    c *= exposure;

    const float a  = 2.51f;
    const float b  = 0.03f;
    const float c1 = 2.43f;
    const float d  = 0.59f;
    const float e  = 0.14f;

    float3 x = (c * (a * c + b)) / (c * (c1 * c + d) + e);
    x = make_float3(
        fminf(fmaxf(x.x, 0.0f), 1.0f),
        fminf(fmaxf(x.y, 0.0f), 1.0f),
        fminf(fmaxf(x.z, 0.0f), 1.0f)
    );

    x = make_float3(
        powf(x.x, 1.0f / 2.2f),
        powf(x.y, 1.0f / 2.2f),
        powf(x.z, 1.0f / 2.2f)
    );
    return x;
}

__device__ __inline__ float3 cosineSampleHemisphere(float u1, float u2)
{
    float r   = sqrtf(u1);
    float phi = 2.0f * PI * u2;

    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

    return make_float3(x, y, z);
}

__device__ __inline__ void buildOrthonormalBasis(float3 n, float3* tangent, float3* bitangent)
{
    float3 N = make_float3(n.x, n.y, n.z);
    float length_N = sqrtf(N.x * N.x + N.y * N.y + N.z * N.z);
    if (length_N > 0.0f) {
        N.x /= length_N; N.y /= length_N; N.z /= length_N;
    }

    float3 helper;
    if (fabsf(N.z) < 0.999f) {
        helper = make_float3(0.0f, 0.0f, 1.0f);
    } else {
        helper = make_float3(1.0f, 0.0f, 0.0f);
    }

    *tangent = make_float3(
        helper.y * N.z - helper.z * N.y,
        helper.z * N.x - helper.x * N.z,
        helper.x * N.y - helper.y * N.x
    );

    float length_T = sqrtf(tangent->x * tangent->x + tangent->y * tangent->y + tangent->z * tangent->z);
    if (length_T > 0.0f) {
        tangent->x /= length_T; tangent->y /= length_T; tangent->z /= length_T;
    }

    *bitangent = make_float3(
        N.y * tangent->z - N.z * tangent->y,
        N.z * tangent->x - N.x * tangent->z,
        N.x * tangent->y - N.y * tangent->x
    );
}

__device__ __inline__ float3 lerp3(float3 a, float3 b, float t)
{
    return make_float3(
        a.x + t * (b.x - a.x),
        a.y + t * (b.y - a.y),
        a.z + t * (b.z - a.z)
    );
}

__device__ __inline__ float3 environmentColor(float3 dir)
{
    float t = clamp01(dir.y * 0.5f + 0.5f);

    float3 ground = make_float3(0.05f, 0.05f, 0.06f);
    float3 skyBot = make_float3(0.08f, 0.08f, 0.10f);
    float3 skyTop = make_float3(0.14f, 0.13f, 0.17f);

    float3 sky  = lerp3(skyBot, skyTop, t);
    float3 base = (dir.y >= 0.0f) ? sky : lerp3(ground, sky, 0.5f);

    return make_float3(base.x * ENV_INTENSITY_GPU,
                      base.y * ENV_INTENSITY_GPU,
                      base.z * ENV_INTENSITY_GPU);
}

// ======================
// PBR функции
// ======================

__device__ __inline__ float3 fresnelSchlick(float cosTheta, float3 F0)
{
    float factor = powf(1.0f - cosTheta, 5.0f);
    return make_float3(
        F0.x + (1.0f - F0.x) * factor,
        F0.y + (1.0f - F0.y) * factor,
        F0.z + (1.0f - F0.z) * factor
    );
}

__device__ __inline__ float distributionGGX(float NdH, float alpha)
{
    float a2    = alpha * alpha;
    float denom = NdH * NdH * (a2 - 1.0f) + 1.0f;
    denom       = PI * denom * denom;
    return a2 / fmaxf(denom, 1e-7f);
}

__device__ __inline__ float geometrySchlickGGX(float NdV, float k)
{
    return NdV / (NdV * (1.0f - k) + k);
}

__device__ __inline__ float geometrySmith(float NdV, float NdL, float k)
{
    float g1 = geometrySchlickGGX(NdV, k);
    float g2 = geometrySchlickGGX(NdL, k);
    return g1 * g2;
}

__device__ __inline__ float gaussianWeight(float dist2, float sigma)
{
    float denom = 2.0f * sigma * sigma;
    if (denom <= 0.0f) return 1.0f;
    return expf(-dist2 / denom);
}

// ======================
// Генерация лучей
// ======================

__device__ __inline__ Ray makePrimaryRayJittered(int px, int py,
                                                int width, int height,
                                                float2 jitter,
                                                float3 camPos,
                                                float3 camForward,
                                                float3 camUp,
                                                float3 camRight,
                                                float fovY)
{
    float fx = (float(px) + 0.5f + jitter.x);
    float fy = (float(py) + 0.5f + jitter.y);

    float ndcX = (fx / float(width))  * 2.0f - 1.0f;
    float ndcY = (fy / float(height)) * 2.0f - 1.0f;
    ndcY = -ndcY;

    float aspect     = float(width) / float(height);
    float tanHalfFov = tanf(0.5f * fovY);

    float3 dirCam = make_float3(
        ndcX * aspect * tanHalfFov,
        ndcY * tanHalfFov,
        -1.0f
    );

    float3 dirWorld = make_float3(
        dirCam.x * camRight.x + dirCam.y * camUp.x - dirCam.z * camForward.x,
        dirCam.x * camRight.y + dirCam.y * camUp.y - dirCam.z * camForward.y,
        dirCam.x * camRight.z + dirCam.y * camUp.z - dirCam.z * camForward.z
    );

    float length_dir = sqrtf(dirWorld.x * dirWorld.x + dirWorld.y * dirWorld.y + dirWorld.z * dirWorld.z);
    if (length_dir > 0.0f) {
        dirWorld.x /= length_dir;
        dirWorld.y /= length_dir;
        dirWorld.z /= length_dir;
    }

    Ray r;
    r.origin    = camPos;
    r.direction = dirWorld;
    return r;
}

// ======================
// Пересечения
// ======================

__device__ __inline__ bool processAxis(float origin, float dir,
                                      float minB, float maxB,
                                      float* tEnter,
                                      float* tExit)
{
    const float eps = 1e-8f;
    if (fabsf(dir) < eps)
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

__device__ __inline__ bool intersectTriangle(const Ray ray,
                                            const Triangle tri,
                                            float* tHitOut)
{
    const float EPS = 1e-6f;

    float3 v0 = tri.v0;
    float3 v1 = tri.v1;
    float3 v2 = tri.v2;

    float3 edge1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 edge2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

    float3 pvec = make_float3(
        ray.direction.y * edge2.z - ray.direction.z * edge2.y,
        ray.direction.z * edge2.x - ray.direction.x * edge2.z,
        ray.direction.x * edge2.y - ray.direction.y * edge2.x
    );

    float det = edge1.x * pvec.x + edge1.y * pvec.y + edge1.z * pvec.z;

    if (fabsf(det) < EPS)
        return false;

    float invDet = 1.0f / det;

    float3 tvec = make_float3(ray.origin.x - v0.x, ray.origin.y - v0.y, ray.origin.z - v0.z);

    float u = (tvec.x * pvec.x + tvec.y * pvec.y + tvec.z * pvec.z) * invDet;
    if (u < 0.0f || u > 1.0f)
        return false;

    float3 qvec = make_float3(
        tvec.y * edge1.z - tvec.z * edge1.y,
        tvec.z * edge1.x - tvec.x * edge1.z,
        tvec.x * edge1.y - tvec.y * edge1.x
    );

    float v = (ray.direction.x * qvec.x + ray.direction.y * qvec.y + ray.direction.z * qvec.z) * invDet;
    if (v < 0.0f || u + v > 1.0f)
        return false;

    float t = (edge2.x * qvec.x + edge2.y * qvec.y + edge2.z * qvec.z) * invDet;
    if (t <= EPS)
        return false;

    *tHitOut = t;
    return true;
}

__device__ __inline__ bool intersectAABBWithRay(const AABB box,
                                               const Ray ray,
                                               float* tEnter,
                                               float* tExit)
{
    const float tMinInit = 0.0f;
    const float tMaxInit = CUDART_INF_F;

    *tEnter = tMinInit;
    *tExit  = tMaxInit;

    float3 bmin = box.v0;
    float3 bmax = box.v1;

    float3 o = ray.origin;
    float3 d = ray.direction;

    if (!processAxis(o.x, d.x, bmin.x, bmax.x, tEnter, tExit)) return false;
    if (!processAxis(o.y, d.y, bmin.y, bmax.y, tEnter, tExit)) return false;
    if (!processAxis(o.z, d.z, bmin.z, bmax.z, tEnter, tExit)) return false;

    if (*tExit < 0.0f)
        return false;

    return true;
}

// ======================
// Обход BVH
// ======================

__device__ __inline__ HitInfo traceRayBVH(const BVHNode* nodes,
                                         const Triangle* tris,
                                         uint nodeCount,
                                         int rootIndex,
                                         const Ray ray)
{
    HitInfo result;
    result.hit      = false;
    result.triIndex = -1;
    result.t        = CUDART_INF_F;
    result.position = make_float3(0.0f);
    result.normal   = make_float3(0.0f);
    result.color    = make_float3(1.0f);
    result.emission = make_float3(0.0f);
    result.metallic = 0.0f;
    result.roughness = 0.0f;

    if (nodeCount == 0 || rootIndex < 0 || rootIndex >= int(nodeCount))
    {
        return result;
    }

    struct StackEntry
    {
        int   index;
        float tEnter;
        float tExit;
    };

    const int MAX_STACK = 64;
    StackEntry stack[MAX_STACK];
    int sp = 0;

    float rootEnter = 0.0f;
    float rootExit  = 0.0f;
    if (!intersectAABBWithRay(nodes[rootIndex].box, ray, &rootEnter, &rootExit))
    {
        return result;
    }

    stack[sp++] = StackEntry{ rootIndex, rootEnter, rootExit };
    float bestT = CUDART_INF_F;

    while (sp > 0)
    {
        StackEntry entry = stack[--sp];

        if (entry.tEnter > bestT)
            continue;

        int nodeIndex = entry.index;
        if (nodeIndex < 0 || nodeIndex >= int(nodeCount))
            continue;

        const BVHNode& node = nodes[nodeIndex];

        if (node.tri >= 0)
        {
            float tHit = 0.0f;
            if (intersectTriangle(ray, tris[node.tri], &tHit) &&
                tHit >= 0.0f && tHit < bestT)
            {
                bestT           = tHit;
                result.hit      = true;
                result.triIndex = node.tri;
                result.t        = tHit;

                result.position = make_float3(
                    ray.origin.x + ray.direction.x * tHit,
                    ray.origin.y + ray.direction.y * tHit,
                    ray.origin.z + ray.direction.z * tHit
                );

                float3 normal = tris[node.tri].normal;
                float length_n = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
                if (length_n > 0.0f) {
                    normal.x /= length_n; normal.y /= length_n; normal.z /= length_n;
                }
                result.normal = normal;

                result.color    = tris[node.tri].color;
                result.emission = tris[node.tri].emission;
                result.metallic  = tris[node.tri].metallic;
                result.roughness = tris[node.tri].roughness;
            }
            continue;
        }

        struct ChildInfo
        {
            int   index;
            float tEnter;
            float tExit;
        };

        ChildInfo children[2];
        int childCount = 0;

        if (node.left >= 0)
        {
            float tL0, tL1;
            if (intersectAABBWithRay(nodes[node.left].box, ray, &tL0, &tL1))
            {
                if (tL1 >= 0.0f && tL0 <= bestT)
                {
                    children[childCount].index  = node.left;
                    children[childCount].tEnter = tL0;
                    children[childCount].tExit  = tL1;
                    ++childCount;
                }
            }
        }

        if (node.right >= 0)
        {
            float tR0, tR1;
            if (intersectAABBWithRay(nodes[node.right].box, ray, &tR0, &tR1))
            {
                if (tR1 >= 0.0f && tR0 <= bestT)
                {
                    children[childCount].index  = node.right;
                    children[childCount].tEnter = tR0;
                    children[childCount].tExit  = tR1;
                    ++childCount;
                }
            }
        }

        if (childCount == 0)
            continue;

        if (childCount == 2 && children[0].tEnter < children[1].tEnter)
        {
            ChildInfo tmp = children[0];
            children[0]   = children[1];
            children[1]   = tmp;
        }

        for (int i = 0; i < childCount; ++i)
        {
            if (children[i].tEnter > bestT)
                continue;

            if (sp < MAX_STACK)
            {
                stack[sp++] = StackEntry{
                    children[i].index,
                    children[i].tEnter,
                    children[i].tExit
                };
            }
        }
    }

    return result;
}

// ======================
// Теневые лучи
// ======================

__device__ __inline__ bool isShadowed(const BVHNode* bvhNodes,
                                     const Triangle* triangles,
                                     uint nodeCount,
                                     int rootIndex,
                                     float3 origin,
                                     float3 dir,
                                     float maxDist)
{
    Ray shadowRay;
    shadowRay.origin    = origin;
    shadowRay.direction = dir;

    HitInfo sh = traceRayBVH(bvhNodes, triangles, nodeCount, rootIndex, shadowRay);
    if (!sh.hit)
        return false;

    if (sh.t <= 1e-4f)
        return false;

    if (maxDist > 0.0f)
        return sh.t < maxDist;
    else
        return true;
}

// ======================
// Псевдо-рандом для CUDA
// ======================

__device__ __inline__ float rand01(uint* seed)
{
    *seed ^= (*seed << 13);
    *seed ^= (*seed >> 17);
    *seed ^= (*seed << 5);
    return (float)(*seed) / 4294967295.0f;
}

// ======================
// Освещение
// ======================

__device__ __inline__ float3 computeLightingAtPoint(
    float3                 hitPos,
    float3                 N,
    float3                 baseColor,
    float                  metallic,
    float                  roughness,
    float3                 V,
    const BVHNode*         bvhNodes,
    const Triangle*        triangles,
    uint                   nodeCount,
    int                    rootIndex,
    const LightGPU*        lights,
    uint                   lightCount,
    uint*                  seed)
{
    float3 result = make_float3(0.0f);
    if (lights == nullptr || lightCount == 0u)
        return result;

    float3 shadowOrigin = make_float3(
        hitPos.x + N.x * SHADOW_EPS,
        hitPos.y + N.y * SHADOW_EPS,
        hitPos.z + N.z * SHADOW_EPS
    );

    float r = clamp01(roughness);
    metallic = clamp01(metallic);

    float3 F0 = lerp3(make_float3(0.04f, 0.04f, 0.04f), baseColor, metallic);

    float alpha = r * r;
    float NdV   = fmaxf(N.x * V.x + N.y * V.y + N.z * V.z, 0.0f);

    float k = (r + 1.0f);
    k = (k * k) * 0.125f;

    for (uint i = 0u; i < lightCount; ++i)
    {
        const LightGPU& light = lights[i];

        int   type    = light.type;
        float3 lp     = light.position;
        float3 ld     = light.direction;
        float ld_length = sqrtf(ld.x * ld.x + ld.y * ld.y + ld.z * ld.z);
        if (ld_length > 0.0f) {
            ld.x /= ld_length; ld.y /= ld_length; ld.z /= ld_length;
        }
        float  radius = light.radius;

        float3 L;
        float  dist      = 1.0f;
        bool   hasFinite = true;

        if (type == LIGHT_TYPE_DIRECTIONAL)
        {
            L = make_float3(-ld.x, -ld.y, -ld.z);
            hasFinite = false;
        }
        else
        {
            float3 toLightCenter = make_float3(lp.x - hitPos.x, lp.y - hitPos.y, lp.z - hitPos.z);

            if (radius > 0.0f)
            {
                float u1 = rand01(seed);
                float u2 = rand01(seed);

                float z   = 1.0f - 2.0f * u1;
                float phi = 2.0f * PI * u2;
                float rxy = sqrtf(fmaxf(0.0f, 1.0f - z * z));

                float3 offset = make_float3(
                    rxy * cosf(phi),
                    rxy * sinf(phi),
                    z
                ) * radius;

                float3 samplePos = make_float3(lp.x + offset.x, lp.y + offset.y, lp.z + offset.z);
                float3 toLight   = make_float3(samplePos.x - hitPos.x, samplePos.y - hitPos.y, samplePos.z - hitPos.z);
                dist = sqrtf(toLight.x * toLight.x + toLight.y * toLight.y + toLight.z * toLight.z);
                if (dist <= 0.0f)
                    continue;

                L = make_float3(toLight.x / dist, toLight.y / dist, toLight.z / dist);
                lp = samplePos;
            }
            else
            {
                dist = sqrtf(toLightCenter.x * toLightCenter.x + toLightCenter.y * toLightCenter.y + toLightCenter.z * toLightCenter.z);
                if (dist <= 0.0f)
                    continue;
                L = make_float3(toLightCenter.x / dist, toLightCenter.y / dist, toLightCenter.z / dist);
            }
        }

        float NdL = N.x * L.x + N.y * L.y + N.z * L.z;
        if (NdL <= 0.0f)
            continue;

        float attenuation = 1.0f;
        if (hasFinite)
        {
            float r2 = dist * dist;
            if (r2 <= 0.0f)
                continue;
            attenuation = INV_4PI / r2;
        }

        float intensity = light.intensity * attenuation * LIGHT_EXPOSURE_GPU;

        if (type == LIGHT_TYPE_SPOT)
        {
            float  spotSize  = fmaxf(light.spotSize, 1e-4f);
            float  spotBlend = clamp01(light.spotBlend);

            float outerAngle = 0.5f * spotSize;
            float innerAngle = outerAngle * (1.0f - spotBlend);

            float cosOuter = cosf(outerAngle);
            float cosInner = cosf(innerAngle);

            float3 dirToPoint = make_float3(hitPos.x - lp.x, hitPos.y - lp.y, hitPos.z - lp.z);
            float length_dirToPoint = sqrtf(dirToPoint.x * dirToPoint.x + dirToPoint.y * dirToPoint.y + dirToPoint.z * dirToPoint.z);
            if (length_dirToPoint > 0.0f) {
                dirToPoint.x /= length_dirToPoint; dirToPoint.y /= length_dirToPoint; dirToPoint.z /= length_dirToPoint;
            }

            float cosTheta = ld.x * dirToPoint.x + ld.y * dirToPoint.y + ld.z * dirToPoint.z;

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
            float3 dirFromLight = make_float3(hitPos.x - lp.x, hitPos.y - lp.y, hitPos.z - lp.z);
            float length_dirFromLight = sqrtf(dirFromLight.x * dirFromLight.x + dirFromLight.y * dirFromLight.y + dirFromLight.z * dirFromLight.z);
            if (length_dirFromLight > 0.0f) {
                dirFromLight.x /= length_dirFromLight; dirFromLight.y /= length_dirFromLight; dirFromLight.z /= length_dirFromLight;
            }

            float cosEmit = ld.x * dirFromLight.x + ld.y * dirFromLight.y + ld.z * dirFromLight.z;
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

        float3 lightColor = make_float3(light.color.x * intensity,
                                       light.color.y * intensity,
                                       light.color.z * intensity);

        float3 H = make_float3(L.x + V.x, L.y + V.y, L.z + V.z);
        float length_H = sqrtf(H.x * H.x + H.y * H.y + H.z * H.z);
        if (length_H > 0.0f) {
            H.x /= length_H; H.y /= length_H; H.z /= length_H;
        }

        float NdH  = fmaxf(N.x * H.x + N.y * H.y + N.z * H.z, 0.0f);
        float HdV  = fmaxf(V.x * H.x + V.y * H.y + V.z * H.z, 0.0f);
        NdL        = fmaxf(NdL, 0.0f);

        if (NdL <= 0.0f || NdH <= 0.0f || NdV <= 0.0f)
            continue;

        float3 F = fresnelSchlick(HdV, F0);
        float  D = distributionGGX(NdH, alpha);
        float  G = geometrySmith(NdV, NdL, k);

        float3 numerator = make_float3(D * G * F.x, D * G * F.y, D * G * F.z);
        float  denom     = fmaxf(4.0f * NdV * NdL, 1e-4f);
        float3 spec      = make_float3(numerator.x / denom, numerator.y / denom, numerator.z / denom);

        float3 kd      = make_float3((1.0f - F.x) * (1.0f - metallic),
                                    (1.0f - F.y) * (1.0f - metallic),
                                    (1.0f - F.z) * (1.0f - metallic));
        float3 diffuse = make_float3(kd.x * baseColor.x * INV_PI,
                                    kd.y * baseColor.y * INV_PI,
                                    kd.z * baseColor.z * INV_PI);

        float3 brdf = make_float3(diffuse.x + spec.x, diffuse.y + spec.y, diffuse.z + spec.z);

        result = make_float3(
            result.x + lightColor.x * brdf.x * NdL,
            result.y + lightColor.y * brdf.y * NdL,
            result.z + lightColor.z * brdf.z * NdL
        );
    }

    return result;
}

// ======================
// Трассировка пути
// ======================

__device__ __inline__ float3 tracePathPixel(
    uint2                  gid,
    uint2                  imgSize,
    const BVHNode*         bvhNodes,
    const Triangle*        triangles,
    uint                   nodeCount,
    int                    rootIndex,
    float3                 camPos,
    float3                 camForward,
    float3                 camUp,
    float3                 camRight,
    float                  fovY,
    int                    samplesPerPixel,
    const LightGPU*        lights,
    uint                   lightCount,
    uint                   frameIndex)
{
    const int   MAX_BOUNCES = 4;
    const float EPSILON_POS = SHADOW_EPS;

    uint w = imgSize.x;
    uint h = imgSize.y;
    if (gid.x >= w || gid.y >= h)
        return make_float3(0.0f);

    int spp = samplesPerPixel;
    if (spp <= 0) spp = 1;

    int strataDim   = (int)floorf(sqrtf((float)spp));
    if (strataDim < 1) strataDim = 1;
    int strataCount = strataDim * strataDim;

    float3 pixelColor = make_float3(0.0f);

    for (int s = 0; s < spp; ++s)
    {
        uint seedBase = gid.x * 73856093u
                      ^ gid.y * 19349663u
                      ^ (uint)s * 83492791u
                      ^ frameIndex * 2654435761u;

        float jx = 0.0f;
        float jy = 0.0f;

        if (s < strataCount)
        {
            int sx = s % strataDim;
            int sy = s / strataDim;

            uint seedJx = seedBase ^ 0x1234u;
            uint seedJy = seedBase ^ 0x5678u;
            float u1 = rand01(&seedJx);
            float u2 = rand01(&seedJy);

            float subX = (float(sx) + u1) / float(strataDim);
            float subY = (float(sy) + u2) / float(strataDim);

            jx = subX - 0.5f;
            jy = subY - 0.5f;
        }
        else
        {
            uint seedJx = seedBase ^ 0xABCDEFu;
            uint seedJy = seedBase ^ 0x13579Bu;
            jx = rand01(&seedJx) - 0.5f;
            jy = rand01(&seedJy) - 0.5f;
        }

        float2 jitter = make_float2(jx, jy);

        Ray ray = makePrimaryRayJittered(
            (int)gid.x, (int)gid.y,
            (int)w, (int)h,
            jitter,
            camPos, camForward, camUp, camRight,
            fovY
        );

        float3 throughput = make_float3(1.0f);
        float3 radiance   = make_float3(0.0f);

        for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce)
        {
            HitInfo hit = traceRayBVH(bvhNodes, triangles, nodeCount, rootIndex, ray);
            if (!(hit.hit && hit.triIndex >= 0))
            {
                float3 env = environmentColor(make_float3(
                    ray.direction.x, ray.direction.y, ray.direction.z
                ));
                radiance = make_float3(
                    radiance.x + throughput.x * env.x,
                    radiance.y + throughput.y * env.y,
                    radiance.z + throughput.z * env.z
                );
                break;
            }

            float3 hitPos    = hit.position;
            float3 N         = hit.normal;
            float3 baseColor = hit.color;
            float3 emissive  = make_float3(
                hit.emission.x * EMISSION_EXPOSURE_GPU,
                hit.emission.y * EMISSION_EXPOSURE_GPU,
                hit.emission.z * EMISSION_EXPOSURE_GPU
            );

            if ((N.x * ray.direction.x + N.y * ray.direction.y + N.z * ray.direction.z) > 0.0f)
            {
                N = make_float3(-N.x, -N.y, -N.z);
            }

            float metallic  = clamp01(hit.metallic);
            float roughness = clamp01(hit.roughness);
            float3 V        = make_float3(-ray.direction.x, -ray.direction.y, -ray.direction.z);
            float length_V = sqrtf(V.x * V.x + V.y * V.y + V.z * V.z);
            if (length_V > 0.0f) {
                V.x /= length_V; V.y /= length_V; V.z /= length_V;
            }

            uint lightSeed = seedBase ^ (0x9E3779B9u * ((uint)bounce + 1u));

            float3 ambient = make_float3(
                baseColor.x * AMBIENT_STRENGTH,
                baseColor.y * AMBIENT_STRENGTH,
                baseColor.z * AMBIENT_STRENGTH
            );

            float3 direct  = computeLightingAtPoint(
                                hitPos, N, baseColor,
                                metallic, roughness, V,
                                bvhNodes, triangles,
                                nodeCount, rootIndex,
                                lights, lightCount,
                                &lightSeed);

            if (bounce == 0)
            {
                float3 contrib = make_float3(
                    ambient.x + direct.x + emissive.x,
                    ambient.y + direct.y + emissive.y,
                    ambient.z + direct.z + emissive.z
                );
                radiance = make_float3(
                    radiance.x + throughput.x * contrib.x,
                    radiance.y + throughput.y * contrib.y,
                    radiance.z + throughput.z * contrib.z
                );
            }
            else
            {
                float3 contrib = make_float3(direct.x + emissive.x, direct.y + emissive.y, direct.z + emissive.z);
                radiance = make_float3(
                    radiance.x + throughput.x * contrib.x,
                    radiance.y + throughput.y * contrib.y,
                    radiance.z + throughput.z * contrib.z
                );
            }

            if (bounce >= 1)
            {
                float maxChannel = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
                maxChannel = clamp01(maxChannel);

                uint  seedRR = seedBase ^ (0x10000u * (uint)bounce ^ 0x10u);
                float rr     = rand01(&seedRR);

                if (rr > maxChannel)
                    break;

                throughput = make_float3(
                    throughput.x / maxChannel,
                    throughput.y / maxChannel,
                    throughput.z / maxChannel
                );
            }

            uint  seedU1 = seedBase ^ (0x10000u * (uint)bounce ^ 0x21u);
            uint  seedU2 = seedBase ^ (0x10000u * (uint)bounce ^ 0x43u);
            float u1 = rand01(&seedU1);
            float u2 = rand01(&seedU2);

            float3 newDir;

            if (metallic < 0.5f)
            {
                float3 localDir = cosineSampleHemisphere(u1, u2);

                float3 tangent, bitangent;
                buildOrthonormalBasis(N, &tangent, &bitangent);

                newDir = make_float3(
                    localDir.x * tangent.x + localDir.y * bitangent.x + localDir.z * N.x,
                    localDir.x * tangent.y + localDir.y * bitangent.y + localDir.z * N.y,
                    localDir.x * tangent.z + localDir.y * bitangent.z + localDir.z * N.z
                );

                float length_newDir = sqrtf(newDir.x * newDir.x + newDir.y * newDir.y + newDir.z * newDir.z);
                if (length_newDir > 0.0f) {
                    newDir.x /= length_newDir; newDir.y /= length_newDir; newDir.z /= length_newDir;
                }

                float3 albedo = make_float3(
                    baseColor.x * (1.0f - metallic),
                    baseColor.y * (1.0f - metallic),
                    baseColor.z * (1.0f - metallic)
                );
                throughput = make_float3(
                    throughput.x * albedo.x,
                    throughput.y * albedo.y,
                    throughput.z * albedo.z
                );
            }
            else
            {
                float glossRough = clamp01(roughness);
                float a          = glossRough;
                float a2         = a * a;

                float phi       = 2.0f * PI * u2;
                float cosTheta  = sqrtf((1.0f - u1) / (1.0f + (a2 - 1.0f) * u1));
                float sinTheta  = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

                float3 tangent, bitangent;
                buildOrthonormalBasis(N, &tangent, &bitangent);

                float3 H = make_float3(
                    tangent.x * (sinTheta * cosf(phi)) +
                    bitangent.x * (sinTheta * sinf(phi)) +
                    N.x * cosTheta,
                    tangent.y * (sinTheta * cosf(phi)) +
                    bitangent.y * (sinTheta * sinf(phi)) +
                    N.y * cosTheta,
                    tangent.z * (sinTheta * cosf(phi)) +
                    bitangent.z * (sinTheta * sinf(phi)) +
                    N.z * cosTheta
                );

                float length_H = sqrtf(H.x * H.x + H.y * H.y + H.z * H.z);
                if (length_H > 0.0f) {
                    H.x /= length_H; H.y /= length_H; H.z /= length_H;
                }

                float3 Rdir = make_float3(
                    ray.direction.x - 2.0f * (ray.direction.x * H.x + ray.direction.y * H.y + ray.direction.z * H.z) * H.x,
                    ray.direction.y - 2.0f * (ray.direction.x * H.x + ray.direction.y * H.y + ray.direction.z * H.z) * H.y,
                    ray.direction.z - 2.0f * (ray.direction.x * H.x + ray.direction.y * H.y + ray.direction.z * H.z) * H.z
                );

                float length_Rdir = sqrtf(Rdir.x * Rdir.x + Rdir.y * Rdir.y + Rdir.z * Rdir.z);
                if (length_Rdir > 0.0f) {
                    Rdir.x /= length_Rdir; Rdir.y /= length_Rdir; Rdir.z /= length_Rdir;
                }
                newDir = Rdir;

                if ((newDir.x * N.x + newDir.y * N.y + newDir.z * N.z) <= 0.0f)
                    break;

                float3 specColor = lerp3(make_float3(0.04f, 0.04f, 0.04f), baseColor, metallic);
                specColor = make_float3(
                    fminf(fmaxf(specColor.x, 0.0f), 1.0f),
                    fminf(fmaxf(specColor.y, 0.0f), 1.0f),
                    fminf(fmaxf(specColor.z, 0.0f), 1.0f)
                );
                throughput = make_float3(
                    throughput.x * specColor.x,
                    throughput.y * specColor.y,
                    throughput.z * specColor.z
                );
            }

            ray.origin = make_float3(
                hitPos.x + newDir.x * EPSILON_POS,
                hitPos.y + newDir.y * EPSILON_POS,
                hitPos.z + newDir.z * EPSILON_POS
            );
            ray.direction = newDir;
        }

        pixelColor = make_float3(
            pixelColor.x + radiance.x,
            pixelColor.y + radiance.y,
            pixelColor.z + radiance.z
        );
    }

    pixelColor = make_float3(
        pixelColor.x / float(spp),
        pixelColor.y / float(spp),
        pixelColor.z / float(spp)
    );
    return pixelColor;
}

// ======================
// CUDA Ядра
// ======================

__global__ void RayTraceKernel(
    const BVHNode*   bvhNodes,
    const Triangle*  triangles,
    const uint*      nodeCountPtr,
    const int*       rootIndexPtr,
    const CameraData* camPtr,
    const uint2*     imageSizePtr,
    const uint*      triCountPtr,
    float3*          framebuffer,
    const LightGPU*  lights,
    const uint*      lightCountPtr)
{
    uint2 gid = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                          blockIdx.y * blockDim.y + threadIdx.y);

    uint2 imgSize = *imageSizePtr;
    uint w = imgSize.x;
    uint h = imgSize.y;

    if (gid.x >= w || gid.y >= h)
        return;

    const uint pixelIndex = gid.y * w + gid.x;

    uint nodeCount = *nodeCountPtr;
    int rootIndex = *rootIndexPtr;
    CameraData cam = *camPtr;

    if (nodeCount == 0 || rootIndex < 0 || rootIndex >= (int)nodeCount)
    {
        framebuffer[pixelIndex] = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }

    float3 camPos     = cam.position;
    float3 camForward = cam.forward;
    float length_camForward = sqrtf(camForward.x * camForward.x + camForward.y * camForward.y + camForward.z * camForward.z);
    if (length_camForward > 0.0f) {
        camForward.x /= length_camForward; camForward.y /= length_camForward; camForward.z /= length_camForward;
    }

    float3 camUp = cam.up;
    float length_camUp = sqrtf(camUp.x * camUp.x + camUp.y * camUp.y + camUp.z * camUp.z);
    if (length_camUp > 0.0f) {
        camUp.x /= length_camUp; camUp.y /= length_camUp; camUp.z /= length_camUp;
    }

    float3 camRight = cam.right;
    float length_camRight = sqrtf(camRight.x * camRight.x + camRight.y * camRight.y + camRight.z * camRight.z);
    if (length_camRight > 0.0f) {
        camRight.x /= length_camRight; camRight.y /= length_camRight; camRight.z /= length_camRight;
    }

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
    framebuffer[pixelIndex] = c;
}

// ============================================================================
// Host-side (C++) API for SceneRTXTester — CUDA renderer bridge
// ============================================================================

#include <cuda_runtime.h>
#include <iostream>

// Глобальные переменные для прогрессивного рендера
static float3 *g_accumBuffer = nullptr;
static uint32_t g_frameIndex = 0;
static int g_accumWidth = 0;
static int g_accumHeight = 0;

// ============================================================================
// Сброс накопительного буфера
// ============================================================================
extern "C" void CUDA_ResetAccumulation_C()
{
    if (g_accumBuffer)
    {
        cudaFree(g_accumBuffer);
        g_accumBuffer = nullptr;
    }
    g_frameIndex = 0;
    g_accumWidth = 0;
    g_accumHeight = 0;
}

// ============================================================================
// Однократный рендер кадра (без накопления)
// ============================================================================
extern "C" bool CUDA_RenderFrame_C(
    const void *nodes,
    std::size_t nodeCount,
    const void *triangles,
    std::size_t triCount,
    const void *lights,
    std::size_t lightCount,
    int rootIndex,
    const void *camera,
    void *framebuffer)
{
    if (!nodes || !triangles || !camera || !framebuffer)
    {
        std::cerr << "CUDA_RenderFrame_C: null pointer\n";
        return false;
    }

    const CameraData *camCPU = reinterpret_cast<const CameraData *>(camera);
    int width = camCPU->width;
    int height = camCPU->height;
    std::size_t fbSize = static_cast<std::size_t>(width) * height * sizeof(float3);

    BVHNode *d_nodes = nullptr;
    Triangle *d_tris = nullptr;
    LightGPU *d_lights = nullptr;
    uint32_t *d_nodeCount = nullptr;
    int *d_rootIndex = nullptr;
    CameraData *d_camera = nullptr;
    uint2 *d_imageSize = nullptr;
    uint32_t *d_triCount = nullptr;
    uint32_t *d_lightCount = nullptr;
    float3 *d_fb = nullptr;

    cudaError_t err;

#define CUDA_CHECK(x)                                                       \
    do                                                                      \
    {                                                                       \
        err = (x);                                                          \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; \
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

    CUDA_CHECK(cudaMalloc(&d_nodes, nodeCount * sizeof(BVHNode)));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes, nodeCount * sizeof(BVHNode), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_tris, triCount * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(d_tris, triangles, triCount * sizeof(Triangle), cudaMemcpyHostToDevice));

    if (lightCount > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_lights, lightCount * sizeof(LightGPU)));
        CUDA_CHECK(cudaMemcpy(d_lights, lights, lightCount * sizeof(LightGPU), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_nodeCount, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_nodeCount, &nodeCount, sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_rootIndex, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_rootIndex, &rootIndex, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_camera, sizeof(CameraData)));
    CUDA_CHECK(cudaMemcpy(d_camera, camera, sizeof(CameraData), cudaMemcpyHostToDevice));

    uint2 imgSize = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    CUDA_CHECK(cudaMalloc(&d_imageSize, sizeof(uint2)));
    CUDA_CHECK(cudaMemcpy(d_imageSize, &imgSize, sizeof(uint2), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_triCount, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_triCount, &triCount, sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_lightCount, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_lightCount, &lightCount, sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_fb, fbSize));

    // Сетка и блоки
    dim3 block(8, 8, 1);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y, 1);

    // Запуск ядра
    RayTraceKernel<<<grid, block>>>(
        d_nodes, d_tris, d_nodeCount, d_rootIndex,
        d_camera, d_imageSize, d_triCount, d_fb,
        d_lights, d_lightCount);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(framebuffer, d_fb, fbSize, cudaMemcpyDeviceToHost));

cleanup:
    cudaFree(d_nodes);
    cudaFree(d_tris);
    cudaFree(d_lights);
    cudaFree(d_nodeCount);
    cudaFree(d_rootIndex);
    cudaFree(d_camera);
    cudaFree(d_imageSize);
    cudaFree(d_triCount);
    cudaFree(d_lightCount);
    cudaFree(d_fb);

#undef CUDA_CHECK
    return (err == cudaSuccess);
}

// ============================================================================
// Прогрессивный рендер (с накоплением)
// ============================================================================
extern "C" bool CUDA_RenderFrameAccum_C(
    const void *nodes,
    std::size_t nodeCount,
    const void *triangles,
    std::size_t triCount,
    const void *lights,
    std::size_t lightCount,
    int rootIndex,
    const void *camera,
    void *framebuffer)
{
    if (!nodes || !triangles || !camera || !framebuffer)
    {
        std::cerr << "CUDA_RenderFrameAccum_C: null pointer\n";
        return false;
    }

    const CameraData *camCPU = reinterpret_cast<const CameraData *>(camera);
    int width = camCPU->width;
    int height = camCPU->height;
    std::size_t fbSize = static_cast<std::size_t>(width) * height * sizeof(float3);

    // Выделяем или ресайзим накопительный буфер
    if (!g_accumBuffer || width != g_accumWidth || height != g_accumHeight)
    {
        if (g_accumBuffer)
            cudaFree(g_accumBuffer);
        cudaMalloc(&g_accumBuffer, fbSize);
        cudaMemset(g_accumBuffer, 0, fbSize);
        g_accumWidth = width;
        g_accumHeight = height;
        g_frameIndex = 0;
    }

    BVHNode *d_nodes = nullptr;
    Triangle *d_tris = nullptr;
    LightGPU *d_lights = nullptr;
    uint32_t *d_nodeCount = nullptr;
    int *d_rootIndex = nullptr;
    CameraData *d_camera = nullptr;
    uint2 *d_imageSize = nullptr;
    uint32_t *d_triCount = nullptr;
    uint32_t *d_lightCount = nullptr;
    float3 *d_fb = nullptr;

    cudaError_t err;
#define CUDA_CHECK(x)                                                       \
    do                                                                      \
    {                                                                       \
        err = (x);                                                          \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; \
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

    CUDA_CHECK(cudaMalloc(&d_nodes, nodeCount * sizeof(BVHNode)));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes, nodeCount * sizeof(BVHNode), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_tris, triCount * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(d_tris, triangles, triCount * sizeof(Triangle), cudaMemcpyHostToDevice));

    if (lightCount > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_lights, lightCount * sizeof(LightGPU)));
        CUDA_CHECK(cudaMemcpy(d_lights, lights, lightCount * sizeof(LightGPU), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_nodeCount, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_nodeCount, &nodeCount, sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_rootIndex, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_rootIndex, &rootIndex, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_camera, sizeof(CameraData)));
    CUDA_CHECK(cudaMemcpy(d_camera, camera, sizeof(CameraData), cudaMemcpyHostToDevice));

    uint2 imgSize = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    CUDA_CHECK(cudaMalloc(&d_imageSize, sizeof(uint2)));
    CUDA_CHECK(cudaMemcpy(d_imageSize, &imgSize, sizeof(uint2), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_triCount, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_triCount, &triCount, sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_lightCount, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_lightCount, &lightCount, sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_fb, fbSize));

    dim3 block(8, 8, 1);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y, 1);

    // Можешь добавить отдельное ядро RayTraceAccumulateKernel с поддержкой g_accumBuffer и frameIndex,
    // но пока запустим обычное ядро.
    RayTraceKernel<<<grid, block>>>(
        d_nodes, d_tris, d_nodeCount, d_rootIndex,
        d_camera, d_imageSize, d_triCount, d_fb,
        d_lights, d_lightCount);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Копируем кадр обратно
    CUDA_CHECK(cudaMemcpy(framebuffer, d_fb, fbSize, cudaMemcpyDeviceToHost));

    g_frameIndex++;

cleanup:
    cudaFree(d_nodes);
    cudaFree(d_tris);
    cudaFree(d_lights);
    cudaFree(d_nodeCount);
    cudaFree(d_rootIndex);
    cudaFree(d_camera);
    cudaFree(d_imageSize);
    cudaFree(d_triCount);
    cudaFree(d_lightCount);
    cudaFree(d_fb);
#undef CUDA_CHECK

    return (err == cudaSuccess);
}
