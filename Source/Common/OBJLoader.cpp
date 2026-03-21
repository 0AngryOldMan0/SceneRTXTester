#include "OBJLoader.h"
#include "MathUtils.h"

#include <fstream>
#include <stdexcept>
#include <cstring>   // std::memcmp, std::strlen
#include <cstdlib>   // std::strtof
#include <algorithm>

namespace
{
    inline const char* skipWs(const char* p)
    {
        while (*p == ' ' || *p == '\t' || *p == '\r')
            ++p;
        return p;
    }

    inline bool startsWith(const char* p, const char* lit)
    {
        const std::size_t n = std::strlen(lit);
        return std::memcmp(p, lit, n) == 0;
    }

    inline bool parseFloat3(const char* p, float& x, float& y, float& z)
    {
        p = skipWs(p);
        char* e = nullptr;
        x = std::strtof(p, &e);
        if (e == p) return false;
        p = skipWs(e);
        y = std::strtof(p, &e);
        if (e == p) return false;
        p = skipWs(e);
        z = std::strtof(p, &e);
        if (e == p) return false;
        return true;
    }

    inline bool parseFloat2(const char* p, float& x, float& y)
    {
        p = skipWs(p);
        char* e = nullptr;
        x = std::strtof(p, &e);
        if (e == p) return false;
        p = skipWs(e);
        y = std::strtof(p, &e);
        if (e == p) return false;
        return true;
    }

    inline int parseObjIndexField(const char*& p, const char* end, std::size_t count)
    {
        // Empty field => -1
        if (p >= end || *p == '/' || *p == ' ' || *p == '\t' || *p == '\0')
            return -1;

        bool neg = false;
        if (*p == '-')
        {
            neg = true;
            ++p;
        }

        int val = 0;
        bool any = false;
        while (p < end && *p >= '0' && *p <= '9')
        {
            any = true;
            val = val * 10 + (*p - '0');
            ++p;
        }

        if (!any || val == 0)
            return -1;

        const int idx = neg ? -val : val;

        // OBJ: 1-based; negative indices are relative to the end
        int out = -1;
        if (idx > 0)
            out = idx - 1;
        else
            out = static_cast<int>(count) + idx;

        if (out < 0 || static_cast<std::size_t>(out) >= count)
            return -1;

        return out;
    }

    struct VertexIdx
    {
        int position = -1;
        int texcoord = -1;
        int normal   = -1;
    };

    inline VertexIdx parseVertexIdxToken(const char* b,
                                         const char* e,
                                         std::size_t positionCount,
                                         std::size_t normalCount,
                                         std::size_t texcoordCount)
    {
        VertexIdx v{};
        const char* p = b;

        // v
        v.position = parseObjIndexField(p, e, positionCount);

        if (p < e && *p == '/')
        {
            ++p;
            // vt (may be empty)
            v.texcoord = parseObjIndexField(p, e, texcoordCount);

            if (p < e && *p == '/')
            {
                ++p;
                // vn (may be empty)
                v.normal = parseObjIndexField(p, e, normalCount);
            }
        }
        return v;
    }

    inline void computeTriAABB(Triangle& t)
    {
        const float minX = std::min(t.v0.x, std::min(t.v1.x, t.v2.x));
        const float minY = std::min(t.v0.y, std::min(t.v1.y, t.v2.y));
        const float minZ = std::min(t.v0.z, std::min(t.v1.z, t.v2.z));

        const float maxX = std::max(t.v0.x, std::max(t.v1.x, t.v2.x));
        const float maxY = std::max(t.v0.y, std::max(t.v1.y, t.v2.y));
        const float maxZ = std::max(t.v0.z, std::max(t.v1.z, t.v2.z));

        t.ABoBa.v0 = Vec3{minX, minY, minZ};
        t.ABoBa.v1 = Vec3{maxX, maxY, maxZ};
    }
}

bool OBJLoader::supportsFormat(const std::string &extension) const
{
    return toLower(extension) == ".obj";
}

std::string OBJLoader::getName() const
{
    return "OBJ Loader";
}

std::vector<SceneObject> OBJLoader::load(const std::string &path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Не удалось открыть OBJ-файл: " + path);

    std::vector<SceneObject> objects;
    SceneObject currentObject("default");

    int currentMaterialId = -1;

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> texcoords;

    // Heuristic reserves (helps large OBJ, cheap for small)
    positions.reserve(1 << 18);
    normals.reserve(1 << 18);
    texcoords.reserve(1 << 18);

    std::string line;
    line.reserve(256);

    while (std::getline(file, line))
    {
        const char* p0 = line.c_str();
        p0 = skipWs(p0);
        if (*p0 == '\0' || *p0 == '#')
            continue;

        // v / vn / vt
        if (p0[0] == 'v')
        {
            if (p0[1] == ' ' || p0[1] == '\t')
            {
                float x, y, z;
                if (!parseFloat3(p0 + 1, x, y, z))
                    throw std::runtime_error("Ошибка парсинга вершины: " + line);
                positions.push_back(Vec3{x, y, z});
                continue;
            }
            if (p0[1] == 'n' && (p0[2] == ' ' || p0[2] == '\t'))
            {
                float x, y, z;
                if (!parseFloat3(p0 + 2, x, y, z))
                    throw std::runtime_error("Ошибка парсинга нормали: " + line);
                normals.push_back(Vec3{x, y, z});
                continue;
            }
            if (p0[1] == 't' && (p0[2] == ' ' || p0[2] == '\t'))
            {
                float u, v;
                if (!parseFloat2(p0 + 2, u, v))
                    throw std::runtime_error("Ошибка парсинга текстурной координаты: " + line);
                texcoords.push_back(Vec2{u, v});
                continue;
            }
        }

        // object/group
        if ((p0[0] == 'o' || p0[0] == 'g') && (p0[1] == ' ' || p0[1] == '\t'))
        {
            if (!currentObject.isEmpty())
            {
                objects.push_back(currentObject);
                currentObject = SceneObject();
            }

            std::string name(p0 + 1);
            trimString(name);
            currentObject.setName(name.empty() ? "unnamed" : name);
            currentMaterialId = -1;
            continue;
        }

        // usemtl
        if (startsWith(p0, "usemtl") && (p0[6] == ' ' || p0[6] == '\t'))
        {
            std::string matName(p0 + 6);
            trimString(matName);

            if (!matName.empty())
                currentMaterialId = currentObject.registerMaterialName(matName);
            else
                currentMaterialId = -1;
            continue;
        }

        // face
        if (p0[0] == 'f' && (p0[1] == ' ' || p0[1] == '\t'))
        {
            const char* p = p0 + 1;
            p = skipWs(p);
            if (*p == '\0')
                continue;

            std::vector<OBJVertex> faceVertices;
            faceVertices.reserve(8);

            const std::size_t posCount = positions.size();
            const std::size_t nrmCount = normals.size();
            const std::size_t texCount = texcoords.size();

            auto parseOBJVertex = [&](const char* b, const char* e) -> OBJVertex
            {
                const VertexIdx idx = parseVertexIdxToken(b, e, posCount, nrmCount, texCount);
                OBJVertex v{};
                v.position = idx.position;
                v.texcoord = idx.texcoord;
                v.normal   = idx.normal;
                return v;
            };

            while (*p != '\0')
            {
                p = skipWs(p);
                if (*p == '\0')
                    break;

                const char* b = p;
                while (*p != '\0' && *p != ' ' && *p != '\t' && *p != '\r')
                    ++p;

                const char* e = p;
                if (e > b)
                    faceVertices.push_back(parseOBJVertex(b, e));
            }

            if (faceVertices.size() >= 3)
                processFace(faceVertices, positions, normals, texcoords, currentObject, currentMaterialId);

            continue;
        }

        // ignore other tags (mtllib, s, etc.)
    }

    if (!currentObject.isEmpty())
        objects.push_back(currentObject);
    else if (objects.empty())
        objects.push_back(currentObject);

    return objects;
}

OBJLoader::OBJVertex OBJLoader::parseVertex(const std::string &token,
                                            std::size_t positionCount,
                                            std::size_t normalCount,
                                            std::size_t texcoordCount) const
{
    const char* b = token.c_str();
    const char* e = b + token.size();
    const VertexIdx idx = parseVertexIdxToken(b, e, positionCount, normalCount, texcoordCount);

    OBJVertex v{};
    v.position = idx.position;
    v.texcoord = idx.texcoord;
    v.normal   = idx.normal;
    return v;
}

void OBJLoader::processFace(const std::vector<OBJVertex> &vertices,
                            const std::vector<Vec3> &positions,
                            const std::vector<Vec3> &normals,
                            const std::vector<Vec2> &texcoords,
                            SceneObject &object,
                            int materialId) const
{
    const std::size_t n = vertices.size();
    if (n < 3)
        return;

    auto fetchPos = [&](const OBJVertex& v) -> Vec3
    {
        const int idx = v.position;
        if (idx < 0 || static_cast<std::size_t>(idx) >= positions.size())
            return Vec3{0,0,0};
        return positions[static_cast<std::size_t>(idx)];
    };

    auto fetchNorm = [&](const OBJVertex& v) -> Vec3
    {
        const int idx = v.normal;
        if (idx < 0 || static_cast<std::size_t>(idx) >= normals.size())
            return Vec3{0,0,0};
        return normals[static_cast<std::size_t>(idx)];
    };

    auto fetchUV = [&](const OBJVertex& v) -> Vec2
    {
        const int idx = v.texcoord;
        if (idx < 0 || static_cast<std::size_t>(idx) >= texcoords.size())
            return Vec2{0.0f, 0.0f};
        return texcoords[static_cast<std::size_t>(idx)];
    };

    const OBJVertex v0 = vertices[0];

    for (std::size_t i = 1; i + 1 < n; ++i)
    {
        const OBJVertex v1 = vertices[i];
        const OBJVertex v2 = vertices[i + 1];

        Triangle t{};

        t.v0 = fetchPos(v0);
        t.v1 = fetchPos(v1);
        t.v2 = fetchPos(v2);

        const Vec2 uv0 = fetchUV(v0);
        const Vec2 uv1 = fetchUV(v1);
        const Vec2 uv2 = fetchUV(v2);
        for (int uvSet = 0; uvSet < 3; ++uvSet)
        {
            TriangleUV(t, uvSet, 0) = uv0;
            TriangleUV(t, uvSet, 1) = uv1;
            TriangleUV(t, uvSet, 2) = uv2;
        }

        const Vec3 n0 = fetchNorm(v0);
        const Vec3 n1 = fetchNorm(v1);
        const Vec3 n2 = fetchNorm(v2);

        Vec3 nrm = Vec3{n0.x + n1.x + n2.x, n0.y + n1.y + n2.y, n0.z + n1.z + n2.z};

        // If normals are missing/zero -> compute geometric normal
        if (nrm.x == 0.0f && nrm.y == 0.0f && nrm.z == 0.0f)
        {
            const Vec3 e1{t.v1.x - t.v0.x, t.v1.y - t.v0.y, t.v1.z - t.v0.z};
            const Vec3 e2{t.v2.x - t.v0.x, t.v2.y - t.v0.y, t.v2.z - t.v0.z};
            nrm = normalize(cross(e1, e2));
        }
        else
        {
            nrm = normalize(nrm);
        }

        t.normal = nrm;

        computeTriAABB(t);

        // дефолты — потом SceneMetaLoader перезапишет
        t.color     = Vec3{1.0f, 1.0f, 1.0f};
        t.emission  = Vec3{0.0f, 0.0f, 0.0f};
        t.metallic  = 0.0f;
        t.roughness = 0.5f;

        object.addTriangle(t, materialId);
    }
}
