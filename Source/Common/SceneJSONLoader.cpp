#include "../Headers/Classes/SceneJSONLoader.h"

#include "../Headers/MathUtils.h"

#include <../ExternalLibs/json-develop/single_include/nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace
{
    using json = nlohmann::json;

    struct SectionInfo
    {
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
        int32_t  materialIndex = -1;
        std::string materialName;
    };

    struct MeshData
    {
        std::vector<Vec3> positions;
        std::vector<Vec3> normals;
        std::vector<Vec2> uv0;
        std::vector<uint32_t> indices;
        std::vector<SectionInfo> defaultSections;
    };


    static std::array<float,16> ReadMatrix(const json& j)
    {
        std::array<float,16> m{
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1
        };

        if (!j.is_array() || j.size() < 16)
            return m;

        for (std::size_t i = 0; i < 16; ++i)
            m[i] = (float)j[i].get<double>();
        return m;
    }


    static Vec3 NormalizeSafe(const Vec3& v, const Vec3& def)
    {
        const float lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
        if (!(lenSq > 0.0f) || !std::isfinite(lenSq))
            return def;

        const float invLen = 1.0f / std::sqrt(lenSq);
        return Vec3{v.x * invLen, v.y * invLen, v.z * invLen};
    }

    static Vec3 TransformPoint(const std::array<float,16>& m, const Vec3& p)
    {
        return Vec3{
            m[0] * p.x + m[1] * p.y + m[2]  * p.z + m[3],
            m[4] * p.x + m[5] * p.y + m[6]  * p.z + m[7],
            m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]
        };
    }

    static Vec3 TransformDirection(const std::array<float,16>& m, const Vec3& v)
    {
        return Vec3{
            m[0] * v.x + m[1] * v.y + m[2]  * v.z,
            m[4] * v.x + m[5] * v.y + m[6]  * v.z,
            m[8] * v.x + m[9] * v.y + m[10] * v.z
        };
    }

    static bool Invert3x3Transpose(const std::array<float,16>& m, float outIT[9])
    {
        const float a00 = m[0], a01 = m[1], a02 = m[2];
        const float a10 = m[4], a11 = m[5], a12 = m[6];
        const float a20 = m[8], a21 = m[9], a22 = m[10];

        const float c00 =  a11 * a22 - a12 * a21;
        const float c01 = -(a10 * a22 - a12 * a20);
        const float c02 =  a10 * a21 - a11 * a20;

        const float c10 = -(a01 * a22 - a02 * a21);
        const float c11 =  a00 * a22 - a02 * a20;
        const float c12 = -(a00 * a21 - a01 * a20);

        const float c20 =  a01 * a12 - a02 * a11;
        const float c21 = -(a00 * a12 - a02 * a10);
        const float c22 =  a00 * a11 - a01 * a10;

        const float det = a00 * c00 + a01 * c01 + a02 * c02;
        if (!(std::abs(det) > 1e-20f) || !std::isfinite(det))
            return false;

        const float invDet = 1.0f / det;

        // inverse-transpose = cofactors / det
        outIT[0] = c00 * invDet; outIT[1] = c10 * invDet; outIT[2] = c20 * invDet;
        outIT[3] = c01 * invDet; outIT[4] = c11 * invDet; outIT[5] = c21 * invDet;
        outIT[6] = c02 * invDet; outIT[7] = c12 * invDet; outIT[8] = c22 * invDet;
        return true;
    }

    static Vec3 TransformNormalInvTranspose(const std::array<float,16>& m, const Vec3& n)
    {
        float it[9];
        if (!Invert3x3Transpose(m, it))
            return NormalizeSafe(TransformDirection(m, n), Vec3{0.0f, 0.0f, 1.0f});

        const Vec3 r{
            it[0] * n.x + it[1] * n.y + it[2] * n.z,
            it[3] * n.x + it[4] * n.y + it[5] * n.z,
            it[6] * n.x + it[7] * n.y + it[8] * n.z
        };
        return NormalizeSafe(r, Vec3{0.0f, 0.0f, 1.0f});
    }
    static void ComputeTriAABB(Triangle& t)
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

    template <typename T>
    static void ReadVectorAt(std::ifstream& file, std::uint64_t absOffset, std::vector<T>& out, std::size_t count)
    {
        out.resize(count);
        if (count == 0)
            return;

        file.seekg((std::streamoff)absOffset, std::ios::beg);
        file.read(reinterpret_cast<char*>(out.data()),
                  (std::streamsize)(sizeof(T) * count));

        if (!file)
            throw std::runtime_error("SceneJSONLoader: failed to read binary payload");
    }

    static MeshData LoadMeshAsset(std::ifstream& binFile,
                                  const json& meshAsset,
                                  const std::unordered_map<std::string, int32_t>& materialIndexById,
                                  const std::unordered_map<std::string, int32_t>& materialIndexByName)
    {
        MeshData mesh{};

        auto itLods = meshAsset.find("lods");
        if (itLods == meshAsset.end() || !itLods->is_array() || itLods->empty())
            throw std::runtime_error("SceneJSONLoader: mesh asset has no LODs");

        const json& lod = (*itLods)[0];

        const uint32_t vertexCount = (uint32_t)lod.value("vertex_count", 0u);
        const uint32_t indexCount  = (uint32_t)lod.value("index_count", 0u);
        const uint32_t uvSetCount  = (uint32_t)lod.value("uv_set_count", 0u);

        const std::uint64_t positionsOffset = (std::uint64_t)lod.value("positions_offset", 0ull);
        const std::uint64_t normalsOffset   = (std::uint64_t)lod.value("normals_offset", 0ull);
        const std::uint64_t uvsOffset       = (std::uint64_t)lod.value("uvs_offset", 0ull);
        const std::uint64_t indicesOffset   = (std::uint64_t)lod.value("indices_offset", 0ull);

        ReadVectorAt(binFile, positionsOffset, mesh.positions, vertexCount);
        ReadVectorAt(binFile, normalsOffset,   mesh.normals,   vertexCount);
        ReadVectorAt(binFile, indicesOffset,   mesh.indices,   indexCount);

        if (uvSetCount > 0)
        {
            std::vector<Vec2> allUvs;
            ReadVectorAt(binFile, uvsOffset, allUvs, (std::size_t)vertexCount * (std::size_t)uvSetCount);

            // meshes.bin stores UVs channel-major:
            // [UV0 for all vertices][UV1 for all vertices]...
            // Keep only the first UV set here because the current renderer/material pipeline
            // samples a single UV channel.
            mesh.uv0.resize(vertexCount, Vec2{0.0f, 0.0f});
            const std::size_t firstChannelBase = 0;
            for (uint32_t vi = 0; vi < vertexCount; ++vi)
                mesh.uv0[vi] = allUvs[firstChannelBase + (std::size_t)vi];
        }
        else
        {
            mesh.uv0.assign(vertexCount, Vec2{0.0f, 0.0f});
        }

        auto itSections = lod.find("sections");
        if (itSections != lod.end() && itSections->is_array())
        {
            mesh.defaultSections.reserve(itSections->size());
            for (const json& js : *itSections)
            {
                SectionInfo s{};
                s.firstIndex = (uint32_t)js.value("first_index", 0u);
                s.indexCount = (uint32_t)js.value("index_count", 0u);
                s.materialName = js.value("material_name", js.value("material_slot_name", std::string{}));

                const std::string matId = js.value("material_id", std::string{});
                auto itId = materialIndexById.find(matId);
                if (itId != materialIndexById.end())
                    s.materialIndex = itId->second;
                else
                {
                    auto itName = materialIndexByName.find(s.materialName);
                    if (itName != materialIndexByName.end())
                        s.materialIndex = itName->second;
                }
                mesh.defaultSections.push_back(std::move(s));
            }

            std::sort(mesh.defaultSections.begin(), mesh.defaultSections.end(),
                      [](const SectionInfo& a, const SectionInfo& b){ return a.firstIndex < b.firstIndex; });
        }

        return mesh;
    }

    static std::vector<SectionInfo> BuildPrimitiveSections(const json& primitive,
                                                           const MeshData& mesh,
                                                           const std::unordered_map<std::string, int32_t>& materialIndexById,
                                                           const std::unordered_map<std::string, int32_t>& materialIndexByName)
    {
        auto itSections = primitive.find("sections");
        if (itSections == primitive.end() || !itSections->is_array() || itSections->empty())
            return mesh.defaultSections;

        std::vector<SectionInfo> out;
        out.reserve(itSections->size());

        for (const json& js : *itSections)
        {
            SectionInfo s{};
            s.firstIndex = (uint32_t)js.value("first_index", 0u);
            s.indexCount = (uint32_t)js.value("index_count", 0u);

            s.materialName = js.value("material_name",
                             js.value("material_slot_name", std::string{}));

            const std::string matId = js.value("material_id", std::string{});
            auto itId = materialIndexById.find(matId);
            if (itId != materialIndexById.end())
                s.materialIndex = itId->second;
            else
            {
                auto itName = materialIndexByName.find(s.materialName);
                if (itName != materialIndexByName.end())
                    s.materialIndex = itName->second;
            }

            out.push_back(std::move(s));
        }

        std::sort(out.begin(), out.end(),
                  [](const SectionInfo& a, const SectionInfo& b){ return a.firstIndex < b.firstIndex; });
        return out;
    }

    static const SectionInfo* FindSectionForTriangle(const std::vector<SectionInfo>& sections, uint32_t triFirstIndex)
    {
        for (const SectionInfo& s : sections)
        {
            if (triFirstIndex >= s.firstIndex && triFirstIndex + 2u < s.firstIndex + s.indexCount)
                return &s;
        }
        return sections.empty() ? nullptr : &sections.front();
    }
}

bool SceneJSONLoader::supportsFormat(const std::string& extension) const
{
    return toLower(extension) == ".json";
}

std::string SceneJSONLoader::getName() const
{
    return "SceneRTX Scene JSON Loader";
}

std::vector<SceneObject> SceneJSONLoader::load(const std::string& path)
{
    json root;
    {
        std::ifstream jsonFile(path);
        if (!jsonFile)
            throw std::runtime_error("Не удалось открыть Scene JSON: " + path);

        try
        {
            jsonFile >> root;
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(std::string("Ошибка парсинга Scene JSON: ") + e.what());
        }
    }

    const std::string format = root.value("format", std::string{});
    if (format != "SceneRTXSceneExport")
        throw std::runtime_error("SceneJSONLoader: JSON не является SceneRTXSceneExport: " + path);

    const std::filesystem::path scenePath(path);
    const std::filesystem::path sceneDir = scenePath.parent_path();

    const json& world = root.at("world");
    const std::string meshesFileName = world.value("meshes_file", std::string("meshes.bin"));
    const std::filesystem::path meshesPath = (sceneDir / meshesFileName).lexically_normal();

    std::ifstream binFile(meshesPath, std::ios::binary);
    if (!binFile)
        throw std::runtime_error("Не удалось открыть meshes.bin: " + meshesPath.string());

    // quick signature check
    {
        char magic[8] = {};
        binFile.read(magic, 8);
        if (!binFile || std::memcmp(magic, "SRTXMSH", 7) != 0)
            throw std::runtime_error("SceneJSONLoader: invalid meshes.bin signature: " + meshesPath.string());
        binFile.clear();
    }

    std::unordered_map<std::string, int32_t> materialIndexById;
    std::unordered_map<std::string, int32_t> materialIndexByName;
    if (auto itM = root.find("materials"); itM != root.end() && itM->is_array())
    {
        materialIndexById.reserve(itM->size());
        materialIndexByName.reserve(itM->size());

        for (std::size_t i = 0; i < itM->size(); ++i)
        {
            const json& jm = (*itM)[i];
            const std::string id = jm.value("stable_id", std::string{});
            const std::string name = jm.value("name", std::string{});
            if (!id.empty())
                materialIndexById[id] = (int32_t)i;
            if (!name.empty())
                materialIndexByName[name] = (int32_t)i;
        }
    }

    std::unordered_map<std::string, const json*> meshAssetById;
    if (auto itAssets = root.find("mesh_assets"); itAssets != root.end() && itAssets->is_array())
    {
        meshAssetById.reserve(itAssets->size());
        for (const json& ja : *itAssets)
        {
            const std::string id = ja.value("stable_id", std::string{});
            if (!id.empty())
                meshAssetById[id] = &ja;
        }
    }

    std::unordered_map<std::string, MeshData> meshCache;
    meshCache.reserve(meshAssetById.size());

    std::vector<SceneObject> objects;

    auto itPrims = root.find("primitives");
    if (itPrims == root.end() || !itPrims->is_array())
        return objects;

    objects.reserve(itPrims->size());

    for (const json& jp : *itPrims)
    {
        if (!jp.is_object())
            continue;

        if (!jp.value("geometry_exported", true))
            continue;

        const std::string meshAssetId = jp.value("mesh_asset_id", std::string{});
        if (meshAssetId.empty())
            continue;

        auto itAsset = meshAssetById.find(meshAssetId);
        if (itAsset == meshAssetById.end() || !itAsset->second)
            continue;

        auto itCached = meshCache.find(meshAssetId);
        if (itCached == meshCache.end())
        {
            MeshData loaded = LoadMeshAsset(binFile, *itAsset->second, materialIndexById, materialIndexByName);
            itCached = meshCache.emplace(meshAssetId, std::move(loaded)).first;
        }

        const MeshData& mesh = itCached->second;
        if (mesh.positions.empty() || mesh.indices.size() < 3)
            continue;

        const std::vector<SectionInfo> sections = BuildPrimitiveSections(jp, mesh, materialIndexById, materialIndexByName);

        const std::string meshSpace = jp.value("mesh_space", std::string{});

        std::vector<std::array<float,16>> instanceMatrices;
        if (meshSpace == "scene_world_baked")
        {
            instanceMatrices.push_back(std::array<float,16>{
                1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1
            });
        }
        else if (auto itInst = jp.find("instance_transforms"); itInst != jp.end() && itInst->is_array() && !itInst->empty())
        {
            instanceMatrices.reserve(itInst->size());
            for (const json& ji : *itInst)
                instanceMatrices.push_back(ReadMatrix(ji.at("matrix")));
        }
        else
        {
            instanceMatrices.push_back(std::array<float,16>{
                1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1
            });
        }

        const std::string baseName = jp.value("component_path",
                                 jp.value("actor_path",
                                 jp.value("name", std::string("ScenePrimitive"))));

        for (std::size_t instIdx = 0; instIdx < instanceMatrices.size(); ++instIdx)
        {
            SceneObject obj(instanceMatrices.size() > 1
                ? (baseName + "_" + std::to_string(instIdx))
                : baseName);

            const auto& matrix = instanceMatrices[instIdx];

            for (std::size_t ii = 0; ii + 2 < mesh.indices.size(); ii += 3)
            {
                const uint32_t triFirstIndex = (uint32_t)ii;
                const uint32_t i0 = mesh.indices[ii + 0];
                const uint32_t i1 = mesh.indices[ii + 1];
                const uint32_t i2 = mesh.indices[ii + 2];

                if (i0 >= mesh.positions.size() || i1 >= mesh.positions.size() || i2 >= mesh.positions.size())
                    continue;

                Triangle t{};

                t.v0 = TransformPoint(matrix, mesh.positions[i0]);
                t.v1 = TransformPoint(matrix, mesh.positions[i1]);
                t.v2 = TransformPoint(matrix, mesh.positions[i2]);

                t.uv0 = (i0 < mesh.uv0.size()) ? mesh.uv0[i0] : Vec2{0.0f, 0.0f};
                t.uv1 = (i1 < mesh.uv0.size()) ? mesh.uv0[i1] : Vec2{0.0f, 0.0f};
                t.uv2 = (i2 < mesh.uv0.size()) ? mesh.uv0[i2] : Vec2{0.0f, 0.0f};

                Vec3 nrm{0.0f, 0.0f, 0.0f};
                if (i0 < mesh.normals.size()) { nrm.x += mesh.normals[i0].x; nrm.y += mesh.normals[i0].y; nrm.z += mesh.normals[i0].z; }
                if (i1 < mesh.normals.size()) { nrm.x += mesh.normals[i1].x; nrm.y += mesh.normals[i1].y; nrm.z += mesh.normals[i1].z; }
                if (i2 < mesh.normals.size()) { nrm.x += mesh.normals[i2].x; nrm.y += mesh.normals[i2].y; nrm.z += mesh.normals[i2].z; }

                nrm = TransformNormalInvTranspose(matrix, nrm);
                if (!(std::abs(nrm.x) > 1e-8f || std::abs(nrm.y) > 1e-8f || std::abs(nrm.z) > 1e-8f))
                {
                    const Vec3 e1{t.v1.x - t.v0.x, t.v1.y - t.v0.y, t.v1.z - t.v0.z};
                    const Vec3 e2{t.v2.x - t.v0.x, t.v2.y - t.v0.y, t.v2.z - t.v0.z};
                    nrm = normalize(cross(e1, e2));
                }
                t.normal = nrm;

                ComputeTriAABB(t);

                t.color     = Vec3{1.0f, 1.0f, 1.0f};
                t.emission  = Vec3{0.0f, 0.0f, 0.0f};
                t.metallic  = 0.0f;
                t.roughness = 0.5f;

                const SectionInfo* sec = FindSectionForTriangle(sections, triFirstIndex);
                if (sec)
                {
                    t.materialIndex = sec->materialIndex;
                    if (!sec->materialName.empty())
                        obj.addTriangle(t, sec->materialName);
                    else
                        obj.addTriangle(t);
                }
                else
                {
                    obj.addTriangle(t);
                }
            }

            if (!obj.isEmpty())
                objects.push_back(std::move(obj));
        }
    }

    return objects;
}
