#include "TestFramework.h"

#include "SceneJSONLoader.h"

#include "TestFsUtils.h"

#include <filesystem>
#include <string>

namespace
{
    struct Offsets
    {
        std::uint64_t positions = 0;
        std::uint64_t normals = 0;
        std::uint64_t indices = 0;
    };

    Offsets createMinimalMeshBinary(const std::filesystem::path &binPath, bool validSignature)
    {
        std::vector<std::uint8_t> bytes;
        bytes.reserve(256);

        const std::uint8_t magicGood[8] = {'S', 'R', 'T', 'X', 'M', 'S', 'H', 0};
        const std::uint8_t magicBad[8] = {'B', 'A', 'D', '_', 'M', 'S', 'H', 0};
        bytes.insert(bytes.end(),
                     validSignature ? std::begin(magicGood) : std::begin(magicBad),
                     validSignature ? std::end(magicGood) : std::end(magicBad));

        Offsets off{};

        off.positions = static_cast<std::uint64_t>(bytes.size());
        testfs::appendBytes(bytes, Vec3{0.0f, 0.0f, 0.0f});
        testfs::appendBytes(bytes, Vec3{1.0f, 0.0f, 0.0f});
        testfs::appendBytes(bytes, Vec3{0.0f, 1.0f, 0.0f});

        off.normals = static_cast<std::uint64_t>(bytes.size());
        testfs::appendBytes(bytes, Vec3{0.0f, 0.0f, 1.0f});
        testfs::appendBytes(bytes, Vec3{0.0f, 0.0f, 1.0f});
        testfs::appendBytes(bytes, Vec3{0.0f, 0.0f, 1.0f});

        off.indices = static_cast<std::uint64_t>(bytes.size());
        testfs::appendBytes(bytes, std::uint32_t{0u});
        testfs::appendBytes(bytes, std::uint32_t{1u});
        testfs::appendBytes(bytes, std::uint32_t{2u});

        testfs::writeBinaryFile(binPath, bytes);
        return off;
    }

    std::string buildSceneJson(const Offsets &offsets)
    {
        return "{\n"
               "  \"format\": \"SceneRTXSceneExport\",\n"
               "  \"world\": {\"meshes_file\": \"meshes.bin\"},\n"
               "  \"materials\": [\n"
               "    {\"stable_id\": \"mat_1\", \"name\": \"Mat_One\"}\n"
               "  ],\n"
               "  \"mesh_assets\": [\n"
               "    {\n"
               "      \"stable_id\": \"mesh_1\",\n"
               "      \"lods\": [\n"
               "        {\n"
               "          \"vertex_count\": 3,\n"
               "          \"index_count\": 3,\n"
               "          \"uv_set_count\": 0,\n"
               "          \"positions_offset\": " + std::to_string(offsets.positions) + ",\n"
               "          \"normals_offset\": " + std::to_string(offsets.normals) + ",\n"
               "          \"tangents_offset\": 0,\n"
               "          \"colors_offset\": 0,\n"
               "          \"uvs_offset\": 0,\n"
               "          \"indices_offset\": " + std::to_string(offsets.indices) + ",\n"
               "          \"sections\": [\n"
               "            {\"first_index\": 0, \"index_count\": 3, \"material_name\": \"Mat_One\"}\n"
               "          ]\n"
               "        }\n"
               "      ]\n"
               "    }\n"
               "  ],\n"
               "  \"primitives\": [\n"
               "    {\n"
               "      \"name\": \"PrimA\",\n"
               "      \"mesh_asset_id\": \"mesh_1\",\n"
               "      \"geometry_exported\": true\n"
               "    }\n"
               "  ]\n"
               "}\n";
    }
}

TEST_CASE(IntegrationSceneJSONLoader, LoadsMinimalValidScene)
{
    testfs::TempDir tmp;
    const auto scenePath = tmp.path() / "scene.json";
    const auto meshesPath = tmp.path() / "meshes.bin";

    const Offsets offsets = createMinimalMeshBinary(meshesPath, true);
    testfs::writeTextFile(scenePath, buildSceneJson(offsets));

    SceneJSONLoader loader;
    const auto objects = loader.load(scenePath.string());

    CHECK_EQ(objects.size(), static_cast<std::size_t>(1));
    const SceneObject &obj = objects[0];

    CHECK_EQ(obj.getTriangleCount(), static_cast<std::size_t>(1));
    CHECK_EQ(obj.getInstanceCount(), static_cast<std::size_t>(1));

    const std::string *materialName = obj.getTriangleMaterialName(0);
    CHECK(materialName != nullptr);
    CHECK_EQ(*materialName, std::string("Mat_One"));
}

TEST_CASE(IntegrationSceneJSONLoader, RejectsInvalidMeshSignature)
{
    testfs::TempDir tmp;
    const auto scenePath = tmp.path() / "scene.json";
    const auto meshesPath = tmp.path() / "meshes.bin";

    const Offsets offsets = createMinimalMeshBinary(meshesPath, false);
    testfs::writeTextFile(scenePath, buildSceneJson(offsets));

    SceneJSONLoader loader;
    CHECK_THROWS(loader.load(scenePath.string()));
}
