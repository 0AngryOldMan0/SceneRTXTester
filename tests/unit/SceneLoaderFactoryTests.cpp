#include "TestFramework.h"

#include "SceneLoaderFactory.h"

TEST_CASE(UnitSceneLoaderFactory, SelectsLoaderByExtension)
{
    SceneLoaderFactory factory;

    auto objLoader = factory.createLoader("/tmp/demo_scene.OBJ");
    CHECK(objLoader != nullptr);
    CHECK_EQ(objLoader->getName(), std::string("OBJ Loader"));

    auto jsonLoader = factory.createLoader("/tmp/demo_scene.json");
    CHECK(jsonLoader != nullptr);
    CHECK_EQ(jsonLoader->getName(), std::string("SceneRTX Scene JSON Loader"));

    auto noLoader = factory.createLoader("/tmp/demo_scene.abc");
    CHECK(noLoader == nullptr);
}

TEST_CASE(UnitSceneLoaderFactory, RejectsFilesWithoutExtension)
{
    SceneLoaderFactory factory;
    auto loader = factory.createLoader("scene_without_ext");
    CHECK(loader == nullptr);
}
