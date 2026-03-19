#pragma once

#include "SceneLoader.h"
#include <memory>
#include <vector>

class SceneLoaderFactory
{
public:
    SceneLoaderFactory();

    std::unique_ptr<SceneLoader> createLoader(const std::string &filePath) const;
    const std::vector<std::unique_ptr<SceneLoader>> &getLoaders() const { return loaders_; }
    void registerLoader(std::unique_ptr<SceneLoader> loader);

private:
    std::vector<std::unique_ptr<SceneLoader>> loaders_;
};
