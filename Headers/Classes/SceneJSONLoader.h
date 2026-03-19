#pragma once

#include "SceneLoader.h"

class SceneJSONLoader : public SceneLoader
{
public:
    std::vector<SceneObject> load(const std::string& path) override;
    bool supportsFormat(const std::string& extension) const override;
    std::string getName() const override;
};
