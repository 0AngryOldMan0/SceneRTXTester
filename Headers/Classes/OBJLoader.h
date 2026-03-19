#pragma once

#include "SceneLoader.h"
#include "UV.h"
#include <string>
#include <vector>

class OBJLoader : public SceneLoader
{
public:
    std::vector<SceneObject> load(const std::string &path) override;
    bool supportsFormat(const std::string &extension) const override;
    std::string getName() const override;

private:
    struct OBJVertex
    {
        int position = -1;
        int texcoord = -1;
        int normal = -1;
    };

    OBJVertex parseVertex(const std::string &token,
                          std::size_t positionCount,
                          std::size_t normalCount,
                          std::size_t texcoordCount) const;

    // ВАЖНО: texcoords -> Vec2
    void processFace(const std::vector<OBJVertex> &vertices,
                     const std::vector<Vec3> &positions,
                     const std::vector<Vec3> &normals,
                     const std::vector<Vec2> &texcoords,
                     SceneObject &object,
                     int materialId) const;
};
