#include "../Headers/Classes/SceneLoader.h"

#include <algorithm>
#include <cctype>
#include <cmath>

std::uint64_t SceneLoader::determineFileSize(std::ifstream &file) const
{
    const auto pos = file.tellg();
    file.seekg(0, std::ios::end);
    const auto end = file.tellg();
    file.seekg(pos, std::ios::beg);

    if (end < 0)
        return 0;

    return static_cast<std::uint64_t>(end);
}

void SceneLoader::trimString(std::string &str) const
{
    auto isWs = [](unsigned char c) { return std::isspace(c) != 0; };

    auto b = std::find_if(str.begin(), str.end(), [&](char c) { return !isWs(static_cast<unsigned char>(c)); });
    auto e = std::find_if(str.rbegin(), str.rend(), [&](char c) { return !isWs(static_cast<unsigned char>(c)); }).base();

    if (b >= e)
    {
        str.clear();
        return;
    }

    str.assign(b, e);
}

std::string SceneLoader::toLower(const std::string &str) const
{
    std::string result;
    result.resize(str.size());

    for (std::size_t i = 0; i < str.size(); ++i)
        result[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(str[i])));

    return result;
}

bool SceneLoader::isFiniteVec(const Vec3 &v) const
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
