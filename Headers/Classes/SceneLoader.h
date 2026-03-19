#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

#include "SceneObject.h"
#include "Point3D.h"

class SceneLoader
{
public:
    virtual ~SceneLoader() = default;

    // Основной метод загрузки сцены
    virtual std::vector<SceneObject> load(const std::string &path) = 0;

    // Проверка поддержки формата
    virtual bool supportsFormat(const std::string &extension) const = 0;

    // Получение имени загрузчика
    virtual std::string getName() const = 0;

protected:
    // Вспомогательные методы, которые могут использоваться всеми загрузчиками
    std::uint64_t determineFileSize(std::ifstream &file) const;
    void trimString(std::string &str) const;
    std::string toLower(const std::string &str) const;
    bool isFiniteVec(const Vec3 &v) const;
};
