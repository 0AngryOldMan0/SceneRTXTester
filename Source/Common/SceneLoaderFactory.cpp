#include "../Headers/Classes/SceneLoaderFactory.h"
#include "../Headers/Classes/OBJLoader.h"
#include "../Headers/Classes/SceneJSONLoader.h"

#include <string>

void SceneLoaderFactory::registerLoader(std::unique_ptr<SceneLoader> loader)
{
    if (loader)
        loaders_.push_back(std::move(loader));
}

SceneLoaderFactory::SceneLoaderFactory()
{
    // Регистрируем доступные загрузчики
    registerLoader(std::make_unique<OBJLoader>());
    registerLoader(std::make_unique<SceneJSONLoader>());
}

static inline std::string ExtractLowerExtension(const std::string& filePath)
{
    const std::size_t dotPos = filePath.find_last_of('.');
    if (dotPos == std::string::npos)
        return {};

    std::string ext = filePath.substr(dotPos);
    for (char& c : ext)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    return ext;
}

std::unique_ptr<SceneLoader> SceneLoaderFactory::createLoader(const std::string &filePath) const
{
    const std::string extension = ExtractLowerExtension(filePath);
    if (extension.empty())
        return nullptr;

    for (const auto &loader : loaders_)
    {
        if (!loader)
            continue;

        if (!loader->supportsFormat(extension))
            continue;

        // Для простоты возвращаем новый экземпляр того же типа
        if (dynamic_cast<OBJLoader *>(loader.get()))
            return std::make_unique<OBJLoader>();
        if (dynamic_cast<SceneJSONLoader *>(loader.get()))
            return std::make_unique<SceneJSONLoader>();
    }

    return nullptr;
}
