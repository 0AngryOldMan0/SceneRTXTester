#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace testfs
{
    class TempDir
    {
    public:
        TempDir()
        {
            const auto now = static_cast<unsigned long long>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::mt19937_64 rng(now);
            const auto suffix = static_cast<unsigned long long>(rng());

            path_ = std::filesystem::temp_directory_path() /
                    ("scene_rtx_tests_" + std::to_string(now) + "_" + std::to_string(suffix));
            std::filesystem::create_directories(path_);
        }

        ~TempDir()
        {
            std::error_code ec;
            std::filesystem::remove_all(path_, ec);
        }

        const std::filesystem::path &path() const { return path_; }

    private:
        std::filesystem::path path_;
    };

    inline void writeTextFile(const std::filesystem::path &path, const std::string &content)
    {
        std::ofstream file(path);
        file << content;
    }

    inline void writeBinaryFile(const std::filesystem::path &path, const std::vector<std::uint8_t> &data)
    {
        std::ofstream file(path, std::ios::binary);
        if (!data.empty())
            file.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    }

    template <typename T>
    inline void appendBytes(std::vector<std::uint8_t> &out, const T &value)
    {
        const std::uint8_t *ptr = reinterpret_cast<const std::uint8_t *>(&value);
        out.insert(out.end(), ptr, ptr + sizeof(T));
    }
}
