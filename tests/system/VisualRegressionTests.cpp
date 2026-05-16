#include "TestFramework.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma clang diagnostic pop

namespace
{
    namespace fs = std::filesystem;

    struct ImageRGB8
    {
        int width = 0;
        int height = 0;
        std::vector<std::uint8_t> pixels;
    };

    struct Metrics
    {
        double psnr = 0.0;
        double ssim = 0.0;
    };

    bool equalsIgnoreCase(std::string a, std::string b)
    {
        if (a.size() != b.size())
            return false;

        for (std::size_t i = 0; i < a.size(); ++i)
        {
            const unsigned char ca = static_cast<unsigned char>(a[i]);
            const unsigned char cb = static_cast<unsigned char>(b[i]);
            if (std::tolower(ca) != std::tolower(cb))
                return false;
        }
        return true;
    }

    bool hasImageExtension(const fs::path &path)
    {
        const std::string ext = path.extension().string();
        return equalsIgnoreCase(ext, ".png") ||
               equalsIgnoreCase(ext, ".jpg") ||
               equalsIgnoreCase(ext, ".jpeg");
    }

    fs::path repoRoot()
    {
#ifdef SCENERTX_SOURCE_DIR
        return fs::path(SCENERTX_SOURCE_DIR);
#else
        return fs::current_path();
#endif
    }

    bool isStrictVisualModeEnabled()
    {
        const char *value = std::getenv("SCENERTX_VISUAL_REQUIRE_ACTUAL");
        if (value == nullptr)
            return false;

        return equalsIgnoreCase(value, "1") ||
               equalsIgnoreCase(value, "true") ||
               equalsIgnoreCase(value, "yes") ||
               equalsIgnoreCase(value, "on");
    }

    ImageRGB8 makeConstantImage(int width, int height, std::uint8_t value)
    {
        ImageRGB8 img;
        img.width = width;
        img.height = height;
        img.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u, value);
        return img;
    }

    ImageRGB8 loadImageRGB8(const fs::path &path)
    {
        int width = 0;
        int height = 0;
        int channels = 0;

        stbi_uc *raw = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
        if (raw == nullptr)
        {
            const char *reason = stbi_failure_reason();
            throw std::runtime_error("Failed to load image: " + path.string() +
                                     (reason ? (" (" + std::string(reason) + ")") : ""));
        }

        const std::size_t size = static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(height) * 3u;

        ImageRGB8 img;
        img.width = width;
        img.height = height;
        img.pixels.assign(raw, raw + size);
        stbi_image_free(raw);
        return img;
    }

    double computePSNR(const ImageRGB8 &lhs, const ImageRGB8 &rhs)
    {
        if (lhs.width != rhs.width || lhs.height != rhs.height)
            throw std::runtime_error("PSNR requires images with identical dimensions");
        if (lhs.pixels.size() != rhs.pixels.size())
            throw std::runtime_error("PSNR requires images with equal pixel buffers");

        if (lhs.pixels.empty())
            return std::numeric_limits<double>::infinity();

        long double sumSquared = 0.0L;
        for (std::size_t i = 0; i < lhs.pixels.size(); ++i)
        {
            const long double a = static_cast<long double>(lhs.pixels[i]) / 255.0L;
            const long double b = static_cast<long double>(rhs.pixels[i]) / 255.0L;
            const long double d = a - b;
            sumSquared += d * d;
        }

        const long double mse = sumSquared / static_cast<long double>(lhs.pixels.size());
        if (mse <= std::numeric_limits<long double>::epsilon())
            return std::numeric_limits<double>::infinity();

        return 10.0 * std::log10(1.0 / static_cast<double>(mse));
    }

    double computeSSIM(const ImageRGB8 &lhs, const ImageRGB8 &rhs)
    {
        if (lhs.width != rhs.width || lhs.height != rhs.height)
            throw std::runtime_error("SSIM requires images with identical dimensions");
        if (lhs.pixels.size() != rhs.pixels.size())
            throw std::runtime_error("SSIM requires images with equal pixel buffers");

        if (lhs.pixels.empty())
            return 1.0;

        long double meanX = 0.0L;
        long double meanY = 0.0L;
        long double sqX = 0.0L;
        long double sqY = 0.0L;
        long double xy = 0.0L;

        const long double inv255 = 1.0L / 255.0L;

        for (std::size_t i = 0; i < lhs.pixels.size(); ++i)
        {
            const long double x = static_cast<long double>(lhs.pixels[i]) * inv255;
            const long double y = static_cast<long double>(rhs.pixels[i]) * inv255;
            meanX += x;
            meanY += y;
            sqX += x * x;
            sqY += y * y;
            xy += x * y;
        }

        const long double n = static_cast<long double>(lhs.pixels.size());
        meanX /= n;
        meanY /= n;

        long double varX = (sqX / n) - (meanX * meanX);
        long double varY = (sqY / n) - (meanY * meanY);
        long double covXY = (xy / n) - (meanX * meanY);

        varX = std::max(0.0L, varX);
        varY = std::max(0.0L, varY);

        constexpr long double kC1 = 0.01L * 0.01L;
        constexpr long double kC2 = 0.03L * 0.03L;

        const long double numerator = (2.0L * meanX * meanY + kC1) * (2.0L * covXY + kC2);
        const long double denominator = (meanX * meanX + meanY * meanY + kC1) * (varX + varY + kC2);

        if (denominator <= std::numeric_limits<long double>::epsilon())
            return 1.0;

        return static_cast<double>(numerator / denominator);
    }

    Metrics computeMetrics(const ImageRGB8 &lhs, const ImageRGB8 &rhs)
    {
        Metrics out;
        out.psnr = computePSNR(lhs, rhs);
        out.ssim = computeSSIM(lhs, rhs);
        return out;
    }

    std::vector<fs::path> collectReferenceImages(const fs::path &referenceDir)
    {
        std::vector<fs::path> files;
        for (const auto &entry : fs::directory_iterator(referenceDir))
        {
            if (!entry.is_regular_file())
                continue;
            if (!hasImageExtension(entry.path()))
                continue;
            files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end());
        return files;
    }

    fs::path resolveActualFramePath(const fs::path &actualDir, const fs::path &referenceImagePath)
    {
        const fs::path direct = actualDir / referenceImagePath.filename();
        if (fs::exists(direct))
            return direct;

        const fs::path pngVariant = actualDir / (referenceImagePath.stem().string() + ".png");
        if (fs::exists(pngVariant))
            return pngVariant;

        const fs::path jpgVariant = actualDir / (referenceImagePath.stem().string() + ".jpg");
        if (fs::exists(jpgVariant))
            return jpgVariant;

        const fs::path jpegVariant = actualDir / (referenceImagePath.stem().string() + ".jpeg");
        if (fs::exists(jpegVariant))
            return jpegVariant;

        return fs::path();
    }

    std::string formatMetric(double value)
    {
        if (std::isinf(value))
            return "inf";

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << value;
        return oss.str();
    }
}

TEST_CASE(SystemVisualRegression, MetricsSanityCheck)
{
    const ImageRGB8 a = makeConstantImage(16, 16, 120u);
    const ImageRGB8 b = makeConstantImage(16, 16, 120u);
    const ImageRGB8 c = makeConstantImage(16, 16, 80u);

    const Metrics same = computeMetrics(a, b);
    CHECK(std::isinf(same.psnr));
    CHECK_NEAR(same.ssim, 1.0, 1e-9);

    const Metrics shifted = computeMetrics(a, c);
    CHECK(shifted.psnr < 20.0);
    CHECK(shifted.ssim < 0.95);
}

TEST_CASE(SystemVisualRegression, CompareReferenceFramesWithRenderedFrames)
{
    const bool strictMode = isStrictVisualModeEnabled();
    const double minPSNR = 30.0;
    const double minSSIM = 0.95;

    const fs::path root = repoRoot();
    const fs::path preferredReferenceDir = root / "Scene" / "Reference";
    const fs::path fallbackReferenceDir = root / "Scene" / "UE5" / "SubwayTonnel" / "Reference";
    const fs::path actualDir = root / "Results" / "GPUFrames";

    fs::path referenceDir = preferredReferenceDir;
    if (!fs::exists(referenceDir) && fs::exists(fallbackReferenceDir))
        referenceDir = fallbackReferenceDir;

    if (!fs::exists(referenceDir))
    {
        if (strictMode)
        {
            TEST_FAIL("Reference directory was not found: " + preferredReferenceDir.string());
        }

        std::cout << "[INFO] Visual regression skipped: reference directory does not exist: "
                  << preferredReferenceDir.string() << "\n";
        return;
    }

    const std::vector<fs::path> references = collectReferenceImages(referenceDir);
    if (references.empty())
    {
        if (strictMode)
        {
            TEST_FAIL("No reference images found in: " + referenceDir.string());
        }

        std::cout << "[INFO] Visual regression skipped: no reference images in "
                  << referenceDir.string() << "\n";
        return;
    }

    if (!fs::exists(actualDir))
    {
        if (strictMode)
        {
            TEST_FAIL("Rendered frame directory was not found: " + actualDir.string());
        }

        std::cout << "[INFO] Visual regression skipped: rendered frame directory does not exist: "
                  << actualDir.string() << "\n";
        return;
    }

    std::vector<std::string> missingFrames;
    std::vector<std::string> belowThreshold;
    std::size_t comparedCount = 0;

    for (const fs::path &referencePath : references)
    {
        const fs::path actualPath = resolveActualFramePath(actualDir, referencePath);
        if (actualPath.empty())
        {
            missingFrames.push_back(referencePath.filename().string());
            continue;
        }

        const ImageRGB8 refImage = loadImageRGB8(referencePath);
        const ImageRGB8 actualImage = loadImageRGB8(actualPath);
        const Metrics metrics = computeMetrics(refImage, actualImage);
        ++comparedCount;

        if (metrics.psnr < minPSNR || metrics.ssim < minSSIM)
        {
            std::ostringstream oss;
            oss << referencePath.filename().string()
                << " (actual: " << actualPath.filename().string()
                << ", PSNR=" << formatMetric(metrics.psnr)
                << ", SSIM=" << formatMetric(metrics.ssim) << ")";
            belowThreshold.push_back(oss.str());
        }
    }

    if (comparedCount == 0)
    {
        if (strictMode)
        {
            TEST_FAIL("No frame pairs were compared. Check naming in Results/GPUFrames.");
        }

        std::cout << "[INFO] Visual regression skipped: no matching frame names between "
                  << referenceDir.string() << " and " << actualDir.string() << "\n";
        return;
    }

    if (!belowThreshold.empty())
    {
        std::ostringstream oss;
        oss << "Visual regression failed for " << belowThreshold.size()
            << " frame(s). Thresholds: PSNR >= " << minPSNR
            << ", SSIM >= " << minSSIM << ". ";

        for (const std::string &line : belowThreshold)
            oss << line << "; ";

        TEST_FAIL(oss.str());
    }

    if (strictMode && !missingFrames.empty())
    {
        std::ostringstream oss;
        oss << "Missing rendered frames for " << missingFrames.size()
            << " reference image(s): ";

        for (const std::string &name : missingFrames)
            oss << name << "; ";

        TEST_FAIL(oss.str());
    }
}
