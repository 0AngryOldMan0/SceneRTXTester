#include "TestFramework.h"

#include "RendererRegistry.h"
#include "RendererFactory.h"

#include <memory>
#include <string>

namespace
{
    class DummyRenderer final : public Renderer
    {
    public:
        explicit DummyRenderer(std::string rendererName) : name_(std::move(rendererName)) {}

        bool render(const Scene &, const Camera &, std::vector<Vec3> &) override
        {
            return true;
        }

        std::string getName() const override
        {
            return name_;
        }

        void resetAccumulation() override
        {
        }

    private:
        std::string name_;
    };

    std::string nextRendererName()
    {
        static int counter = 0;
        ++counter;
        return "UnitTestRenderer_" + std::to_string(counter);
    }
}

TEST_CASE(UnitRendererRegistry, RegistersAndCreatesRenderer)
{
    RendererRegistry &registry = RendererRegistry::getInstance();
    const std::string name = nextRendererName();

    const bool firstRegistration = registry.registerRenderer(name, [name]()
                                                             { return std::make_unique<DummyRenderer>(name); });
    CHECK(firstRegistration);
    CHECK(registry.hasRenderer(name));

    const bool duplicateRegistration = registry.registerRenderer(name, [name]()
                                                                 { return std::make_unique<DummyRenderer>(name); });
    CHECK(!duplicateRegistration);

    auto renderer = RendererFactory::createRenderer(name);
    CHECK(renderer != nullptr);
    CHECK_EQ(renderer->getName(), name);
}
