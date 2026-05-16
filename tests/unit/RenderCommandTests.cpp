#include "TestFramework.h"

#include "RenderCommand.h"

TEST_CASE(UnitRenderCommand, DefaultsAndFluentSetters)
{
    RenderCommand cmd;

    CHECK_EQ(cmd.getImageWidth(), 1920);
    CHECK_EQ(cmd.getImageHeight(), 1080);
    CHECK_EQ(cmd.getSamplesPerPixel(), 1);
    CHECK(!cmd.hasMetadata());
    CHECK(!cmd.isDebugViewEnabled());

    cmd.setImageSize(800, 600)
        .setSamplesPerPixel(16)
        .setRenderMode(RenderCommand::RenderMode::Preview)
        .setAccumulationMode(RenderCommand::AccumulationMode::FinalStill)
        .setDebugView(RenderCommand::DebugView::BaseColor)
        .setExportDebugViews(true);

    CHECK_EQ(cmd.getImageWidth(), 800);
    CHECK_EQ(cmd.getImageHeight(), 600);
    CHECK_EQ(cmd.getSamplesPerPixel(), 16);
    CHECK(cmd.getRenderMode() == RenderCommand::RenderMode::Preview);
    CHECK(cmd.getAccumulationMode() == RenderCommand::AccumulationMode::FinalStill);
    CHECK(cmd.getDebugView() == RenderCommand::DebugView::BaseColor);
    CHECK(cmd.isDebugViewEnabled());
    CHECK(cmd.shouldExportDebugViews());
}

TEST_CASE(UnitRenderCommand, ResetRestoresDefaults)
{
    RenderCommand cmd;
    cmd.setImageSize(1, 1)
        .setSamplesPerPixel(99)
        .setRenderMode(RenderCommand::RenderMode::Preview)
        .setDebugView(RenderCommand::DebugView::MaterialModel)
        .setExportDebugViews(true);

    cmd.reset();

    CHECK_EQ(cmd.getImageWidth(), 1920);
    CHECK_EQ(cmd.getImageHeight(), 1080);
    CHECK_EQ(cmd.getSamplesPerPixel(), 1);
    CHECK(cmd.getRenderMode() == RenderCommand::RenderMode::Progressive);
    CHECK(cmd.getAccumulationMode() == RenderCommand::AccumulationMode::PreviewProgressive);
    CHECK(cmd.getDebugView() == RenderCommand::DebugView::Disabled);
    CHECK(!cmd.shouldExportDebugViews());
}
