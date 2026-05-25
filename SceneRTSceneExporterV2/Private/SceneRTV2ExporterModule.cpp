#include "SceneRTV2ExporterModule.h"
#include "SceneRTV2ExporterCore.h"
#include "SceneRTV2ExportSettings.h"

#include "Modules/ModuleManager.h"
#include "Framework/MultiBox/MultiBoxBuilder.h"
#include "ToolMenus.h"

#define LOCTEXT_NAMESPACE "SceneRTV2"

void FSceneRTV2ExporterModule::StartupModule()
{
    UToolMenus::RegisterStartupCallback(
        FSimpleMulticastDelegate::FDelegate::CreateRaw(this, &FSceneRTV2ExporterModule::RegisterMenus));
}

void FSceneRTV2ExporterModule::ShutdownModule()
{
    UnregisterMenus();
}

void FSceneRTV2ExporterModule::RegisterMenus()
{
    FToolMenuOwnerScoped OwnerScoped(this);
    UToolMenu* Menu = UToolMenus::Get()->ExtendMenu("LevelEditor.MainMenu.File");
    if (!Menu) { return; }

    FToolMenuSection& Section = Menu->FindOrAddSection("SceneRTV2");
    Section.AddMenuEntry(
        "ExportSceneRTV2",
        LOCTEXT("ExportSceneRTV2", "Export SceneRT V2..."),
        LOCTEXT("ExportSceneRTV2Tooltip", "Export the current level into the SceneRT V2 bundle format."),
        FSlateIcon(),
        FUIAction(FExecuteAction::CreateLambda([]()
        {
            USceneRTV2ExportSettings* Settings = GetMutableDefault<USceneRTV2ExportSettings>();
            FString Error;
            const bool bOk = FSceneRTV2ExporterCore::ExportEditorWorld(Settings, &Error);
            if (!bOk)
            {
                UE_LOG(LogTemp, Error, TEXT("SceneRT V2 export failed: %s"), *Error);
            }
        })));
}

void FSceneRTV2ExporterModule::UnregisterMenus()
{
    UToolMenus::UnregisterOwner(this);
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FSceneRTV2ExporterModule, SceneRTSceneExporterV2)
