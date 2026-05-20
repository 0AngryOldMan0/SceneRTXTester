#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class IDetailsView;
class SDockTab;
class USceneRTSceneExportSettings;

class FSceneRTSceneExporterModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

private:
    void RegisterMenus();
    void PluginButtonClicked();
    TSharedRef<SDockTab> OnSpawnPluginTab(const class FSpawnTabArgs& SpawnTabArgs);
    FReply OnExportClicked();
    void SaveSettings() const;

private:
    TSharedPtr<IDetailsView> SettingsDetailsView;
    TWeakObjectPtr<USceneRTSceneExportSettings> SettingsObject;
};
