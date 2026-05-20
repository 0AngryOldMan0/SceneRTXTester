#include "SceneRTSceneExporterModule.h"

#include "SceneRTSceneExportSettings.h"
#include "SceneRTSceneExporterCore.h"

#include "Framework/Docking/TabManager.h"
#include "Framework/Notifications/NotificationManager.h"
#include "Interfaces/IPluginManager.h"
#include "LevelEditor.h"
#include "Modules/ModuleManager.h"
#include "PropertyEditorModule.h"
#include "Styling/AppStyle.h"
#include "ToolMenus.h"
#include "Widgets/Docking/SDockTab.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Notifications/SNotificationList.h"
#include "Widgets/Text/STextBlock.h"

static const FName SceneRTSceneExporterTabName(TEXT("SceneRTSceneExporter"));

#define LOCTEXT_NAMESPACE "FSceneRTSceneExporterModule"

void FSceneRTSceneExporterModule::StartupModule()
{
    FGlobalTabmanager::Get()->RegisterNomadTabSpawner(
        SceneRTSceneExporterTabName,
        FOnSpawnTab::CreateRaw(this, &FSceneRTSceneExporterModule::OnSpawnPluginTab))
        .SetDisplayName(LOCTEXT("SceneRTSceneExporterTabTitle", "SceneRT Scene Exporter"))
        .SetMenuType(ETabSpawnerMenuType::Hidden)
        .SetIcon(FSlateIcon(FAppStyle::GetAppStyleSetName(), "LevelEditor.Tabs.Details"));

    UToolMenus::RegisterStartupCallback(
        FSimpleMulticastDelegate::FDelegate::CreateRaw(this, &FSceneRTSceneExporterModule::RegisterMenus));
}

void FSceneRTSceneExporterModule::ShutdownModule()
{
    if (UToolMenus::IsToolMenuUIEnabled())
    {
        UToolMenus::UnRegisterStartupCallback(this);
        UToolMenus::UnregisterOwner(this);
    }

    FGlobalTabmanager::Get()->UnregisterNomadTabSpawner(SceneRTSceneExporterTabName);
}

void FSceneRTSceneExporterModule::RegisterMenus()
{
    FToolMenuOwnerScoped OwnerScoped(this);

    if (UToolMenu* Menu = UToolMenus::Get()->ExtendMenu("LevelEditor.MainMenu.Window"))
    {
        FToolMenuSection& Section = Menu->FindOrAddSection("WindowLayout");
        Section.AddMenuEntry(
            "OpenSceneRTSceneExporter",
            LOCTEXT("SceneRTSceneExporterMenuLabel", "SceneRT Scene Exporter"),
            LOCTEXT("SceneRTSceneExporterMenuTooltip", "Open the SceneRT scene exporter panel."),
            FSlateIcon(FAppStyle::GetAppStyleSetName(), "LevelEditor.Tabs.Details"),
            FUIAction(FExecuteAction::CreateRaw(this, &FSceneRTSceneExporterModule::PluginButtonClicked)));
    }

    if (UToolMenu* ToolbarMenu = UToolMenus::Get()->ExtendMenu("LevelEditor.LevelEditorToolBar"))
    {
        FToolMenuSection& Section = ToolbarMenu->FindOrAddSection("Settings");
        FToolMenuEntry Entry = FToolMenuEntry::InitToolBarButton(
            "SceneRTSceneExporterToolbar",
            FUIAction(FExecuteAction::CreateRaw(this, &FSceneRTSceneExporterModule::PluginButtonClicked)),
            LOCTEXT("SceneRTSceneExporterToolbarLabel", "SceneRT Export"),
            LOCTEXT("SceneRTSceneExporterToolbarTooltip", "Open the SceneRT scene exporter panel."),
            FSlateIcon(FAppStyle::GetAppStyleSetName(), "LevelEditor.Tabs.Details"));
        Section.AddEntry(Entry);
    }
}

void FSceneRTSceneExporterModule::PluginButtonClicked()
{
    FGlobalTabmanager::Get()->TryInvokeTab(SceneRTSceneExporterTabName);
}

TSharedRef<SDockTab> FSceneRTSceneExporterModule::OnSpawnPluginTab(const FSpawnTabArgs& SpawnTabArgs)
{
    USceneRTSceneExportSettings* MutableSettings = GetMutableDefault<USceneRTSceneExportSettings>();
    SettingsObject = MutableSettings;

    FPropertyEditorModule& PropertyEditorModule = FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");

    FDetailsViewArgs DetailsArgs;
    DetailsArgs.bAllowSearch = true;
    DetailsArgs.bHideSelectionTip = true;
    DetailsArgs.bLockable = false;
    DetailsArgs.bShowScrollBar = true;
    DetailsArgs.bUpdatesFromSelection = false;
    DetailsArgs.NameAreaSettings = FDetailsViewArgs::HideNameArea;

    SettingsDetailsView = PropertyEditorModule.CreateDetailView(DetailsArgs);
    SettingsDetailsView->SetObject(MutableSettings);

    return SNew(SDockTab)
        .TabRole(ETabRole::NomadTab)
        [
            SNew(SBorder)
                .Padding(8.0f)
                [
                    SNew(SVerticalBox)
                        + SVerticalBox::Slot()
                        .AutoHeight()
                        .Padding(0.0f, 0.0f, 0.0f, 8.0f)
                        [
                            SNew(STextBlock)
                                .Text(LOCTEXT("SceneRTSceneExporterTitle", "Export current editor world to scene.json + meshes.bin"))
                        ]
                        + SVerticalBox::Slot()
                        .FillHeight(1.0f)
                        [
                            SettingsDetailsView.ToSharedRef()
                        ]
                        + SVerticalBox::Slot()
                        .AutoHeight()
                        .Padding(0.0f, 8.0f, 0.0f, 0.0f)
                        [
                            SNew(SButton)
                                .Text(LOCTEXT("SceneRTExportNow", "Export Current Level"))
                                .OnClicked_Raw(this, &FSceneRTSceneExporterModule::OnExportClicked)
                        ]
                ]
        ];
}

FReply FSceneRTSceneExporterModule::OnExportClicked()
{
    USceneRTSceneExportSettings* MutableSettings = SettingsObject.IsValid() ? SettingsObject.Get() : GetMutableDefault<USceneRTSceneExportSettings>();
    SaveSettings();

    FString ErrorText;
    const bool bOk = FSceneRTSceneExporterCore::ExportEditorWorld(MutableSettings, &ErrorText);

    FNotificationInfo Info(
        bOk
        ? LOCTEXT("SceneRTExportSuccess", "Scene export completed")
        : FText::FromString(ErrorText.IsEmpty() ? TEXT("Scene export failed") : ErrorText));
    Info.ExpireDuration = bOk ? 4.0f : 8.0f;
    Info.bUseSuccessFailIcons = true;

    const TSharedPtr<SNotificationItem> Notification = FSlateNotificationManager::Get().AddNotification(Info);
    if (Notification.IsValid())
    {
        Notification->SetCompletionState(bOk ? SNotificationItem::CS_Success : SNotificationItem::CS_Fail);
    }

    return FReply::Handled();
}

void FSceneRTSceneExporterModule::SaveSettings() const
{
    if (USceneRTSceneExportSettings* MutableSettings = SettingsObject.IsValid() ? SettingsObject.Get() : GetMutableDefault<USceneRTSceneExportSettings>())
    {
        MutableSettings->SaveConfig();
    }
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FSceneRTSceneExporterModule, SceneRTSceneExporter)
