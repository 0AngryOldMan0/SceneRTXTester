#pragma once

#include "CoreMinimal.h"

class USceneRTSceneExportSettings;

class SCENERTSCENEEXPORTER_API FSceneRTSceneExporterCore
{
public:
    static bool ExportEditorWorld(const USceneRTSceneExportSettings* InSettings, FString* OutError = nullptr);
};
