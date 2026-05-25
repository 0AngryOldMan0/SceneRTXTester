#pragma once

#include "CoreMinimal.h"

class USceneRTV2ExportSettings;

/**
 * Public entry point for the V2 exporter. Designed to be invoked from the Editor
 * (toolbar / commandlet / Python) with a settings object that fully describes
 * the desired bundle.
 *
 * The exporter is scene-agnostic: no hardcoded asset paths, no "if name contains
 * subway" branches. All adaptation happens through reflection of UE engine types
 * (UPrimitiveComponent subclasses, FPostProcessSettings struct, material
 * parameter associations) so any UE 5.x level can flow through unchanged.
 *
 * Coordinate handling: by default writes Unreal-native LH Z-up cm coordinates;
 * RH conversion is a flag, applied as a final pass before serialization.
 */
class SCENERTSCENEEXPORTERV2_API FSceneRTV2ExporterCore
{
public:
    /** Exports the currently open editor world. Returns false on hard failure. */
    static bool ExportEditorWorld(const USceneRTV2ExportSettings* InSettings, FString* OutError = nullptr);

    /** Exports an explicit world (used by commandlet / batch). */
    static bool ExportWorld(class UWorld* InWorld,
                            const USceneRTV2ExportSettings* InSettings,
                            FString* OutError = nullptr);
};
