using UnrealBuildTool;

public class SceneRTSceneExporter : ModuleRules
{
    public SceneRTSceneExporter(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "CoreUObject",
                "Engine",
                "CinematicCamera",
                "Landscape"
            }
        );

        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                "Json",
                "JsonUtilities",
                "ImageCore",
                "RenderCore",
                "RHI",
                "Slate",
                "SlateCore",
                "ToolMenus",
                "PropertyEditor",
                "LevelEditor",
                "Projects",
                "UnrealEd",
                "EditorFramework"
            }
        );
    }
}
