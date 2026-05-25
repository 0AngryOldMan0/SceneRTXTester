using UnrealBuildTool;

public class SceneRTSceneExporterV2 : ModuleRules
{
    public SceneRTSceneExporterV2(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
        PrivatePCHHeaderFile = "Private/SceneRTV2PCH.h";

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
        });

        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "InputCore",
            "Json",
            "JsonUtilities",
            "ImageCore",
            "ImageWrapper",
            "RenderCore",
            "RHI",
            "Slate",
            "SlateCore",
            "ToolMenus",
            "PropertyEditor",
            "LevelEditor",
            "Projects",
            "UnrealEd",
            "EditorFramework",
            "MeshDescription",
            "StaticMeshDescription",
            "SkeletalMeshDescription",
            "Landscape",
            "LandscapeEditor",
            "Foliage",
            "InstancedFoliage",
            "MaterialBaking",
            "MaterialEditor",
            "MaterialUtilities",
            "Renderer",
            "Niagara",
        });
    }
}
