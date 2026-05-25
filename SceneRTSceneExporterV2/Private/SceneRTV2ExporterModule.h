#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"

class FSceneRTV2ExporterModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

private:
    void RegisterMenus();
    void UnregisterMenus();
    FDelegateHandle MenuExtenderHandle;
};
