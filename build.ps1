param(
    [Parameter(Position = 0)]
    [ValidateSet("auto", "msvc", "hip", "cuda", "metal")]
    [string]$Backend = "auto",

    [Parameter(Position = 1)]
    [ValidateSet("Debug", "Release")]
    [string]$Config = "Debug",

    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

function Test-IsWindowsHost {
    return [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
        [System.Runtime.InteropServices.OSPlatform]::Windows
    )
}

function Test-IsMacOSHost {
    return [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
        [System.Runtime.InteropServices.OSPlatform]::OSX
    )
}

function Find-HipPathCandidate {
    if ($env:HIP_PATH -and (Test-Path $env:HIP_PATH)) {
        return $env:HIP_PATH
    }

    $rocmRoot = "C:\Program Files\AMD\ROCm"
    if (-not (Test-Path $rocmRoot)) {
        return $null
    }

    $candidate = Get-ChildItem -Path $rocmRoot -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName "bin\hipcc.bat") } |
        Sort-Object Name -Descending |
        Select-Object -First 1

    if ($candidate) {
        return $candidate.FullName
    }

    return $null
}

function Test-CudaAvailable {
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
        return $true
    }

    return $null -ne (Get-Command nvcc.exe -ErrorAction SilentlyContinue)
}

function Resolve-Backend {
    param(
        [string]$RequestedBackend
    )

    if (Test-IsWindowsHost) {
        if ($RequestedBackend -eq "auto") {
            $hipPath = Find-HipPathCandidate
            if ($hipPath) {
                return "hip"
            }

            if (Test-CudaAvailable) {
                return "cuda"
            }

            return "msvc"
        }

        if ($RequestedBackend -eq "metal") {
            throw "Metal build is only available on macOS."
        }

        return $RequestedBackend
    }

    if (Test-IsMacOSHost) {
        if ($RequestedBackend -eq "auto") {
            return "metal"
        }

        if ($RequestedBackend -ne "metal") {
            throw "On macOS this project is expected to build through the Metal backend."
        }

        return $RequestedBackend
    }

    throw "Unsupported host OS for this project."
}

function Resolve-PresetName {
    param(
        [string]$ResolvedBackend,
        [string]$BuildConfig
    )

    $suffix = $BuildConfig.ToLowerInvariant()

    switch ($ResolvedBackend) {
        "msvc"  { return "windows-msvc-$suffix" }
        "hip"   { return "windows-hip-$suffix" }
        "cuda"  { return "windows-cuda-$suffix" }
        "metal" { return "macos-$suffix" }
        default { throw "Unsupported backend '$ResolvedBackend'." }
    }
}

function Find-VsDevCmd {
    if ($env:VSINSTALLDIR) {
        $fromEnv = Join-Path $env:VSINSTALLDIR "Common7\Tools\VsDevCmd.bat"
        if (Test-Path $fromEnv) {
            return $fromEnv
        }
    }

    $vsWhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        $installationPath = & $vsWhere `
            -latest `
            -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath

        if ($LASTEXITCODE -eq 0 -and $installationPath) {
            $candidate = Join-Path $installationPath.Trim() "Common7\Tools\VsDevCmd.bat"
            if (Test-Path $candidate) {
                return $candidate
            }
        }
    }

    $knownRoots = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
    )

    foreach ($root in $knownRoots) {
        $candidate = Join-Path $root "Common7\Tools\VsDevCmd.bat"
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Visual Studio developer environment was not found. Install Visual Studio C++ tools or open a Developer PowerShell."
}

function Import-VsDevEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$VsDevCmdPath
    )

    $tempFile = [System.IO.Path]::GetTempFileName()

    try {
        $command = "`"$VsDevCmdPath`" -no_logo -arch=x64 -host_arch=x64 >nul && set > `"$tempFile`""
        & cmd.exe /d /s /c $command | Out-Null

        if ($LASTEXITCODE -ne 0) {
            throw "Failed to import the Visual Studio developer environment."
        }

        foreach ($line in [System.IO.File]::ReadAllLines($tempFile)) {
            $separatorIndex = $line.IndexOf("=")
            if ($separatorIndex -le 0) {
                continue
            }

            $name = $line.Substring(0, $separatorIndex)
            $value = $line.Substring($separatorIndex + 1)
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
    finally {
        Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
    }
}

$resolvedBackend = Resolve-Backend -RequestedBackend $Backend
$preset = Resolve-PresetName -ResolvedBackend $resolvedBackend -BuildConfig $Config
$buildDir = Join-Path $PSScriptRoot "build\$preset"

if (Test-IsWindowsHost) {
    if (-not $env:VSCMD_VER -and -not $env:VCINSTALLDIR) {
        $vsDevCmd = Find-VsDevCmd
        Import-VsDevEnvironment -VsDevCmdPath $vsDevCmd
    }

    if ($resolvedBackend -eq "hip" -and -not $env:HIP_PATH) {
        $hipPath = Find-HipPathCandidate
        if (-not $hipPath) {
            throw "HIP backend requested, but HIP_PATH is not set and ROCm was not found under C:\Program Files\AMD\ROCm."
        }

        $env:HIP_PATH = $hipPath
    }
}

if ($Clean -and (Test-Path $buildDir)) {
    Remove-Item -Recurse -Force $buildDir
}

Write-Host "Configuring preset $preset"
if ($resolvedBackend -eq "hip" -and $env:HIP_PATH) {
    Write-Host "Using HIP_PATH=$($env:HIP_PATH)"
}

Push-Location $PSScriptRoot
try {
    & cmake --preset $preset
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed for preset '$preset'."
    }

    & cmake --build --preset $preset
    if ($LASTEXITCODE -ne 0) {
        throw "CMake build failed for preset '$preset'."
    }
}
finally {
    Pop-Location
}
