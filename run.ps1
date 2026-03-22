param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Scene,

    [ValidateSet("auto", "msvc", "hip", "cuda", "metal")]
    [string]$Backend = "auto",

    [ValidateSet("Debug", "Release")]
    [string]$Config = "Debug",

    [switch]$Build,

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

function Test-IsWindowsHost {
    return [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
        [System.Runtime.InteropServices.OSPlatform]::Windows
    )
}

$scenePath = $Scene
if (-not [System.IO.Path]::IsPathRooted($scenePath)) {
    $candidate = Join-Path $PSScriptRoot $scenePath
    if (Test-Path $candidate) {
        $scenePath = $candidate
    }
}

if (-not (Test-Path $scenePath)) {
    throw "Scene file '$Scene' was not found."
}

$scenePath = (Resolve-Path $scenePath).Path

$buildScript = Join-Path $PSScriptRoot "build.ps1"
$exeName = if (Test-IsWindowsHost) { "SceneRTXTester.exe" } else { "SceneRTXTester" }
$exePath = Join-Path $PSScriptRoot "build\$exeName"

if ($Build -or -not (Test-Path $exePath)) {
    & $buildScript $Backend $Config
    if ($LASTEXITCODE -ne 0) {
        throw "Build step failed."
    }
}

if (-not (Test-Path $exePath)) {
    throw "Executable '$exePath' was not found. Build the project first."
}

Push-Location $PSScriptRoot
try {
    & $exePath $scenePath @AppArgs
    if ($LASTEXITCODE -ne 0) {
        throw "SceneRTXTester exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
