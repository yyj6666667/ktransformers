<#
.SYNOPSIS
    kt-kernel Windows build script.
    Activates venv -> prerequisite checks -> sets env vars -> pip install
.EXAMPLE
    .\scripts\build_win.ps1
    .\scripts\build_win.ps1 -Instruct AVX512 -Cuda
    .\scripts\build_win.ps1 -VenvDir .\.venv -VcpkgDir D:\vcpkg
#>
param(
    [ValidateSet("AVX2","AVX512","NATIVE","FANCY")]
    [string]$Instruct  = "AVX2",
    [switch]$Cuda,
    [string]$VenvDir   = ".venv",
    [string]$VcpkgDir  = ""
)
$ErrorActionPreference = "Stop"

$repoRoot    = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ktKernelDir = Join-Path $repoRoot "kt-kernel"
$venvPath    = Join-Path $repoRoot $VenvDir
$pip         = Join-Path $venvPath "Scripts\pip.exe"

# ============================================================
# Prerequisite checks
# ============================================================
Write-Host "`n=== kt-kernel Windows Build ===" -ForegroundColor Cyan

# --- venv ---
if (-not (Test-Path $pip)) {
    Write-Host "[FAIL] venv not found: $venvPath" -ForegroundColor Red
    Write-Host "       Run setup_win_deps.ps1 first." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] venv : $venvPath" -ForegroundColor Green

# --- Activate venv (prepend Scripts to PATH) ---
$venvScripts = Join-Path $venvPath "Scripts"
if ($env:PATH -notlike "*$venvScripts*") {
    $env:PATH = "$venvScripts;$env:PATH"
}

# --- cmake ---
$cmakeExe = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmakeExe) {
    Write-Host "[FAIL] cmake not found. Run setup_win_deps.ps1 or install cmake manually." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] cmake: $($cmakeExe.Source)" -ForegroundColor Green

# --- MSVC (cl.exe) ---
$clExe = Get-Command cl -ErrorAction SilentlyContinue
if (-not $clExe) {
    Write-Host "[WARN] cl.exe not in PATH." -ForegroundColor Yellow
    Write-Host "       Attempting to load VS Developer environment ..." -ForegroundColor Yellow
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $loaded = $false
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($vsPath) {
            $devShell = Join-Path $vsPath "Common7\Tools\Launch-VsDevShell.ps1"
            if (Test-Path $devShell) {
                $savedVcpkgRoot = $env:VCPKG_ROOT
                & $devShell -Arch amd64 -SkipAutomaticLocation
                if ($savedVcpkgRoot) { $env:VCPKG_ROOT = $savedVcpkgRoot }
                else { Remove-Item Env:\VCPKG_ROOT -ErrorAction SilentlyContinue }
                $loaded = $true
                Write-Host "[OK] Loaded VS Developer environment: $vsPath" -ForegroundColor Green
            }
        }
    }
    if (-not $loaded) {
        Write-Host "[FAIL] Cannot load MSVC environment. Please run this script from a Developer PowerShell," -ForegroundColor Red
        Write-Host "       or install Visual Studio Build Tools (C++ Desktop workload)." -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "[OK] cl   : $($clExe.Source)" -ForegroundColor Green
}

# --- kt-kernel directory ---
if (-not (Test-Path (Join-Path $ktKernelDir "CMakeLists.txt"))) {
    Write-Host "[FAIL] kt-kernel directory not found: $ktKernelDir" -ForegroundColor Red
    exit 1
}

# ============================================================
# vcpkg toolchain
# ============================================================
if (-not $VcpkgDir) {
    $localVcpkg = Join-Path $repoRoot "vcpkg"
    $VcpkgDir = if (Test-Path (Join-Path $localVcpkg "scripts\buildsystems\vcpkg.cmake")) { $localVcpkg }
                elseif ($env:VCPKG_ROOT) { $env:VCPKG_ROOT }
                else { $localVcpkg }
}
$tcFile = (Join-Path $VcpkgDir "scripts\buildsystems\vcpkg.cmake") -replace '\\','/'
if (-not (Test-Path $tcFile)) {
    Write-Host "[FAIL] vcpkg toolchain not found: $tcFile" -ForegroundColor Red
    Write-Host "       Run setup_win_deps.ps1 first." -ForegroundColor Red
    exit 1
}
$env:CMAKE_ARGS = "-DCMAKE_TOOLCHAIN_FILE=$tcFile"
Write-Host "[OK] vcpkg: $tcFile" -ForegroundColor Green

# ============================================================
# CPU instruction set
# ============================================================
$env:CPUINFER_CPU_INSTRUCT = $Instruct

# ============================================================
# CUDA (optional)
# ============================================================
if ($Cuda) {
    $env:CPUINFER_USE_CUDA = "1"
    if (-not $env:CUDA_HOME) {
        $cudaBase = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
        if (Test-Path $cudaBase) {
            $latest = Get-ChildItem $cudaBase -Directory |
                      Where-Object { $_.Name -match '^v' } |
                      Sort-Object Name -Descending |
                      Select-Object -First 1
            if ($latest) { $env:CUDA_HOME = $latest.FullName }
        }
    }
    if ($env:CUDA_HOME) {
        Write-Host "[OK] CUDA : $env:CUDA_HOME" -ForegroundColor Green
    } else {
        Write-Host "[WARN] -Cuda specified but CUDA Toolkit not found." -ForegroundColor Yellow
    }
} else {
    $env:CPUINFER_USE_CUDA = "0"
}

# ============================================================
# Build
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Building kt-kernel" -ForegroundColor Cyan
Write-Host "   Instruct: $Instruct" -ForegroundColor Cyan
Write-Host "   CUDA    : $(if($Cuda){'ON'}else{'OFF'})" -ForegroundColor Cyan
Write-Host "   WorkDir : $ktKernelDir" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Push-Location $ktKernelDir
try {
    & $pip install . --no-build-isolation
    if ($LASTEXITCODE -ne 0) { throw "pip install failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " kt-kernel build complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Usage:" -ForegroundColor Yellow
Write-Host "  & $venvPath\Scripts\Activate.ps1"
Write-Host "  kt --help"
