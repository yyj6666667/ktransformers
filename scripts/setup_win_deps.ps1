<#
.SYNOPSIS
    kt-kernel Windows prerequisite installer.
    Detects toolchain -> submodules -> venv -> PyTorch -> vcpkg + hwloc
.EXAMPLE
    .\scripts\setup_win_deps.ps1
    .\scripts\setup_win_deps.ps1 -CudaVersion cu124
    .\scripts\setup_win_deps.ps1 -CudaVersion cpu
#>
param(
    [string]$VenvDir = ".venv",
    [ValidateSet("cu124","cu126","cpu")]
    [string]$CudaVersion = "cu124"
)
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Write-Step($step, $total, $msg) {
    Write-Host "`n[$step/$total] $msg" -ForegroundColor Cyan
}

$totalSteps = 6

# ============================================================
# 1. Check Git
# ============================================================
Write-Step 1 $totalSteps "Checking Git ..."
$gitExe = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitExe) {
    Write-Host "  [FAIL] Git not found. Please install Git for Windows: https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] git = $($gitExe.Source)" -ForegroundColor Green

# ============================================================
# 2. Check Python
# ============================================================
Write-Step 2 $totalSteps "Checking Python ..."
$pyExe = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyExe) {
    Write-Host "  [FAIL] Python not found. Please install Python 3.8+: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
$pyVer = & python --version 2>&1
Write-Host "  [OK] $pyVer ($($pyExe.Source))" -ForegroundColor Green

# ============================================================
# 3. Check Visual Studio / MSVC toolchain
# ============================================================
Write-Step 3 $totalSteps "Checking MSVC toolchain ..."
$vsFound = $false
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if ($vsPath) {
        Write-Host "  [OK] Visual Studio: $vsPath" -ForegroundColor Green
        $vsFound = $true
    }
}
if (-not $vsFound) {
    $clExe = Get-Command cl -ErrorAction SilentlyContinue
    if ($clExe) {
        Write-Host "  [OK] cl.exe found in PATH: $($clExe.Source)" -ForegroundColor Green
        $vsFound = $true
    }
}
if (-not $vsFound) {
    Write-Host "  [WARN] MSVC not detected (Visual Studio C++ Desktop workload)." -ForegroundColor Yellow
    Write-Host "         cl.exe is required for compilation. Please install Visual Studio Build Tools:" -ForegroundColor Yellow
    Write-Host "         https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Yellow
    Write-Host "         Then re-run this script from a Developer PowerShell." -ForegroundColor Yellow
}

# ============================================================
# 4. Git submodules
# ============================================================
Write-Step 4 $totalSteps "Updating Git submodules ..."
Push-Location $repoRoot
try {
    git submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) { throw "git submodule update failed" }
    Write-Host "  [OK] Submodules ready" -ForegroundColor Green
} finally {
    Pop-Location
}

# ============================================================
# 5. Create venv + install pip dependencies + PyTorch
# ============================================================
Write-Step 5 $totalSteps "Python venv + dependencies ($VenvDir) ..."

$venvPath = Join-Path $repoRoot $VenvDir
$pip      = Join-Path $venvPath "Scripts\pip.exe"

if (-not (Test-Path $venvPath)) {
    Write-Host "  Creating venv ..." -ForegroundColor DarkGray
    python -m venv $venvPath
}

Write-Host "  Installing build tools (setuptools, wheel, cmake, pybind11, cpufeature) ..." -ForegroundColor DarkGray
& $pip install --upgrade pip setuptools wheel cmake pybind11 cpufeature

Write-Host "  Installing PyTorch ($CudaVersion) ..." -ForegroundColor DarkGray
if ($CudaVersion -eq "cpu") {
    & $pip install torch --index-url https://download.pytorch.org/whl/cpu
} else {
    & $pip install torch --index-url "https://download.pytorch.org/whl/$CudaVersion"
}
Write-Host "  [OK] Python dependencies installed" -ForegroundColor Green

# ============================================================
# 6. vcpkg + hwloc
# ============================================================
Write-Step 6 $totalSteps "vcpkg + hwloc ..."

$vcpkgDir = Join-Path $repoRoot "vcpkg"

if (-not (Test-Path (Join-Path $vcpkgDir "vcpkg.exe"))) {
    if (-not (Test-Path $vcpkgDir)) {
        Write-Host "  Cloning vcpkg ..." -ForegroundColor DarkGray
        git clone https://github.com/microsoft/vcpkg.git $vcpkgDir
    }
    Write-Host "  Bootstrapping vcpkg ..." -ForegroundColor DarkGray
    & (Join-Path $vcpkgDir "bootstrap-vcpkg.bat") -disableMetrics
}

Write-Host "  Installing hwloc:x64-windows ..." -ForegroundColor DarkGray
& (Join-Path $vcpkgDir "vcpkg.exe") install hwloc:x64-windows
Write-Host "  [OK] hwloc ready" -ForegroundColor Green

# ============================================================
# Summary
# ============================================================
$tcFile = (Join-Path $vcpkgDir "scripts/buildsystems/vcpkg.cmake") -replace '\\','/'
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " Prerequisites installed successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  venv     : $venvPath"
Write-Host "  vcpkg    : $vcpkgDir"
Write-Host "  toolchain: $tcFile"
Write-Host ""
Write-Host "Next step - build kt-kernel:" -ForegroundColor Yellow
Write-Host "  .\scripts\build_win.ps1"
Write-Host "  .\scripts\build_win.ps1 -Instruct AVX512 -Cuda"
Write-Host "Or one-click build:" -ForegroundColor Yellow
Write-Host "  .\scripts\build_all.ps1"
