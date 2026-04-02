<#
.SYNOPSIS
    kt-kernel Windows one-click build (install deps + compile).
    Run this single script to go from zero to a fully built kt-kernel.
.EXAMPLE
    .\scripts\build_all.ps1
    .\scripts\build_all.ps1 -Instruct AVX512 -Cuda
    .\scripts\build_all.ps1 -CudaVersion cu126 -Instruct AVX2
    .\scripts\build_all.ps1 -CudaVersion cpu
#>
param(
    [ValidateSet("AVX2","AVX512","NATIVE","FANCY")]
    [string]$Instruct    = "AVX2",
    [switch]$Cuda,
    [ValidateSet("cu124","cu126","cpu")]
    [string]$CudaVersion = "cu124",
    [string]$VenvDir     = ".venv"
)
$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot

Write-Host ""
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host "  kt-kernel Windows One-Click Build" -ForegroundColor Magenta
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host "  Instruct : $Instruct"
Write-Host "  CUDA     : $(if($Cuda){'ON'}else{'OFF'})"
Write-Host "  PyTorch  : $CudaVersion"
Write-Host "  venv     : $VenvDir"
Write-Host "========================================================" -ForegroundColor Magenta

$timer = [System.Diagnostics.Stopwatch]::StartNew()

# ============================================================
# Phase 1: Install prerequisites
# ============================================================
Write-Host "`n>>> Phase 1/2: Installing prerequisites <<<" -ForegroundColor Magenta
& "$scriptDir\setup_win_deps.ps1" -VenvDir $VenvDir -CudaVersion $CudaVersion
if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
    Write-Host "setup_win_deps.ps1 failed, aborting." -ForegroundColor Red
    exit 1
}

# ============================================================
# Phase 2: Build kt-kernel
# ============================================================
Write-Host "`n>>> Phase 2/2: Building kt-kernel <<<" -ForegroundColor Magenta
$buildArgs = @{ VenvDir = $VenvDir; Instruct = $Instruct }
if ($Cuda) { $buildArgs.Cuda = $true }
& "$scriptDir\build_win.ps1" @buildArgs
if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
    Write-Host "build_win.ps1 failed, aborting." -ForegroundColor Red
    exit 1
}

# ============================================================
# Done
# ============================================================
$timer.Stop()
$elapsed = $timer.Elapsed
Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  All done! Elapsed: $($elapsed.Minutes)m $($elapsed.Seconds)s" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Activate the environment and get started:" -ForegroundColor Yellow
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
Write-Host "  & $repoRoot\$VenvDir\Scripts\Activate.ps1"
Write-Host "  kt --help"
