param(
    [string]$KaggleOwner = '',
    [string]$OpenFwiOwner = 'ayushdebnath0123'
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'

. (Join-Path $PSScriptRoot 'kaggle_scheduler_common.ps1')

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$resultRoot = Join-Path $repoRoot 'results\ssgen'
$scheduleLog = Join-Path $resultRoot 'kaggle_schedule.log'
$cachePackage = Join-Path $repoRoot 'tmp\kaggle_ssgen_cache'
$probePackage = Join-Path $repoRoot 'tmp\kaggle_ssgen_probe'
$KaggleOwner = Resolve-KaggleOwner $KaggleOwner
$cacheSlug = "$KaggleOwner/subsurfacegen-cache-600-100-80"
$probeSlug = "$KaggleOwner/subsurfacegen-four-model-gpu-probe"
$openfwiSlugs = @(
    "$OpenFwiOwner/openfwi-deeponet-gno-flatvel-b",
    "$OpenFwiOwner/openfwi-deeponet-gno-curvevel-b"
)
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

function Write-ScheduleLog([string]$message) {
    $line = '{0:o}  {1}' -f (Get-Date), $message
    Add-Content -LiteralPath $scheduleLog -Value $line
    Write-Output $line
}

function Get-KernelStatus([string]$slug) {
    return (& kaggle kernels status $slug 2>&1 | Out-String).Trim()
}

function Wait-Kernel([string]$slug) {
    while ($true) {
        $status = Get-KernelStatus $slug
        Write-ScheduleLog $status | Out-Null
        if ($status -match 'KernelWorkerStatus\.COMPLETE') { return $true }
        if ($status -match 'KernelWorkerStatus\.(ERROR|CANCELLED|CANCELED)') { return $false }
        if ($status -match '(?i)Cannot access kernel|Permission .* denied|not found|unauthorized|forbidden') {
            Write-ScheduleLog "cannot monitor $slug; stopping this wait instead of polling forever" | Out-Null
            return $false
        }
        Start-Sleep -Seconds 60
    }
}

Write-ScheduleLog "using authenticated Kaggle owner $KaggleOwner"
Write-ScheduleLog 'waiting for both OpenFWI DeepONet/GNO jobs to finish'
foreach ($slug in $openfwiSlugs) {
    if (-not (Wait-Kernel $slug)) {
        Write-ScheduleLog "OpenFWI predecessor ended unsuccessfully: $slug; continuing with independent SubsurfaceGen cache"
    }
}

Write-ScheduleLog 'pushing shared SubsurfaceGen CPU cache builder'
Set-KaggleKernelMetadata -Package $cachePackage -KernelSlug $cacheSlug
$cachePushOutput = @(& kaggle kernels push -p $cachePackage 2>&1)
$cachePushExitCode = $LASTEXITCODE
$cachePushOutput | ForEach-Object { Write-ScheduleLog ([string]$_) }
if ($cachePushExitCode -ne 0) {
    Write-ScheduleLog "cache push failed with rc=$cachePushExitCode"
    exit 1
}
$cacheSlug = Resolve-PushedKernelSlug $cachePushOutput $cacheSlug
if (-not (Wait-Kernel $cacheSlug)) {
    & kaggle kernels logs $cacheSlug 2>&1 |
        Set-Content -LiteralPath (Join-Path $resultRoot 'kaggle_cache_kernel_logs.json')
    Write-ScheduleLog 'cache kernel failed; stopping before GPU work'
    exit 1
}

$cacheStatusDir = Join-Path $resultRoot 'kaggle_cache_status'
New-Item -ItemType Directory -Force -Path $cacheStatusDir | Out-Null
& kaggle kernels output $cacheSlug -p $cacheStatusDir --page-size 200 --force `
    --file-pattern 'cache_status.json' 2>&1 |
    ForEach-Object { Write-ScheduleLog ([string]$_) }

Write-ScheduleLog 'pushing four-model SubsurfaceGen GPU probe'
Set-KaggleKernelMetadata -Package $probePackage -KernelSlug $probeSlug -KernelSources @($cacheSlug)
$probePushOutput = @(& kaggle kernels push -p $probePackage --accelerator GPU 2>&1)
$probePushExitCode = $LASTEXITCODE
$probePushOutput | ForEach-Object { Write-ScheduleLog ([string]$_) }
if ($probePushExitCode -ne 0) {
    Write-ScheduleLog "probe push failed with rc=$probePushExitCode"
    exit 1
}
$probePushText = (($probePushOutput | ForEach-Object { [string]$_ }) -join "`n")
$probeSlug = Resolve-PushedKernelSlug $probePushOutput $probeSlug
if ($probePushText -match '(?i)not valid kernel sources') {
    Write-ScheduleLog "probe was pushed without required source $cacheSlug; stopping before GPU work"
    & kaggle kernels logs $probeSlug 2>&1 |
        Set-Content -LiteralPath (Join-Path $resultRoot 'kaggle_probe_kernel_logs.json')
    exit 1
}
if (-not (Wait-Kernel $probeSlug)) {
    & kaggle kernels logs $probeSlug 2>&1 |
        Set-Content -LiteralPath (Join-Path $resultRoot 'kaggle_probe_kernel_logs.json')
    Write-ScheduleLog 'GPU probe failed'
    exit 1
}

$probeOutput = Join-Path $resultRoot 'kaggle_four_model_probe'
New-Item -ItemType Directory -Force -Path $probeOutput | Out-Null
& kaggle kernels output $probeSlug -p $probeOutput --page-size 200 --force 2>&1 |
    ForEach-Object { Write-ScheduleLog ([string]$_) }
& kaggle kernels logs $probeSlug 2>&1 |
    Set-Content -LiteralPath (Join-Path $resultRoot 'kaggle_probe_kernel_logs.json')
Write-ScheduleLog "cache and GPU probe complete: $probeOutput"
