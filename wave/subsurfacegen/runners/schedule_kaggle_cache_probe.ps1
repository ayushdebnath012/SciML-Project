$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$resultRoot = Join-Path $repoRoot 'results\ssgen'
$scheduleLog = Join-Path $resultRoot 'kaggle_schedule.log'
$cachePackage = Join-Path $repoRoot 'tmp\kaggle_ssgen_cache'
$probePackage = Join-Path $repoRoot 'tmp\kaggle_ssgen_probe'
$cacheSlug = 'ayushdebnath0123/subsurfacegen-cache-600-100-80'
$probeSlug = 'ayushdebnath0123/subsurfacegen-four-model-gpu-probe'
$openfwiSlugs = @(
    'ayushdebnath0123/openfwi-deeponet-gno-flatvel-b',
    'ayushdebnath0123/openfwi-deeponet-gno-curvevel-b'
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
        Write-ScheduleLog $status
        if ($status -match 'KernelWorkerStatus\.COMPLETE') { return $true }
        if ($status -match 'KernelWorkerStatus\.(ERROR|CANCELLED)') { return $false }
        Start-Sleep -Seconds 60
    }
}

Write-ScheduleLog 'waiting for both OpenFWI DeepONet/GNO jobs to finish'
foreach ($slug in $openfwiSlugs) {
    if (-not (Wait-Kernel $slug)) {
        Write-ScheduleLog "OpenFWI predecessor ended unsuccessfully: $slug; continuing with independent SubsurfaceGen cache"
    }
}

Write-ScheduleLog 'pushing shared SubsurfaceGen CPU cache builder'
& kaggle kernels push -p $cachePackage 2>&1 |
    ForEach-Object { Write-ScheduleLog ([string]$_) }
if ($LASTEXITCODE -ne 0) {
    Write-ScheduleLog "cache push failed with rc=$LASTEXITCODE"
    exit 1
}
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
& kaggle kernels push -p $probePackage --accelerator GPU 2>&1 |
    ForEach-Object { Write-ScheduleLog ([string]$_) }
if ($LASTEXITCODE -ne 0) {
    Write-ScheduleLog "probe push failed with rc=$LASTEXITCODE"
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
