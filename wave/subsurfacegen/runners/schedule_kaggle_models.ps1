param(
    [string]$KaggleOwner = '',
    [string]$CacheSlug = ''
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'

. (Join-Path $PSScriptRoot 'kaggle_scheduler_common.ps1')

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$resultRoot = Join-Path $repoRoot 'results\ssgen'
$scheduleLog = Join-Path $resultRoot 'kaggle_model_schedule.log'
$templatePackage = Join-Path $repoRoot 'tmp\kaggle_ssgen_model'
$generatedRoot = Join-Path $repoRoot ('tmp\kaggle_ssgen_generated_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$KaggleOwner = Resolve-KaggleOwner $KaggleOwner
$probeSlug = "$KaggleOwner/subsurfacegen-four-model-gpu-probe"
if ([string]::IsNullOrWhiteSpace($CacheSlug)) {
    $CacheSlug = "$KaggleOwner/subsurfacegen-cache-600-100-80"
}
$probeOutput = Join-Path $resultRoot 'kaggle_four_model_probe'
$probeSummary = Join-Path $probeOutput 'ssgen_probe\openfwi_summary.json'
New-Item -ItemType Directory -Force -Path $resultRoot,$generatedRoot | Out-Null

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
            Write-ScheduleLog "cannot monitor $slug; stopping this job instead of polling forever" | Out-Null
            return $false
        }
        Start-Sleep -Seconds 60
    }
}

function Get-GpuQuota {
    $raw = (& kaggle quota --format json 2>&1 | Out-String)
    try {
        $items = $raw | ConvertFrom-Json
        $gpu = $items | Where-Object { $_.resource -eq 'GPU' }
        return [pscustomobject]@{
            RemainingHours = [double]($gpu.remaining.TrimEnd('h'))
            RefreshAt = [DateTimeOffset]::Parse($gpu.refreshAt)
        }
    } catch {
        Write-ScheduleLog "could not parse quota response: $raw"
        return $null
    }
}

function Wait-ForQuota([double]$requiredHours) {
    $checks = 0
    while ($true) {
        $quota = Get-GpuQuota
        if ($null -ne $quota -and $quota.RemainingHours -ge $requiredHours) {
            Write-ScheduleLog ("quota ready: {0:N2} h remaining, {1:N2} h required" -f $quota.RemainingHours,$requiredHours)
            return
        }
        if (($checks % 10) -eq 0) {
            if ($null -eq $quota) {
                Write-ScheduleLog 'waiting for readable GPU quota'
            } else {
                Write-ScheduleLog ("waiting for quota refresh at {0:o}: {1:N2} h remaining, {2:N2} h required" -f $quota.RefreshAt,$quota.RemainingHours,$requiredHours)
            }
        }
        $checks += 1
        Start-Sleep -Seconds 60
    }
}

function New-ModelPackage($job) {
    $jobDir = Join-Path $generatedRoot $job.Name.ToLower()
    New-Item -ItemType Directory -Force -Path $jobDir | Out-Null
    Copy-Item -LiteralPath (Join-Path $templatePackage 'subsurfacegen-model-benchmark.py') -Destination $jobDir
    Copy-Item -LiteralPath (Join-Path $templatePackage 'kernel-metadata.json') -Destination $jobDir

    $scriptPath = Join-Path $jobDir 'subsurfacegen-model-benchmark.py'
    $script = [IO.File]::ReadAllText($scriptPath)
    $script = $script.Replace('MODEL = "FNO"', ('MODEL = "' + $job.Name + '"'))
    $script = $script.Replace('INIT_SEED = 42', ('INIT_SEED = ' + $job.Seed))
    [IO.File]::WriteAllText($scriptPath, $script, (New-Object Text.UTF8Encoding($false)))

    Set-KaggleKernelMetadata `
        -Package $jobDir `
        -KernelSlug $job.Slug `
        -Title ('SubsurfaceGen ' + $job.Name + ' Benchmark') `
        -KernelSources @($CacheSlug)
    return $jobDir
}

Write-ScheduleLog "using authenticated Kaggle owner $KaggleOwner and cache source $CacheSlug"
while (-not (Test-Path -LiteralPath $probeSummary)) {
    Write-ScheduleLog 'waiting for the SubsurfaceGen four-model GPU probe'
    if (-not (Wait-Kernel $probeSlug)) {
        Write-ScheduleLog 'probe failed; full models will not be launched blindly'
        exit 1
    }
    Write-ScheduleLog "downloading completed probe output to $probeOutput"
    New-Item -ItemType Directory -Force -Path $probeOutput | Out-Null
    & kaggle kernels output $probeSlug -p $probeOutput --page-size 200 --force 2>&1 |
        ForEach-Object { Write-ScheduleLog ([string]$_) }
    & kaggle kernels logs $probeSlug 2>&1 |
        Set-Content -LiteralPath (Join-Path $resultRoot 'kaggle_probe_kernel_logs.json')
    if (-not (Test-Path -LiteralPath $probeSummary)) {
        Write-ScheduleLog "probe summary still unavailable: $probeSummary"
    }
    Start-Sleep -Seconds 60
}

$summary = Get-Content -Raw -LiteralPath $probeSummary | ConvertFrom-Json
$projected = @{}
# Probe epoch: 4 train + 2 validation batches. Full epoch: 300 + 50.
$batchScale = 350.0 / 6.0
foreach ($result in $summary.results) {
    $hours = [double]$result.seconds_per_epoch * $batchScale * 80.0 / 3600.0
    # 25% timing margin plus setup/cache/final OOD scoring allowance.
    $projected[$result.model] = 1.25 * $hours + 0.35
    Write-ScheduleLog ("projected {0}: {1:N2} h including safety margin" -f $result.model,$projected[$result.model])
}

$jobs = @(
    [pscustomobject]@{Name='GNO'; Seed=342; Slug="$KaggleOwner/subsurfacegen-gno-benchmark"},
    [pscustomobject]@{Name='PFNO'; Seed=142; Slug="$KaggleOwner/subsurfacegen-pfno-benchmark"},
    [pscustomobject]@{Name='DeepONet'; Seed=242; Slug="$KaggleOwner/subsurfacegen-deeponet-benchmark"},
    [pscustomobject]@{Name='FNO'; Seed=42; Slug="$KaggleOwner/subsurfacegen-fno-benchmark"}
)

$cacheStatus = Get-KernelStatus $CacheSlug
Write-ScheduleLog $cacheStatus | Out-Null
if ($cacheStatus -notmatch 'KernelWorkerStatus\.COMPLETE') {
    Write-ScheduleLog "required cache $CacheSlug is not complete or accessible; run schedule_kaggle_cache_probe.ps1 first"
    exit 1
}

foreach ($job in $jobs) {
    $required = [double]$projected[$job.Name]
    if ($required -gt 11.5) {
        Write-ScheduleLog ("{0} projects to {1:N2} h, above safe 12 h session capacity; not launching" -f $job.Name,$required)
        continue
    }
    $modelName = $job.Name.ToLower()
    $existingSummaries = @(
        (Join-Path $resultRoot ("kaggle_{0}\bench_results\{0}\openfwi_summary.json" -f $modelName)),
        (Join-Path $resultRoot ("kaggle_{0}_benchmark\bench_results\{0}\openfwi_summary.json" -f $modelName))
    )
    $existingSummary = $existingSummaries | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($null -ne $existingSummary) {
        Write-ScheduleLog "$($job.Name) already has a completed local summary at $existingSummary; skipping"
        continue
    }
    Wait-ForQuota $required
    $package = New-ModelPackage $job
    Write-ScheduleLog "pushing $($job.Name) kernel from $package"
    $pushOutput = @(& kaggle kernels push -p $package --accelerator GPU 2>&1)
    $pushExitCode = $LASTEXITCODE
    $pushOutput | ForEach-Object { Write-ScheduleLog ([string]$_) }
    if ($pushExitCode -ne 0) {
        Write-ScheduleLog "$($job.Name) push failed with rc=$pushExitCode"
        continue
    }

    $pushText = (($pushOutput | ForEach-Object { [string]$_ }) -join "`n")
    $kernelSlug = Resolve-PushedKernelSlug $pushOutput $job.Slug
    if ($kernelSlug -ne $job.Slug) {
        Write-ScheduleLog "Kaggle created $kernelSlug (metadata requested $($job.Slug)); using the created slug"
    }
    if ($pushText -match '(?i)not valid kernel sources') {
        Write-ScheduleLog "$($job.Name) was pushed without a required notebook source; stopping before more GPU jobs are launched"
        & kaggle kernels logs $kernelSlug 2>&1 |
            Set-Content -LiteralPath (Join-Path $resultRoot ("kaggle_{0}_kernel_logs.json" -f $job.Name.ToLower()))
        exit 1
    }

    if (-not (Wait-Kernel $kernelSlug)) {
        & kaggle kernels logs $kernelSlug 2>&1 |
            Set-Content -LiteralPath (Join-Path $resultRoot ("kaggle_{0}_kernel_logs.json" -f $job.Name.ToLower()))
        Write-ScheduleLog "$($job.Name) kernel failed"
        continue
    }

    $outputDir = Join-Path $resultRoot ("kaggle_{0}" -f $job.Name.ToLower())
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    & kaggle kernels output $kernelSlug -p $outputDir --page-size 200 --force 2>&1 |
        ForEach-Object { Write-ScheduleLog ([string]$_) }
    & kaggle kernels logs $kernelSlug 2>&1 |
        Set-Content -LiteralPath (Join-Path $resultRoot ("kaggle_{0}_kernel_logs.json" -f $job.Name.ToLower()))
    Write-ScheduleLog "$($job.Name) complete from $kernelSlug and downloaded to $outputDir"
}

Write-ScheduleLog 'SubsurfaceGen full-model schedule finished'
