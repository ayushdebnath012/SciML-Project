param(
    [Parameter(Mandatory = $true)][string]$Slug,
    [Parameter(Mandatory = $true)][string]$OutputSubdir
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$resultRoot = Join-Path $repoRoot 'results\openfwi'
$outputDir = Join-Path $resultRoot $OutputSubdir
$monitorLog = Join-Path $resultRoot ($OutputSubdir + '_monitor.log')
$kernelLog = Join-Path $resultRoot ($OutputSubdir + '_kernel_logs.json')
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

function Write-MonitorLog([string]$message) {
    $line = '{0:o}  {1}' -f (Get-Date), $message
    Add-Content -LiteralPath $monitorLog -Value $line
    Write-Output $line
}

Write-MonitorLog "monitor started for $Slug"
while ($true) {
    $status = (& kaggle kernels status $Slug 2>&1 | Out-String).Trim()
    Write-MonitorLog $status

    if ($status -match 'COMPLETE') {
        Write-MonitorLog 'downloading Kaggle output files'
        & kaggle kernels output $Slug -p $outputDir --page-size 200 --force 2>&1 |
            ForEach-Object { Write-MonitorLog ([string]$_) }
        & kaggle kernels logs $Slug 2>&1 | Set-Content -LiteralPath $kernelLog
        Write-MonitorLog "download complete: $outputDir"
        exit 0
    }

    if ($status -match 'ERROR|CANCEL') {
        & kaggle kernels logs $Slug 2>&1 | Set-Content -LiteralPath $kernelLog
        Write-MonitorLog "kernel ended unsuccessfully; logs saved to $kernelLog"
        exit 1
    }

    Start-Sleep -Seconds 60
}
