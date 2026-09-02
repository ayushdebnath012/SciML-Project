function Resolve-KaggleOwner([string]$RequestedOwner = '') {
    # Prefer the owner returned by an authenticated API request.  This also
    # works with OAuth logins, where KAGGLE_USERNAME/kaggle.json may be absent.
    $listOutput = @(& kaggle kernels list -m --page-size 1 -v --sort-by dateRun 2>&1)
    $listExitCode = $LASTEXITCODE
    $detectedOwner = ''
    if ($listExitCode -eq 0) {
        $listText = (($listOutput | ForEach-Object { [string]$_ }) -join "`n")
        $match = [regex]::Match(
            $listText,
            '(?m)^\s*"?([A-Za-z0-9_-]+)/[A-Za-z0-9_-]+'
        )
        if ($match.Success) {
            $detectedOwner = $match.Groups[1].Value
        }
    }

    if ([string]::IsNullOrWhiteSpace($detectedOwner) -and
        -not [string]::IsNullOrWhiteSpace($env:KAGGLE_USERNAME)) {
        $detectedOwner = $env:KAGGLE_USERNAME.Trim()
    }

    if ([string]::IsNullOrWhiteSpace($detectedOwner)) {
        $configRoot = if ([string]::IsNullOrWhiteSpace($env:KAGGLE_CONFIG_DIR)) {
            Join-Path $HOME '.kaggle'
        } else {
            $env:KAGGLE_CONFIG_DIR
        }
        $configPath = Join-Path $configRoot 'kaggle.json'
        if (Test-Path -LiteralPath $configPath) {
            try {
                $config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
                if (-not [string]::IsNullOrWhiteSpace([string]$config.username)) {
                    $detectedOwner = ([string]$config.username).Trim()
                }
            } catch {
                # Do not print the credential file: it also contains the API key.
            }
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($RequestedOwner)) {
        $requested = $RequestedOwner.Trim()
        if (-not [string]::IsNullOrWhiteSpace($detectedOwner) -and
            $detectedOwner -ne $requested) {
            throw "Kaggle is authenticated as '$detectedOwner', not requested owner '$requested'."
        }
        return $requested
    }

    if ([string]::IsNullOrWhiteSpace($detectedOwner)) {
        throw 'Could not determine the authenticated Kaggle owner. Set KAGGLE_USERNAME or pass -KaggleOwner.'
    }
    return $detectedOwner
}

function Resolve-PushedKernelSlug([object[]]$PushOutput, [string]$Fallback) {
    $pushText = (($PushOutput | ForEach-Object { [string]$_ }) -join "`n")
    $match = [regex]::Match(
        $pushText,
        'https://www\.kaggle\.com/code/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)'
    )
    if ($match.Success) {
        return '{0}/{1}' -f $match.Groups[1].Value,$match.Groups[2].Value
    }
    return $Fallback
}

function Set-KaggleKernelMetadata {
    param(
        [Parameter(Mandatory=$true)][string]$Package,
        [Parameter(Mandatory=$true)][string]$KernelSlug,
        [string]$Title = '',
        [string[]]$KernelSources
    )

    $metadataPath = Join-Path $Package 'kernel-metadata.json'
    if (-not (Test-Path -LiteralPath $metadataPath)) {
        throw "Missing Kaggle metadata: $metadataPath"
    }
    $metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
    $metadata.id = $KernelSlug
    if (-not [string]::IsNullOrWhiteSpace($Title)) {
        $metadata.title = $Title
    }
    if ($PSBoundParameters.ContainsKey('KernelSources')) {
        $sources = @($KernelSources)
        if ($null -eq $metadata.PSObject.Properties['kernel_sources']) {
            $metadata | Add-Member -NotePropertyName 'kernel_sources' -NotePropertyValue $sources
        } else {
            $metadata.kernel_sources = $sources
        }
    }
    $metadataJson = $metadata | ConvertTo-Json -Depth 10
    [IO.File]::WriteAllText(
        $metadataPath,
        $metadataJson,
        (New-Object Text.UTF8Encoding($false))
    )
}
