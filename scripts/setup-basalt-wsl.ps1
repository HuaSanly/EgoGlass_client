[CmdletBinding()]
param(
    [string]$Distribution = 'Nvidia_SDKM_Ubuntu_22.04_JetPack_7.2',
    [string]$SourceDirectory = '/home/nvidia/egoglass/tools/basalt-src',
    [string]$BuildDirectory = '/home/nvidia/egoglass/tools/basalt-build',
    [string]$BasaltRevision = '0f3b2b52c807f70ff4e2973ce253c73329eea7bc',
    [string]$ProxyUrl = ''
)

$ErrorActionPreference = 'Stop'
$setupScript = (Resolve-Path (Join-Path $PSScriptRoot 'wsl\setup-basalt.sh')).Path
$patchDirectory = (Resolve-Path (Join-Path $PSScriptRoot 'wsl\patches')).Path

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw 'WSL is required and wsl.exe was not found.'
}
$distributions = @(wsl.exe --list --quiet) | ForEach-Object { ($_ -replace "`0", '').Trim() }
if ($Distribution -notin $distributions) {
    throw "WSL distribution not found: $Distribution"
}

$setupScriptWsl = (& wsl.exe --distribution $Distribution --exec wslpath -a $setupScript).Trim()
$patchDirectoryWsl = (& wsl.exe --distribution $Distribution --exec wslpath -a $patchDirectory).Trim()
if (-not $setupScriptWsl -or -not $patchDirectoryWsl) {
    throw 'Failed to translate the WSL setup paths.'
}

if (-not $ProxyUrl -and (Get-Command git -ErrorAction SilentlyContinue)) {
    $ProxyUrl = (git config --global --get https.proxy)
    if (-not $ProxyUrl) {
        $ProxyUrl = (git config --global --get http.proxy)
    }
}
if ($ProxyUrl) {
    try {
        $proxyUri = [Uri]$ProxyUrl
    }
    catch {
        throw "Invalid proxy URL: $ProxyUrl"
    }
    if ($proxyUri.Scheme -notin @('http', 'https')) {
        throw 'The WSL setup proxy must use http or https.'
    }
    if ($proxyUri.IsLoopback) {
        $defaultRoute = (& wsl.exe --distribution $Distribution --exec ip route show default)
        $routeMatch = [regex]::Match(($defaultRoute -join ' '), 'default via ([0-9.]+)')
        if (-not $routeMatch.Success) {
            throw 'Could not map the Windows loopback proxy into WSL.'
        }
        $proxyBuilder = [UriBuilder]$proxyUri
        $proxyBuilder.Host = $routeMatch.Groups[1].Value
        $ProxyUrl = $proxyBuilder.Uri.AbsoluteUri.TrimEnd('/')
    }
}
$proxyArgument = if ($ProxyUrl) { $ProxyUrl } else { '-' }

& wsl.exe --distribution $Distribution --exec /bin/bash `
    $setupScriptWsl `
    $SourceDirectory `
    $BuildDirectory `
    $patchDirectoryWsl `
    $BasaltRevision `
    $proxyArgument
if ($LASTEXITCODE -ne 0) {
    throw "Basalt WSL setup failed with exit code $LASTEXITCODE"
}
