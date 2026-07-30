[CmdletBinding()]
param(
    [int]$IngestPort = 8770,
    [int]$DiscoveryPort = 8771,
    [string]$EnvironmentName = 'egoglass'
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot

function Assert-TcpPortAvailable([int] $Port) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    if ($null -ne $listener) {
        throw "TCP port $Port is already in use. Close the existing EgoGlass client first."
    }
}

function Assert-UdpPortAvailable([int] $Port) {
    $endpoint = Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue
    if ($null -ne $endpoint) {
        throw "UDP port $Port is already in use. Close the existing EgoGlass client first."
    }
}

if ($env:OS -ne 'Windows_NT') {
    throw 'The EgoGlass client launcher requires Windows.'
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw 'conda is required and was not found on PATH'
}
if ($IngestPort -notin 1..65535 -or $DiscoveryPort -notin 1..65535) {
    throw 'IngestPort and DiscoveryPort must be valid ports.'
}

$condaBase = (conda info --base).Trim()
$workspacePython = Join-Path $condaBase "envs\$EnvironmentName\python.exe"
if (-not (Test-Path -LiteralPath $workspacePython -PathType Leaf)) {
    throw (
        "Missing client environment: $workspacePython. " +
        "Run .\scripts\setup_client.ps1 -EnvironmentName $EnvironmentName first."
    )
}

Assert-TcpPortAvailable -Port $IngestPort
Assert-UdpPortAvailable -Port $DiscoveryPort
$recordingsDirectory = Join-Path $repositoryRoot 'local-data\recordings'
New-Item -ItemType Directory -Force -Path $recordingsDirectory | Out-Null

Write-Host 'Starting the unified EgoGlass native client.'
Write-Host 'Close the native window or press Ctrl+C to stop gateway, inference, and UI.'
Push-Location $repositoryRoot
try {
    & $workspacePython -m ui `
        --host 0.0.0.0 `
        --port $IngestPort `
        --discovery-port $DiscoveryPort `
        --recordings-root $recordingsDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "EgoGlass client exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
