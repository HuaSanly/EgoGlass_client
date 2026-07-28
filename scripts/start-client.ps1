[CmdletBinding()]
param(
    [int]$IngestPort = 8770,
    [int]$DiscoveryPort = 8771,
    [string]$EnvironmentName = "egoglass"
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'client-process-lifecycle.ps1')
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw 'conda is required and was not found on PATH'
}
$condaBase = (conda info --base).Trim()
$workspacePython = Join-Path $condaBase "envs\$EnvironmentName\python.exe"
$desktopPython = Join-Path $condaBase "envs\$EnvironmentName\pythonw.exe"
$dataRoot = if ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA 'EgoGlass'
} else {
    Join-Path $HOME '.egoglass'
}
$logDirectory = Join-Path $dataRoot 'logs'
$recordingsDirectory = Join-Path $repositoryRoot 'local-data\recordings'
$ingestStdout = Join-Path $logDirectory 'ingest.stdout.log'
$ingestStderr = Join-Path $logDirectory 'ingest.stderr.log'

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

function New-RuntimePairingToken {
    $bytes = New-Object byte[] 32
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    } finally {
        $generator.Dispose()
    }
    return [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
}

function Wait-IngestHealth([int] $Port, [int] $TimeoutSeconds = 15) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        Start-Sleep -Milliseconds 200
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/v1/health" `
                -TimeoutSec 1
        } catch {
            $health = $null
        }
    } while ($null -eq $health -and (Get-Date) -lt $deadline)

    if ($null -eq $health -or $health.status -ne 'ok') {
        throw "EgoGlass ingest gateway did not become ready. Log: $ingestStderr"
    }
}

if ($env:OS -ne 'Windows_NT') {
    throw 'The EgoGlass client launcher requires Windows.'
}
foreach ($executable in @($workspacePython, $desktopPython)) {
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Missing client environment: $executable. Run .\scripts\setup_client.ps1 -EnvironmentName $EnvironmentName from the client root."
    }
}
if ($IngestPort -notin 1..65535 -or $DiscoveryPort -notin 1..65535) {
    throw 'IngestPort and DiscoveryPort must be valid ports.'
}

Assert-TcpPortAvailable -Port $IngestPort
Assert-UdpPortAvailable -Port $DiscoveryPort
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $recordingsDirectory | Out-Null

$pairingToken = New-RuntimePairingToken
$previousPairingToken = $env:EGOGLASS_PAIRING_TOKEN
$previousRecordingsRoot = $env:EGOGLASS_RECORDINGS_ROOT
$ingestProcess = $null
$desktopProcess = $null
$processJob = New-EgoGlassProcessJob
try {
    try {
        $env:EGOGLASS_PAIRING_TOKEN = $pairingToken
        $ingestProcess = Start-Process -FilePath $workspacePython -ArgumentList @(
            '-m',
            'ingest_gateway.app',
            '--host',
            '0.0.0.0',
            '--port',
            $IngestPort,
            '--discovery-port',
            $DiscoveryPort,
            '--recordings-root',
            $recordingsDirectory,
            '--hide-pairing-token'
        ) -WorkingDirectory $repositoryRoot -WindowStyle Hidden `
            -RedirectStandardOutput $ingestStdout -RedirectStandardError $ingestStderr -PassThru
        Add-ProcessTreeToJob -Job $processJob -ProcessId $ingestProcess.Id
    } finally {
        $env:EGOGLASS_PAIRING_TOKEN = $previousPairingToken
    }

    Wait-IngestHealth -Port $IngestPort
    try {
        $env:EGOGLASS_RECORDINGS_ROOT = $recordingsDirectory
        $desktopProcess = Start-Process -FilePath $desktopPython -ArgumentList @(
            '-m',
            'operator_console.desktop'
        ) -WorkingDirectory $repositoryRoot -PassThru
        Add-ProcessTreeToJob -Job $processJob -ProcessId $desktopProcess.Id
    } finally {
        $env:EGOGLASS_RECORDINGS_ROOT = $previousRecordingsRoot
    }

    Write-Host 'EgoGlass client is ready.'
    Write-Host 'Now open EgoGlass directly from the Glass3 application list.'
    Write-Host 'Close the EgoGlass window or press Ctrl+C here to stop the client.'
    Wait-Process -Id $desktopProcess.Id
} finally {
    if ($null -ne $ingestProcess -and -not $ingestProcess.HasExited) {
        try {
            Complete-EgoGlassCaptureSession -IngestPort $IngestPort
        } catch {
            Write-Warning (
                "Capture-session finalization failed; forcing process cleanup: " +
                $_.Exception.Message
            )
        }
    }
    Stop-ClientProcesses -Processes @($desktopProcess, $ingestProcess) `
        -ProcessJob $processJob
    Write-Host 'EgoGlass client stopped. Runtime ports have been released.'
}
