[CmdletBinding(DefaultParameterSetName = 'Hours')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Hours')]
    [ValidateRange(0.001, 192)]
    [double]$DurationHours,
    [Parameter(Mandatory = $true, ParameterSetName = 'UntilInterrupted')]
    [switch]$UntilInterrupted,
    [int]$IngestPort = 8770,
    [int]$DiscoveryPort = 8771,
    [string]$EnvironmentName = 'egoglass',
    [string]$OutputRoot = '',
    [string]$AdbSerial = '',
    [switch]$DisableAdbPreparation
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ($env:OS -ne 'Windows_NT') { throw 'The IMU calibration launcher requires Windows.' }
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) { throw 'conda is required and was not found on PATH' }

if (Get-NetTCPConnection -State Listen -LocalPort $IngestPort -ErrorAction SilentlyContinue) {
    throw "TCP port $IngestPort is already in use. Close the normal EgoGlass client first."
}
if (Get-NetUDPEndpoint -LocalPort $DiscoveryPort -ErrorAction SilentlyContinue) {
    throw "UDP port $DiscoveryPort is already in use. Close the normal EgoGlass client first."
}

$condaBase = (conda info --base).Trim()
$workspacePython = Join-Path $condaBase "envs\$EnvironmentName\python.exe"
if (-not (Test-Path -LiteralPath $workspacePython -PathType Leaf)) {
    throw "Missing client environment: $workspacePython"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repositoryRoot 'local-data\imu-calibration'
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$arguments = @('-m', 'ui.imu_calibration.app', '--port', $IngestPort, '--discovery-port', $DiscoveryPort, '--output-root', $OutputRoot)
if (-not $DisableAdbPreparation) {
    $adb = Get-Command adb -ErrorAction SilentlyContinue
    if (-not [string]::IsNullOrWhiteSpace($AdbSerial)) {
        if ($null -eq $adb) { throw 'adb is required when -AdbSerial is supplied' }
    } elseif ($null -ne $adb) {
        $AdbSerials = @(adb devices | Select-Object -Skip 1 | ForEach-Object {
            if ($_ -match '^([^\s]+)\s+device$') { $Matches[1] }
        })
        if ($AdbSerials.Count -gt 1) {
            throw 'Multiple ADB devices are connected. Select the Glass3 with -AdbSerial.'
        }
        if ($AdbSerials.Count -eq 1) { $AdbSerial = $AdbSerials[0] }
    }
    if (-not [string]::IsNullOrWhiteSpace($AdbSerial)) {
        $arguments += @('--adb-serial', $AdbSerial)
    }
}
if ($UntilInterrupted) { $arguments += '--until-interrupted' }
else { $arguments += @('--duration-seconds', [string]($DurationHours * 3600)) }

Push-Location $repositoryRoot
try {
    & $workspacePython @arguments
    exit $LASTEXITCODE
} finally { Pop-Location }
