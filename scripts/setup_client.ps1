[CmdletBinding()]
param(
    [string]$EnvironmentName = "egoglass"
)

$ErrorActionPreference = "Stop"
$clientRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$environmentFile = Join-Path $clientRoot "environment.yml"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )

    & $Executable @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $ArgumentList"
    }
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda is required and was not found on PATH"
}

$condaBase = (conda info --base).Trim()
$pythonPath = Join-Path $condaBase "envs\$EnvironmentName\python.exe"
if (Test-Path -LiteralPath $pythonPath -PathType Leaf) {
    Invoke-Checked -Executable conda -ArgumentList @(
        "env", "update", "--name", $EnvironmentName, "--file", $environmentFile
    )
} else {
    Invoke-Checked -Executable conda -ArgumentList @(
        "env", "create", "--name", $EnvironmentName, "--file", $environmentFile
    )
}

Push-Location $clientRoot
try {
    Invoke-Checked -Executable $pythonPath -ArgumentList @(
        "-m", "pip", "install", "--no-deps", "-e", "."
    )
} finally {
    Pop-Location
}
