param(
    [string]$EnvironmentName = "egoglass"
)

$ErrorActionPreference = "Stop"
$clientRoot = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda is required and was not found on PATH"
}
$condaBase = (conda info --base).Trim()
$pyinstaller = Join-Path $condaBase "envs\$EnvironmentName\Scripts\pyinstaller.exe"

Push-Location $clientRoot
try {
    if (-not (Test-Path -LiteralPath $pyinstaller -PathType Leaf)) {
        throw "PyInstaller was not found at $pyinstaller. Run .\scripts\setup_client.ps1 -EnvironmentName $EnvironmentName first."
    }

    & $pyinstaller --noconfirm --clean packaging\egoglass-client.spec
    if ($LASTEXITCODE -ne 0) {
        throw "Desktop build failed with exit code $LASTEXITCODE"
    }

    $executable = Join-Path $clientRoot "dist\EgoGlass\EgoGlass.exe"
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Desktop executable was not created at $executable"
    }

    $noticesSource = Join-Path $clientRoot "packaging\THIRD_PARTY_NOTICES.txt"
    $noticesTarget = Join-Path (Split-Path -Parent $executable) "THIRD_PARTY_NOTICES.txt"
    Copy-Item -LiteralPath $noticesSource -Destination $noticesTarget -Force
    if (-not (Test-Path -LiteralPath $noticesTarget -PathType Leaf)) {
        throw "Third-party notices were not copied to $noticesTarget"
    }

    & $executable --smoke-test
    if ($LASTEXITCODE -ne 0) {
        throw "Packaged desktop smoke test failed with exit code $LASTEXITCODE"
    }

    Write-Output "Desktop build verified: $executable"
}
finally {
    Pop-Location
}
