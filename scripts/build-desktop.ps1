param(
    [switch]$SkipSync
)

$ErrorActionPreference = "Stop"
$clientRoot = Split-Path -Parent $PSScriptRoot
$pyinstaller = Join-Path $clientRoot ".venv\Scripts\pyinstaller.exe"

Push-Location $clientRoot
try {
    if (-not $SkipSync) {
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if ($null -eq $uv) {
            throw "uv.exe was not found on PATH. Install uv, or use -SkipSync with an existing .venv."
        }

        & $uv.Source sync --group dev
        if ($LASTEXITCODE -ne 0) {
            throw "Dependency synchronization failed with exit code $LASTEXITCODE"
        }
    }

    if (-not (Test-Path -LiteralPath $pyinstaller -PathType Leaf)) {
        throw "PyInstaller was not found at $pyinstaller. Run uv sync --group dev first."
    }

    & $pyinstaller --noconfirm --clean packaging\egoglass-console.spec
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
