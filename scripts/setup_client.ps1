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
        "env", "update", "--name", $EnvironmentName, "--file", $environmentFile, "--prune"
    )
} else {
    Invoke-Checked -Executable conda -ArgumentList @(
        "env", "create", "--name", $EnvironmentName, "--file", $environmentFile
    )
}

Invoke-Checked -Executable $pythonPath -ArgumentList @(
    "-m", "pip", "install", "--no-build-isolation", "chumpy==0.70"
)
Invoke-Checked -Executable $pythonPath -ArgumentList @(
    "-m", "pip", "install", "--no-deps",
    "hamer @ git+https://github.com/geopavlakos/hamer.git@3a01849f4148352e9260b69bf28b65d1671a4905"
)
Invoke-Checked -Executable $pythonPath -ArgumentList @(
    "-m", "pip", "install", "--no-deps",
    "easy_ViTPose @ git+https://github.com/JunkyByte/easy_ViTPose.git@bb9860359e55b099a507c8000e360d48a27cc36d"
)
$previousSam2BuildCuda = $env:SAM2_BUILD_CUDA
try {
    $env:SAM2_BUILD_CUDA = "0"
    Invoke-Checked -Executable $pythonPath -ArgumentList @(
        "-m", "pip", "install", "--no-deps",
        "sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
    )
}
finally {
    if ($null -eq $previousSam2BuildCuda) {
        Remove-Item Env:SAM2_BUILD_CUDA -ErrorAction SilentlyContinue
    }
    else {
        $env:SAM2_BUILD_CUDA = $previousSam2BuildCuda
    }
}
Invoke-Checked -Executable $pythonPath -ArgumentList @(
    "-m", "pip", "install", "--no-deps",
    "cotracker @ git+https://github.com/facebookresearch/co-tracker.git@82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
)

$sitePackages = & $pythonPath -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
$chumpyCode = Join-Path $sitePackages "chumpy\ch.py"
$chumpyInit = Join-Path $sitePackages "chumpy\__init__.py"
$hamerDataset = Join-Path $sitePackages "hamer\datasets\vitdet_dataset.py"

$content = Get-Content -LiteralPath $chumpyCode -Raw
$content = $content.Replace("inspect.getargspec", "inspect.getfullargspec")
Set-Content -LiteralPath $chumpyCode -Value $content -Encoding UTF8

$content = Get-Content -LiteralPath $chumpyInit -Raw
$content = $content.Replace(
    "from numpy import bool, int, float, complex, object, unicode, str, nan, inf",
    "from numpy import nan, inf"
)
Set-Content -LiteralPath $chumpyInit -Value $content -Encoding UTF8

$content = Get-Content -LiteralPath $hamerDataset -Raw
$content = $content.Replace("            print(f'{downsampling_factor=}')`r`n", "")
$content = $content.Replace("            print(f'{downsampling_factor=}')`n", "")
Set-Content -LiteralPath $hamerDataset -Value $content -Encoding UTF8

Push-Location $clientRoot
try {
    Invoke-Checked -Executable $pythonPath -ArgumentList @(
        "-m", "pip", "install", "--no-deps", "-e", "."
    )
    Invoke-Checked -Executable $pythonPath -ArgumentList @(
        "-c",
        "import torch; assert torch.cuda.is_available(); value=(torch.tensor([2.0], device='cuda')**2).item(); assert value == 4.0; print(torch.__version__, torch.cuda.get_device_name(0))"
    )
} finally {
    Pop-Location
}
