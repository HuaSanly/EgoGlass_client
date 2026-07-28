param(
    [string]$EnvironmentName = "egoglass-hamer"
)

$ErrorActionPreference = "Stop"
$clientRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda is required and was not found on PATH"
}
$condaBase = (conda info --base).Trim()
$pythonPath = Join-Path $condaBase "envs\$EnvironmentName\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    conda create -n $EnvironmentName python=3.11 -y
} else {
    conda install -n $EnvironmentName python=3.11 -y
}

& $pythonPath -m pip install `
    torch==2.5.1 torchvision==0.20.1 `
    --index-url https://download.pytorch.org/whl/cu121

& $pythonPath -m pip install `
    mediapipe==1.0.0 huggingface_hub==1.25.1 yacs==0.1.8 `
    pytorch-lightning==2.6.5 gdown==6.1.0 xtcocotools==1.14.3 `
    webdataset==1.0.2 filterpy==1.4.5 ffmpeg-python==0.2.0 `
    smplx==0.1.28 pyrender==0.1.45 scikit-image==0.26.0 `
    timm==1.0.28 einops==0.8.2 pandas==3.0.5 hydra-core==1.3.4 `
    pyrootutils==1.0.4 rich==15.0.0 ultralytics==8.3.107 roma==1.5.7 `
    pydantic==2.13.4 av==16.1.0 hatchling==1.27.0 pytest==8.4.2

& $pythonPath -m pip install --no-build-isolation chumpy==0.70
& $pythonPath -m pip install --no-deps `
    "hamer @ git+https://github.com/geopavlakos/hamer.git@3a01849f4148352e9260b69bf28b65d1671a4905"
& $pythonPath -m pip install --no-deps `
    "easy_ViTPose @ git+https://github.com/JunkyByte/easy_ViTPose.git@bb9860359e55b099a507c8000e360d48a27cc36d"
& $pythonPath -m pip install --no-deps `
    "wilor-mini @ git+https://github.com/warmshao/WiLoR-mini.git@ebec42f94c389070cdd7dda6fd1bf0b4a659c960"

$sitePackages = & $pythonPath -c "import site; print(site.getsitepackages()[0])"
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
    & $pythonPath -m pip install --no-deps -e .
    & $pythonPath -c "import torch; assert torch.cuda.is_available(); value=(torch.tensor([2.0], device='cuda')**2).item(); assert value == 4.0; print(torch.__version__, torch.cuda.get_device_name(0))"
} finally {
    Pop-Location
}
