from pathlib import Path

service_root = Path(SPECPATH).parent
source_root = service_root / "src"
static_root = source_root / "egoglass_operator_console" / "static"
version_file = service_root / "packaging" / "windows-version-info.txt"
notices_file = service_root / "packaging" / "THIRD_PARTY_NOTICES.txt"

a = Analysis(
    [str(service_root / "packaging" / "desktop-entry.py")],
    pathex=[str(source_root)],
    binaries=[],
    datas=[
        (str(static_root), "egoglass_operator_console/static"),
        (str(notices_file), "."),
    ],
    hiddenimports=[
        "webview.platforms.edgechromium",
        "webview.platforms.win32",
        "webview.platforms.winforms",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EgoGlass",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=str(version_file),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="EgoGlass",
)
