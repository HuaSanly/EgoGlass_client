from pathlib import Path

client_root = Path(SPECPATH).parent
source_root = client_root / "src"
static_root = source_root / "operator_console" / "static"
version_file = client_root / "packaging" / "windows-version-info.txt"
notices_file = client_root / "packaging" / "THIRD_PARTY_NOTICES.txt"

a = Analysis(
    [str(client_root / "packaging" / "desktop-entry.py")],
    pathex=[str(source_root)],
    binaries=[],
    datas=[
        (str(static_root), "operator_console/static"),
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
