# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui_wist\\main.py'],
    pathex=[],
    binaries=[],
    datas=[('agents', 'agents'), ('environments', 'environments'), ('intelligence', 'intelligence'), ('gui_wist', 'gui_wist')],
    hiddenimports=['pygame', 'agents.wist_rule_based.rule_based_agent', 'agents.wist_learning.learning_agent', 'environments.wist', 'intelligence.core'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SudaneseWist_v1.2.2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
