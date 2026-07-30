# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui_pygame\\main.py'],
    pathex=[],
    binaries=[],
    datas=[('agents', 'agents'), ('environments', 'environments'), ('intelligence', 'intelligence'), ('gui_pygame', 'gui_pygame')],
    hiddenimports=['pygame', 'agents.rule_based.rule_based_agent', 'agents.learning.learning_agent', 'environments.wist', 'intelligence.core'],
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
    name='PlaySudaneseWist-v.1.1.0',
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
    icon=['gui_pygame\\icon.ico'],
)
