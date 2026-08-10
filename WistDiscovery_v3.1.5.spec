# -*- mode: python ; coding: utf-8 -*-
a = Analysis(['gui_wist_discovery\\main.py'], pathex=[], binaries=[],
    datas=[('agents', 'agents'), ('environments', 'environments'), ('intelligence', 'intelligence'), ('gui_wist_discovery', 'gui_wist_discovery'), ('gui_wist', 'gui_wist')],
    hiddenimports=['pygame', 'agents.wist_discovery.discovery_agent', 'agents.wist_discovery.neural_net', 'agents.wist_discovery.mcts', 'environments.wist', 'environments.wist.environment', 'environments.wist.round', 'environments.wist.rules', 'environments.wist.scoring', 'environments.wist.setup', 'environments.wist.tasmiya_engine', 'environments.wist.trick', 'intelligence.core', 'intelligence.core.cards', 'gui_wist_discovery.game_engine', 'gui_wist_discovery.training', 'gui_wist_discovery.milestones', 'gui_wist_discovery.insights', 'gui_wist_discovery.renderer', 'gui_wist_discovery.constants'],
    hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=[], noarchive=False, optimize=0,)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='WistDiscovery_v3.1.5', debug=False, bootloader_ignore_signals=False, strip=False, upx=True, upx_exclude=[], runtime_tmpdir=None, console=False, disable_windowed_traceback=False, argv_emulation=False, target_arch=None, codesign_identity=None, entitlements_file=None,)
