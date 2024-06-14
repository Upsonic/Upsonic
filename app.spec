# -*- mode: python ; coding: utf-8 -*-

import site
import os
site_pkgs_dir = site.getsitepackages()[0]  # Get the path to site-packages directory

block_cipher = None

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[('gpt_computer_assistant/utils/media/*', 'gpt_computer_assistant/utils/media')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Adding manually all the libraries installed in site-packages
a.binaries += [("ext_libs", name, 'BINARY') for name in os.listdir(site_pkgs_dir) if not os.path.isdir(os.path.join(site_pkgs_dir, name))]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='GPT_Computer_Assistant',
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
    icon='gpt_computer_assistant/utils/media/icon.ico'
)
