from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect data files located in the embedchain module
datas = collect_data_files('embedchain.llm')

# Collect all submodules of embedchain to ensure they are included
hiddenimports = collect_submodules('embedchain.llm')