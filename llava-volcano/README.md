```powershell
# create env
conda create --name llava python=3.12


# list environments (same effects)
conda env list
conda info --envs

# activate environment
conda activate llava

# list packages installed
conda list

# install pip packages (you can copy/paste and run em all!)
pip install --upgrade pip
pip install --upgrade Pillow

# run jupyter notebook
jupyter notebook

# deactivate environment (already does this when you close the terminal)
conda deactivate llava
```