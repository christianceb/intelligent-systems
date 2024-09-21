```powershell
# create env
conda create --name cnn python=3.12


# list environments (same effects)
conda env list
conda info --envs

# activate environment
conda activate cnn

# list packages installed
conda list

# install jupyter
conda install -c anaconda jupyter

# install pip packages (you can copy/paste and run em all!)
# to test: might better if you could use tensorflow-gpu instead of tensorflow!
pip install scikit-learn imutils tqdm pillow scikit-image seaborn;
pip install tensorflow;
pip install opencv-python;
pip install opencv-contrib-python;

# run jupyter notebook
jupyter notebook

# deactivate environment (already does this when you close the terminal)
conda deactivate cnn
```