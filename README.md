# Setup

conda env create -f environment.yml --prefix ./.env

# Original Setup

conda create --prefix ./.env python=3.9
conda activate ./.env
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
python -m pip install tensorflow-datasets
conda install jupyter pandas numpy matplotlib scikit-learn
