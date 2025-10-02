# Benchmarking Method Set up

## ESM3
```bash
pip install esm
```

## FoldSeek
```bash
pip install mini3di
```

## ProTokens 
```bash
# download model weights from source

gdown --id '1z2X_Ly-HXpDryqJIGtnOCuddQi_eIoHS' # => ProToken-1.0.tar.xz
tar -xJf './ProToken-1.0.tar.xz'
cp -r ProToken-1.0/models/ ./ProToken/
rm -r ProToken-1.0/
```

## Cheap
```bash
git clone https://github.com/amyxlu/cheap-proteins.git
mv cheap-proteins cheap_proteins
pip install --no-deps git+https://github.com/amyxlu/openfold.git 
pip install dm-tree
pip install modelcif
pip install ml_collections

# to install both fair-esm and esm
pip uninstall esm
pip install fair-esm
mv /home/mila/x/xinyu.yuan/anaconda/envs/evo2/lib/python3.11/site-packages/esm/ /home/mila/x/xinyu.yuan/anaconda/envs/evo2/lib/python3.11/site-packages/esm2/
# modify all "esm." imports to "esm2."
vi /network/scratch/x/xinyu.yuan/StrucTokenBench/src/baselines/cheap-proteins/src/cheap/esmfold/_esmfold.py
# modify all "esm." imports to "esm2."
pip install esm

export CHEAP_CACHE=$DIR/cheap_cache
```
test env with:

```python
from src.cheap.pretrained import CHEAP_shorten_1_dim_64, CHEAP_shorten_2_dim_1024
pipeline = CHEAP_shorten_1_dim_64(return_pipeline=True)
```