For downloading the full data

Refer to: 


# Download
## Model Checkpoints
```bash
CKPT_DIR=$DIR/struct_token_bench_release_ckpt
cd $CKPT_DIR
gdown https://drive.google.com/drive/folders/1s6mz6MQ7x1XLjt4veET7QT5fZ43_xO7n -O ./codebook_512x1024-1e+19-linear-fixed-last.ckpt --folder
gdown https://drive.google.com/drive/folders/1hl7gAe_Hn1pYQ3ow790ArISVbJ2lmJ8b -O ./codebook_512x1024-1e+19-PST-last.ckpt --folder
```


## Pre-training Datasets
First download all the pdb files, which would also be useful for downstreams: 
```bash
DOWNLOAD_DIR=$DIR/pdb_data/mmcif_files
cd $DOWNLOAD_DIR
aws s3 cp s3://openfold/pdb_mmcif.zip $DOWNLOAD_DIR --no-sign-request
unzip pdb_mmcif.zip
wget https://files.pdbj.org/pub/pdb/data/status/obsolete.dat
```
which should result in the following file structure:
```
├── pdb_data
│   └── mmcif_files
│       ├── mmcif_files
│       │   └──xxx.cif
│       ├── obsolete.dat
```

Then download the pretraining subsampled pdb indices list:
```bash
DOWNLOAD_DIR=$DIR/pdb_data/
cd $DOWNLOAD_DIR
gdown https://drive.google.com/uc?id=1UGPbnxeNwlg1jt514J6Foo07pQJEizHy
unzip pretrain.zip
mv pretrain_zip pretrain
```

## Downstream Datasets
Using the following command: 

```
cd $DIR
gdown https://drive.google.com/uc?id=1wJ4dSNdMyuF0985ET4UuwViHgV-clF4K
unzip struct_token_bench_release_data_download.zip
mv struct_token_bench_release_data_download struct_token_bench_release_data
```
which should result in the following file structure:
```
├── struct_token_bench_release_data
│   ├── data
│       ├── CATH
│       │   ├── cath-classification-data
│       │   └── sequence-data
│       ├── functional
│       │   └── local
│       │       ├── biolip2
│       │       ├── interpro
│       │       ├── proteinglue_epitoperegion
│       │       └── proteinshake_bindingsite
│       ├── physicochemical
│       │   ├── atlas
│       ├── sensitivity
│       │   ├── conformational
│       ├── structural
│       │   ├── remote_homology
│       ├── utility
│       │   ├── cameo
│       │   └── casp14
