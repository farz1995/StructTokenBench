# Zeroshot Variant Effect Prediction

Zeroshot variant effect prediction refers to the task of predicting the functional impact of genetic variants, especially single nucleotide polymorphisms (SNPs), without requiring additional task-specific fine-tuning of the model.
AIDO.ModelGenerator implements the procedure proposed by [Nucleotide Transformer](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1)
We use this to predict the effects of single nucleotide polymorphisms (SNPs) in the [AIDO.DNA-300M](https://huggingface.co/genbio-ai/AIDO.DNA-300M) model.
This task uses the pre-trained models directly, and does not require finetuning.

To do zeroshot variant effect prediction,

1. Download human reference genome and raw clinvar data to `GENBIO_DATA_DIR/`
**hg38.ml**: [https://hgdownload.soe.ucsc.edu/downloads.html](https://hgdownload.soe.ucsc.edu/downloads.html)

**clinvar data**: [https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/clinvar/](https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/clinvar/)

2. Run `python preprocess_clinvar.py` to process data into `Clinvar_Processed.tsv` file with  `gene_sequence`,`variant_sequence` and `label` column.
3. Run `mgen test --config config.yaml`.
If you want to take the norm distance between reference and variant sequence embeddings as prediction, the config should be
```
model:
  class_path: modelgenerator.tasks.ZeroshotPredictionDistance
  init_args:
    backbone: <you-choose>
data:
  class_path: ClinvarRetrieve
  init_args:
    method: Distance
    window: <window size centered around the SNPs>
    test_split_files:
      - <my_sequences.tsv>
    reference_file: <human_reference_genome.ml.fa> # Example: hg38.ml.fa
trainer:
  callbacks:
  - class_path: modelgenerator.callbacks.PredictionWriter
    dict_kwargs:
      output_dir: predictions
      filetype: tsv
      write_cols: ['score','norm_type','labels','num_layer']
```
If you take the loglikelihood ratio between reference and variant sequence embeddings at the mutation position as prediction, then the config should be
```
model:
  class_path: modelgenerator.tasks.ZeroshotPredictionDiff
  init_args:
    backbone: <you-choose>
data:
  class_path: ClinvarRetrieve
  init_args:
    method: Diff
    window: <window size centered around the SNPs>
    test_split_files:
      - <my_sequences.tsv>
    reference_file: <human_reference_genome.ml.fa> # Example: hg38.ml.fa
trainer:
  callbacks:
  - class_path: modelgenerator.callbacks.PredictionWriter
    dict_kwargs:
      output_dir: predictions
      filetype: tsv
      write_cols: ['score','label']
```
The labels and scores are also saved in `test_predictions.tsv` under the dir specified by `--trainer.callbacks.output_dir`.

Here are two examples of how to load HF model for inference
For norm distance mode
```
mgen test --config Clinvar_300M_zeroshot_Distance.yaml
```
For loglikelihood ratio mode
```
mgen test --config Clinvar_300M_zeroshot_Diff.yaml
```