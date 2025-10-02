# RNA Secondary Structure Prediction
As with proteins, structure determines RNA function. RNA secondary structure, formed by base pairing, is more stable and accessible than its tertiary form within cells. Accurate prediction of RNA secondary structure is essential for tasks such as higher-order structure prediction and function prediction. As discussed in our paper [AIDO.RNA](https://doi.org/10.1101/2024.11.28.625345), we finetune the [AIDO.RNA-1.6B](https://huggingface.co/genbio-ai/AIDO.RNA-1.6B) model on the training splits of the following two datasets:
1. [bpRNA](https://doi.org/10.1093/nar/gky285)
2. [Archive-II](http://www.rnajournal.org/cgi/doi/10.1261/rna.053694.115)

We preprocessed and split the datasets (into train, test, and validation splits) in the same way as done in a previous study [RiNALMo](https://doi.org/10.48550/arXiv.2403.00043). 


#### To finetune [AIDO.RNA-1.6B](https://huggingface.co/genbio-ai/AIDO.RNA-1.6B) on RNA SS:

- Set the environment variable for ModelGenerator's data directory:
    ```
    export MGEN_DATA_DIR=~/mgen_data # or any other local directory of your choice
    ```

- Download the preprocessed data (provided as zip file named `rna_ss_data.zip`) from [here](https://huggingface.co/datasets/genbio-ai/rna-secondary-structure-prediction/blob/main/rna_ss_data.zip). Unzip `rna_ss_data.zip` inside the directory `${MGEN_DATA_DIR}/modelgenerator/datasets/`. 
    
    **Alternatively**, you can simply run the following script to do this:
    ```
    mkdir -p ${MGEN_DATA_DIR}/modelgenerator/datasets/
    wget -P ${MGEN_DATA_DIR}/modelgenerator/datasets/ https://huggingface.co/datasets/genbio-ai/rna-secondary-structure-prediction/resolve/main/rna_ss_data.zip
    unzip ${MGEN_DATA_DIR}/modelgenerator/datasets/rna_ss_data.zip -d ${MGEN_DATA_DIR}/modelgenerator/datasets/
    ```
    
    You should find two sub-folders containing the preprocessed datasets:
    1. bpRNA: `${MGEN_DATA_DIR}/modelgenerator/datasets/rna_ss_data/bpRNA`
    2. Archive-II: `${MGEN_DATA_DIR}/modelgenerator/datasets/rna_ss_data/archiveII`

- Then run a finetuning job on either dataset as following (Note that here we are using finetuning scheduler. See [this tutorial](https://github.com/genbio-ai/ModelGenerator/blob/main/docs/docs/tutorials/finetuning_scheduler.md) for details):
    1. To train on bpRNA dataset, run the following command:
    ```
    bash rna_secondary_structure_prediction.sh train bpRNA
    ```
    2. Alternatively, to finetune on Archive-II datasets (for the inter-family generalization experiment discussed in the paper [AIDO.RNA](https://doi.org/10.1101/2024.11.28.625345)), run the following command:
    ```
    bash rna_secondary_structure_prediction.sh train archiveII_<FamilyName>
    ```
    Here, `<FamilyName>` is any of the following nine strings (representing different RNA families in Archive-II dataset): `5s, 16s, 23s, grp1, srp, telomerase, RNaseP, tmRNA, tRNA`. Note that, following the conventioned using by [RiNALMo's code repository](https://github.com/lbcb-sci/RiNALMo/tree/main), when a `<FamilyName>` is chosen, it will only be used as the **test set** and the rest of the families are used for training and validation. One example finetuning run with `5s` family:
    ```
    bash rna_secondary_structure_prediction.sh train archiveII_5s
    ```
    Here, the [AIDO.RNA-1.6B](https://huggingface.co/genbio-ai/AIDO.RNA-1.6B) model will be finetuned using **all other splits except archiveII_5s**.

#### To test a finetuned checkpoint on RNA SS:
- Finetune [AIDO.RNA-1.6B](https://huggingface.co/genbio-ai/AIDO.RNA-1.6B) as discussed above, **or** download the `model.ckpt` checkpoint from [here](https://huggingface.co/genbio-ai/AIDO.RNA-1.6B-inv-fold).
- Test the checkpoint on the _corresponding dataset_ as following (replace `/path/to/checkpoint` with the actual path to the finetuned checkpoint):
    1. To test on bpRNA dataset, run the following command:
    ```
    bash rna_secondary_structure_prediction.sh test bpRNA /path/to/checkpoint
    ```
    2. Alternatively, to test on Archive-II datasets, run the following command:
    ```
    bash rna_secondary_structure_prediction.sh test archiveII_<FamilyName> /path/to/checkpoint
    ```
    See the previous section for details on `<FamilyName>`.

#### Outputs:
- The evaluation scores will be printed on the console.

