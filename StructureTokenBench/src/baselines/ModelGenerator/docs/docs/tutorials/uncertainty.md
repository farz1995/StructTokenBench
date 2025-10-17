# Uncertainty Estimation

Datasets implementing the `DataInterface` support semi-automatic k-fold crossvalidation for uncertainty estimation. 

First, create your experiment config.
Then, train a model on each of the folds.

```
# 10-fold cross-validation
for FOLD in {0..9}
do
    RUN_NAME=${FOLD}
    CKPT_SAVE_DIR=logs/my_experiment/${FOLD}
    mgen fit --config my_config.yaml \
        --cv_num_folds: 10 \
        --cv_enable_val_fold: true \
        --data.cv_test_fold_id $FOLD \
        --trainer.callbacks.dirpath $CKPT_SAVE_DIR
done
```

After training, you can evaluate the model on each fold.

```
for FOLD in {0..9}
do
    CKPT_PATH=logs/my_experiment/${FOLD}/best_val*
    mgen test --config my_config.yaml \
        --cv_num_folds: 10 \
        --cv_enable_val_fold: true \
        --data.cv_test_fold_id $FOLD \
        --ckpt_path $CKPT_PATH
        --model.strict_loading true \
        --model.reset_optimizer_states true
done
```