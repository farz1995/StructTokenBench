import os
import sys
from glob import glob
import hydra
import omegaconf

import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter

# local imports
exc_dir = os.path.dirname(os.path.dirname(__file__))  # "src/"
sys.path.append(exc_dir)
exc_dir_baseline = os.path.join(os.path.abspath(exc_dir), "baselines")
all_baseline_names = glob(exc_dir_baseline + "/*")
for name in all_baseline_names:
    if name != "cheap_proteins":
        sys.path.append(os.path.join(exc_dir_baseline, name))
sys.path.append(exc_dir_baseline)

import data_module
from util import setup_loggings

def setup_trainer(cfg):
    trainer_logger = hydra.utils.instantiate(cfg.lightning.logger)

    # strategy determines distributed training
    if cfg.deepspeed_path:
        strategy = hydra.utils.instantiate(cfg.lightning.strategy)
    else:
        strategy = "ddp"  # distributed data parallel

    # callbacks
    callbacks = [
        hydra.utils.instantiate(cfg.lightning.callbacks.checkpoint),
        hydra.utils.instantiate(cfg.lightning.callbacks.lr_monitor),
        hydra.utils.instantiate(cfg.lightning.callbacks.progress_bar),
    ]

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        plugins=[],
        strategy=strategy,
        logger=trainer_logger,
    )
    return trainer


@hydra.main(version_base=None, config_path="config")
def main(cfg):
    """
    Launch supervised fine-tuning using a hydra config, for protein classification
    (Homo / Remote Homology). This version is adjusted to work with a continuous
    H5 tokenizer and a 45-class remapped label space for Table-2 comparability.
    """
    omegaconf.OmegaConf.resolve(cfg)

    # set up python loggings
    logger = setup_loggings(cfg)

    # forbidden restart model
    assert cfg.model.ckpt_path is None
    # set seed before initializing models
    pl.seed_everything(cfg.optimization.seed)

    # set up trainer
    trainer = setup_trainer(cfg)

    # ------------------- DataModule wiring -------------------
    # Make sure the dataset looks at the correct label key by default.
    # Homo uses fold labels; our DataModule later canonicalizes to "label".
    if not hasattr(cfg.data, "target_field") or cfg.data.target_field is None:
        cfg.data.target_field = "fold_label"

    # these are mirrored into the DataModule as expected by the repo
    cfg.data.multi_label = cfg.model.multi_label
    cfg.data.is_global_or_local = cfg.model.is_global_or_local

    # (Only used for the optional pretrained tokenizer path not your current case)
    if cfg.tokenizer == "WrappedOurPretrainedTokenizer":
        assert cfg.tokenizer_pretrained_ckpt_path is not None
        assert cfg.tokenizer_ckpt_name is not None
        tmp_cfg = omegaconf.OmegaConf.load(
            os.path.join(exc_dir, "./script/config/pretrain.yaml")
        )["model"]
        tmp_cfg.quantizer.freeze_codebook = True
        tmp_cfg.quantizer._need_init = False
        tmp_cfg.quantizer.use_linear_project = cfg.quantizer_use_linear_project
        tmp_cfg.encoder.d_model = cfg.model_encoder_dmodel
        tmp_cfg.encoder.n_layers = cfg.model_encoder_nlayers
        tmp_cfg.encoder.v_heads = cfg.model_encoder_vheads
        tmp_cfg.quantizer.codebook_size = cfg.quantizer_codebook_size
        tmp_cfg.quantizer.codebook_embed_size = cfg.quantizer_codebook_embed_size
        tmp_cfg.encoder.d_out = cfg.model_encoder_dout

        pretrained_model_cfg = {
            "model_cfg": tmp_cfg,
            "pretrained_ckpt_path": cfg.tokenizer_pretrained_ckpt_path,
            "ckpt_name": cfg.tokenizer_ckpt_name,
        }
    else:
        pretrained_model_cfg = {}

    # Build datamodule
    # Prefer explicit tokenizer_kwargs from cfg if provided; otherwise fall back to
    # the pretrained_model_cfg (used only for WrappedOurPretrainedTokenizer).
    tk_kwargs = {}
    if isinstance(getattr(cfg, "tokenizer_kwargs", None), (dict, omegaconf.dictconfig.DictConfig)):
        tk_kwargs = dict(cfg.tokenizer_kwargs)
    elif isinstance(pretrained_model_cfg, dict):
        tk_kwargs = dict(pretrained_model_cfg)

    datamodule = data_module.ProteinDataModule(
        tokenizer_name=cfg.tokenizer,
        tokenizer_device=getattr(cfg, "tokenizer_device", "cpu"),
        seed=cfg.optimization.seed,
        micro_batch_size=cfg.optimization.micro_batch_size,
        data_args=cfg.data,
        py_logger=logger,
        test_only=getattr(cfg, "test_only", False),
        precompute_tokens=getattr(cfg, "precompute_tokens", False),
        tokenizer_kwargs=tk_kwargs,
    )
    datamodule.setup()  # our edits inside DataModule do: fold_label->label, 45-class remap before sharding

    # ------------------- Tokenizer & model shape sync -------------------
    # Detect whether this tokenizer is continuous (H5 features) or discrete (LM tokens).
    try:
        _num_tokens = datamodule.get_tokenizer().get_num_tokens()
    except Exception:
        _num_tokens = None

    # If discrete, pass vocab size to the model; if continuous, skip.
    if _num_tokens is not None:
        cfg.model.num_tokens = _num_tokens
        cfg.data.use_continuous = False
        logger.info(f"[run] Discrete tokenizer detected (num_tokens={_num_tokens}).")
    else:
        cfg.data.use_continuous = True
        logger.info("[run] Continuous tokenizer detected (H5 features).")

    # If the DataModule computed a 45-class mapping, force the head size to 45
    # so it's consistent with Table-2 Homo setting.
    if hasattr(datamodule, "num_labels_for_model") and datamodule.num_labels_for_model:
        cfg.model.num_labels = int(datamodule.num_labels_for_model)
        logger.info(f"[run] Setting model.num_labels = {cfg.model.num_labels} (45-class remap)")

    # Pass sequence usage flag through
    cfg.model.use_sequence = cfg.data.use_sequence

    # ------------------- LightningModule instantiation -------------------
    model = hydra.utils.instantiate(
        cfg.lightning.model_module,
        _recursive_=False,
        model_cfg=cfg.model,
        trainer=trainer,
        py_logger=logger,
        optimizer_cfg=cfg.optimization,
        all_split_names=datamodule.all_split_names,
        codebook_embedding=None if cfg.data.use_continuous else datamodule.get_codebook_embedding(),
    )

    # ------------------- Train / Validate flow -------------------
    if not getattr(cfg, "validate_only", False) and not getattr(cfg, "test_only", False):
        logger.info("*********** start training ***********\n")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.model.ckpt_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        logger.info(f"Saving final model weights to {cfg.save_dir_path}")
        trainer.save_checkpoint(cfg.save_dir_path, weights_only=True)
        logger.info("Finished saving final model")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        trainer.validate(model=model, datamodule=datamodule, ckpt_path="best")
    else:
        logger.info("*********** start validation ***********\n")
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.model.ckpt_path)


if __name__ == "__main__":
    main()
