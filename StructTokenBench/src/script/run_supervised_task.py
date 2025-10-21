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

    # Merge optional tokenizer_kwargs from Hydra (e.g., h5_path for continuous reps)
    tok_kwargs = {}
    try:
        extra = getattr(cfg, "tokenizer_kwargs", None)
        if extra is not None:
            tok_kwargs = omegaconf.OmegaConf.to_container(extra, resolve=True)
            if not isinstance(tok_kwargs, dict):
                tok_kwargs = {}
    except Exception:
        tok_kwargs = {}
    # For sensitivity / continuous runs, default to using raw features (no quantization)
    if getattr(cfg.data, "use_continuous", False):
        tok_kwargs.setdefault("quantize_continuous", False)
        tok_kwargs.setdefault("fallback_to_any_chain", True)
    # Let explicit Hydra kwargs override defaults from pretrained_model_cfg
    tok_kwargs = {**pretrained_model_cfg, **tok_kwargs}

    # Build datamodule
    datamodule = data_module.ProteinDataModule(
        tokenizer_name=cfg.tokenizer,
        tokenizer_device=getattr(cfg, "tokenizer_device", "cpu"),
        seed=cfg.optimization.seed,
        micro_batch_size=cfg.optimization.micro_batch_size,
        data_args=cfg.data,
        py_logger=logger,
        test_only=getattr(cfg, "test_only", False),
        precompute_tokens=getattr(cfg, "precompute_tokens", False),
        tokenizer_kwargs=tok_kwargs,
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

        # For continuous tokenizers, align model.d_model to tokenizer embed_dim automatically
        try:
            tok = datamodule.get_tokenizer()
            # support multiple tokenizer impls: prefer 'embed_dim', fallback to 'd_model'
            embed_dim = getattr(tok, "embed_dim", None)
            if embed_dim is None:
                embed_dim = getattr(tok, "d_model", None)
            if embed_dim is not None:
                if getattr(cfg.model, "d_model", None) is None or int(cfg.model.d_model) != int(embed_dim):
                    logger.info(f"[run] Setting model.d_model to tokenizer embed_dim={int(embed_dim)}")
                    cfg.model.d_model = int(embed_dim)
        except Exception:
            pass

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
    def _pretty_log_results(results, header="validation"):
        """Log metrics returned by trainer.validate/test in a readable way."""
        try:
            import torch
        except Exception:
            torch = None

        if results is None:
            logger.info(f"No {header} results returned.")
            return

        # Lightning returns a list[dict]
        if isinstance(results, dict):
            results = [results]

        merged = {}
        for d in results or []:
            if not isinstance(d, dict):
                continue
            merged.update(d)

        if not merged:
            logger.info(f"No {header} metrics to display.")
            return

        # Convert tensors to scalars
        flat = {}
        for k, v in merged.items():
            val = v
            if torch is not None and isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    val = v.item()
                else:
                    val = float(v.detach().float().mean().item())
            flat[k] = val

        # Stable order: split metrics first
        ordered = sorted(flat.items(), key=lambda kv: kv[0])
        msg = ", ".join([f"{k}={v}" for k, v in ordered])
        logger.info(f"Final {header} metrics -> {msg}")
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

        # In fast_dev_run, Lightning forbids ckpt_path="best". Fallback to in-memory weights.
        ckpt_for_val = None
        if not getattr(trainer, "fast_dev_run", False):
            try:
                cb = getattr(trainer, "checkpoint_callback", None)
                # Prefer explicit best-model path if available; otherwise allow "best" alias.
                if cb is not None and getattr(cb, "best_model_path", None):
                    ckpt_for_val = cb.best_model_path
                else:
                    ckpt_for_val = "best"
            except Exception:
                ckpt_for_val = None

        val_results = trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_for_val)
        _pretty_log_results(val_results, header="validation")
    else:
        logger.info("*********** start validation ***********\n")
        val_results = trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.model.ckpt_path)
        _pretty_log_results(val_results, header="validation")


if __name__ == "__main__":
    main()
