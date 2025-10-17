import os
import math
import time
import json
import hydra
from tqdm import tqdm

import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
import safetensors

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_pt_utils import get_parameter_names
from transformers import AutoConfig, AdamW, EsmModel
from torch.optim import Adam
import torch.nn.functional as F
import deepspeed
from pytorch_lightning.utilities import grad_norm

from esm.models.esm3 import ESM3
from tape import ProteinBertModel
from biotite.sequence import Alphabet, Sequence, GeneralSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

from util import accuracy, get_optimizer, calculate_regression_metric, calculate_binary_clf_metric, \
    calculate_multiclass_clf_metric
from modeling_util import model_init_fn
from vqvae.quantizer_module import get_codebook_utility


class SequenceClassificationHead(nn.Module):

    def __init__(self,
                 in_dim: int, hid_dim: int, out_dim: int,
                 num_layer: int = 1, dropout: float = 0.0
                 ):
        super().__init__()

        dims = [in_dim] + [hid_dim] * num_layer + [out_dim]
        layer_list = []
        for i in range(num_layer + 1):
            layer_list.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layer:
                layer_list.append(nn.ReLU())
                layer_list.append(nn.Dropout(dropout, inplace=False))

        self.classify = nn.Sequential(*layer_list)

    def forward(self, pooled_output, targets=None):
        logits = self.classify(pooled_output)
        outputs = (logits,)

        return outputs


class ProceedingBaseModel(nn.Module):
    # src/model_module.py

    import torch

    def inference_feature(self, input_list):
        """
        Normalize input_list to a 3-tuple: (input_ids, input_mask, feature)

        Accepts:
          - (input_ids, input_mask, feature)
          - {"feature"/"input_embeds"/"inputs": (B,L,D)  [preferred for continuous]
             "token_ids": (B,L)                          [discrete ids]
             "attention_mask"/"input_mask": (B,L)
             "input_ids": (B,L)}
          - [ { ...dict as above... } ]  (list of one dict)
        """
        import torch

        # unwrap [ {dict} ] -> dict
        if isinstance(input_list, (list, tuple)) and len(input_list) == 1 and isinstance(input_list[0], dict):
            input_list = input_list[0]

        # already a 3-tuple/list
        if isinstance(input_list, (list, tuple)) and len(input_list) == 3:
            ids, mask, feat = input_list
            # normalize to tensors
            if feat is not None and not torch.is_tensor(feat):
                feat = torch.as_tensor(feat, dtype=torch.float32)
            if mask is not None and not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, dtype=torch.long)
            if ids is not None and not torch.is_tensor(ids):
                ids = torch.as_tensor(ids, dtype=torch.long)

            # If this is the continuous path and mask is missing/empty, repair it
            if feat is not None and feat.ndim == 3:
                dev = feat.device
                if mask is None or (mask.numel() > 0 and mask.sum() == 0):
                    with torch.no_grad():
                        mask = (feat.abs().sum(dim=-1) > 0).to(dtype=torch.long, device=dev)
                if ids is None:
                    B, L, _ = feat.shape
                    ids = torch.zeros(B, L, dtype=torch.long, device=feat.device)
            return ids, mask, feat

        # dict-style payload
        if isinstance(input_list, dict):
            # Prefer explicit continuous features
            feat = None
            for k in ("feature", "input_embeds", "inputs"):
                v = input_list.get(k, None)
                if v is not None:
                    feat = v
                    break

            ids = input_list.get("input_ids", None)
            tok = input_list.get("token_ids", None)

            # If no explicit feature, interpret token_ids dynamically:
            # - If token_ids is 3D or float dtype -> it's actually continuous features
            # - Else -> treat as discrete ids
            if feat is None and tok is not None:
                if torch.is_tensor(tok):
                    tok_is_feature = (tok.ndim == 3) or (tok.dtype.is_floating_point)
                else:
                    tok_is_feature = False
                if tok_is_feature:
                    feat = tok
                else:
                    ids = tok if ids is None else ids

            # to tensors
            if feat is not None and not torch.is_tensor(feat):
                feat = torch.as_tensor(feat, dtype=torch.float32)
            if ids is not None and not torch.is_tensor(ids):
                ids = torch.as_tensor(ids, dtype=torch.long)

            # mask
            mask = None
            for k in ("attention_mask", "input_mask"):
                v = input_list.get(k, None)
                if v is not None:
                    mask = v
                    break
            if mask is not None and not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, dtype=torch.long)

            # ----- continuous path -----
            if feat is not None and feat.ndim == 3:
                dev = feat.device
                if mask is None or (mask.numel() > 0 and mask.sum() == 0):
                    with torch.no_grad():
                        mask = (feat.abs().sum(dim=-1) > 0).to(dtype=torch.long, device=dev)
                if ids is None:
                    B, L, _ = feat.shape
                    ids = torch.zeros(B, L, dtype=torch.long, device=dev)
                return ids, mask, feat

            # ----- discrete path -----
            if ids is None:
                raise ValueError(
                    "inference_feature: neither continuous 'feature' nor discrete 'token_ids'/'input_ids' found.")
            if mask is None:
                mask = torch.ones_like(ids, dtype=torch.long, device=ids.device)
            return ids, mask, None

        raise ValueError(f"inference_feature: unexpected format type={type(input_list)}")


class SequenceClassificationModel(ProceedingBaseModel):

    def __init__(self,
                 model_cfg, codebook_embedding=None,
                 ):
        super().__init__()

        pretrained_ckpt_path = model_cfg.pretrained_ckpt_path
        num_labels = model_cfg.num_labels
        dropout = model_cfg.dropout
        num_layer = model_cfg.num_layer
        hidden_size = model_cfg.hidden_size
        num_tokens = model_cfg.num_tokens
        d_model = model_cfg.d_model
        is_global_or_local = model_cfg.is_global_or_local

        self.multi_label = model_cfg.multi_label
        self.regression = model_cfg.regression
        self.num_labels = model_cfg.num_labels

        self.pretrained_ckpt_path = pretrained_ckpt_path
        assert pretrained_ckpt_path == ""

        self.codebook_embedding = codebook_embedding
        self.use_sequence = model_cfg.use_sequence
        self.sequence_only = model_cfg.sequence_only
        self.add_noise = model_cfg.add_noise

        if self.add_noise is not None:
            self.noise_embedding = nn.Embedding(1, d_model)
            if self.use_sequence:  # prohibit seq + struct tokens for adding noise (not tested)
                assert self.sequence_only

        if self.sequence_only:
            assert self.use_sequence

        self.entering_step = 0

        # load simple neural layers for benchmarking
        if num_tokens is not None:
            self.tokens_embed = nn.Embedding(num_tokens, d_model)
            self.tokens_embed.weight.requires_grad = False
            self.tokens_embed.weight[:len(self.codebook_embedding)] = self.codebook_embedding
        else:
            pass  # use the model encoded continuous representations instead

        if self.use_sequence:
            self.sequence_embed = nn.Embedding(26 + 1, d_model)

        layer_norm_eps = 1e-12
        self.tokens_layernorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.tokens_dropout = nn.Dropout(dropout)

        max_position_embeddings = 700  # bounded by cfg.data.filter_length
        self.position_embed = nn.Embedding(max_position_embeddings, d_model)
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False
        )

        self.is_global_or_local = is_global_or_local
        if self.is_global_or_local == "global":  # protein-wise multi-class classification
            pass
        elif self.is_global_or_local == "local":  # residue-wise binary classification
            assert num_labels == 1
        else:
            raise NotImplementedError

        self.d_model = d_model
        self.classify = SequenceClassificationHead(
            d_model, hidden_size, num_labels, num_layer, dropout
        )

    def proceed_global_prediction(self, input_ids, input_mask, feature, targets):

        # Build token mask (True for tokens, False for pads)
        if feature is not None and feature.ndim == 3:
            # robust for continuous features: non-zero rows are tokens
            token_mask = (feature.abs().sum(dim=-1) > 0)  # [B,L] bool
        else:
            # fall back to provided mask semantics: True=pad, False=token
            if input_mask is None:
                raise ValueError("input_mask is required when feature embeddings are unavailable")
            token_mask = ~(input_mask.to(torch.bool))

        # count tokens and guard
        num_tokens = token_mask.sum(dim=1, keepdim=True)  # [B,1]
        keep_index = num_tokens.squeeze(1) != 0
        assert keep_index.any(), "All sequences are empty after masking"

        # pooled sum over tokens
        if token_mask.ndim == 3:
            pooled = (feature * token_mask[:, :, 0:1]).sum(dim=1)
        else:
            pooled = (feature * token_mask.unsqueeze(-1)).sum(dim=1)

        # normalize by token counts
        pooled = pooled[keep_index]
        num_tokens = num_tokens[keep_index].clamp(min=1)
        pooled = pooled / num_tokens
        targets = targets[keep_index]

        # head + loss/metrics
        logits = self.classify(pooled, targets)[0]
        ret = (logits,)
        if targets is not None:
            valid = (targets != -100).nonzero().squeeze(-1)
            logits_v, targets_v = logits[valid], targets[valid]
            loss = nn.CrossEntropyLoss()(logits_v, targets_v)
            metrics = calculate_multiclass_clf_metric(logits_v, targets_v)
            ret = ((loss, metrics), logits_v, targets_v)
        return ret

        # averge pool for the last hidden state to get seq-level reprs.
        # num_of_tokens = (~input_mask).sum(dim=1, keepdim=True)  # (B, 1) or (B, 1, dim)
        # if len(input_ids.shape) == 3:
        #     assert (num_of_tokens[:, :, :-1] == num_of_tokens[:, :, :-1]).all()
        #     num_of_tokens = num_of_tokens[:, :, 0]  # (B, 1)
        #
        # assert not (num_of_tokens == 0).any()
        # keep_index = num_of_tokens.squeeze(1) != 0
        # num_of_tokens = num_of_tokens[keep_index]  # (B, 1)
        # if len(input_mask.shape) == 3:
        #     assert (input_mask[:, :, :-1] == input_mask[:, :, 1:]).all()
        #     pooled_hidden_state = (feature * (~input_mask)[:, :, 0:1]).sum(dim=1)  # (B, dim)
        # else:
        #     pooled_hidden_state = (feature * (~input_mask).unsqueeze(dim=-1)).sum(dim=1)
        # pooled_hidden_state = pooled_hidden_state[keep_index]
        # pooled_hidden_state = pooled_hidden_state / num_of_tokens  # (B, hidden_size)
        # targets = targets[keep_index]
        #
        # # transform to binary classification tasks
        # if self.multi_label:
        #     raise NotImplementedError
        # else:
        #     logits = self.classify(pooled_hidden_state, targets)[0]
        #     ret = (logits,)
        #
        #     if targets is not None:
        #         valid_indices = (targets != -100).nonzero().squeeze(-1)  # (B, ...)
        #         logits, targets = logits[valid_indices], targets[valid_indices]  # (B', ...)
        #
        #         loss_fct = nn.CrossEntropyLoss()
        #         classification_loss = loss_fct(logits, targets)
        #         metrics = calculate_multiclass_clf_metric(logits, targets)
        #         loss_and_metrics = (classification_loss, metrics)
        #         ret = (loss_and_metrics, logits, targets)
        #
        # return ret

    def proceed_local_prediction(self, input_ids, input_mask, feature, targets):
        if self.regression:
            # input_ids: (B, L) or (B, L, hidden_dim) for ProteinMPNN; feature: (B, L, hidden_size)
            logits = self.classify(feature)[0]  # (B, L, 1)
            ret = (logits,)

            if targets is not None:
                # ignore_index default to NaN and the padded -100
                ignore_index = torch.logical_or(targets.isnan(), (torch.abs(targets - (-100)) < 1e-6))
                logits = logits.squeeze(-1)[~ignore_index]  # (M, )
                targets = targets[~ignore_index].float()  # (M, )

                loss_fct = nn.MSELoss()
                regression_loss = loss_fct(logits, targets)
                metrics = calculate_regression_metric(logits, targets)
                loss_and_metrics = (regression_loss, metrics)

                ret = (loss_and_metrics, logits, targets)
        else:

            logits = self.classify(feature)[0]  # [B, L, hidden_size] -> [B, L, 1]
            ret = (logits,)

            if targets is not None:
                # ignore_index default to -100
                ignore_index = targets == -100  # [B, L]
                logits = logits.squeeze(-1)[~ignore_index]  # [M, ]
                targets = targets[~ignore_index].float()  # [M, ]

                # data imbalance issue
                ## per-batch class weight
                pos_weight = targets.shape[-1] / targets.sum() * 0.5
                neg_weight = targets.shape[-1] / (targets.shape[-1] - targets.sum()) * 0.5
                ## overal class weight
                class_weight = torch.ones_like(targets) * neg_weight
                class_weight[targets.long() == 1] = pos_weight

                loss_fct = nn.BCEWithLogitsLoss(weight=class_weight)
                classification_loss = loss_fct(logits, targets)
                metrics = calculate_binary_clf_metric(logits, targets)

                loss_fct_tmp = nn.BCEWithLogitsLoss()
                metrics["unweighted_loss"] = loss_fct_tmp(logits, targets)
                loss_and_metrics = (classification_loss, metrics)
                metrics["pos_weight"] = pos_weight
                metrics["neg_weight"] = neg_weight

                ret = (loss_and_metrics, logits, targets)
        return ret

    def forward(self, input_list, targets=None):

        input_ids, input_mask, feature = self.inference_feature(input_list)

        # if continuous features are missing (discrete path), embed ids
        if feature is None:
            if hasattr(self, "tokens_embed") and self.tokens_embed is not None:
                feature = self.tokens_embed(input_ids)
            else:
                raise ValueError("No continuous features provided and no tokens_embed available to embed input_ids.")

        # ensure targets are long class indices
        targets = targets.long()

        # try to discover number of classes from common attributes on *this* module
        num_labels = getattr(self, "num_labels", None)
        if num_labels is None and hasattr(self, "classifier"):
            num_labels = getattr(self.classifier, "out_features", None)
        if num_labels is None and hasattr(self, "head"):
            num_labels = getattr(self.head, "out_features", None)

        # if we discovered it, enforce label range to avoid CUDA asserts
        if num_labels is not None:
            bad_lo = (targets < 0).nonzero(as_tuple=False).flatten()
            bad_hi = (targets >= num_labels).nonzero(as_tuple=False).flatten()
            if bad_lo.numel() or bad_hi.numel():
                raise ValueError(
                    f"[LabelRangeError] {bad_lo.numel()} labels < 0 and "
                    f"{bad_hi.numel()} labels >= num_labels ({num_labels}). "
                    f"min={int(targets.min())}, max={int(targets.max())}"
                )

        if self.is_global_or_local == "global":
            assert not self.regression
            ret = self.proceed_global_prediction(input_ids, input_mask, feature, targets)
        elif self.is_global_or_local == "local":
            ret = self.proceed_local_prediction(input_ids, input_mask, feature, targets)
        else:
            raise NotImplementedError

        return ret

    def configure_optimizers(self):
        """
        AdamW with sensible defaults, weight-decay on non-norm/bias params.
        Optional cosine scheduler if provided in cfg.optimization.scheduler.
        """
        # ---- read from hydra cfg (with safe defaults) ----
        cfg = self.optimizer_cfg if isinstance(self.optimizer_cfg, dict) else {}
        lr = float(cfg.get("lr", 1e-3))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        eps = float(cfg.get("eps", 1e-8))

        # ---- param groups: no weight decay on bias/LayerNorm ----
        no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight", "ln.weight")
        decay_params, nodecay_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (nodecay_params if any(k in n for k in no_decay_keys) else decay_params).append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)

        # ---- optional LR scheduler ----
        scfg = cfg.get("scheduler", {}) or {}
        name = str(scfg.get("name", "")).lower()

        if name in ("cosine", "cosineannealinglr"):
            T_max = int(scfg.get("t_max", scfg.get("T_max", 100)))
            eta_min = float(scfg.get("eta_min", 0.0))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            interval = "step" if str(scfg.get("interval", "epoch")).lower() == "step" else "epoch"
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": interval}
            }

        if name in ("onecycle", "onecyclelr"):
            # needs total_steps; fall back to max_steps if present
            total_steps = int(scfg.get("total_steps", getattr(self.trainer, "max_steps", 0) or 1000))
            pct_start = float(scfg.get("pct_start", 0.3))
            anneal = str(scfg.get("anneal_strategy", "cos")).lower()
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, total_steps=total_steps,
                pct_start=pct_start, anneal_strategy=anneal
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
            }

        # default: no scheduler
        return optimizer


class ZeroShotCodebookUtilityModel(ProceedingBaseModel):

    def __init__(self, model_cfg, codebook_embedding=None):
        super().__init__()

        d_model = model_cfg.d_model

        self.d_model = d_model
        self.codebook_embedding = codebook_embedding  # [codebook_size, codebook_embed_size]

    def forward(self, input_list, target=None):
        input_ids, input_mask = input_list
        # metrics = get_codebook_utility(input_ids[~input_mask], self.codebook_embedding.to(input_ids.device))
        metrics = get_codebook_utility(input_ids[~input_mask], self.codebook_embedding)


        # compression bits
        reduced_bits_ratio = []
        for i in range(len(input_ids)):
            tmp = input_ids[i][~input_mask[i]]
            new_bits = sum([len(str(x.item())) for x in tmp])
            # (for each residue, there are 4 backbone atoms,
            # 3 numbers for xyz, and on average 6 bytes for each float number)
            old_bits = len(tmp) * 4 * 3 * 6
            reduced_bits_ratio.append(new_bits / old_bits)

        metrics["compression_ratio"] = torch.mean(torch.tensor(reduced_bits_ratio, device=input_ids.device))

        loss_and_metrics = (torch.zeros(1, device=input_ids.device), metrics)
        return (loss_and_metrics, input_ids[~input_mask], None)


class ZeroshotProximityModel(ProceedingBaseModel):

    def __init__(self, model_cfg, codebook_embedding=None, ):
        super().__init__()

        d_model = model_cfg.d_model

        self.d_model = d_model

        if codebook_embedding is not None:
            # for discretized tokenizers
            self.codebook_embedding = codebook_embedding.detach().cpu()
            embed = self.codebook_embedding
            embed = F.normalize(embed, p=2, dim=-1)
            embed = embed.to(torch.float16)
            sim_score = torch.matmul(embed, embed.T)
            sim_score = sim_score.numpy() * 100
            sim_score = sim_score.astype(np.int32)
            real_num_tokens = sim_score.shape[0]
            self.real_num_tokens = real_num_tokens
            self.alphabet = Alphabet(list(range(real_num_tokens)))
            self.substitution_matrix = SubstitutionMatrix(self.alphabet, self.alphabet, sim_score)
        else:
            # for continuous tokenizers
            self.codebook_embedding = None

    def forward(self, input_list, targets=None):
        """This could be slow because alignment algorithm is currently on CPU.
        Running on GPU could be possible but it's stophisticated
        """

        prot1_input_ids, prot2_input_ids = input_list
        # [B, L1], [B, L2] for discretized tokenizers
        # [B, L1, hidden_dim], [B, L2, hidden_dim] for continuous tokenizers

        bsz = prot1_input_ids.shape[0]
        score_list = []
        for i in range(bsz):
            lst1, lst2 = prot1_input_ids[i], prot2_input_ids[i]
            if self.codebook_embedding is not None:
                lst1 = lst1[lst1 < self.real_num_tokens].cpu().numpy()
                lst2 = lst2[lst2 < self.real_num_tokens].cpu().numpy()
                seq1 = GeneralSequence(self.alphabet, lst1)
                seq2 = GeneralSequence(self.alphabet, lst2)
                align_score = align_optimal(seq1, seq2, self.substitution_matrix)[0].score
                score_list.append(align_score)
            else:
                print('**************************************************')
                # print(type(sim_score), isinstance(sim_score, np.ndarray), sim_score.dtype, sim_score.dtype.kind)

                # lst1: [L1, hidden_dim], lst2: [L2, hidden_dim]
                lst1_embed = F.normalize(lst1, p=2, dim=-1)
                lst2_embed = F.normalize(lst2, p=2, dim=-1)
                sim = torch.matmul(lst1_embed, lst2_embed.T)  # [L1, L2]
                sim = sim.detach().cpu().numpy() * 100
                sim = sim.astype(np.int32)
                L1, L2 = len(lst1_embed), len(lst2_embed)
                sim_score = np.zeros((L1 + L2, L1 + L2), dtype=np.int32)
                sim_score[:L1, L1:] = sim

                print('**************************************************')
                print(type(sim_score), isinstance(sim_score, np.ndarray), sim_score.dtype, sim_score.dtype.kind)
                alphabet = Alphabet(list(range(L1 + L2)))

                substitution_matrix = SubstitutionMatrix(alphabet, alphabet, sim_score)

                seq1 = GeneralSequence(alphabet, np.arange(L1))
                seq2 = GeneralSequence(alphabet, np.arange(L2) + L1)
                align_score = align_optimal(seq1, seq2, substitution_matrix)[0].score
                score_list.append(align_score)

        score_list = torch.tensor(score_list, device=targets.device)
        metrics = calculate_regression_metric(score_list.float(), targets)

        loss_and_metrics = (torch.zeros(1, device=targets.device), {k: v for k, v in metrics.items() if k != "r2"})

        return (loss_and_metrics, score_list.float(), targets)


class PlModel(pl.LightningModule):
    """
    Pytorch Lightning wrapper class for model training
    """

    def __init__(
            self,
            model_cfg,
            trainer,
            py_logger,
            optimizer_cfg,
            all_split_names,
            codebook_embedding,

            # may be a Tensor or None
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.trainer = trainer
        self.py_logger = py_logger
        self.optimizer_cfg = optimizer_cfg
        self.all_split_names = all_split_names
        for split in self.all_split_names:
            setattr(self, f"{split}_step_outputs", [])

        import torch

        # --- SAFE BUFFER CREATION (no duplicates) ---
        # 1) Create the buffer only if it doesn't already exist
        if "codebook_embedding" not in self._buffers:
            # start with empty; we will fill it here or later in `setup`
            self.register_buffer("codebook_embedding", torch.empty(0), persistent=False)

        # 2) If a codebook was passed in, fill the (already registered) buffer
        if isinstance(codebook_embedding, torch.Tensor) and codebook_embedding.numel() > 0:
            self.codebook_embedding = codebook_embedding.detach().float()

    def _log_split_metrics(self, metrics: dict, split: str, on_step=False, on_epoch=True):
        """Log dict with consistent flags; convert tensors to scalars."""
        import torch
        for k, v in metrics.items():
            key = f"{split}/{k}"  # e.g., "validation/loss"

            # scalarize
            if isinstance(v, torch.Tensor):
                if v.numel() > 1:
                    v = v.float().mean()
                v = v.detach()

            self.log(
                key, v,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=("loss" in k),
                logger=True,
                sync_dist=True,
            )

    def setup(self, stage: str):
        """
        Set up the module, including model creation
        Args:
            stage: Pytorch Lightning stage train/val/test can be used to induce different
                    behavior only used for inheritance
        """

        self.trainer.strategy.config["train_micro_batch_size_per_gpu"] = self.optimizer_cfg.micro_batch_size

        # Build the inner model; pass a tensor if you want, or rely on the buffer we fill below
        self.model = model_init_fn(self.trainer, self.model_cfg,
                                   codebook_embedding=self.codebook_embedding)

        # If the buffer is empty, try to get the codebook from the tokenizer
        import torch, time
        needs_fill = (not isinstance(self.codebook_embedding, torch.Tensor)) or (self.codebook_embedding.numel() == 0)
        if needs_fill:
            dm = self.trainer.datamodule
            tok = getattr(dm, "tokenizer", None)
            if tok is not None and getattr(tok, "codebook", None) is not None:
                cb = tok.codebook.detach().float()
                # assign to the already-registered buffer (no re-registration!)
                self.codebook_embedding = cb
            else:
                raise RuntimeError(
                    "codebook_embedding is empty. Datamodule must expose `tokenizer.codebook` "
                    "(e.g., from '/codebook' in the H5)."
                )

        # timing, etc.
        self._last_logged_batch_start_time = time.monotonic()
        super().setup(stage)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["input_list"], batch["targets"])
        loss, metrics = outputs[0]

        self.log(
            "training_loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=self.optimizer_cfg.micro_batch_size,
            logger=True,
            sync_dist=True,
        )

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Log time/step
        Args:
            outputs: outputs of train_step, not used, required for hook
            batch: use batch to get input/output sequence length
            batch_idx: batch number, not used required for hook
        """

        if batch_idx > 0 and batch_idx % self.trainer.log_every_n_steps == 0:
            # get the time for this iteration
            elapsed_time = time.monotonic() - self._last_logged_batch_start_time
            # start timeer for the next iteration
            self._last_logged_batch_start_time = time.monotonic()

            time_per_step = elapsed_time / self.trainer.log_every_n_steps

            self.log(
                "sec/step",
                time_per_step,
                on_step=True,
                prog_bar=True,
                logger=True,
                rank_zero_only=True,
            )
        torch.cuda.empty_cache()

    def _valid_or_test_step(self, batch, batch_idx, split="validation"):
        outputs = self.model(batch["input_list"], batch["targets"])
        loss, metrics = outputs[0]

        # For epoch-end aggregation, keep split-prefixed keys in the returned dict
        metrics_for_return = {f"{split}_{k}": v for k, v in metrics.items()}

        # For logging, pass a flat dict (no nested dicts)
        metrics_for_log = {"loss": loss}
        metrics_for_log.update(metrics)  # e.g., {"accuracy": ..., "f1_score": ...}

        # Single pathway logging to avoid duplicates
        self._log_split_metrics(metrics_for_log, split, on_step=False, on_epoch=True)

        opt_ret = {}
        if batch["targets"] is not None:
            num_sequences = torch.tensor(batch["targets"].shape[0], device=batch["targets"].device)
            opt_ret["num_sequences"] = num_sequences

        logits, targets = outputs[1], outputs[2]
        return {
            f"{split}_loss": loss,
            **metrics_for_return,
            "logits": logits,
            "targets": targets,
            **opt_ret,
        }

    def _valid_or_test_epoch_end(self, outputs, split="validation"):
        # Safeguard: nothing was collected for this split
        if outputs is None or len(outputs) == 0:
            return
        agg_result = {k: [] for k in outputs[0].keys() if k.startswith(split)}
        logits, targets = [], []
        for out in outputs:
            for k in out.keys():
                if "logits" in k:
                    logits.append(out[k])
                elif "targets" in k:
                    targets.append(out[k])
                elif k.startswith(split):
                    agg_result[k].append(out[k])

        if logits[0] is not None and targets[0] is not None:
            logits, targets = torch.concatenate(logits), torch.concatenate(targets)
        elif logits[0] is not None:

            if self.model_cfg.task_goal == "codebook_utilization":
                device = logits[0].device
                utilization_rate = round(np.mean([x.item() for x in agg_result["test_use_ratio"]]), 6)
                perplexity = round(np.mean([x.item() for x in agg_result["test_perplexity_normalized"]]), 6)
                entropy = round(np.mean([x.item() for x in agg_result["test_entropy_normalized"]]), 6)
                agg_result["UR"] = [torch.tensor(utilization_rate, device=device)]
                agg_result["perplexity"] = [torch.tensor(perplexity, device=device)]
                agg_result["entropy"] = [torch.tensor(entropy, device=device)]

            elif self.model_cfg.task_goal == "codebook_diversity":

                tk_name = self.trainer.datamodule.tokenizer_name.replace("Wrapped", "").replace("Tokenizer", "").lower()
                if tk_name == "ourpretrained":
                    tk_name += "_" + self.trainer.datamodule.tokenizer_kwargs["ckpt_name"]
                data_name = self.trainer.datamodule.data_args["data_name"].replace("Dataset", "").lower()

                # save codebook
                codevec = self.codebook_embedding

                dir_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp_codebook_embedding")
                os.makedirs(dir_name, exist_ok=True)
                codebook_file = os.path.join(dir_name, f"codebook_{tk_name}_{data_name}")
                print("Save codebook embeddings to: ", codebook_file)
                torch.save(codevec, codebook_file)

                # save codebook pairwise similarities
                sim_cos_score = F.cosine_similarity(codevec.cpu().unsqueeze(1), codevec.cpu().unsqueeze(0), dim=-1)

                dir_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp_simscore_dist")
                os.makedirs(dir_name, exist_ok=True)
                similarity_file = os.path.join(dir_name, f"simscore_cos_{tk_name}_{data_name}")
                torch.save(sim_cos_score, similarity_file)
                print("Save codebook similarities to: ", similarity_file)

                # save codebook pairwise similarities weighted by token usage frequency
                input_ids = torch.concatenate(logits)

                def transform_sim(sims):
                    sims = sims.cpu().numpy()
                    index_count = torch.bincount(input_ids, minlength=len(sims))
                    return (index_count, sims)

                used_sim_cos_score = transform_sim(sim_cos_score)

                dir_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp_simscore_used_dist")
                os.makedirs(dir_name, exist_ok=True)
                used_similarity_file = os.path.join(dir_name, f"simscore_cos_{tk_name}_{data_name}")
                torch.save(used_sim_cos_score, used_similarity_file)
                print("Save codebook similarities weighted by token frequency to: ", used_similarity_file)

                exit(0)
            else:
                raise NotImplementedError

        for k in agg_result.keys():
            agg_result[k] = torch.stack(agg_result[k]).mean()

            # recalculate for regression metrics
            if "spearmanr" in k or "pearsonr" in k or "r2" in k:
                tmp = calculate_regression_metric(logits, targets)
                agg_result[k] = tmp[k.split("_")[-1]]

        self.log_dict(
            agg_result,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,  # reduce metrics across devices
            batch_size=self.optimizer_cfg.micro_batch_size,
            add_dataloader_idx=False,
        )

    def on_validation_epoch_end(self):
        for split in self.all_split_names:
            self._valid_or_test_epoch_end(getattr(self, f"{split}_step_outputs"), split=split)
        for split in self.all_split_names:
            getattr(self, f"{split}_step_outputs").clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        if batch is None:
            return None
            # optional: use your configured split names if available
        split_name = "validation"
        try:
            if hasattr(self, "all_split_names") and self.all_split_names:
                split_name = str(self.all_split_names[dataloader_idx])
        except Exception:
            pass
        # Only pass supported kwargs to avoid signature mismatch; do not forward dataloader_idx.
        res = self._valid_or_test_step(batch, batch_idx, split=split_name)
        # Collect outputs for custom epoch_end aggregator
        try:
            getattr(self, f"{split_name}_step_outputs").append(res)
        except Exception:
            # if buffer missing, create and append
            setattr(self, f"{split_name}_step_outputs", [res])
        return res

    def configure_optimizers(self):
        """AdamW with optional cosine/one-cycle scheduler driven by self.optimizer_cfg."""
        import torch

        cfg = self.optimizer_cfg if isinstance(self.optimizer_cfg, dict) else {}
        lr = float(cfg.get("lr", 1e-3))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        eps = float(cfg.get("eps", 1e-8))

        # no weight decay for norm/bias
        no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight", "ln.weight")
        decay_params, nodecay_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (nodecay_params if any(k in n for k in no_decay_keys) else decay_params).append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)

        # optional scheduler
        scfg = cfg.get("scheduler", {}) or {}
        name = str(scfg.get("name", "")).lower()

        if name in ("cosine", "cosineannealinglr"):
            T_max = int(scfg.get("t_max", scfg.get("T_max", getattr(self.trainer, "max_epochs", 100) or 100)))
            eta_min = float(scfg.get("eta_min", 0.0))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            interval = "step" if str(scfg.get("interval", "epoch")).lower() == "step" else "epoch"
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": interval}}

        if name in ("onecycle", "onecyclelr"):
            total_steps = int(scfg.get("total_steps", getattr(self.trainer, "max_steps", 0) or 1000))
            pct_start = float(scfg.get("pct_start", 0.3))
            anneal = str(scfg.get("anneal_strategy", "cos")).lower()
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, total_steps=total_steps, pct_start=pct_start, anneal_strategy=anneal
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        return optimizer

