"""Training orchestration (steps 1-6 of the pipeline).

Usage::

    python scripts/train.py --config configs/default.yaml
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..config import Config, FeatureSpec
from ..data.ingestion import IngestResult, ingest
from ..data.normalization import FeatureNormalizer
from ..data.preprocessing import clean_all
from ..data.windows import (
    DEMAND_TARGETS,
    WindowBatch,
    build_windows,
    channel_names,
    concat_batches,
    with_deltas,
)
from ..logging_utils import get_logger
from ..models.fingerprint import FingerprintNet, reconstruction_errors
from ..models.pca import PCAHeartbeat
from ..utils.helpers import ensure_dir, get_device, set_seed, timer, write_json
from .checkpoint import save_checkpoint
from .validate import evaluate_model


class Trainer:
    """End-to-end training: ingest → clean → windows → normalize → fit → save."""

    def __init__(self, cfg: Config, features: list[FeatureSpec], classes: list[str]):
        self.cfg = cfg
        self.features = features
        self.feature_names = [f.name for f in features]
        self.classes = classes
        self.demand_idx = [self.feature_names.index(c) for c in DEMAND_TARGETS]
        self.logger = get_logger("cluster_heartbeat.train", cfg.paths.logs)
        set_seed(cfg.seed)
        self.device = get_device(cfg.train.device)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        with timer("ingestion", self.logger):
            result = ingest(cfg.data.source, cfg, self.features)
            result.frames = clean_all(
                result.frames, self.features, cfg.data.synthetic.interval_seconds
            )

        with timer("window generation", self.logger):
            train_batch, val_batch = self._make_splits(result)
        self.logger.info("windows: train=%d val=%d", len(train_batch), len(val_batch))

        normalizer = FeatureNormalizer(self.feature_names).fit(train_batch.X)

        if cfg.model.type == "pca":
            return self._run_pca(train_batch, val_batch, normalizer, result)
        return self._run_neural(train_batch, val_batch, normalizer, result)

    # ------------------------------------------------------------------
    def _make_splits(self, result: IngestResult) -> tuple[WindowBatch, WindowBatch]:
        cfg = self.cfg
        interval = cfg.data.synthetic.interval_seconds
        rng = np.random.default_rng(cfg.seed)
        nodes = sorted(result.frames)
        rng.shuffle(nodes)
        n_val = max(1, int(round(cfg.train.val_split * len(nodes))))
        val_nodes, train_nodes = set(nodes[:n_val]), set(nodes[n_val:])

        # Failure-aware stratification: guarantee the validation set contains
        # at least one failing node so detection/lead-time metrics are defined.
        failing = set(result.failures)
        if failing and not (val_nodes & failing):
            swap_in = sorted(failing & train_nodes)[:1]
            if swap_in:
                swap_out = sorted(val_nodes - failing)[:1]
                train_nodes -= set(swap_in)
                val_nodes |= set(swap_in)
                if swap_out:
                    val_nodes -= set(swap_out)
                    train_nodes |= set(swap_out)
                self.logger.info("stratified split: moved %s into validation", swap_in)

        def build(node_ids) -> WindowBatch:
            batches = []
            for node in sorted(node_ids):
                failure = result.failures.get(node)
                batches.append(build_windows(
                    node_id=node, df=result.frames[node],
                    feature_names=self.feature_names, classes=self.classes,
                    size=cfg.window.size, stride=cfg.window.stride,
                    interval_seconds=interval,
                    ttf_horizon_hours=cfg.window.ttf_horizon_hours,
                    failure_step=failure["failure_step"] if failure else None,
                ))
            return concat_batches(batches)

        self._val_nodes = sorted(val_nodes)
        return build(train_nodes), build(val_nodes)

    # ------------------------------------------------------------------
    def _tensors(self, batch: WindowBatch, normalizer: FeatureNormalizer):
        X_np = normalizer.transform_X(batch.X)
        if self.cfg.model.use_deltas and self.cfg.model.type != "pca":
            X_np = with_deltas(X_np)
        X = torch.from_numpy(X_np)
        cls = torch.from_numpy(batch.cls)
        demand = torch.from_numpy(normalizer.scale_demand(batch.demand, self.demand_idx))
        ttf = torch.from_numpy(batch.ttf)
        gt = torch.from_numpy(batch.anomaly_gt)
        return X, cls, demand, ttf, gt

    # ------------------------------------------------------------------
    def _run_neural(self, train_batch, val_batch, normalizer, result) -> dict:
        cfg = self.cfg
        w = cfg.train.loss_weights

        n_channels = len(channel_names(self.feature_names, cfg.model.use_deltas))
        model = FingerprintNet(
            n_features=n_channels,
            window_size=cfg.window.size,
            latent_dim=cfg.model.latent_dim,
            hidden_dims=list(cfg.model.hidden_dims),
            num_classes=len(self.classes),
            dropout=cfg.model.dropout,
        ).to(self.device)
        self.logger.info("model params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

        Xtr, Ctr, Dtr, Ttr, _ = self._tensors(train_batch, normalizer)
        Xva, Cva, Dva, Tva, Gva = self._tensors(val_batch, normalizer)
        train_loader = DataLoader(
            TensorDataset(Xtr, Ctr, Dtr, Ttr),
            batch_size=cfg.train.batch_size, shuffle=True, drop_last=False,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                weight_decay=cfg.train.weight_decay)

        history, best_val, best_state, bad_epochs = [], float("inf"), None, 0
        with timer("training", self.logger):
            for epoch in range(1, cfg.train.epochs + 1):
                model.train()
                totals = {"recon": 0.0, "cls": 0.0, "demand": 0.0, "ttf": 0.0}
                for xb, cb, db, tb in train_loader:
                    xb, cb, db, tb = (t.to(self.device) for t in (xb, cb, db, tb))
                    # Denoising objective: randomly mask whole channels (0 is
                    # the fleet mean in normalized space) so the model must
                    # reconstruct each sensor from the *others* — this is what
                    # makes correlation breaks (temp up, power flat) visible
                    # in the reconstruction error early.
                    x_in = xb
                    if cfg.train.mask_prob > 0:
                        mask = torch.rand(xb.shape[0], xb.shape[1], 1,
                                          device=xb.device) < cfg.train.mask_prob
                        x_in = xb.masked_fill(mask, 0.0)
                    out = model(x_in)
                    loss_r = F.mse_loss(out["recon"], xb)
                    loss_c = F.cross_entropy(out["cls_logits"], cb)
                    loss_d = F.smooth_l1_loss(out["demand"], db)
                    loss_t = F.smooth_l1_loss(out["ttf"], tb)
                    loss = (w["reconstruction"] * loss_r + w["classification"] * loss_c
                            + w["demand"] * loss_d + w["ttf"] * loss_t)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    for k, v in (("recon", loss_r), ("cls", loss_c),
                                 ("demand", loss_d), ("ttf", loss_t)):
                        totals[k] += v.item() * len(xb)

                model.eval()
                with torch.no_grad():
                    val_recon = F.mse_loss(model(Xva.to(self.device))["recon"],
                                           Xva.to(self.device)).item()
                row = {"epoch": epoch, "val_recon": val_recon,
                       **{k: v / len(train_batch) for k, v in totals.items()}}
                history.append(row)
                self.logger.info(
                    "epoch %02d | recon %.4f | cls %.4f | demand %.4f | ttf %.4f | val_recon %.4f",
                    epoch, row["recon"], row["cls"], row["demand"], row["ttf"], val_recon,
                )
                if val_recon < best_val - 1e-4:
                    best_val, best_state, bad_epochs = val_recon, copy.deepcopy(model.state_dict()), 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg.train.patience:
                        self.logger.info("early stopping at epoch %d", epoch)
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        # --- anomaly calibration on the training distribution -----------
        stats = self._calibrate(model, Xtr)

        # --- evaluation ---------------------------------------------------
        metrics = evaluate_model(
            model=model, batch=val_batch, normalizer=normalizer,
            stats=stats, cfg=self.cfg, classes=self.classes,
            failures=result.failures, device=self.device,
        )
        self.logger.info("validation metrics: %s", json.dumps(metrics, default=str)[:800])

        out_dir = Path(cfg.paths.checkpoints) / "best"
        save_checkpoint(model, normalizer, cfg, self.feature_names, self.classes,
                        stats, metrics, out_dir)
        write_json(history, Path(cfg.paths.reports) / "history.json")
        self.logger.info("checkpoint saved to %s", out_dir)
        return metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _calibrate(self, model: FingerprintNet, X: torch.Tensor) -> dict:
        """Reconstruction-error distribution of healthy training windows.

        Stores per-channel error stats (explainability), raw aggregate stats,
        and the distribution of the *aggregated top-k z-score* itself — the
        score the pipeline actually emits — so inference can normalize it
        against the healthy fleet baseline.
        """
        from ..inference.scores import aggregate_z

        model.eval()
        loader = DataLoader(X, batch_size=512)
        per_feat, agg = [], []
        for xb in loader:
            xb = xb.to(self.device)
            pf, ag = reconstruction_errors(model(xb)["recon"], xb)
            per_feat.append(pf.cpu().numpy())
            agg.append(ag.cpu().numpy())
        per_feat = np.concatenate(per_feat)      # (N, C)
        agg = np.concatenate(agg)                # (N,)
        pf_mean = per_feat.mean(axis=0)
        pf_std = np.maximum(per_feat.std(axis=0), 1e-8)
        agg_z = np.array([aggregate_z(pf, pf_mean, pf_std) for pf in per_feat])
        return {
            "err_mean": float(agg.mean()),
            "err_std": float(max(agg.std(), 1e-8)),
            "per_feature_mean": pf_mean.tolist(),
            "per_feature_std": pf_std.tolist(),
            "agg_z_mean": float(agg_z.mean()),
            "agg_z_std": float(max(agg_z.std(), 1e-8)),
            "agg_z_p99": float(np.percentile(agg_z, 99)),
            "threshold": float(agg.mean() + 3 * max(agg.std(), 1e-8)),
        }

    # ------------------------------------------------------------------
    def _run_pca(self, train_batch, val_batch, normalizer, result) -> dict:
        """Phase-1 baseline: PCA embedding + reconstruction anomaly only."""
        cfg = self.cfg
        Xtr = normalizer.transform_X(train_batch.X).reshape(len(train_batch), -1)
        pca = PCAHeartbeat(cfg.model.pca_components).fit(Xtr)
        ev = float(np.sum(pca.pca.explained_variance_ratio_))
        self.logger.info("PCA fitted: %d components, explained variance %.3f",
                         cfg.model.pca_components, ev)

        _, agg = pca.errors(Xtr)
        stats = {
            "err_mean": float(agg.mean()),
            "err_std": float(max(agg.std(), 1e-8)),
            "threshold": float(agg.mean() + 3 * max(agg.std(), 1e-8)),
        }

        Xva = normalizer.transform_X(val_batch.X).reshape(len(val_batch), -1)
        _, va_agg = pca.errors(Xva)
        z = (va_agg - stats["err_mean"]) / stats["err_std"]
        scores = np.clip(z / cfg.scoring.anomaly["z_full_scale"], 0, 1)
        from .metrics import anomaly_metrics
        metrics = {
            "backend": "pca",
            "explained_variance": ev,
            "anomaly": anomaly_metrics(scores, val_batch.anomaly_gt),
            "note": "PCA backend: classification/demand/TTF heads unavailable",
        }
        out_dir = ensure_dir(Path(cfg.paths.checkpoints) / "best")
        save_checkpoint(pca, normalizer, cfg, self.feature_names, self.classes,
                        stats, metrics, out_dir)
        write_json(stats, out_dir / "pca_stats.json")
        self.logger.info("PCA baseline saved to %s", out_dir)
        return metrics
