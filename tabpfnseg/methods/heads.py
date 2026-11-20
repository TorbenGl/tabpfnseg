"""In-context 'heads': given context rows (X_ctx, y_ctx) predict p(fg) for query rows.

All heads are fit *per episode* on the support rows only (no meta-training on any
segmentation data) -- this is what makes the comparison an in-context one.

  tabpfn      TabPFN classifier (v3.5 by default; any ModelVersion string works)
  tabpfn_reg  TabPFN regressor on the soft fg fraction of each cell (uses mixed boundary cells)
  logreg      L2 logistic regression (standardised with support statistics only)
  knn         cosine k-NN vote
  proto       masked-average-pooling prototypes (fg / bg), cosine + softmax  (classic FSS baseline)
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F


class Head:
    name = "head"
    uses_soft_labels = False

    def fit_predict(self, Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def info(self) -> dict:
        return {}


def _constant(yc, n):
    return np.full(n, float(np.mean(yc >= 0.5)), dtype=np.float32)


class TabPFNHead(Head):
    name = "tabpfn"

    def __init__(self, version: str = "v3.5", n_estimators: int | str = 4, device: str = "cuda",
                 model_path: str | None = None, chunk: int = 32768, regression: bool = False,
                 softmax_temperature: float | str = "auto", fit_mode: str = "fit_with_cache", seed: int = 0):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        cls = TabPFNRegressor if regression else TabPFNClassifier
        kw = dict(n_estimators=n_estimators, device=device, ignore_pretraining_limits=True,
                  random_state=seed, fit_mode=fit_mode)
        if not regression:
            kw["softmax_temperature"] = softmax_temperature
        if model_path:
            self.model = cls(model_path=model_path, **kw)
        else:
            self.model = cls.create_default_for_version(ModelVersion(version), **kw)
        self.regression, self.chunk = regression, chunk
        self.uses_soft_labels = regression
        self.name = "tabpfn_reg" if regression else "tabpfn"
        self._info: dict = {"version": version, "n_estimators": n_estimators, "model_path": model_path}

    def fit_predict(self, Xc, yc, Xq):
        if not self.regression and len(np.unique(yc >= 0.5)) < 2:
            return _constant(yc, len(Xq))
        if self.regression:
            self.model.fit(Xc, yc.astype(np.float32))
        else:
            self.model.fit(Xc, (yc >= 0.5).astype(np.int64))
        self._record_effective_config()
        out = []
        for s in range(0, len(Xq), self.chunk):
            xq = Xq[s : s + self.chunk]
            if self.regression:
                out.append(np.clip(self.model.predict(xq), 0, 1))
            else:
                p = self.model.predict_proba(xq)
                out.append(p[:, list(self.model.classes_).index(1)])
        return np.concatenate(out).astype(np.float32)

    def _record_effective_config(self):
        """Log what TabPFN *actually* does with our 1k-20k features (feature subsampling per
        estimator etc.) -- essential for interpreting the feature-budget experiments."""
        if "effective" in self._info:
            return
        eff = {}
        cfg = getattr(self.model, "inference_config_", None)
        if cfg is not None:
            for k in ("MAX_NUMBER_OF_FEATURES", "MAX_NUMBER_OF_SAMPLES", "FEATURE_SUBSAMPLING_METHOD"):
                eff[k] = getattr(cfg, k, None)
            try:
                eff["max_features_per_estimator"] = [
                    getattr(p, "max_features_per_estimator", None) for p in cfg.PREPROCESS_TRANSFORMS
                ]
            except Exception:  # noqa: BLE001
                pass
        self._info["effective"] = json.loads(json.dumps(eff, default=str))

    def info(self):
        return self._info


class LogRegHead(Head):
    name = "logreg"

    def __init__(self, C: float = 1.0, device: str = "cuda", iters: int = 200, balanced: bool = False):
        self.C, self.device, self.iters, self.balanced = C, device, iters, balanced

    def fit_predict(self, Xc, yc, Xq):
        y = (yc >= 0.5).astype(np.float32)
        if len(np.unique(y)) < 2:
            return _constant(yc, len(Xq))
        dev = self.device
        Xc_t, Xq_t = torch.from_numpy(Xc).float().to(dev), torch.from_numpy(Xq).float().to(dev)
        mu, sd = Xc_t.mean(0), Xc_t.std(0) + 1e-6  # support statistics only
        Xc_t, Xq_t = (Xc_t - mu) / sd, (Xq_t - mu) / sd
        y_t = torch.from_numpy(y).to(dev)
        w = torch.zeros(Xc_t.shape[1], device=dev, requires_grad=True)
        b = torch.zeros(1, device=dev, requires_grad=True)
        pos_w = (1 - y_t.mean()) / y_t.mean() if self.balanced else torch.tensor(1.0, device=dev)
        opt = torch.optim.LBFGS([w, b], lr=1, max_iter=self.iters, line_search_fn="strong_wolfe")
        lam = 1.0 / (self.C * len(y))

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(Xc_t @ w + b, y_t, pos_weight=pos_w) + 0.5 * lam * (w**2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            return torch.sigmoid(Xq_t @ w + b).cpu().numpy().astype(np.float32)


class KNNHead(Head):
    name = "knn"

    def __init__(self, k: int = 16, device: str = "cuda"):
        self.k, self.device = k, device

    def fit_predict(self, Xc, yc, Xq):
        dev = self.device
        C = F.normalize(torch.from_numpy(Xc).float().to(dev), dim=1)
        y = torch.from_numpy((yc >= 0.5).astype(np.float32)).to(dev)
        out = []
        for s in range(0, len(Xq), 8192):
            q = F.normalize(torch.from_numpy(Xq[s : s + 8192]).float().to(dev), dim=1)
            sim = q @ C.T
            v, i = sim.topk(min(self.k, C.shape[0]), dim=1)
            wgt = torch.softmax(v / 0.07, dim=1)
            out.append((wgt * y[i]).sum(1))
        return torch.cat(out).cpu().numpy().astype(np.float32)


class ProtoHead(Head):
    name = "proto"

    def __init__(self, tau: float = 20.0, device: str = "cuda"):
        self.tau, self.device = tau, device

    def fit_predict(self, Xc, yc, Xq):
        y = yc >= 0.5
        if y.all() or (~y).all():
            return _constant(yc, len(Xq))
        dev = self.device
        Xc_t = F.normalize(torch.from_numpy(Xc).float().to(dev), dim=1)
        pf = F.normalize(Xc_t[torch.from_numpy(y).to(dev)].mean(0), dim=0)
        pb = F.normalize(Xc_t[torch.from_numpy(~y).to(dev)].mean(0), dim=0)
        q = F.normalize(torch.from_numpy(Xq).float().to(dev), dim=1)
        logits = self.tau * torch.stack([q @ pb, q @ pf], 1)
        return torch.softmax(logits, 1)[:, 1].cpu().numpy().astype(np.float32)


def build_head(name: str, args) -> Head:
    dev = args.device
    if name == "tabpfn":
        return TabPFNHead(args.tabpfn_version, args.n_estimators, dev, args.tabpfn_model_path, seed=args.seed,
                          fit_mode=args.tabpfn_fit_mode)
    if name == "tabpfn_reg":
        return TabPFNHead(args.tabpfn_version, args.n_estimators, dev, args.tabpfn_model_path, regression=True,
                          seed=args.seed, fit_mode=args.tabpfn_fit_mode)
    if name == "logreg":
        return LogRegHead(C=args.logreg_C, device=dev)
    if name == "knn":
        return KNNHead(k=args.knn_k, device=dev)
    if name == "proto":
        return ProtoHead(device=dev)
    raise ValueError(name)
