"""Deterministic episode generation shared by ALL methods.

An episode = (class, query image, K support images of the same class).
Episodes are written once to a JSON file and every method (TabPFN variants, baselines,
SegGPT, SAM3) reads the *same* file -> identical support/query pairs for all methods.

Leakage guards implemented here:
  * the query is never one of its own supports (asserted),
  * supports are sampled without replacement,
  * optional near-duplicate rejection: a support whose perceptual hash is (almost)
    identical to the query is re-drawn (some benchmarks contain duplicate images,
    which would turn segmentation into copy-paste).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from .datasets import FewShotDataset


@dataclass
class Episode:
    idx: int
    cls: str
    query: tuple[str, str]
    supports: list[tuple[str, str]]


def dhash(img: np.ndarray, size: int = 8) -> int:
    g = np.asarray(Image.fromarray(img).convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def make_episodes(
    ds: FewShotDataset,
    shots: int,
    n_episodes: int | None,
    seed: int,
    dedup_hamming: int = -1,
) -> list[Episode]:
    """HSNet/PATNet style sampling.

    * if n_episodes is None or equals the number of images: every image is the query once
      (FSS-1000 protocol, 2400 episodes)
    * otherwise: classes are visited round-robin (balanced), the query is drawn uniformly.
    """
    rng = np.random.default_rng(seed)
    all_q = [(c, it) for c in ds.classes for it in ds.items[c]]
    if n_episodes is None or n_episodes == len(all_q):
        queries = all_q
    else:
        queries = []
        for i in range(n_episodes):
            c = ds.classes[i % len(ds.classes)]
            queries.append((c, ds.items[c][rng.integers(len(ds.items[c]))]))

    hash_cache: dict[str, int] = {}

    def h(item):
        if item[0] not in hash_cache:
            hash_cache[item[0]] = dhash(ds.load(*item)[0])
        return hash_cache[item[0]]

    episodes = []
    n_rejected = 0
    for i, (c, q) in enumerate(queries):
        pool = [it for it in ds.items[c] if it[0] != q[0]]
        if len(pool) < shots:
            raise ValueError(f"class {c} has only {len(pool)} candidate supports for {shots}-shot")
        order = rng.permutation(len(pool))
        sup = []
        for j in order:
            cand = pool[j]
            if dedup_hamming >= 0 and hamming(h(cand), h(q)) <= dedup_hamming:
                n_rejected += 1
                continue
            sup.append(cand)
            if len(sup) == shots:
                break
        if len(sup) < shots:
            raise ValueError(f"not enough non-duplicate supports for query {q[0]}")
        assert q[0] not in [s[0] for s in sup], "query leaked into its own support set"
        episodes.append(Episode(i, c, tuple(q), [tuple(s) for s in sup]))
    if dedup_hamming >= 0:
        print(f"[episodes] rejected {n_rejected} near-duplicate supports (dHash hamming <= {dedup_hamming})")
    return episodes


def episode_path(out_dir: str, ds: FewShotDataset, shots: int, seed: int, n: int) -> str:
    return os.path.join(out_dir, f"{ds.name}_{ds.split}_{shots}shot_seed{seed}_n{n}.json")


def save_episodes(path: str, episodes: list[Episode], meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({"meta": meta, "episodes": [asdict(e) for e in episodes]}, f)
    os.replace(tmp, path)  # atomic: safe with SLURM job arrays writing the same file


def load_episodes(path: str) -> tuple[list[Episode], dict]:
    with open(path) as f:
        d = json.load(f)
    eps = [Episode(e["idx"], e["cls"], tuple(e["query"]), [tuple(s) for s in e["supports"]]) for e in d["episodes"]]
    return eps, d["meta"]


def get_or_create_episodes(ds, shots, n_episodes, seed, out_dir="episodes", dedup_hamming=-1):
    n = n_episodes if n_episodes is not None else ds.num_images()
    path = episode_path(out_dir, ds, shots, seed, n)
    if os.path.exists(path):
        eps, meta = load_episodes(path)
        return eps, path
    eps = make_episodes(ds, shots, n_episodes, seed, dedup_hamming)
    meta = {"dataset": ds.name, "split": ds.split, "shots": shots, "seed": seed, "n": len(eps),
            "dedup_hamming": dedup_hamming}
    if ds.name != "synthetic":
        save_episodes(path, eps, meta)
    return eps, path
