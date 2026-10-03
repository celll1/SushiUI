"""Held-out retrieval metrics: Recall@k and mAP@R over cosine similarity (design §10)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .style_dataset import csr_group, gather_csr, splitmix64


@dataclass(frozen=True, eq=False)
class Csr:
    """Per-row id sets: row i holds values[indptr[i]:indptr[i + 1]]."""

    indptr: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class RetrievalSummary:
    queries: int
    used: int
    excluded_no_positive: int
    excluded_no_negative: int  # has a positive but no other-artist item
    recall: dict[int, Optional[float]]
    map_at_r: Optional[float]


@dataclass(frozen=True, eq=False)
class QueryRetrieval:
    """Per-query results; hits and ap_at_r are only meaningful where `eligible` (False / NaN elsewhere)."""

    ks: tuple[int, ...]
    positives: np.ndarray  # R: same-artist gallery items other than the query itself
    negatives: np.ndarray
    hits: np.ndarray       # bool [n_queries, len(ks)]
    ap_at_r: np.ndarray    # float64 [n_queries]

    @property
    def eligible(self) -> np.ndarray:
        return (self.positives > 0) & (self.negatives > 0)

    def summarize(self, select: Optional[np.ndarray] = None) -> RetrievalSummary:
        """Aggregate over the queries in `select` (bool [n_queries]); the gallery is not re-split."""
        n = self.positives.shape[0]
        sel = np.ones(n, dtype=bool) if select is None else np.asarray(select)
        if sel.dtype != bool or sel.shape != (n,):
            raise ValueError(f"select must be bool [{n}]")
        pos, neg = self.positives[sel], self.negatives[sel]
        eligible = (pos > 0) & (neg > 0)
        used = int(eligible.sum())
        hits = self.hits[sel][eligible]
        return RetrievalSummary(
            queries=int(sel.sum()), used=used, excluded_no_positive=int((pos == 0).sum()),
            excluded_no_negative=int(((pos > 0) & (neg == 0)).sum()),
            recall={k: float(hits[:, j].mean()) if used else None for j, k in enumerate(self.ks)},
            map_at_r=float(self.ap_at_r[sel][eligible].mean()) if used else None,
        )


def _check_rows(name: str, csr: Csr, rows: int) -> None:
    if csr.indptr.shape != (rows + 1,) or csr.indptr[0] != 0 or csr.indptr[-1] != csr.values.shape[0]:
        raise ValueError(f"{name} CSR does not describe {rows} rows")


def _tie_keys(seed_word: np.uint64, query_ids: np.ndarray, gallery_ids: np.ndarray) -> np.ndarray:
    """int64 pseudo-random key per (query id, gallery id) pair; gallery_ids is [n_query, k]."""
    q = splitmix64(query_ids.astype(np.int64).view(np.uint64) ^ seed_word)
    return splitmix64(q[:, None] ^ gallery_ids.astype(np.int64).view(np.uint64)).view(np.int64)


def retrieve(query_embeddings: torch.Tensor, query_artists: np.ndarray, query_ids: np.ndarray,
             gallery_embeddings: torch.Tensor, gallery_artists: np.ndarray, gallery_ids: np.ndarray,
             ks: Sequence[int], chunk_size: int, tie_break_seed: int, query_copyrights: Optional[Csr] = None,
             gallery_copyrights: Optional[Csr] = None) -> QueryRetrieval:
    """Rank the gallery for each query, excluding the gallery item with the query's own id. With both
    copyright CSRs, a query's gallery is further restricted to items sharing >= 1 copyright id with it.
    Exact similarity ties are ordered by a key hashed from (tie_break_seed, query id, gallery id), so
    collapsed embeddings score at chance and gallery order does not matter. Chunk invariance holds only
    for bit-identical similarities: GEMM shape can move near-duplicate scores by one ULP.
    Memory is O(chunk_size * n_gallery)."""
    ks = tuple(int(k) for k in ks)
    if not ks or len(set(ks)) != len(ks) or min(ks) < 1:
        raise ValueError(f"ks must be distinct positive integers, got {ks}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if (query_copyrights is None) != (gallery_copyrights is None):
        raise ValueError("query_copyrights and gallery_copyrights must be given together")
    nq, ng = query_embeddings.shape[0], gallery_embeddings.shape[0]
    query_artists, query_ids = np.asarray(query_artists), np.asarray(query_ids, dtype=np.int64)
    gallery_artists, gallery_ids = np.asarray(gallery_artists), np.asarray(gallery_ids, dtype=np.int64)
    if query_artists.shape != (nq,) or query_ids.shape != (nq,):
        raise ValueError("query artists / ids must have one entry per query embedding")
    if gallery_artists.shape != (ng,) or gallery_ids.shape != (ng,):
        raise ValueError("gallery artists / ids must have one entry per gallery embedding")
    id_order = np.argsort(gallery_ids, kind="stable")
    sorted_ids = gallery_ids[id_order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError("gallery ids must be unique")

    if query_copyrights is not None:
        _check_rows("query copyright", query_copyrights, nq)
        _check_rows("gallery copyright", gallery_copyrights, ng)
        n_ids = int(max(query_copyrights.values.max(initial=-1), gallery_copyrights.values.max(initial=-1))) + 1
        q_cp_values = query_copyrights.values.astype(np.int64)
        g_rows = np.repeat(np.arange(ng, dtype=np.int64), np.diff(gallery_copyrights.indptr))
        inv_indptr, inv_rows = csr_group(gallery_copyrights.values.astype(np.int64), g_rows, n_ids)

    seed_word = np.uint64(int.from_bytes(hashlib.blake2b(str(int(tie_break_seed)).encode("ascii"), digest_size=8,
                                                         person=b"style-eval-tie").digest(), "little"))
    device = gallery_embeddings.device
    q_all = F.normalize(query_embeddings.to(device=device, dtype=torch.float32), dim=1)
    g_all = F.normalize(gallery_embeddings.float(), dim=1)
    g_artist = torch.as_tensor(gallery_artists, device=device)
    kmax = max(ks)
    positives = np.zeros(nq, dtype=np.int64)
    negatives = np.zeros(nq, dtype=np.int64)
    hits = np.zeros((nq, len(ks)), dtype=bool)
    ap_at_r = np.full(nq, np.nan, dtype=np.float64)

    for start in range(0, nq, chunk_size):
        stop = min(start + chunk_size, nq)
        rows = np.arange(start, stop, dtype=np.int64)
        c = rows.size
        if query_copyrights is None:
            allowed = np.ones((c, ng), dtype=bool)
        else:
            allowed = np.zeros((c, ng), dtype=bool)
            e_rows, e_ids = gather_csr(query_copyrights.indptr, q_cp_values, rows)
            counts = inv_indptr[e_ids + 1] - inv_indptr[e_ids]
            _, shared = gather_csr(inv_indptr, inv_rows, e_ids)
            allowed[np.repeat(e_rows - start, counts), shared] = True
        pos = np.minimum(np.searchsorted(sorted_ids, query_ids[rows]), max(ng - 1, 0))
        own = (sorted_ids[pos] == query_ids[rows]) if ng else np.zeros(c, dtype=bool)
        allowed[np.flatnonzero(own), id_order[pos[own]]] = False

        allowed_t = torch.from_numpy(allowed).to(device)
        same = (torch.as_tensor(query_artists[rows], device=device)[:, None] == g_artist[None, :])
        rel_all = same & allowed_t
        r = rel_all.sum(dim=1)
        n_neg = (~same & allowed_t).sum(dim=1)
        sims = (q_all[start:stop] @ g_all.T).masked_fill(~allowed_t, float("-inf"))
        width = min(max(kmax, int(r.max().item()) if c else 0), ng)
        if width == 0:
            positives[rows] = r.cpu().numpy()
            negatives[rows] = n_neg.cpu().numpy()
            continue
        # Only the top `width` ranks matter; every item tied with the width-th value is a candidate.
        threshold = torch.topk(sims, width, dim=1).values[:, -1:]
        candidate = allowed_t & (sims >= threshold)
        kc = max(int(candidate.sum(dim=1).max().item()), width)
        vals, idx = torch.topk(sims, kc, dim=1)
        cand = candidate.gather(1, idx)
        vals = vals.masked_fill(~cand, float("-inf"))
        keys = torch.from_numpy(_tie_keys(seed_word, query_ids[rows], gallery_ids[idx.cpu().numpy()])).to(device)
        # Lexicographic (similarity desc, tie key, gallery index) via successive stable sorts.
        perm = torch.argsort(idx, dim=1, stable=True)
        perm = perm.gather(1, torch.argsort(keys.gather(1, perm), dim=1, stable=True))
        perm = perm.gather(1, torch.argsort(vals.gather(1, perm), dim=1, descending=True, stable=True))[:, :width]
        rel = rel_all.gather(1, idx.gather(1, perm)) & cand.gather(1, perm)
        ranks = torch.arange(1, width + 1, device=device, dtype=torch.float64)
        precision = rel.cumsum(dim=1).double() / ranks
        within = ranks[None, :] <= r[:, None].double()
        ap = (precision * (rel & within)).sum(dim=1) / r.clamp_min(1).double()

        ok = ((r > 0) & (n_neg > 0)).cpu().numpy()
        positives[rows] = r.cpu().numpy()
        negatives[rows] = n_neg.cpu().numpy()
        for j, k in enumerate(ks):
            hits[rows, j] = rel[:, :k].any(dim=1).cpu().numpy() & ok
        ap_at_r[rows[ok]] = ap.cpu().numpy()[ok]

    return QueryRetrieval(ks=ks, positives=positives, negatives=negatives, hits=hits, ap_at_r=ap_at_r)
