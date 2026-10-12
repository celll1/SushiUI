"""Shared stable attention for full-canvas global summaries."""
import torch


def global_summary_attention(summary_queries, keys, values, queries):
    output_dtype = queries.dtype
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in
                                (summary_queries, keys, values, queries)) else torch.float32
    # A FP32 softmax alone still leaves QK, probabilities and PV in BF16.
    with torch.autocast(device_type=keys.device.type, enabled=False):
        summary_queries, keys, values, queries = (
            x.to(dtype) for x in (summary_queries, keys, values, queries))
        scale = keys.shape[-1] ** -.5
        weights = torch.softmax(torch.einsum("sr,bnr->bsn", summary_queries, keys) * scale, dim=-1)
        summaries = torch.einsum("bsn,bnr->bsr", weights, values)
        weights = torch.softmax(torch.einsum("bnr,bsr->bns", queries, summaries) * scale, dim=-1)
        result = torch.einsum("bns,bsr->bnr", weights, summaries)
    return result.to(output_dtype)
