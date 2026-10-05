"""One CPU pixel lookahead; GPU VAE and prompt processor stay caller-owned."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError


# Includes source decode/convert scratch and normalized output intermediates.
PIXEL_WORKING_BUDGET_BYTES = 512 * 1024**2


def estimate_pixel_bytes(pairs):
    total = 0
    for item, _ in pairs:
        record = item["_ming_record"]
        assets = (*record.targets, *record.references)
        width = item.get("bucket_width") or item["width"]
        height = item.get("bucket_height") or item["height"]
        total += len(assets) * (16 * record.source_size[0] * record.source_size[1] + 96 * width * height)
        try:
            total += sum(asset.path.stat().st_size for asset in assets)
        except OSError:
            return PIXEL_WORKING_BUDGET_BYTES + 1
    return total


def prefetch_pixels(batches, start, end, load, stop_check):
    """Yield original indices; oversized pairs use the existing serial path."""
    if start >= end:
        return
    pending = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ming-pixels") as worker:
        try:
            stop_check()
            current = load(start, batches[start])
            for index in range(start, end):
                stop_check()
                if index + 1 < end and (estimate_pixel_bytes(batches[index]) +
                        estimate_pixel_bytes(batches[index + 1]) <= PIXEL_WORKING_BUDGET_BYTES):
                    pending = worker.submit(load, index + 1, batches[index + 1])
                yield index, current
                del current
                if index + 1 < end:
                    if pending is None:
                        stop_check()
                        current = load(index + 1, batches[index + 1])
                    else:
                        while True:
                            stop_check()
                            try:
                                current = pending.result(timeout=.05)
                                break
                            except TimeoutError:
                                if pending.done():
                                    raise
                        pending = None
        finally:
            if pending is not None:
                pending.cancel()
