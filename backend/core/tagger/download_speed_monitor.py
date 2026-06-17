"""Download-speed monitor for Danbooru augmentation (ban-avoidance safety).

Danbooru typically throttles bandwidth *before* a hard ban, so a sustained drop
in achievable download speed is an early-warning signal. This monitor watches the
*actual* per-download network speed (excluding the client's own bandwidth-limit
sleep) and, when speed stays degraded for a sustained streak, triggers a cooldown
that pauses all Danbooru collection for a configurable window.

Robust to transients: a cooldown fires only after BOTH a minimum consecutive
streak of slow/timed-out downloads AND a minimum elapsed duration; a single
healthy download resets the streak. So an isolated timeout or a brief congestion
dip does not pause collection.

Process-global singleton (``get_speed_monitor()``): every DanbooruClient in the
process (sampler / surveyor / cooc) feeds the same monitor and they all pause
together. Manual resume is delivered cross-process by the worker via a control
file (see the sampler/collector), which calls :meth:`request_resume`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict, List, Tuple


class DownloadSpeedMonitor:
    """Tracks per-download network speed and manages the degradation cooldown."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # --- config (defaults overridden by configure()) ---
        self._enabled = True
        self._degraded_bps = 250 * 1024        # below this = "slow"
        self._min_slow_streak = 8              # consecutive slow downloads to trip
        self._min_slow_seconds = 90.0          # ...sustained at least this long
        self._cooldown_seconds = 3600.0        # pause duration
        self._min_sample_bytes = 64 * 1024     # ignore tiny files (noisy speed)
        # --- state ---
        self._recent: deque[Tuple[float, float]] = deque(maxlen=64)  # (ts, kbps)
        self._last_kbps = 0.0
        self._slow_streak = 0
        self._slow_streak_start = 0.0
        self._cooldown_until = 0.0
        self._cooldown_count = 0
        self._last_reason = ""

    # ------------------------------------------------------------------
    def configure(self, *, enabled: bool = True, degraded_kbps: int = 250,
                  min_slow_streak: int = 8, min_slow_seconds: float = 90.0,
                  cooldown_seconds: float = 3600.0) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            self._degraded_bps = max(1, int(degraded_kbps)) * 1024
            self._min_slow_streak = max(1, int(min_slow_streak))
            self._min_slow_seconds = max(0.0, float(min_slow_seconds))
            self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        print(f"[DownloadSpeedMonitor] configured: enabled={enabled} "
              f"degraded<{degraded_kbps}KB/s streak>={min_slow_streak} "
              f"sustained>={min_slow_seconds:.0f}s cooldown={cooldown_seconds:.0f}s")

    # ------------------------------------------------------------------
    def record(self, num_bytes: int, net_seconds: float, timed_out: bool = False) -> None:
        """Feed one download outcome.

        Parameters
        ----------
        num_bytes : bytes downloaded.
        net_seconds : wall time spent on the actual network read, EXCLUDING the
            client's bandwidth-limit sleep (so it reflects true network speed).
        timed_out : True if the request timed out / failed mid-download. Counted
            as a slow event (a throttle often manifests as stalls/timeouts).
        """
        if not self._enabled:
            return
        now = time.time()
        with self._lock:
            if timed_out:
                is_slow = True
            else:
                if num_bytes < self._min_sample_bytes or net_seconds <= 0:
                    return  # too small / invalid to judge speed reliably
                kbps = (num_bytes / net_seconds) / 1024.0
                self._last_kbps = kbps
                self._recent.append((now, kbps))
                is_slow = (kbps * 1024.0) < self._degraded_bps

            if not is_slow:
                self._slow_streak = 0
                return

            # slow event
            if self._slow_streak == 0:
                self._slow_streak_start = now
            self._slow_streak += 1
            already_cooling = now < self._cooldown_until
            if (not already_cooling
                    and self._slow_streak >= self._min_slow_streak
                    and (now - self._slow_streak_start) >= self._min_slow_seconds):
                self._cooldown_until = now + self._cooldown_seconds
                self._cooldown_count += 1
                self._last_reason = (f"{self._slow_streak} slow downloads over "
                                     f"{int(now - self._slow_streak_start)}s")
                print(f"[DownloadSpeedMonitor] COOLDOWN triggered ({self._last_reason}); "
                      f"pausing Danbooru collection for {int(self._cooldown_seconds)}s")
                self._slow_streak = 0

    # ------------------------------------------------------------------
    def is_in_cooldown(self) -> bool:
        with self._lock:
            return time.time() < self._cooldown_until

    def request_resume(self) -> bool:
        """Clear an active cooldown (manual resume). Returns True if one was cleared."""
        with self._lock:
            if time.time() < self._cooldown_until:
                self._cooldown_until = 0.0
                self._slow_streak = 0
                print("[DownloadSpeedMonitor] Manual resume — cooldown cleared")
                return True
            return False

    # ------------------------------------------------------------------
    def metrics(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            remaining = max(0.0, self._cooldown_until - now)
            recent: List[Tuple[float, float]] = list(self._recent)
            last = self._last_kbps
            streak = self._slow_streak
            cd_count = self._cooldown_count
            reason = self._last_reason
            enabled = self._enabled
        window = [k for (t, k) in recent if now - t <= 60.0]
        avg = sum(window) / len(window) if window else 0.0
        return {
            "dl_speed_check_enabled": enabled,
            "dl_speed_current_kbps": round(last, 1),
            "dl_speed_avg_kbps": round(avg, 1),
            "dl_cooldown_active": remaining > 0,
            "dl_cooldown_remaining_sec": int(remaining),
            "dl_slow_streak": streak,
            "dl_cooldown_count": cd_count,
            "dl_cooldown_reason": reason,
        }


_monitor = DownloadSpeedMonitor()


def get_speed_monitor() -> DownloadSpeedMonitor:
    """Return the process-global download-speed monitor."""
    return _monitor
