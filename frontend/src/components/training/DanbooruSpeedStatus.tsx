"use client";

import { useState } from "react";

export interface DanbooruSpeedMetrics {
  dl_speed_check_enabled?: boolean;
  dl_speed_current_kbps?: number;
  dl_speed_avg_kbps?: number;
  dl_cooldown_active?: boolean;
  dl_cooldown_remaining_sec?: number;
  dl_slow_streak?: number;
  dl_cooldown_count?: number;
  dl_cooldown_reason?: string;
}

/** Live Danbooru download-speed + throttle-cooldown status with a manual-resume
 *  button. Shared by the tagger and image-gen augmentation panels. Hidden when
 *  the speed check is disabled for the run. */
export default function DanbooruSpeedStatus({
  data,
  onResume,
}: {
  data: DanbooruSpeedMetrics;
  onResume: () => Promise<unknown>;
}) {
  const [resuming, setResuming] = useState(false);

  if (data.dl_speed_check_enabled === false) return null;

  const cur = data.dl_speed_current_kbps ?? 0;
  const avg = data.dl_speed_avg_kbps ?? 0;
  const cd = !!data.dl_cooldown_active;
  const remain = data.dl_cooldown_remaining_sec ?? 0;
  const mm = Math.floor(remain / 60);
  const ss = remain % 60;
  const streak = data.dl_slow_streak ?? 0;

  const handleResume = async () => {
    setResuming(true);
    try {
      await onResume();
    } finally {
      setResuming(false);
    }
  };

  return (
    <div
      className={`rounded p-2 text-xs ${
        cd ? "bg-amber-900/20 border border-amber-600" : "bg-gray-800 border border-gray-700"
      }`}
    >
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <span className="text-gray-300">
          DL speed: <span className="text-gray-100">{cur.toFixed(0)}</span> KB/s
          <span className="text-gray-500"> (avg {avg.toFixed(0)})</span>
          {streak > 0 && !cd && (
            <span className="text-amber-400"> · slow streak {streak}</span>
          )}
        </span>
        {cd ? (
          <div className="flex items-center gap-2">
            <span className="text-amber-300">
              ⏸ Cooldown {mm}:{String(ss).padStart(2, "0")} left
            </span>
            <button
              onClick={handleResume}
              disabled={resuming}
              className="px-2 py-0.5 rounded bg-amber-600 hover:bg-amber-500 text-white disabled:opacity-50"
            >
              {resuming ? "Resuming…" : "Resume now"}
            </button>
          </div>
        ) : (
          <span className="text-green-400">● healthy</span>
        )}
      </div>
      {cd && data.dl_cooldown_reason && (
        <div className="text-amber-400/70 mt-1">
          Paused: {data.dl_cooldown_reason} — Danbooru throttle suspected (ban avoidance)
        </div>
      )}
    </div>
  );
}
