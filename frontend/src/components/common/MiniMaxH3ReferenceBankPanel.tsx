"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Button from "./Button";
import {
  cancelMiniMaxH3ReferenceBank,
  getMiniMaxH3TeAgreement,
  MiniMaxH3TeAgreementStatus,
  startMiniMaxH3ReferenceBank,
} from "@/utils/api";

interface MiniMaxH3ReferenceBankPanelProps {
  /** Re-read whenever the loaded model changes. */
  modelVersion?: number;
  className?: string;
}

// While a build runs the only progress a client gets is this document.
const POLL_MS = 1500;

export default function MiniMaxH3ReferenceBankPanel({
  modelVersion,
  className = "",
}: MiniMaxH3ReferenceBankPanelProps) {
  const [status, setStatus] = useState<MiniMaxH3TeAgreementStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const mounted = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const data = await getMiniMaxH3TeAgreement();
      if (mounted.current) setStatus(data);
      return data;
    } catch {
      if (mounted.current) setStatus(null);
      return null;
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    refresh();
    return () => {
      mounted.current = false;
    };
  }, [refresh, modelVersion]);

  const running = status?.job?.state === "running";
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(refresh, POLL_MS);
    return () => clearInterval(timer);
  }, [running, refresh]);

  if (!status?.supported || !status.loaded) return null;

  const { loaded, suite, cost, bank, banks, job, measurements } = status;
  const minutes = Math.round(cost.seconds / 60);
  const otherBanks = banks.filter((entry) => !entry.is_loaded_encoder);
  const localMeasurement = measurements.find(
    (entry) => entry.encoder === loaded.text_encoder && entry.projection === loaded.projection
  );

  const build = async () => {
    if (!loaded.text_encoder_path) return;
    setBusy(true);
    setError(null);
    try {
      await startMiniMaxH3ReferenceBank(loaded.text_encoder_path);
      await refresh();
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(
        (typeof detail === "string" && detail) ||
          err?.message ||
          "The reference bank build could not be started."
      );
    } finally {
      setBusy(false);
    }
  };

  const cancel = async () => {
    setBusy(true);
    try {
      await cancelMiniMaxH3ReferenceBank();
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className={`rounded border border-gray-700 bg-gray-800/60 px-2 py-1.5 text-[11px] leading-relaxed text-gray-300 ${className}`}
    >
      <p className="app-kicker">Text-encoder reference bank</p>

      {bank ? (
        <p className="mt-1">
          A reference bank is stored for {bank.reference}, the loaded encoder: {bank.presentations}{" "}
          presentations of suite {bank.suite_version}
          {bank.built_at ? `, built ${bank.built_at}` : ""}.
        </p>
      ) : (
        <p className="mt-1">No reference bank is stored for {loaded.text_encoder}.</p>
      )}

      {otherBanks.length > 0 && (
        <p className="mt-1 text-gray-400">
          {otherBanks.length === 1
            ? `A bank built from ${otherBanks[0].reference} is also stored. It was built from a different released encoder, so it does not describe the loaded one.`
            : `${otherBanks.length} banks built from other released encoders are also stored. They do not describe the loaded one.`}
        </p>
      )}

      {loaded.is_substitute && loaded.substitution && (
        <p className="mt-1">{loaded.substitution}</p>
      )}

      {loaded.is_substitute && !localMeasurement && bank && (
        <p className="mt-1 text-gray-400">
          No measurement of this pairing against the stored bank is recorded yet. It is measured
          the next time this pairing is loaded.
        </p>
      )}

      {!loaded.is_substitute && !bank && (
        <p className="mt-1">
          Building one encodes suite {suite.version} ({suite.prompts} prompts, plus the long-form
          composites built from them) with the loaded encoder. Measured on the 25 GB released
          encoder for suite {cost.suite_version}: {minutes} minutes, {cost.host_ram_gib_min}-
          {cost.host_ram_gib_max} GiB of host RAM, {cost.stored_mb} MB stored. A generation cannot
          run while it does.
        </p>
      )}

      {job.state === "running" && (
        <p className="mt-1 text-violet-200">
          Building the bank for {job.reference}:{" "}
          {job.total ? `presentation ${job.processed} of ${job.total}` : job.message}.
        </p>
      )}

      {job.state === "cancelled" && (
        <p className="mt-1 text-amber-300">
          The build was {job.message}. No bank was stored.
        </p>
      )}

      {job.state === "failed" && (
        <p className="mt-1 text-red-300">The build failed: {job.error}</p>
      )}

      {!status.can_build && status.reason && job.state !== "running" && (
        <p className="mt-1 text-gray-400">A bank cannot be built now: {status.reason}</p>
      )}

      {error && <p className="mt-1 text-red-300">{error}</p>}

      <div className="mt-1.5 flex gap-2">
        {job.state === "running" ? (
          <Button size="xs" variant="secondary" onClick={cancel} disabled={busy}>
            Cancel build
          </Button>
        ) : (
          <Button
            size="xs"
            variant="secondary"
            onClick={build}
            disabled={busy || !status.can_build}
          >
            {bank ? "Rebuild reference bank" : "Build reference bank"}
          </Button>
        )}
      </div>
    </div>
  );
}
