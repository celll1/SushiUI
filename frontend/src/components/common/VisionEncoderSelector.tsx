"use client";

import { useState, useEffect } from "react";
import Select from "./Select";

interface VisionEncoderEntry {
  name: string;
  path: string;
  size_gb?: number;
  source_dir?: string;
}

interface VisionEncoderSelectorProps {
  value: string | null;
  onChange: (path: string | null) => void;
  label?: string;
  className?: string;
}

export default function VisionEncoderSelector({
  value,
  onChange,
  label = "Vision Encoder",
  className = "",
}: VisionEncoderSelectorProps) {
  const [encoders, setEncoders] = useState<VisionEncoderEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch("/api/models/vision_encoders")
      .then((r) => r.json())
      .then((data) => {
        setEncoders(data.vision_encoders || []);
      })
      .catch(() => {
        setEncoders([]);
      })
      .finally(() => setLoading(false));
  }, []);

  const options = [
    { value: "", label: "-- None --" },
    ...encoders.map((ve) => ({
      value: ve.path,
      label: `${ve.name}${ve.size_gb ? ` (${ve.size_gb} GB)` : ""}`,
    })),
  ];

  return (
    <div className={`space-y-1 ${className}`}>
      <Select
        label={label}
        value={value || ""}
        onChange={(e) => onChange(e.target.value || null)}
        options={options}
        disabled={loading}
      />
      {loading && (
        <p className="text-xs text-gray-500">Scanning model directory...</p>
      )}
      {!loading && encoders.length === 0 && (
        <p className="text-xs text-gray-500">
          No vision encoders found in model directory.
        </p>
      )}
    </div>
  );
}
