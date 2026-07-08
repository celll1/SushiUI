"use client";

import { useState, useEffect } from "react";
import Select from "./Select";
import { fetchTextEncoders, TextEncoderEntry } from "@/utils/api";

interface TextEncoderOverrideSelectorProps {
  value: string | null;
  onChange: (path: string | null) => void;
  label?: string;
  className?: string;
  disabled?: boolean;
}

export default function TextEncoderOverrideSelector({
  value,
  onChange,
  label = "Text encoder override",
  className = "",
  disabled = false,
}: TextEncoderOverrideSelectorProps) {
  const [encoders, setEncoders] = useState<TextEncoderEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchTextEncoders()
      .then((data) => {
        setEncoders(data.text_encoders || []);
      })
      .catch(() => {
        setEncoders([]);
      })
      .finally(() => setLoading(false));
  }, []);

  const options = [
    { value: "", label: "Default (model's text encoder)" },
    ...encoders.map((te) => {
      const dims: string[] = [];
      if (te.out_dim != null) dims.push(`${te.out_dim}d`);
      if (te.te_type) dims.push(te.te_type);
      const suffix = dims.length > 0 ? ` (${dims.join(", ")})` : "";
      return { value: te.path, label: `${te.name}${suffix}` };
    }),
  ];

  return (
    <div className={`space-y-1 ${className}`}>
      <Select
        label={label}
        value={value || ""}
        onChange={(e) => onChange(e.target.value || null)}
        options={options}
        disabled={loading || disabled}
      />
      {loading && (
        <p className="text-xs text-gray-500">Scanning model directory...</p>
      )}
      {!loading && encoders.length === 0 && (
        <p className="text-xs text-gray-500">
          No standalone text encoders found in model directory.
        </p>
      )}
    </div>
  );
}
