"use client";

import { useState } from "react";
import Button from "../common/Button";
import { queueStudioTransfer, type StudioTransferMedia } from "./studioTransfer";

interface SendToStudioButtonProps {
  media?: StudioTransferMedia;
  parameters?: object;
  sendMedia?: boolean;
  sendPrompt?: boolean;
  sendParameters?: boolean;
  source?: "generate" | "gallery";
  className?: string;
}

export default function SendToStudioButton({
  media,
  parameters,
  sendMedia = true,
  sendPrompt = true,
  sendParameters = true,
  source = "generate",
  className,
}: SendToStudioButtonProps) {
  const [sending, setSending] = useState(false);
  const values = (parameters || {}) as Record<string, unknown>;
  const enabled = Boolean((sendMedia && media) || (sendPrompt && values.prompt) || (sendParameters && parameters));

  const send = async () => {
    setSending(true);
    try {
      await queueStudioTransfer({
        source,
        media: sendMedia ? media : undefined,
        prompt: sendPrompt && typeof values.prompt === "string" ? values.prompt : undefined,
        negativePrompt: sendPrompt && typeof values.negative_prompt === "string" ? values.negative_prompt : undefined,
        parameters: sendParameters ? parameters : undefined,
      });
      window.location.assign("/studio");
    } catch (error) {
      console.error("Failed to send result to Studio", error);
      alert("Failed to send this result to Studio.");
      setSending(false);
    }
  };

  return (
    <Button
      onClick={send}
      variant="secondary"
      size="sm"
      disabled={!enabled || sending}
      className={className}
    >
      {sending ? "Sending…" : "Send to Studio"}
    </Button>
  );
}
