"use client";

import { useEffect } from "react";

interface ImageViewerProps {
  imageUrl: string;
  onClose: () => void;
}

export default function ImageViewer({ imageUrl, onClose }: ImageViewerProps) {
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 bg-black bg-opacity-90 flex items-center justify-center"
      onClick={onClose}
    >
      <div className="relative max-w-[95vw] max-h-[95vh]">
        <img
          src={imageUrl}
          alt="Full size preview"
          className="max-w-full max-h-[95vh] object-contain"
          onClick={(e) => e.stopPropagation()}
        />
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-white text-3xl font-bold bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-12 h-12 flex items-center justify-center"
        >
          ×
        </button>
      </div>
    </div>
  );
}
