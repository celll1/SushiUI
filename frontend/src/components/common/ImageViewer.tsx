"use client";

import { useEffect } from "react";

interface ImageViewerProps {
  imageUrl: string;
  onClose: () => void;
  onNavigate?: (direction: 'prev' | 'next') => void;
  hasPrev?: boolean;
  hasNext?: boolean;
}

export default function ImageViewer({ imageUrl, onClose, onNavigate, hasPrev, hasNext }: ImageViewerProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "ArrowLeft" && hasPrev && onNavigate) {
        onNavigate('prev');
      } else if (e.key === "ArrowRight" && hasNext && onNavigate) {
        onNavigate('next');
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose, onNavigate, hasPrev, hasNext]);

  return (
    <div
      className="fixed inset-0 z-50 bg-black bg-opacity-90 flex items-center justify-center"
      onClick={onClose}
    >
      <div className="relative max-w-[95vw] max-h-[95vh] flex items-center">
        {/* Previous button */}
        {hasPrev && onNavigate && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onNavigate('prev');
            }}
            className="absolute left-4 text-white text-4xl font-bold bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-14 h-14 flex items-center justify-center z-10"
            title="Previous (Left Arrow)"
          >
            ‹
          </button>
        )}

        <img
          src={imageUrl}
          alt="Full size preview"
          className="max-w-full max-h-[95vh] object-contain"
          onClick={(e) => e.stopPropagation()}
        />

        {/* Next button */}
        {hasNext && onNavigate && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onNavigate('next');
            }}
            className="absolute right-4 text-white text-4xl font-bold bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-14 h-14 flex items-center justify-center z-10"
            title="Next (Right Arrow)"
          >
            ›
          </button>
        )}

        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-white text-3xl font-bold bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-12 h-12 flex items-center justify-center"
          title="Close (Escape)"
        >
          ×
        </button>
      </div>
    </div>
  );
}
