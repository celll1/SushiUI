"use client";

/**
 * ImageList - Memoized image grid with loading state
 *
 * This component is wrapped with React.memo to prevent re-renders
 * when only filters change. Loading indicator is handled internally
 * to avoid re-rendering the entire gallery.
 */

import React, { memo } from "react";
import { GeneratedImage } from "@/utils/api";

interface ImageListProps {
  images: GeneratedImage[];
  gridColumns: number;
  onImageClick: (image: GeneratedImage) => void;
  loading?: boolean;
}

const ImageList: React.FC<ImageListProps> = memo(({ images, gridColumns, onImageClick, loading = false }) => {
  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center py-8 text-gray-400">Loading images...</div>
      </div>
    );
  }

  return (
    <div className="flex-1">
      <div
        className="grid gap-4"
        style={{
          gridTemplateColumns: `repeat(${gridColumns}, minmax(0, 1fr))`
        }}
      >
        {images.map((image) => {
          // Get base filename without extension for WebP support
          // New thumbnails: baseName.png + baseName.webp
          // Old thumbnails: original filename (e.g., txt2img_001.png)
          const baseName = image.filename.replace(/\.[^/.]+$/, "");
          // Videos share the poster/thumbnail base name with the mp4, so the
          // existing /thumbnails path renders the poster frame. Detect by
          // is_video flag or file extension.
          const isVideo = image.is_video === true || /\.(mp4|webm)$/i.test(image.filename);
          // Audio (ACE-Step) records share a waveform-PNG thumbnail with the
          // .flac base name, generated the same way as the video poster frame.
          const isAudio = image.is_audio === true || /\.(flac|wav|mp3|ogg)$/i.test(image.filename) ||
            image.generation_type === "txt2aud" || image.generation_type === "aud2aud";
          return (
            <div
              key={image.id}
              onClick={() => onImageClick(image)}
              className="cursor-pointer group"
            >
              <div className="aspect-square bg-gray-800 rounded-lg overflow-hidden relative">
                {/* Use picture element for WebP with fallback to original filename
                    - New thumbnails: WebP preferred, PNG fallback
                    - Old thumbnails: Falls back to original filename
                    - Video: poster thumbnail shares the mp4 base name
                    - Audio: waveform thumbnail shares the flac base name */}
                <picture>
                  <source srcSet={`/thumbnails/${baseName}.webp`} type="image/webp" />
                  <source srcSet={`/thumbnails/${baseName}.png`} type="image/png" />
                  <img
                    src={(isVideo || isAudio) ? `/thumbnails/${baseName}.png` : `/thumbnails/${image.filename}`}
                    alt={image.prompt}
                    loading="lazy"
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                  />
                </picture>
                {isVideo && (
                  <>
                    {/* Play badge overlay so videos are distinguishable in the grid */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <span className="flex items-center justify-center w-10 h-10 rounded-full bg-black bg-opacity-60 text-white text-lg">
                        ▶
                      </span>
                    </div>
                    <span className="absolute top-1 right-1 px-1.5 py-0.5 rounded bg-black bg-opacity-70 text-white text-[10px] font-medium pointer-events-none">
                      Video
                    </span>
                  </>
                )}
                {isAudio && (
                  <span className="absolute top-1 right-1 px-1.5 py-0.5 rounded bg-black bg-opacity-70 text-white text-[10px] font-medium pointer-events-none">
                    Audio
                  </span>
                )}
              </div>
              <p className="mt-2 text-xs text-gray-400 truncate hidden lg:block">{image.prompt}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
});

ImageList.displayName = "ImageList";

export default ImageList;
