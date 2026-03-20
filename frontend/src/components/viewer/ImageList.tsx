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
          return (
            <div
              key={image.id}
              onClick={() => onImageClick(image)}
              className="cursor-pointer group"
            >
              <div className="aspect-square bg-gray-800 rounded-lg overflow-hidden">
                {/* Use picture element for WebP with fallback to original filename
                    - New thumbnails: WebP preferred, PNG fallback
                    - Old thumbnails: Falls back to original filename */}
                <picture>
                  <source srcSet={`/thumbnails/${baseName}.webp`} type="image/webp" />
                  <source srcSet={`/thumbnails/${baseName}.png`} type="image/png" />
                  <img
                    src={`/thumbnails/${image.filename}`}
                    alt={image.prompt}
                    loading="lazy"
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                  />
                </picture>
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
