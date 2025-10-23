"use client";

import { useState } from "react";
import ImageViewer from "./ImageViewer";

interface FloatingGalleryProps {
  images: Array<{ url: string; timestamp: number }>;
  maxImages: number;
}

export default function FloatingGallery({ images, maxImages }: FloatingGalleryProps) {
  const [viewerImage, setViewerImage] = useState<string | null>(null);

  // Limit to most recent images
  const displayImages = images.slice(-maxImages);

  if (displayImages.length === 0) {
    return null;
  }

  return (
    <>
      <div className="fixed top-4 right-4 z-40 bg-gray-800 rounded-lg shadow-lg p-2 max-w-[60vw]">
        <div className="flex items-center gap-2 overflow-x-auto scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
          {displayImages.map((image, index) => (
            <div
              key={`${image.timestamp}-${index}`}
              className="flex-shrink-0 cursor-pointer hover:opacity-80 transition-opacity"
              onDoubleClick={() => setViewerImage(image.url)}
            >
              <img
                src={image.url}
                alt={`Generated ${index + 1}`}
                className="h-24 w-auto object-contain rounded border border-gray-700"
              />
            </div>
          ))}
        </div>
      </div>

      {viewerImage && (
        <ImageViewer
          imageUrl={viewerImage}
          onClose={() => setViewerImage(null)}
        />
      )}
    </>
  );
}
