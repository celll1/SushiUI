"use client";

import { useState } from "react";
import { Image as ImageIcon } from "lucide-react";
import ImageViewer from "./ImageViewer";

interface FloatingGalleryProps {
  images: Array<{ url: string; timestamp: number }>;
  maxImages: number;
  hideToggle?: boolean;
}

export default function FloatingGallery({ images, maxImages, hideToggle = false }: FloatingGalleryProps) {
  const [viewerImageIndex, setViewerImageIndex] = useState<number | null>(null);
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);

  // Limit to most recent images
  const displayImages = images.slice(-maxImages);

  if (displayImages.length === 0) {
    return null;
  }

  const handleNavigate = (direction: 'prev' | 'next') => {
    if (viewerImageIndex === null) return;

    if (direction === 'prev' && viewerImageIndex > 0) {
      setViewerImageIndex(viewerImageIndex - 1);
    } else if (direction === 'next' && viewerImageIndex < displayImages.length - 1) {
      setViewerImageIndex(viewerImageIndex + 1);
    }
  };

  return (
    <>
      {/* Mobile gallery toggle button */}
      {!hideToggle && (
        <button
          onClick={() => setIsGalleryOpen(!isGalleryOpen)}
          className="fixed top-4 right-4 z-50 p-3 rounded-lg bg-gray-800 bg-opacity-90 text-white shadow-lg lg:hidden"
          aria-label="Toggle gallery"
        >
          <ImageIcon className="h-5 w-5" />
          {displayImages.length > 0 && (
            <span className="absolute -top-1 -right-1 bg-blue-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
              {displayImages.length}
            </span>
          )}
        </button>
      )}

      {/* Gallery panel - collapsible on mobile, always visible on desktop */}
      <div className={`
        fixed top-4 right-4 z-40 bg-gray-800 rounded-lg shadow-lg p-2
        transition-all duration-200 ease-in-out
        ${isGalleryOpen ? 'translate-x-0' : 'translate-x-[calc(100%+1rem)]'}
        lg:translate-x-0
        max-w-[80vw] lg:max-w-[60vw]
      `}>
        <div className="flex items-center gap-2 overflow-x-auto scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
          {displayImages.map((image, index) => (
            <div
              key={`${image.timestamp}-${index}`}
              className="flex-shrink-0 cursor-pointer hover:opacity-80 transition-opacity"
              onDoubleClick={() => setViewerImageIndex(index)}
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

      {viewerImageIndex !== null && (
        <ImageViewer
          imageUrl={displayImages[viewerImageIndex].url}
          onClose={() => setViewerImageIndex(null)}
          onNavigate={handleNavigate}
          hasPrev={viewerImageIndex > 0}
          hasNext={viewerImageIndex < displayImages.length - 1}
        />
      )}
    </>
  );
}
