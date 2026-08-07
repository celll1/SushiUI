"use client";

import { useState, useEffect } from "react";
import { Image as ImageIcon } from "lucide-react";
import ImageViewer from "./ImageViewer";
import { isVideoUrl, isAudioUrl, posterUrlForVideo } from "@/utils/previewStorage";

interface FloatingGalleryProps {
  images: Array<{ url: string; timestamp: number }>;
  maxImages: number;
}

export default function FloatingGallery({ images, maxImages }: FloatingGalleryProps) {
  const [viewerImageIndex, setViewerImageIndex] = useState<number | null>(null);
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);
  const [isEditorOpen, setIsEditorOpen] = useState(false);

  // Limit to most recent results
  const displayImages = images.slice(-maxImages);

  // Video/audio results reach this strip through the same onImageGenerated
  // callback as images, so entries are keyed by media type rather than assumed
  // to be images. The full-size ImageViewer only understands images, so it is
  // driven by an image-only sub-list (indices below are into that list).
  const viewerImages = displayImages.filter((entry) => !isVideoUrl(entry.url) && !isAudioUrl(entry.url));

  // Monitor editor state changes
  useEffect(() => {
    const checkEditorState = () => {
      const editorOpen =
        document.body.dataset.promptEditorOpen === 'true' ||
        document.body.dataset.imageEditorOpen === 'true';
      setIsEditorOpen(editorOpen);
    };

    // Check immediately
    checkEditorState();

    // Set up a MutationObserver to watch for dataset changes
    const observer = new MutationObserver(checkEditorState);
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ['data-prompt-editor-open', 'data-image-editor-open']
    });

    return () => observer.disconnect();
  }, []);

  if (displayImages.length === 0) {
    return null;
  }

  const handleNavigate = (direction: 'prev' | 'next') => {
    if (viewerImageIndex === null) return;

    if (direction === 'prev' && viewerImageIndex > 0) {
      setViewerImageIndex(viewerImageIndex - 1);
    } else if (direction === 'next' && viewerImageIndex < viewerImages.length - 1) {
      setViewerImageIndex(viewerImageIndex + 1);
    }
  };

  return (
    <>
      {/* Mobile gallery toggle button */}
      {!isEditorOpen && (
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
        ${isEditorOpen ? 'lg:hidden' : ''}
      `}>
        <div className="flex items-center gap-2 overflow-x-auto scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
          {displayImages.map((image, index) => {
            const video = isVideoUrl(image.url);
            const audio = !video && isAudioUrl(image.url);
            const viewerIndex = video || audio ? -1 : viewerImages.indexOf(image);
            return (
              <div
                key={`${image.timestamp}-${index}`}
                className={`flex-shrink-0 hover:opacity-80 transition-opacity ${viewerIndex >= 0 ? "cursor-pointer" : ""}`}
                onDoubleClick={() => {
                  if (viewerIndex >= 0) setViewerImageIndex(viewerIndex);
                }}
              >
                {video ? (
                  // Static tile: `poster` points at the poster PNG the backend
                  // writes next to every clip, so a thumbnail shows without
                  // fetching video data (a missing poster degrades to the
                  // browser's own first-frame handling). Deliberately not
                  // autoplaying -- this strip can hold dozens of results. The
                  // clip is playable full-size in the panel that produced it.
                  <video
                    src={image.url}
                    poster={posterUrlForVideo(image.url)}
                    className="h-24 w-auto object-contain rounded border border-gray-700"
                    preload="metadata"
                    muted
                    playsInline
                  />
                ) : audio ? (
                  <audio src={image.url} controls className="h-24 w-56 rounded border border-gray-700" />
                ) : (
                  <img
                    src={image.url}
                    alt={`Generated ${index + 1}`}
                    className="h-24 w-auto object-contain rounded border border-gray-700"
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {viewerImageIndex !== null && viewerImages[viewerImageIndex] && (
        <ImageViewer
          imageUrl={viewerImages[viewerImageIndex].url}
          onClose={() => setViewerImageIndex(null)}
          onNavigate={handleNavigate}
          hasPrev={viewerImageIndex > 0}
          hasNext={viewerImageIndex < viewerImages.length - 1}
        />
      )}
    </>
  );
}
