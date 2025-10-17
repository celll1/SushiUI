"use client";

import React, { memo } from "react";
import { GeneratedImage } from "@/utils/api";

interface ImageListProps {
  images: GeneratedImage[];
  gridColumns: number;
  onImageClick: (image: GeneratedImage) => void;
}

const ImageList: React.FC<ImageListProps> = memo(({ images, gridColumns, onImageClick }) => {
  return (
    <div className="flex-1">
      <div
        className="grid gap-4"
        style={{
          gridTemplateColumns: `repeat(${gridColumns}, minmax(0, 1fr))`
        }}
      >
        {images.map((image) => (
          <div
            key={image.id}
            onClick={() => onImageClick(image)}
            className="cursor-pointer group"
          >
            <div className="aspect-square bg-gray-800 rounded-lg overflow-hidden">
              <img
                src={`/thumbnails/${image.filename}`}
                alt={image.prompt}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform"
              />
            </div>
            <p className="mt-2 text-xs text-gray-400 truncate">{image.prompt}</p>
          </div>
        ))}
      </div>
    </div>
  );
});

ImageList.displayName = "ImageList";

export default ImageList;
