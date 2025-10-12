"use client";

import { useEffect, useState } from "react";
import { getImages, GeneratedImage } from "@/utils/api";
import Card from "../common/Card";

export default function ImageGrid() {
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null);

  useEffect(() => {
    loadImages();
  }, []);

  const loadImages = async () => {
    try {
      const result = await getImages();
      setImages(result.images);
    } catch (error) {
      console.error("Failed to load images:", error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="text-center py-8">Loading images...</div>;
  }

  return (
    <div>
      {selectedImage ? (
        <div className="space-y-4">
          <button
            onClick={() => setSelectedImage(null)}
            className="text-blue-400 hover:text-blue-300"
          >
            ← Back to gallery
          </button>
          <Card title="Image Details">
            <div className="space-y-4">
              <img
                src={`/outputs/${selectedImage.filename}`}
                alt="Generated"
                className="w-full rounded-lg"
              />
              <div className="space-y-2 text-sm">
                <div>
                  <span className="text-gray-400">Prompt:</span>
                  <p className="text-gray-100">{selectedImage.prompt}</p>
                </div>
                {selectedImage.negative_prompt && (
                  <div>
                    <span className="text-gray-400">Negative Prompt:</span>
                    <p className="text-gray-100">{selectedImage.negative_prompt}</p>
                  </div>
                )}
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-gray-400">Steps:</span> {selectedImage.steps}
                  </div>
                  <div>
                    <span className="text-gray-400">CFG Scale:</span> {selectedImage.cfg_scale}
                  </div>
                  <div>
                    <span className="text-gray-400">Seed:</span> {selectedImage.seed}
                  </div>
                  <div>
                    <span className="text-gray-400">Size:</span> {selectedImage.width}x{selectedImage.height}
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {images.map((image) => (
            <div
              key={image.id}
              onClick={() => setSelectedImage(image)}
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
      )}
    </div>
  );
}
