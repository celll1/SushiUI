"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getImages, GeneratedImage } from "@/utils/api";
import Card from "../common/Card";
import Button from "../common/Button";

export default function ImageGrid() {
  const router = useRouter();
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

  const importToTxt2Img = (image: GeneratedImage) => {
    console.log("Importing image parameters:", image);
    console.log("Image parameters field:", image.parameters);

    const params = {
      prompt: image.prompt,
      negative_prompt: image.negative_prompt,
      steps: image.steps,
      cfg_scale: image.cfg_scale,
      sampler: image.parameters?.sampler || "euler",
      schedule_type: image.parameters?.schedule_type || "uniform",
      seed: image.seed,
      width: image.width,
      height: image.height,
    };

    console.log("Constructed params for import:", params);
    localStorage.setItem("txt2img_params", JSON.stringify(params));
    console.log("Saved to localStorage, navigating to generate page...");
    router.push("/generate");
  };

  const sendToImg2Img = (image: GeneratedImage) => {
    // Save image URL to img2img input storage
    const imageUrl = `/outputs/${image.filename}`;
    localStorage.setItem("img2img_input_image", imageUrl);

    // Trigger event to notify img2img tab
    window.dispatchEvent(new Event("img2img_input_updated"));

    // Navigate to generate page with img2img tab
    router.push("/generate?tab=img2img");
  };

  const importToImg2Img = (image: GeneratedImage) => {
    const params = {
      prompt: image.prompt,
      negative_prompt: image.negative_prompt,
      steps: image.steps,
      cfg_scale: image.cfg_scale,
      sampler: image.parameters?.sampler || "euler",
      schedule_type: image.parameters?.schedule_type || "uniform",
      seed: image.seed,
      width: image.width,
      height: image.height,
      denoising_strength: 0.75,
    };

    localStorage.setItem("img2img_params", JSON.stringify(params));
    router.push("/generate?tab=img2img");
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
                    <span className="text-gray-400">Sampler:</span> {selectedImage.sampler}
                  </div>
                  <div>
                    <span className="text-gray-400">Seed:</span> {selectedImage.seed}
                  </div>
                  <div>
                    <span className="text-gray-400">Size:</span> {selectedImage.width}x{selectedImage.height}
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  onClick={() => importToTxt2Img(selectedImage)}
                  className="w-full"
                >
                  Import to txt2img
                </Button>
                <Button
                  onClick={() => sendToImg2Img(selectedImage)}
                  variant="secondary"
                  className="w-full"
                >
                  Send to img2img
                </Button>
              </div>
              <Button
                onClick={() => importToImg2Img(selectedImage)}
                variant="secondary"
                className="w-full"
              >
                Import to img2img
              </Button>
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
