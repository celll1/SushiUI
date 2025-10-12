"use client";

import { useState } from "react";
import Card from "../common/Card";
import Input from "../common/Input";
import Textarea from "../common/Textarea";
import Button from "../common/Button";
import Slider from "../common/Slider";
import ModelSelector from "../common/ModelSelector";
import { generateTxt2Img, GenerationParams } from "@/utils/api";

export default function Txt2ImgPanel() {
  const [params, setParams] = useState<GenerationParams>({
    prompt: "",
    negative_prompt: "",
    steps: 20,
    cfg_scale: 7.0,
    sampler: "euler_a",
    seed: -1,
    width: 512,
    height: 512,
  });

  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!params.prompt) {
      alert("Please enter a prompt");
      return;
    }

    setIsGenerating(true);
    try {
      const result = await generateTxt2Img(params);
      setGeneratedImage(`/outputs/${result.image.filename}`);
    } catch (error) {
      console.error("Generation failed:", error);
      alert("Generation failed. Please check console for details.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Parameters Panel */}
      <div className="space-y-4">
        <ModelSelector />

        <Card title="Prompt">
          <Textarea
            label="Positive Prompt"
            placeholder="Enter your prompt here..."
            rows={4}
            value={params.prompt}
            onChange={(e) => setParams({ ...params, prompt: e.target.value })}
          />
          <Textarea
            label="Negative Prompt"
            placeholder="Enter negative prompt..."
            rows={3}
            value={params.negative_prompt}
            onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
          />
        </Card>

        <Card title="Parameters">
          <div className="space-y-4">
            <Slider
              label="Steps"
              min={1}
              max={150}
              step={1}
              value={params.steps}
              onChange={(e) => setParams({ ...params, steps: parseInt(e.target.value) })}
            />
            <Slider
              label="CFG Scale"
              min={1}
              max={30}
              step={0.5}
              value={params.cfg_scale}
              onChange={(e) => setParams({ ...params, cfg_scale: parseFloat(e.target.value) })}
            />
            <Slider
              label="Width"
              min={64}
              max={2048}
              step={64}
              value={params.width}
              onChange={(e) => setParams({ ...params, width: parseInt(e.target.value) })}
            />
            <Slider
              label="Height"
              min={64}
              max={2048}
              step={64}
              value={params.height}
              onChange={(e) => setParams({ ...params, height: parseInt(e.target.value) })}
            />
            <Input
              label="Seed"
              type="number"
              value={params.seed}
              onChange={(e) => setParams({ ...params, seed: parseInt(e.target.value) })}
            />
          </div>
        </Card>

        <Button
          onClick={handleGenerate}
          disabled={isGenerating}
          className="w-full"
          size="lg"
        >
          {isGenerating ? "Generating..." : "Generate"}
        </Button>
      </div>

      {/* Preview Panel */}
      <div>
        <Card title="Preview">
          <div className="aspect-square bg-gray-800 rounded-lg flex items-center justify-center">
            {generatedImage ? (
              <img
                src={generatedImage}
                alt="Generated"
                className="max-w-full max-h-full rounded-lg"
              />
            ) : (
              <p className="text-gray-500">No image generated yet</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
