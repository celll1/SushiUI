"use client";

import { createContext, useContext, useState, useCallback, ReactNode } from "react";
import { GenerationParams, Img2ImgParams, InpaintParams } from "@/utils/api";

export interface QueueItem {
  id: string;
  type: "txt2img" | "img2img" | "inpaint";
  params: GenerationParams | Img2ImgParams | InpaintParams;
  inputImage?: string; // For img2img and inpaint
  maskImage?: string; // For inpaint only
  status: "pending" | "generating" | "completed" | "failed";
  addedAt: number;
  prompt: string; // For display purposes
}

interface GenerationQueueContextType {
  queue: QueueItem[];
  currentItem: QueueItem | null;
  addToQueue: (item: Omit<QueueItem, "id" | "status" | "addedAt">) => void;
  removeFromQueue: (id: string) => void;
  startNextInQueue: () => QueueItem | null;
  completeCurrentItem: () => void;
  failCurrentItem: () => void;
  clearQueue: () => void;
  generateForever: boolean;
  setGenerateForever: (enabled: boolean) => void;
}

const GenerationQueueContext = createContext<GenerationQueueContextType | undefined>(undefined);

export function GenerationQueueProvider({ children }: { children: ReactNode }) {
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [currentItem, setCurrentItem] = useState<QueueItem | null>(null);
  const [generateForever, setGenerateForever] = useState<boolean>(false);

  const addToQueue = useCallback((item: Omit<QueueItem, "id" | "status" | "addedAt">) => {
    const newItem: QueueItem = {
      ...item,
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      status: "pending",
      addedAt: Date.now(),
    };
    setQueue((prev) => [...prev, newItem]);
  }, []);

  const removeFromQueue = useCallback((id: string) => {
    setQueue((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const startNextInQueue = useCallback(() => {
    // Find next pending item from current queue state (synchronous)
    const nextItem = queue.find((item) => item.status === "pending");
    console.log("[QueueContext] startNextInQueue - current queue:", queue);
    console.log("[QueueContext] Found next item:", nextItem);

    if (nextItem) {
      const updatedItem = { ...nextItem, status: "generating" as const };
      setCurrentItem(updatedItem);

      // Update the item in queue to generating status
      setQueue((prev) =>
        prev.map((item) =>
          item.id === nextItem.id ? updatedItem : item
        )
      );

      console.log("[QueueContext] Returning updated item:", updatedItem);
      return updatedItem;
    }

    console.log("[QueueContext] No pending items found, setting currentItem to null");
    setCurrentItem(null);
    return null;
  }, [queue]);

  const completeCurrentItem = useCallback(() => {
    if (!currentItem) return;

    console.log("[QueueContext] Completing item:", currentItem.id);
    // Remove completed item from queue
    setQueue((prev) => prev.filter((item) => item.id !== currentItem.id));
    setCurrentItem(null);
  }, [currentItem]);

  const failCurrentItem = useCallback(() => {
    if (!currentItem) return;

    console.log("[QueueContext] Failing item:", currentItem.id);
    // Mark as failed but keep in queue for user to see
    setQueue((prev) =>
      prev.map((item) =>
        item.id === currentItem.id ? { ...item, status: "failed" as const } : item
      )
    );
    setCurrentItem(null);
  }, [currentItem]);

  const clearQueue = useCallback(() => {
    setQueue([]);
    setCurrentItem(null);
  }, []);

  return (
    <GenerationQueueContext.Provider
      value={{
        queue,
        currentItem,
        addToQueue,
        removeFromQueue,
        startNextInQueue,
        completeCurrentItem,
        failCurrentItem,
        clearQueue,
        generateForever,
        setGenerateForever,
      }}
    >
      {children}
    </GenerationQueueContext.Provider>
  );
}

export function useGenerationQueue() {
  const context = useContext(GenerationQueueContext);
  if (!context) {
    throw new Error("useGenerationQueue must be used within a GenerationQueueProvider");
  }
  return context;
}
