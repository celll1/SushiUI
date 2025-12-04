"use client";

import { createContext, useContext, useState, useCallback, ReactNode, useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
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
  loopGroupId?: string; // ID to group loop steps together
  loopStepIndex?: number; // Index of this step in the loop sequence
  isLoopStep?: boolean; // Whether this is a loop step (vs main generation)
  startTime?: number; // When generation started (for timing)
  endTime?: number; // When generation completed (for timing)
}

interface GenerationQueueContextType {
  queue: QueueItem[];
  currentItem: QueueItem | null;
  addToQueue: (item: Omit<QueueItem, "id" | "status" | "addedAt">) => void;
  removeFromQueue: (id: string) => void;
  updateQueueItem: (id: string, updates: Partial<QueueItem>) => void;
  updateQueueItemByLoop: (loopGroupId: string, loopStepIndex: number, updates: Partial<QueueItem> | ((item: QueueItem) => Partial<QueueItem>)) => void;
  cancelLoopGroup: (loopGroupId: string) => void;
  cancelRelatedItems: (itemId: string) => void;
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

  // Use refs that are synchronously updated alongside state
  const queueRef = useRef<QueueItem[]>(queue);
  const currentItemRef = useRef<QueueItem | null>(currentItem);

  // Synchronously update refs whenever state changes
  queueRef.current = queue;
  currentItemRef.current = currentItem;

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

  const updateQueueItem = useCallback((id: string, updates: Partial<QueueItem>) => {
    setQueue((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, ...updates } : item
      )
    );
  }, []);

  const updateQueueItemByLoop = useCallback((loopGroupId: string, loopStepIndex: number, updates: Partial<QueueItem> | ((item: QueueItem) => Partial<QueueItem>)) => {
    setQueue((prev) => {
      const item = prev.find((item) =>
        item.loopGroupId === loopGroupId &&
        item.loopStepIndex === loopStepIndex
      );

      if (item) {
        console.log(`[QueueContext] updateQueueItemByLoop: Found item with loopGroupId=${loopGroupId}, loopStepIndex=${loopStepIndex}`);
        const actualUpdates = typeof updates === 'function' ? updates(item) : updates;
        return prev.map((i) =>
          i.id === item.id ? { ...i, ...actualUpdates} : i
        );
      } else {
        console.log(`[QueueContext] updateQueueItemByLoop: Item not found with loopGroupId=${loopGroupId}, loopStepIndex=${loopStepIndex}`);
        return prev;
      }
    });
  }, []);

  const cancelLoopGroup = useCallback((loopGroupId: string) => {
    console.log(`[QueueContext] Cancelling all pending items in loop group: ${loopGroupId}`);

    // Remove all pending items with this loopGroupId
    setQueue((prev) => prev.filter((item) =>
      !(item.loopGroupId === loopGroupId && item.status === "pending")
    ));

    // If currentItem is part of this loop group, clear its loopGroupId
    // to prevent trying to continue the cancelled loop sequence
    setCurrentItem((prev) => {
      if (prev && prev.loopGroupId === loopGroupId) {
        console.log(`[QueueContext] Clearing loopGroupId from currentItem to break loop sequence`);
        return { ...prev, loopGroupId: undefined, loopStepIndex: undefined };
      }
      return prev;
    });
  }, []);

  // Smart cancel: determine what to cancel based on item type and position
  const cancelRelatedItems = useCallback((itemId: string) => {
    const item = queueRef.current.find((i) => i.id === itemId);
    if (!item || !item.loopGroupId) {
      // No loop group - just remove this item
      console.log(`[QueueContext] Removing single item: ${itemId}`);
      removeFromQueue(itemId);
      return;
    }

    const { loopGroupId, isLoopStep, loopStepIndex } = item;
    const allItemsInGroup = queueRef.current.filter((i) => i.loopGroupId === loopGroupId);

    // Case 1: Cancelling main generation (Base)
    if (!isLoopStep) {
      console.log(`[QueueContext] Cancelling main generation and all related loop steps for group: ${loopGroupId}`);
      // Remove all items in this loop group (main + all loop steps)
      setQueue((prev) => prev.filter((i) => i.loopGroupId !== loopGroupId));

      // Clear loopGroupId from currentItem if it belongs to this group
      setCurrentItem((prev) => {
        if (prev && prev.loopGroupId === loopGroupId) {
          return { ...prev, loopGroupId: undefined, loopStepIndex: undefined };
        }
        return prev;
      });
      return;
    }

    // Case 2 & 3: Cancelling loop step
    const maxLoopStepIndex = Math.max(
      ...allItemsInGroup
        .filter((i) => i.isLoopStep && i.loopStepIndex !== undefined)
        .map((i) => i.loopStepIndex!)
    );

    // Case 2: Cancelling first or middle loop step - cancel this and all following steps
    if (loopStepIndex !== undefined && loopStepIndex < maxLoopStepIndex) {
      console.log(`[QueueContext] Cancelling loop step ${loopStepIndex} and all following steps in group: ${loopGroupId}`);
      setQueue((prev) => prev.filter((i) =>
        !(i.loopGroupId === loopGroupId &&
          i.isLoopStep === true &&
          i.loopStepIndex !== undefined &&
          i.loopStepIndex >= loopStepIndex)
      ));

      // Clear loopGroupId from currentItem if it's a later step
      setCurrentItem((prev) => {
        if (prev &&
            prev.loopGroupId === loopGroupId &&
            prev.isLoopStep === true &&
            prev.loopStepIndex !== undefined &&
            prev.loopStepIndex >= loopStepIndex) {
          return { ...prev, loopGroupId: undefined, loopStepIndex: undefined };
        }
        return prev;
      });
      return;
    }

    // Case 3: Cancelling last loop step - only cancel this step
    console.log(`[QueueContext] Cancelling only the last loop step ${loopStepIndex} in group: ${loopGroupId}`);
    removeFromQueue(itemId);
  }, [removeFromQueue]);

  const startNextInQueue = useCallback(() => {
    // Access latest values from refs (synchronously updated)
    const currentQueue = queueRef.current;
    const currentItemValue = currentItemRef.current;

    console.log("[QueueContext] startNextInQueue - current queue:", currentQueue);
    console.log("[QueueContext] currentItem:", currentItemValue);

    let nextItem: QueueItem | undefined;

    // If current item is part of a loop group, prioritize next step in same group
    if (currentItemValue?.loopGroupId) {
      const currentLoopGroupId = currentItemValue.loopGroupId;
      const currentLoopStepIndex = currentItemValue.loopStepIndex ?? -1;

      // Find next step in the same loop group
      nextItem = currentQueue.find((item) =>
        item.status === "pending" &&
        item.loopGroupId === currentLoopGroupId &&
        (item.loopStepIndex ?? 0) === currentLoopStepIndex + 1
      );

      console.log("[QueueContext] Looking for next loop step in group:", currentLoopGroupId, "after index:", currentLoopStepIndex);
      console.log("[QueueContext] Found loop step:", nextItem);
    }

    // If no loop step found, get next pending item in queue order
    if (!nextItem) {
      nextItem = currentQueue.find((item) => item.status === "pending");
      console.log("[QueueContext] Found next pending item:", nextItem);
    }

    if (nextItem) {
      const updatedItem = { ...nextItem, status: "generating" as const, startTime: Date.now() };
      setCurrentItem(updatedItem);

      // Update the item in queue to generating status
      setQueue((prev) =>
        prev.map((item) =>
          item.id === nextItem!.id ? updatedItem : item
        )
      );

      console.log("[QueueContext] Returning updated item:", updatedItem);
      return updatedItem;
    }

    console.log("[QueueContext] No pending items found, setting currentItem to null");
    setCurrentItem(null);
    return null;
  }, []); // Empty deps - uses refs for latest values

  const completeCurrentItem = useCallback(() => {
    const currentItemValue = currentItemRef.current;
    if (!currentItemValue) return;

    console.log("[QueueContext] Completing item:", currentItemValue.id);
    // Mark completion time before removing
    const endTime = Date.now();
    const elapsedMs = currentItemValue.startTime ? endTime - currentItemValue.startTime : 0;
    console.log(`[QueueContext] Generation took ${(elapsedMs / 1000).toFixed(2)}s`);

    // Remove completed item from queue
    setQueue((prev) => prev.filter((item) => item.id !== currentItemValue.id));
    setCurrentItem(null);
  }, []); // Empty deps - uses refs

  const failCurrentItem = useCallback(() => {
    const currentItemValue = currentItemRef.current;
    if (!currentItemValue) return;

    console.log("[QueueContext] Failing item:", currentItemValue.id);
    // Mark as failed but keep in queue for user to see
    setQueue((prev) =>
      prev.map((item) =>
        item.id === currentItemValue.id ? { ...item, status: "failed" as const } : item
      )
    );
    setCurrentItem(null);
  }, []); // Empty deps - uses refs

  const clearQueue = useCallback(() => {
    setQueue([]);
    setCurrentItem(null);
  }, []);

  // Track pathname to detect page navigation
  const pathname = usePathname();
  const [prevPathname, setPrevPathname] = useState(pathname);

  useEffect(() => {
    // If pathname changed and there's a current item generating
    if (pathname !== prevPathname) {
      const currentItemValue = currentItemRef.current;
      if (currentItemValue) {
        console.log("[QueueContext] Page navigation detected while generating, marking current item as completed");
        console.log(`[QueueContext] Navigated from ${prevPathname} to ${pathname}`);

        // Remove current item from queue (generation continues in background)
        setQueue((prev) => prev.filter((item) => item.id === currentItemValue.id));
        setCurrentItem(null);
      }
      setPrevPathname(pathname);
    }
  }, [pathname, prevPathname]); // Use refs for currentItem

  return (
    <GenerationQueueContext.Provider
      value={{
        queue,
        currentItem,
        addToQueue,
        removeFromQueue,
        updateQueueItem,
        updateQueueItemByLoop,
        cancelLoopGroup,
        cancelRelatedItems,
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
