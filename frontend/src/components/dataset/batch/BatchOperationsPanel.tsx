import React, { useState, useEffect } from "react";
import { Sparkles, ArrowDownUp, Replace, X } from "lucide-react";
import ConfirmDialog from "@/components/common/ConfirmDialog";
import {
  batchTaggerInference,
  batchReorderTags,
  batchReplaceTag,
  cancelBatchOperation,
  BatchTaggerRequest,
  BatchReorderTagsRequest,
  BatchReplaceTagRequest,
} from "@/utils/api";
import { wsClient } from "@/utils/websocket";

interface BatchOperationsPanelProps {
  datasetId: number;
  selectedItemIds: number[];
  totalItems: number;
  captionProcessingConfig: any;
  taggerSettings: any;
  onOperationComplete: () => void;
}

export default function BatchOperationsPanel({
  datasetId,
  selectedItemIds,
  totalItems,
  captionProcessingConfig,
  taggerSettings,
  onOperationComplete,
}: BatchOperationsPanelProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");

  // Confirmation dialogs
  const [showTaggerConfirm, setShowTaggerConfirm] = useState(false);
  const [showReorderConfirm, setShowReorderConfirm] = useState(false);
  const [showReplaceDialog, setShowReplaceDialog] = useState(false);

  // Replace tag dialog state
  const [fromTag, setFromTag] = useState("");
  const [toTag, setToTag] = useState("");

  // Subscribe to progress updates
  useEffect(() => {
    const handleProgress = (step: number, totalSteps: number, message: string) => {
      if (isProcessing) {
        const progress = totalSteps > 0 ? (step / totalSteps) * 100 : 0;
        setProgress(progress);
        setProgressMessage(message || "Processing...");
      }
    };

    wsClient.subscribe(handleProgress);
    return () => wsClient.unsubscribe(handleProgress);
  }, [isProcessing]);

  const targetCount = selectedItemIds.length > 0 ? selectedItemIds.length : totalItems;
  const targetItemIds = selectedItemIds.length > 0 ? selectedItemIds : [];

  // ============================================================
  // Batch Tagger Inference
  // ============================================================

  const handleBatchTagger = async () => {
    if (!taggerSettings) {
      alert("Please configure tagger settings first");
      return;
    }

    setIsProcessing(true);
    setProgress(0);
    setProgressMessage("Starting batch tagger inference...");
    wsClient.connect();

    try {
      const thresholds: Record<string, number> = {};
      taggerSettings.categoryThresholds.forEach((cat: any) => {
        if (cat.enabled) {
          thresholds[cat.id] = cat.addThreshold;
        }
      });

      const request: BatchTaggerRequest = {
        item_ids: targetItemIds,
        gen_threshold: taggerSettings.categoryThresholds.find((c: any) => c.id === "general")?.addThreshold || 0.45,
        char_threshold: taggerSettings.categoryThresholds.find((c: any) => c.id === "character")?.addThreshold || 0.45,
        thresholds,
        model_version: "cl_tagger_1_02",
        merge_with_existing: true,
      };

      const result = await batchTaggerInference(datasetId, request);

      console.log(`[BatchTagger] ${result.message}`);
      console.log(`[BatchTagger] Processed: ${result.processed_count}, Updated: ${result.updated_count}`);

      setProgress(100);
      setProgressMessage(result.message);

      setTimeout(() => {
        setIsProcessing(false);
        onOperationComplete();
      }, 2000);

    } catch (error) {
      console.error("[BatchTagger] Error:", error);
      setProgressMessage("Batch tagger failed. Check console for details.");
      setTimeout(() => setIsProcessing(false), 3000);
    }
  };

  // ============================================================
  // Batch Tag Reordering
  // ============================================================

  const handleBatchReorder = async () => {
    const categoryOrder = captionProcessingConfig?.category_order || [
      "Rating",
      "Quality",
      "Character",
      "Copyright",
      "Artist",
      "General",
      "Meta",
      "Model",
    ];

    setIsProcessing(true);
    setProgress(0);
    setProgressMessage("Starting batch tag reordering...");
    wsClient.connect();

    try {
      const request: BatchReorderTagsRequest = {
        item_ids: targetItemIds,
        category_order: categoryOrder,
      };

      const result = await batchReorderTags(datasetId, request);

      console.log(`[BatchReorder] ${result.message}`);
      console.log(`[BatchReorder] Processed: ${result.processed_count}, Updated: ${result.updated_count}`);

      setProgress(100);
      setProgressMessage(result.message);

      setTimeout(() => {
        setIsProcessing(false);
        onOperationComplete();
      }, 2000);

    } catch (error) {
      console.error("[BatchReorder] Error:", error);
      setProgressMessage("Batch reorder failed. Check console for details.");
      setTimeout(() => setIsProcessing(false), 3000);
    }
  };

  // ============================================================
  // Batch Tag Replacement
  // ============================================================

  const handleBatchReplace = async () => {
    if (!fromTag || !toTag) {
      alert("Please enter both 'from' and 'to' tags");
      return;
    }

    setShowReplaceDialog(false);
    setIsProcessing(true);
    setProgress(0);
    setProgressMessage(`Replacing '${fromTag}' with '${toTag}'...`);
    wsClient.connect();

    try {
      const request: BatchReplaceTagRequest = {
        item_ids: targetItemIds,
        from_tag: fromTag,
        to_tag: toTag,
        normalize_match: true,
      };

      const result = await batchReplaceTag(datasetId, request);

      console.log(`[BatchReplace] ${result.message}`);
      console.log(`[BatchReplace] Processed: ${result.processed_count}, Updated: ${result.updated_count}`);

      setProgress(100);
      setProgressMessage(result.message);

      setTimeout(() => {
        setIsProcessing(false);
        setFromTag("");
        setToTag("");
        onOperationComplete();
      }, 2000);

    } catch (error) {
      console.error("[BatchReplace] Error:", error);
      setProgressMessage("Batch replace failed. Check console for details.");
      setTimeout(() => setIsProcessing(false), 3000);
    }
  };

  // ============================================================
  // Cancel Operation
  // ============================================================

  const handleCancel = async () => {
    try {
      await cancelBatchOperation(datasetId);
      setProgressMessage("Cancelling...");
    } catch (error) {
      console.error("[BatchOps] Failed to cancel:", error);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <h3 className="text-sm font-semibold text-gray-200">Batch Operations</h3>

      {/* Operation Buttons */}
      <div className="grid grid-cols-3 gap-2">
        <button
          onClick={() => setShowTaggerConfirm(true)}
          disabled={isProcessing || targetCount === 0}
          className="flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded text-xs font-medium transition-colors"
          title="Run tagger inference on selected/all items"
        >
          <Sparkles className="h-3.5 w-3.5" />
          <span>Batch Tagger</span>
        </button>

        <button
          onClick={() => setShowReorderConfirm(true)}
          disabled={isProcessing || targetCount === 0}
          className="flex items-center justify-center gap-2 px-3 py-2 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded text-xs font-medium transition-colors"
          title="Reorder tags by category"
        >
          <ArrowDownUp className="h-3.5 w-3.5" />
          <span>Reorder Tags</span>
        </button>

        <button
          onClick={() => setShowReplaceDialog(true)}
          disabled={isProcessing || targetCount === 0}
          className="flex items-center justify-center gap-2 px-3 py-2 bg-orange-600 hover:bg-orange-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded text-xs font-medium transition-colors"
          title="Replace specific tag with another"
        >
          <Replace className="h-3.5 w-3.5" />
          <span>Replace Tag</span>
        </button>
      </div>

      {/* Target Info */}
      <div className="text-xs text-gray-400">
        {selectedItemIds.length > 0 ? (
          <span>Target: {selectedItemIds.length} selected items</span>
        ) : (
          <span>Target: All {totalItems} items</span>
        )}
      </div>

      {/* Progress Bar */}
      {isProcessing && (
        <div className="bg-gray-900 rounded-lg p-3 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Processing...</span>
            <span className="text-sm text-gray-400">{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="flex items-center justify-between">
            <p className="text-xs text-gray-400">{progressMessage}</p>
            <button
              onClick={handleCancel}
              className="px-2 py-1 bg-red-600 hover:bg-red-500 rounded text-xs font-medium transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Confirmation Dialogs */}
      <ConfirmDialog
        isOpen={showTaggerConfirm}
        onClose={() => setShowTaggerConfirm(false)}
        onConfirm={handleBatchTagger}
        title="Batch Tagger Inference"
        message={
          <div>
            <p className="text-sm text-gray-300 mb-2">
              Run tagger inference on {targetCount} items?
            </p>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• Existing tags will be merged with new predictions</li>
              <li>• Tags below threshold will be excluded</li>
              <li>• Changes will be saved to database and txt files</li>
            </ul>
          </div>
        }
        confirmText="Run Tagger"
      />

      <ConfirmDialog
        isOpen={showReorderConfirm}
        onClose={() => setShowReorderConfirm(false)}
        onConfirm={handleBatchReorder}
        title="Batch Tag Reordering"
        message={
          <div>
            <p className="text-sm text-gray-300 mb-2">
              Reorder tags by category for {targetCount} items?
            </p>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• Tags will be sorted by category order from Caption Processing settings</li>
              <li>• Changes will be saved to database and txt files</li>
            </ul>
          </div>
        }
        confirmText="Reorder Tags"
      />

      {/* Replace Tag Dialog */}
      {showReplaceDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-gray-800 rounded-lg shadow-xl max-w-md w-full mx-4 border border-gray-700">
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
              <h3 className="text-lg font-semibold text-white">Replace Tag</h3>
              <button
                onClick={() => setShowReplaceDialog(false)}
                className="p-1 hover:bg-gray-700 rounded transition-colors"
              >
                <X className="h-5 w-5 text-gray-400" />
              </button>
            </div>

            <div className="p-4 space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1">From Tag</label>
                <input
                  type="text"
                  value={fromTag}
                  onChange={(e) => setFromTag(e.target.value)}
                  placeholder="e.g., bad qualllty"
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1">To Tag</label>
                <input
                  type="text"
                  value={toTag}
                  onChange={(e) => setToTag(e.target.value)}
                  placeholder="e.g., bad quality"
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                />
              </div>

              <p className="text-xs text-gray-500">
                Target: {targetCount} items (only items with the tag will be updated)
              </p>
            </div>

            <div className="flex items-center justify-end gap-2 p-4 border-t border-gray-700">
              <button
                onClick={() => setShowReplaceDialog(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleBatchReplace}
                disabled={!fromTag || !toTag}
                className="px-4 py-2 bg-orange-600 hover:bg-orange-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded text-sm font-medium transition-colors"
              >
                Replace
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
