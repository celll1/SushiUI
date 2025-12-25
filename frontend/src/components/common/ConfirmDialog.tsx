import React from "react";
import { X } from "lucide-react";

export interface ConfirmDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string | React.ReactNode;
  confirmText?: string;
  cancelText?: string;
  confirmButtonClass?: string;
  isDestructive?: boolean;
}

/**
 * Generic confirmation dialog component
 *
 * Usage:
 * ```tsx
 * const [isOpen, setIsOpen] = useState(false);
 *
 * <ConfirmDialog
 *   isOpen={isOpen}
 *   onClose={() => setIsOpen(false)}
 *   onConfirm={handleConfirm}
 *   title="Delete Items"
 *   message={`Are you sure you want to delete ${count} items?`}
 *   confirmText="Delete"
 *   isDestructive
 * />
 * ```
 */
export default function ConfirmDialog({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = "Confirm",
  cancelText = "Cancel",
  confirmButtonClass,
  isDestructive = false,
}: ConfirmDialogProps) {
  if (!isOpen) return null;

  const handleConfirm = () => {
    onConfirm();
    onClose();
  };

  const defaultConfirmClass = isDestructive
    ? "bg-red-600 hover:bg-red-500"
    : "bg-blue-600 hover:bg-blue-500";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-800 rounded-lg shadow-xl max-w-md w-full mx-4 border border-gray-700">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white">{title}</h3>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-700 rounded transition-colors"
          >
            <X className="h-5 w-5 text-gray-400" />
          </button>
        </div>

        {/* Message */}
        <div className="p-4">
          {typeof message === "string" ? (
            <p className="text-sm text-gray-300">{message}</p>
          ) : (
            message
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end gap-2 p-4 border-t border-gray-700">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm font-medium transition-colors"
          >
            {cancelText}
          </button>
          <button
            onClick={handleConfirm}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              confirmButtonClass || defaultConfirmClass
            }`}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
