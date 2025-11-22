"use client";

import { useState, useEffect } from "react";
import Sidebar from "@/components/common/Sidebar";
import Card from "@/components/common/Card";
import Button from "@/components/common/Button";
import DirectorySettings from "@/components/settings/DirectorySettings";
import ProtectedRoute from "@/components/common/ProtectedRoute";
import { restartBackend, restartFrontend, restartBoth } from "@/utils/api";

// Default presets
const DEFAULT_ASPECT_RATIO_PRESETS = [
  { label: "1:1", ratio: 1 / 1 },
  { label: "4:3", ratio: 4 / 3 },
  { label: "3:4", ratio: 3 / 4 },
  { label: "16:9", ratio: 16 / 9 },
  { label: "9:16", ratio: 9 / 16 },
  { label: "21:9", ratio: 21 / 9 },
  { label: "9:21", ratio: 9 / 21 },
  { label: "3:2", ratio: 3 / 2 },
  { label: "2:3", ratio: 2 / 3 },
  { label: "5:4", ratio: 5 / 4 },
];

const DEFAULT_FIXED_RESOLUTION_PRESETS = [
  { width: 768, height: 1152 },
  { width: 1152, height: 768 },
  { width: 1248, height: 720 },
  { width: 720, height: 1248 },
  { width: 960, height: 1344 },
  { width: 1344, height: 960 },
  { width: 1024, height: 1152 },
  { width: 1152, height: 1024 },
  { width: 1024, height: 1024 },
  { width: 896, height: 1152 },
  { width: 1152, height: 896 },
  { width: 832, height: 1216 },
  { width: 1216, height: 832 },
  { width: 640, height: 1536 },
  { width: 1536, height: 640 },
  { width: 512, height: 512 },
];

export default function SettingsPage() {
  const [isRestarting, setIsRestarting] = useState(false);
  const [storageInfo, setStorageInfo] = useState({ used: 0, quota: 0 });
  const [restoreOnCancel, setRestoreOnCancel] = useState(false);
  const [resolutionStep, setResolutionStep] = useState(64);
  const [aspectRatioPresets, setAspectRatioPresets] = useState(DEFAULT_ASPECT_RATIO_PRESETS);
  const [fixedResolutionPresets, setFixedResolutionPresets] = useState(DEFAULT_FIXED_RESOLUTION_PRESETS);
  const [includeMetadataInDownloads, setIncludeMetadataInDownloads] = useState(false);

  // Send size mode settings
  const [sendSizeMode, setSendSizeMode] = useState<"absolute" | "scale">("absolute");
  const [sendDefaultScale, setSendDefaultScale] = useState(1.0);

  // Developer mode
  const [developerMode, setDeveloperMode] = useState(false);

  // Panel visibility settings
  const [txt2imgVisibility, setTxt2imgVisibility] = useState({
    lora: true,
    controlnet: true,
    aspectRatioPresets: true,
    fixedResolutionPresets: true,
  });
  const [img2imgVisibility, setImg2imgVisibility] = useState({
    lora: true,
    controlnet: true,
    aspectRatioPresets: true,
    fixedResolutionPresets: true,
  });
  const [inpaintVisibility, setInpaintVisibility] = useState({
    lora: true,
    controlnet: true,
    aspectRatioPresets: true,
    fixedResolutionPresets: true,
  });

  const updateStorageInfo = () => {
    if (typeof window !== 'undefined' && 'storage' in navigator && 'estimate' in navigator.storage) {
      navigator.storage.estimate().then(estimate => {
        setStorageInfo({
          used: estimate.usage || 0,
          quota: estimate.quota || 0,
        });
      });
    }
  };

  const handleClearLocalStorage = () => {
    if (!confirm("Are you sure you want to clear all localStorage data? This will reset all saved settings, images, and panel states.")) {
      return;
    }

    try {
      localStorage.clear();
      alert("localStorage cleared successfully! The page will reload.");
      window.location.reload();
    } catch (error) {
      console.error("Failed to clear localStorage:", error);
      alert("Failed to clear localStorage. Please check the console.");
    }
  };

  const handleClearTempImages = async () => {
    if (!confirm("Are you sure you want to clear all temporary images? This will remove all saved input images and ControlNet references.")) {
      return;
    }

    try {
      const { cleanupTempImages } = await import("@/utils/api");
      const deletedCount = await cleanupTempImages(0); // Delete all images (max age 0 hours)
      alert(`Successfully deleted ${deletedCount} temporary images.`);
      updateStorageInfo();
    } catch (error) {
      console.error("Failed to clear temp images:", error);
      alert("Failed to clear temp images. Please check the console.");
    }
  };

  const handleRestartBackend = async () => {
    if (!confirm("Are you sure you want to restart the backend server?")) {
      return;
    }

    setIsRestarting(true);
    try {
      const result = await restartBackend();
      console.log("Backend restart response:", result);
      alert("Backend restart scheduled. The backend will restart in a moment. You may need to refresh the page in a few seconds.");
    } catch (error: any) {
      console.error("Failed to restart backend:", error);
      console.error("Error details:", error.response?.data);
      const errorMsg = error.response?.data?.detail || error.message || "Unknown error";
      alert(`Failed to restart backend: ${errorMsg}\n\nPlease check the backend console for details.`);
    } finally {
      // Keep the button disabled for a few seconds
      setTimeout(() => {
        setIsRestarting(false);
      }, 5000);
    }
  };

  const handleRestartFrontend = () => {
    if (!confirm("Are you sure you want to restart the frontend? The page will reload.")) {
      return;
    }

    restartFrontend();
  };

  const handleRestartBoth = async () => {
    if (!confirm("Are you sure you want to restart both servers? The page will reload after backend restarts.")) {
      return;
    }

    setIsRestarting(true);
    try {
      await restartBoth();
    } catch (error) {
      console.error("Failed to restart servers:", error);
      alert("Failed to restart servers. Please check the console.");
      setIsRestarting(false);
    }
  };

  useEffect(() => {
    updateStorageInfo();
    // Load settings from localStorage
    if (typeof window !== 'undefined') {
      setRestoreOnCancel(localStorage.getItem('restore_image_on_cancel') === 'true');
      setIncludeMetadataInDownloads(localStorage.getItem('include_metadata_in_downloads') === 'true');

      const savedResolutionStep = localStorage.getItem('resolution_step');
      if (savedResolutionStep) {
        setResolutionStep(parseInt(savedResolutionStep));
      }

      // Load custom presets
      const savedAspectRatioPresets = localStorage.getItem('aspect_ratio_presets');
      if (savedAspectRatioPresets) {
        try {
          setAspectRatioPresets(JSON.parse(savedAspectRatioPresets));
        } catch (e) {
          console.error('Failed to parse aspect ratio presets:', e);
        }
      }

      const savedFixedResolutionPresets = localStorage.getItem('fixed_resolution_presets');
      if (savedFixedResolutionPresets) {
        try {
          setFixedResolutionPresets(JSON.parse(savedFixedResolutionPresets));
        } catch (e) {
          console.error('Failed to parse fixed resolution presets:', e);
        }
      }

      // Load panel visibility settings
      const savedTxt2imgVisibility = localStorage.getItem('txt2img_visibility');
      if (savedTxt2imgVisibility) {
        try {
          setTxt2imgVisibility(JSON.parse(savedTxt2imgVisibility));
        } catch (e) {
          console.error('Failed to parse txt2img visibility:', e);
        }
      }

      const savedImg2imgVisibility = localStorage.getItem('img2img_visibility');
      if (savedImg2imgVisibility) {
        try {
          setImg2imgVisibility(JSON.parse(savedImg2imgVisibility));
        } catch (e) {
          console.error('Failed to parse img2img visibility:', e);
        }
      }

      const savedInpaintVisibility = localStorage.getItem('inpaint_visibility');
      if (savedInpaintVisibility) {
        try {
          setInpaintVisibility(JSON.parse(savedInpaintVisibility));
        } catch (e) {
          console.error('Failed to parse inpaint visibility:', e);
        }
      }

      // Load send size mode settings
      const savedSendSizeMode = localStorage.getItem('send_size_mode');
      if (savedSendSizeMode && (savedSendSizeMode === 'absolute' || savedSendSizeMode === 'scale')) {
        setSendSizeMode(savedSendSizeMode);
      }

      const savedSendDefaultScale = localStorage.getItem('send_default_scale');
      if (savedSendDefaultScale) {
        const scale = parseFloat(savedSendDefaultScale);
        if (!isNaN(scale) && scale > 0) {
          setSendDefaultScale(scale);
        }
      }

      // Load developer mode setting
      const savedDeveloperMode = localStorage.getItem('developer_mode');
      if (savedDeveloperMode === 'true') {
        setDeveloperMode(true);
      }
    }
  }, []);

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  // Aspect ratio preset management
  const handleAddAspectRatioPreset = () => {
    const label = prompt("Enter aspect ratio label (e.g., '16:9'):");
    if (!label) return;

    const ratioStr = prompt("Enter aspect ratio as width:height (e.g., '16:9'):");
    if (!ratioStr) return;

    const [w, h] = ratioStr.split(':').map(n => parseFloat(n.trim()));
    if (isNaN(w) || isNaN(h) || w <= 0 || h <= 0) {
      alert("Invalid ratio format. Please use format like '16:9'");
      return;
    }

    const newPresets = [...aspectRatioPresets, { label, ratio: w / h }];
    setAspectRatioPresets(newPresets);
    localStorage.setItem('aspect_ratio_presets', JSON.stringify(newPresets));
  };

  const handleRemoveAspectRatioPreset = (index: number) => {
    const newPresets = aspectRatioPresets.filter((_, i) => i !== index);
    setAspectRatioPresets(newPresets);
    localStorage.setItem('aspect_ratio_presets', JSON.stringify(newPresets));
  };

  const handleRestoreAspectRatioDefaults = () => {
    if (!confirm("Restore default aspect ratio presets?")) return;
    setAspectRatioPresets(DEFAULT_ASPECT_RATIO_PRESETS);
    localStorage.setItem('aspect_ratio_presets', JSON.stringify(DEFAULT_ASPECT_RATIO_PRESETS));
  };

  // Fixed resolution preset management
  const handleAddFixedResolutionPreset = () => {
    const widthStr = prompt("Enter width (must be multiple of 8):");
    if (!widthStr) return;
    let width = parseInt(widthStr);
    if (isNaN(width) || width < 8) {
      alert("Invalid width");
      return;
    }
    // Round to nearest multiple of 8
    width = Math.round(width / 8) * 8;

    const heightStr = prompt("Enter height (must be multiple of 8):");
    if (!heightStr) return;
    let height = parseInt(heightStr);
    if (isNaN(height) || height < 8) {
      alert("Invalid height");
      return;
    }
    // Round to nearest multiple of 8
    height = Math.round(height / 8) * 8;

    const newPresets = [...fixedResolutionPresets, { width, height }];
    setFixedResolutionPresets(newPresets);
    localStorage.setItem('fixed_resolution_presets', JSON.stringify(newPresets));
  };

  const handleRemoveFixedResolutionPreset = (index: number) => {
    const newPresets = fixedResolutionPresets.filter((_, i) => i !== index);
    setFixedResolutionPresets(newPresets);
    localStorage.setItem('fixed_resolution_presets', JSON.stringify(newPresets));
  };

  const handleRestoreFixedResolutionDefaults = () => {
    if (!confirm("Restore default fixed resolution presets?")) return;
    setFixedResolutionPresets(DEFAULT_FIXED_RESOLUTION_PRESETS);
    localStorage.setItem('fixed_resolution_presets', JSON.stringify(DEFAULT_FIXED_RESOLUTION_PRESETS));
  };

  return (
    <ProtectedRoute>
      <div className="flex h-screen">
        <Sidebar />
        <main className="flex-1 overflow-auto p-6">
          <h1 className="text-2xl font-bold mb-6">Settings</h1>

        <div className="space-y-6">
          <Card title="Server Control">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Restart the backend or frontend servers without manually stopping them.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Button
                  onClick={handleRestartBackend}
                  disabled={isRestarting}
                  variant="secondary"
                  className="w-full"
                >
                  {isRestarting ? "Restarting..." : "Restart Backend"}
                </Button>

                <Button
                  onClick={handleRestartFrontend}
                  disabled={isRestarting}
                  variant="secondary"
                  className="w-full"
                >
                  Restart Frontend
                </Button>

                <Button
                  onClick={handleRestartBoth}
                  disabled={isRestarting}
                  className="w-full"
                >
                  {isRestarting ? "Restarting..." : "Restart Both"}
                </Button>
              </div>

              <div className="mt-4 p-4 bg-gray-800 rounded-lg">
                <h3 className="text-sm font-semibold mb-2">Notes:</h3>
                <ul className="text-sm text-gray-400 space-y-1 list-disc list-inside">
                  <li><strong>Backend:</strong> Restarts the Python FastAPI server. Use this after code changes in backend/.</li>
                  <li><strong>Frontend:</strong> Reloads the page. Use this to refresh the UI state.</li>
                  <li><strong>Both:</strong> Restarts backend first, then reloads the page after 2 seconds.</li>
                </ul>
              </div>
            </div>
          </Card>

          <Card title="Storage Management">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Manage browser storage and temporary files to free up space.
              </p>

              {storageInfo.quota > 0 && (
                <div className="p-4 bg-gray-800 rounded-lg mb-4">
                  <h3 className="text-sm font-semibold mb-2">Storage Usage</h3>
                  <div className="space-y-2 text-sm text-gray-400">
                    <div className="flex justify-between">
                      <span>Used:</span>
                      <span className="font-mono">{formatBytes(storageInfo.used)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Quota:</span>
                      <span className="font-mono">{formatBytes(storageInfo.quota)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Available:</span>
                      <span className="font-mono">{formatBytes(storageInfo.quota - storageInfo.used)}</span>
                    </div>
                    <div className="mt-2 bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full"
                        style={{ width: `${(storageInfo.used / storageInfo.quota) * 100}%` }}
                      />
                    </div>
                    <div className="text-xs text-center text-gray-500">
                      {((storageInfo.used / storageInfo.quota) * 100).toFixed(1)}% used
                    </div>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Button
                  onClick={handleClearLocalStorage}
                  variant="secondary"
                  className="w-full"
                >
                  Clear localStorage
                </Button>

                <Button
                  onClick={handleClearTempImages}
                  variant="secondary"
                  className="w-full"
                >
                  Clear Temp Images
                </Button>
              </div>

              <div className="mt-4 p-4 bg-gray-800 rounded-lg">
                <h3 className="text-sm font-semibold mb-2">What gets cleared:</h3>
                <ul className="text-sm text-gray-400 space-y-1 list-disc list-inside">
                  <li><strong>localStorage:</strong> All saved settings, prompts, parameters, and image references. The page will reload after clearing.</li>
                  <li><strong>Temp Images:</strong> All temporary images stored on the server (input images, ControlNet references). References in localStorage will become invalid.</li>
                </ul>
              </div>
            </div>
          </Card>

          <Card title="Model Directories">
            <DirectorySettings />
          </Card>

          <Card title="Tag Suggestions">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Configure tag autocompletion behavior in prompt fields.
              </p>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Minimum Tag Count
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="10000"
                    value={
                      typeof window !== 'undefined'
                        ? parseInt(localStorage.getItem('tag_suggestion_min_count') || '50')
                        : 50
                    }
                    onChange={(e) => {
                      localStorage.setItem('tag_suggestion_min_count', e.target.value);
                    }}
                    className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Only show tags that appear at least this many times in the dataset. Lower values show more tags but may include uncommon or misspelled tags. Default: 50
                  </p>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Generation Gallery">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Configure the floating gallery that shows recent generated images.
              </p>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Maximum Images in Gallery
                  </label>
                  <input
                    type="number"
                    min="5"
                    max="100"
                    value={
                      typeof window !== 'undefined'
                        ? parseInt(localStorage.getItem('floating_gallery_max_images') || '30')
                        : 30
                    }
                    onChange={(e) => {
                      localStorage.setItem('floating_gallery_max_images', e.target.value);
                    }}
                    className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Maximum number of recent images to keep in the floating gallery. Older images will be removed automatically. Default: 30
                  </p>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Generation Behavior">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Configure how the UI behaves during and after generation.
              </p>

              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <input
                    type="checkbox"
                    id="restore_on_cancel"
                    checked={restoreOnCancel}
                    onChange={(e) => {
                      const newValue = e.target.checked;
                      setRestoreOnCancel(newValue);
                      localStorage.setItem('restore_image_on_cancel', newValue.toString());
                    }}
                    className="mt-1 w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  />
                  <div>
                    <label htmlFor="restore_on_cancel" className="text-sm font-medium text-gray-300 cursor-pointer">
                      Restore previous image on generation cancel
                    </label>
                    <p className="text-xs text-gray-500 mt-1">
                      When enabled, cancelling a generation will restore the previously completed image instead of showing the intermediate TAESD preview. Disable this if you want to see the generation progress at the point of cancellation.
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  <label htmlFor="resolution_step" className="text-sm font-medium text-gray-300">
                    Resolution slider step size
                  </label>
                  <div className="flex items-center space-x-4">
                    <input
                      type="number"
                      id="resolution_step"
                      value={resolutionStep}
                      onChange={(e) => {
                        let value = parseInt(e.target.value);
                        // Ensure it's a multiple of 8
                        if (value < 8) value = 8;
                        if (value % 8 !== 0) {
                          value = Math.round(value / 8) * 8;
                        }
                        setResolutionStep(value);
                        localStorage.setItem('resolution_step', value.toString());
                      }}
                      min="8"
                      step="8"
                      className="w-24 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:ring-blue-500 focus:border-blue-500"
                    />
                    <span className="text-sm text-gray-400">pixels (must be multiple of 8)</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Controls the step size for width and height sliders in generation panels. Default is 64.
                  </p>
                </div>

                <div className="flex items-start space-x-3">
                  <input
                    type="checkbox"
                    id="include_metadata_in_downloads"
                    checked={includeMetadataInDownloads}
                    onChange={(e) => {
                      const newValue = e.target.checked;
                      setIncludeMetadataInDownloads(newValue);
                      localStorage.setItem('include_metadata_in_downloads', newValue.toString());
                    }}
                    className="mt-1 w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  />
                  <div>
                    <label htmlFor="include_metadata_in_downloads" className="text-sm font-medium text-gray-300 cursor-pointer">
                      Include metadata in manual downloads
                    </label>
                    <p className="text-xs text-gray-500 mt-1">
                      When enabled, images downloaded using the download button will include generation metadata (prompt, parameters, etc.). Note: Images automatically saved to the output folder always include metadata regardless of this setting.
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <input
                    type="checkbox"
                    id="developer_mode"
                    checked={developerMode}
                    onChange={(e) => {
                      const newValue = e.target.checked;
                      setDeveloperMode(newValue);
                      localStorage.setItem('developer_mode', newValue.toString());
                    }}
                    className="mt-1 w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  />
                  <div>
                    <label htmlFor="developer_mode" className="text-sm font-medium text-gray-300 cursor-pointer">
                      Developer Mode
                    </label>
                    <p className="text-xs text-gray-500 mt-1">
                      Enable developer features including CFG metrics visualization during generation. Shows noise prediction magnitudes, guidance strength, and other diagnostic information below the preview panel.
                    </p>
                  </div>
                </div>

              </div>
            </div>
          </Card>

          <Card title="Send Settings">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Configure default size mode when sending images between panels (txt2img → img2img/inpaint, etc.)
              </p>

              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">
                    Default Size Mode
                  </label>
                  <div className="flex gap-4">
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="radio"
                        name="send_size_mode"
                        value="absolute"
                        checked={sendSizeMode === "absolute"}
                        onChange={(e) => {
                          setSendSizeMode("absolute");
                          localStorage.setItem('send_size_mode', "absolute");
                        }}
                        className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-300">Absolute</span>
                    </label>
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="radio"
                        name="send_size_mode"
                        value="scale"
                        checked={sendSizeMode === "scale"}
                        onChange={(e) => {
                          setSendSizeMode("scale");
                          localStorage.setItem('send_size_mode', "scale");
                        }}
                        className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-300">Scale</span>
                    </label>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    When set to "Scale", receiving panels will use scale mode with the default scale value below. When "Absolute", the exact pixel dimensions are used.
                  </p>
                </div>

                <div className="space-y-2">
                  <label htmlFor="send_default_scale" className="text-sm font-medium text-gray-300">
                    Default Scale Value (for Scale mode)
                  </label>
                  <div className="flex items-center space-x-4">
                    <input
                      type="number"
                      id="send_default_scale"
                      value={sendDefaultScale}
                      onChange={(e) => {
                        const value = parseFloat(e.target.value);
                        if (!isNaN(value) && value > 0) {
                          setSendDefaultScale(value);
                          localStorage.setItem('send_default_scale', value.toString());
                        }
                      }}
                      min="0.1"
                      max="4.0"
                      step="0.1"
                      className="w-24 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:ring-blue-500 focus:border-blue-500"
                    />
                    <span className="text-sm text-gray-400">×</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Default scale multiplier when receiving images in scale mode. Default is 1.0 (same size as source).
                  </p>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Resolution Presets">
            <div className="space-y-6">
              <p className="text-gray-400 text-sm mb-4">
                Customize aspect ratio and fixed resolution presets shown in generation panels.
              </p>

              {/* Aspect Ratio Presets */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-200">Aspect Ratio Presets</h3>
                  <div className="flex gap-2">
                    <Button onClick={handleAddAspectRatioPreset} size="sm" variant="secondary">
                      Add
                    </Button>
                    <Button onClick={handleRestoreAspectRatioDefaults} size="sm" variant="secondary">
                      Restore Defaults
                    </Button>
                  </div>
                </div>
                <div className="grid grid-cols-5 gap-2">
                  {aspectRatioPresets.map((preset, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded border border-gray-700"
                    >
                      <span className="text-sm text-gray-300 flex-1">{preset.label}</span>
                      <button
                        onClick={() => handleRemoveAspectRatioPreset(index)}
                        className="text-red-400 hover:text-red-300 text-xs"
                        title="Remove"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Fixed Resolution Presets */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-200">Fixed Resolution Presets</h3>
                  <div className="flex gap-2">
                    <Button onClick={handleAddFixedResolutionPreset} size="sm" variant="secondary">
                      Add
                    </Button>
                    <Button onClick={handleRestoreFixedResolutionDefaults} size="sm" variant="secondary">
                      Restore Defaults
                    </Button>
                  </div>
                </div>
                <div className="grid grid-cols-6 gap-2">
                  {fixedResolutionPresets.map((preset, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-2 px-2 py-2 bg-gray-800 rounded border border-gray-700"
                    >
                      <span className="text-xs text-gray-300 flex-1">
                        {preset.width}×{preset.height}
                      </span>
                      <button
                        onClick={() => handleRemoveFixedResolutionPreset(index)}
                        className="text-red-400 hover:text-red-300 text-xs"
                        title="Remove"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>

          <Card title="Panel Visibility Settings">
            <div className="space-y-6">
              <p className="text-gray-400 text-sm mb-4">
                Control which features are visible in each generation panel.
              </p>

              {/* Txt2Img Panel */}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-gray-200">Text to Image Panel</h3>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={txt2imgVisibility.lora}
                      onChange={(e) => {
                        const newVisibility = { ...txt2imgVisibility, lora: e.target.checked };
                        setTxt2imgVisibility(newVisibility);
                        localStorage.setItem('txt2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show LoRA</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={txt2imgVisibility.controlnet}
                      onChange={(e) => {
                        const newVisibility = { ...txt2imgVisibility, controlnet: e.target.checked };
                        setTxt2imgVisibility(newVisibility);
                        localStorage.setItem('txt2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show ControlNet</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={txt2imgVisibility.aspectRatioPresets}
                      onChange={(e) => {
                        const newVisibility = { ...txt2imgVisibility, aspectRatioPresets: e.target.checked };
                        setTxt2imgVisibility(newVisibility);
                        localStorage.setItem('txt2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show Aspect Ratio Presets</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={txt2imgVisibility.fixedResolutionPresets}
                      onChange={(e) => {
                        const newVisibility = { ...txt2imgVisibility, fixedResolutionPresets: e.target.checked };
                        setTxt2imgVisibility(newVisibility);
                        localStorage.setItem('txt2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show Fixed Resolution Presets</span>
                  </label>
                </div>
              </div>

              {/* Img2Img Panel */}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-gray-200">Image to Image Panel</h3>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={img2imgVisibility.lora}
                      onChange={(e) => {
                        const newVisibility = { ...img2imgVisibility, lora: e.target.checked };
                        setImg2imgVisibility(newVisibility);
                        localStorage.setItem('img2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show LoRA</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={img2imgVisibility.controlnet}
                      onChange={(e) => {
                        const newVisibility = { ...img2imgVisibility, controlnet: e.target.checked };
                        setImg2imgVisibility(newVisibility);
                        localStorage.setItem('img2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show ControlNet</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={img2imgVisibility.aspectRatioPresets}
                      onChange={(e) => {
                        const newVisibility = { ...img2imgVisibility, aspectRatioPresets: e.target.checked };
                        setImg2imgVisibility(newVisibility);
                        localStorage.setItem('img2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show Aspect Ratio Presets</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={img2imgVisibility.fixedResolutionPresets}
                      onChange={(e) => {
                        const newVisibility = { ...img2imgVisibility, fixedResolutionPresets: e.target.checked };
                        setImg2imgVisibility(newVisibility);
                        localStorage.setItem('img2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show Fixed Resolution Presets</span>
                  </label>
                </div>
              </div>

              {/* Inpaint Panel */}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-gray-200">Inpaint Panel</h3>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={inpaintVisibility.lora}
                      onChange={(e) => {
                        const newVisibility = { ...inpaintVisibility, lora: e.target.checked };
                        setInpaintVisibility(newVisibility);
                        localStorage.setItem('inpaint_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show LoRA</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={inpaintVisibility.controlnet}
                      onChange={(e) => {
                        const newVisibility = { ...inpaintVisibility, controlnet: e.target.checked };
                        setInpaintVisibility(newVisibility);
                        localStorage.setItem('inpaint_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show ControlNet</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={inpaintVisibility.aspectRatioPresets}
                      onChange={(e) => {
                        const newVisibility = { ...inpaintVisibility, aspectRatioPresets: e.target.checked };
                        setInpaintVisibility(newVisibility);
                        localStorage.setItem('inpaint_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show Aspect Ratio Presets</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={inpaintVisibility.fixedResolutionPresets}
                      onChange={(e) => {
                        const newVisibility = { ...inpaintVisibility, fixedResolutionPresets: e.target.checked };
                        setInpaintVisibility(newVisibility);
                        localStorage.setItem('inpaint_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show Fixed Resolution Presets</span>
                  </label>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Other Settings">
            <p className="text-gray-400">Additional settings will be implemented in future updates.</p>
          </Card>
        </div>
      </main>
    </div>
    </ProtectedRoute>
  );
}
