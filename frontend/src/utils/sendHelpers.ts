import { saveTempImage, deleteTempImageRef } from "./tempImageStorage";
import {
  INPAINT_VIDEO_INPUT_KEY,
  INPAINT_VIDEO_PENDING_KEY,
  OUTPAINT_VIDEO_INPUT_KEY,
  OUTPAINT_VIDEO_PENDING_KEY,
  saveMediaInput,
} from "./mediaInputStorage";
import { roundFloat } from "./numberUtils";

/**
 * Common parameters that can be sent between panels
 */
interface BaseSendParams {
  prompt: string;
  negative_prompt?: string;
  steps?: number;
  cfg_scale?: number;
  sampler?: string;
  schedule_type?: string;
  seed?: number;
  ancestral_seed?: number;
  width?: number;
  height?: number;
  // Advanced CFG parameters
  cfg_schedule_type?: string;
  cfg_schedule_min?: number;
  cfg_schedule_max?: number;
  cfg_schedule_power?: number;
  cfg_rescale_snr_alpha?: number;
  dynamic_threshold_percentile?: number;
  dynamic_threshold_mimic_scale?: number;
  // NAG parameters
  nag_enable?: boolean;
  nag_scale?: number;
  nag_tau?: number;
  nag_alpha?: number;
  nag_sigma_end?: number;
  nag_negative_prompt?: string;
  // Attention processor type
  attention_type?: string;
  attention_impl?: string;
}

/**
 * Extended parameters with denoising strength for img2img/inpaint
 */
interface ExtendedSendParams extends BaseSendParams {
  denoising_strength?: number;
}

/**
 * Sends prompt and/or parameters to target panel's localStorage
 *
 * @param sourceParams - Source parameters to send
 * @param targetStorageKey - Target panel's localStorage key
 * @param options - Options for what to send
 */
export function sendToPanel(
  sourceParams: ExtendedSendParams,
  targetStorageKey: string,
  options: {
    sendPrompt?: boolean;
    sendParameters?: boolean;
    includeDenoising?: boolean;
    dispatchEvent?: string;
  } = {}
): void {
  const {
    sendPrompt = true,
    sendParameters = true,
    includeDenoising = false,
    dispatchEvent
  } = options;

  console.log("[sendToPanel] targetStorageKey:", targetStorageKey);
  console.log("[sendToPanel] sendPrompt:", sendPrompt, "sendParameters:", sendParameters);
  console.log("[sendToPanel] sourceParams.prompt:", sourceParams.prompt);

  // Load existing params and merge
  const targetParams = JSON.parse(localStorage.getItem(targetStorageKey) || "{}");
  console.log("[sendToPanel] Existing targetParams:", targetParams);

  // Send prompt if requested
  if (sendPrompt) {
    targetParams.prompt = sourceParams.prompt;
    targetParams.negative_prompt = sourceParams.negative_prompt;
    console.log("[sendToPanel] Set prompt to:", targetParams.prompt);
  }

  // Send parameters if requested
  if (sendParameters) {
    targetParams.steps = sourceParams.steps;
    targetParams.cfg_scale = sourceParams.cfg_scale !== undefined ? roundFloat(sourceParams.cfg_scale, 2) : sourceParams.cfg_scale;
    targetParams.sampler = sourceParams.sampler;
    targetParams.schedule_type = sourceParams.schedule_type;
    targetParams.seed = sourceParams.seed;
    targetParams.ancestral_seed = sourceParams.ancestral_seed;
    targetParams.width = sourceParams.width;
    targetParams.height = sourceParams.height;

    // Add Advanced CFG parameters
    if (sourceParams.cfg_schedule_type !== undefined) {
      targetParams.cfg_schedule_type = sourceParams.cfg_schedule_type;
    }
    if (sourceParams.cfg_schedule_min !== undefined) {
      targetParams.cfg_schedule_min = sourceParams.cfg_schedule_min;
    }
    if (sourceParams.cfg_schedule_max !== undefined) {
      targetParams.cfg_schedule_max = sourceParams.cfg_schedule_max;
    }
    if (sourceParams.cfg_schedule_power !== undefined) {
      targetParams.cfg_schedule_power = sourceParams.cfg_schedule_power;
    }
    if (sourceParams.cfg_rescale_snr_alpha !== undefined) {
      targetParams.cfg_rescale_snr_alpha = sourceParams.cfg_rescale_snr_alpha;
    }
    if (sourceParams.dynamic_threshold_percentile !== undefined) {
      targetParams.dynamic_threshold_percentile = sourceParams.dynamic_threshold_percentile;
    }
    if (sourceParams.dynamic_threshold_mimic_scale !== undefined) {
      targetParams.dynamic_threshold_mimic_scale = sourceParams.dynamic_threshold_mimic_scale;
    }

    // Add NAG parameters
    if (sourceParams.nag_enable !== undefined) {
      targetParams.nag_enable = sourceParams.nag_enable;
    }
    if (sourceParams.nag_scale !== undefined) {
      targetParams.nag_scale = sourceParams.nag_scale;
    }
    if (sourceParams.nag_tau !== undefined) {
      targetParams.nag_tau = sourceParams.nag_tau;
    }
    if (sourceParams.nag_alpha !== undefined) {
      targetParams.nag_alpha = sourceParams.nag_alpha;
    }
    if (sourceParams.nag_sigma_end !== undefined) {
      targetParams.nag_sigma_end = sourceParams.nag_sigma_end;
    }
    if (sourceParams.nag_negative_prompt !== undefined) {
      targetParams.nag_negative_prompt = sourceParams.nag_negative_prompt;
    }

    // Add attention processor type
    if (sourceParams.attention_type !== undefined) {
      targetParams.attention_type = sourceParams.attention_type;
    }
    if (sourceParams.attention_impl !== undefined) {
      targetParams.attention_impl = sourceParams.attention_impl;
    }

    if (includeDenoising && sourceParams.denoising_strength !== undefined) {
      targetParams.denoising_strength = roundFloat(sourceParams.denoising_strength, 2);
    }
  }

  // Save merged params once
  if (sendPrompt || sendParameters) {
    console.log("[sendToPanel] Saving merged params:", targetParams);
    localStorage.setItem(targetStorageKey, JSON.stringify(targetParams));

    // Dispatch custom event if specified
    if (dispatchEvent) {
      console.log("[sendToPanel] Dispatching event:", dispatchEvent);
      window.dispatchEvent(new Event(dispatchEvent));
    }
  }
}

/**
 * @deprecated Use sendToPanel instead
 * Sends prompt to target panel's localStorage
 */
export function sendPromptToPanel(
  sourceParams: BaseSendParams,
  targetStorageKey: string
): void {
  sendToPanel(sourceParams, targetStorageKey, {
    sendPrompt: true,
    sendParameters: false
  });
}

/**
 * @deprecated Use sendToPanel instead
 * Sends parameters to target panel's localStorage
 */
export function sendParametersToPanel(
  sourceParams: ExtendedSendParams,
  targetStorageKey: string,
  includeDenoising: boolean = false
): void {
  sendToPanel(sourceParams, targetStorageKey, {
    sendPrompt: false,
    sendParameters: true,
    includeDenoising
  });
}

/**
 * Sends image to img2img panel
 */
export async function sendImageToImg2Img(
  imageUrl: string,
  storageKey: string = "img2img_input_image"
): Promise<void> {
  const response = await fetch(imageUrl);
  const blob = await response.blob();

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = async () => {
      try {
        const base64data = reader.result as string;
        const oldRef = localStorage.getItem(storageKey);
        if (oldRef) {
          await deleteTempImageRef(oldRef);
        }
        const ref = await saveTempImage(base64data);
        localStorage.setItem(storageKey, ref);
        window.dispatchEvent(new Event("img2img_input_updated"));
        resolve();
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Sends image to inpaint panel (clears mask)
 */
export async function sendImageToInpaint(
  imageUrl: string,
  inputStorageKey: string = "inpaint_input_image",
  maskStorageKey: string = "inpaint_mask_image"
): Promise<void> {
  const response = await fetch(imageUrl);
  const blob = await response.blob();

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = async () => {
      try {
        const base64data = reader.result as string;
        const ref = await saveTempImage(base64data);
        localStorage.setItem(inputStorageKey, ref);
        localStorage.removeItem(maskStorageKey);
        window.dispatchEvent(new Event("inpaint_input_updated"));
        resolve();
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Sends image to outpaint panel (no mask -- outpaint builds its own canvas +
 * mask server-side from the placement fields).
 */
export async function sendImageToOutpaint(
  imageUrl: string,
  storageKey: string = "outpaint_input_image"
): Promise<void> {
  const response = await fetch(imageUrl);
  const blob = await response.blob();

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = async () => {
      try {
        const base64data = reader.result as string;
        const oldRef = localStorage.getItem(storageKey);
        if (oldRef) {
          await deleteTempImageRef(oldRef);
        }
        const ref = await saveTempImage(base64data);
        localStorage.setItem(storageKey, ref);
        window.dispatchEvent(new Event("outpaint_input_updated"));
        resolve();
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Sends base64 image to outpaint panel (no fetching needed)
 */
export async function sendBase64ImageToOutpaint(
  base64Image: string,
  storageKey: string = "outpaint_input_image"
): Promise<void> {
  const tempRef = await saveTempImage(base64Image);
  localStorage.setItem(storageKey, tempRef);
  window.dispatchEvent(new Event("outpaint_input_updated"));
}

/**
 * Sends base64 image to img2img panel (no fetching needed)
 */
export async function sendBase64ImageToImg2Img(
  base64Image: string,
  storageKey: string = "img2img_input_image"
): Promise<void> {
  const tempRef = await saveTempImage(base64Image);
  localStorage.setItem(storageKey, tempRef);
  window.dispatchEvent(new Event("img2img_input_updated"));
}

/**
 * Sends base64 image from txt2img to inpaint (no fetching needed)
 */
export async function sendBase64ImageToInpaint(
  base64Image: string,
  inputStorageKey: string = "inpaint_input_image",
  maskStorageKey: string = "inpaint_mask_image"
): Promise<void> {
  const tempRef = await saveTempImage(base64Image);
  localStorage.setItem(inputStorageKey, tempRef);
  localStorage.removeItem(maskStorageKey);
  window.dispatchEvent(new Event("inpaint_input_updated"));
}

/**
 * Sends image to upscale panel
 */
export async function sendImageToUpscale(
  imageUrl: string,
  storageKey: string = "upscale_input_image"
): Promise<void> {
  const response = await fetch(imageUrl);
  const blob = await response.blob();

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = async () => {
      try {
        const base64data = reader.result as string;
        const oldRef = localStorage.getItem(storageKey);
        if (oldRef) {
          await deleteTempImageRef(oldRef);
        }
        const ref = await saveTempImage(base64data);
        localStorage.setItem(storageKey, ref);
        window.dispatchEvent(new Event("upscale_input_updated"));
        resolve();
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Sends base64 image to upscale panel (no fetching needed)
 */
export async function sendBase64ImageToUpscale(
  base64Image: string,
  storageKey: string = "upscale_input_image"
): Promise<void> {
  const tempRef = await saveTempImage(base64Image);
  localStorage.setItem(storageKey, tempRef);
  window.dispatchEvent(new Event("upscale_input_updated"));
}

/**
 * Sends image to img2vid panel (keyframe). P3b gallery frame-grab targets this.
 */
export async function sendImageToImg2Vid(
  imageUrl: string,
  storageKey: string = "img2vid_input_image"
): Promise<void> {
  const response = await fetch(imageUrl);
  const blob = await response.blob();

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = async () => {
      try {
        const base64data = reader.result as string;
        const oldRef = localStorage.getItem(storageKey);
        if (oldRef) {
          await deleteTempImageRef(oldRef);
        }
        const ref = await saveTempImage(base64data);
        localStorage.setItem(storageKey, ref);
        window.dispatchEvent(new Event("img2vid_input_updated"));
        resolve();
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Sends base64 image to img2vid panel (no fetching needed)
 */
export async function sendBase64ImageToImg2Vid(
  base64Image: string,
  storageKey: string = "img2vid_input_image"
): Promise<void> {
  const tempRef = await saveTempImage(base64Image);
  localStorage.setItem(storageKey, tempRef);
  window.dispatchEvent(new Event("img2vid_input_updated"));
}

/**
 * Fetches a URL (e.g. `/outputs/<filename>` or a blob: URL) and materializes
 * it as a File. Video/audio results are too large for the base64 +
 * localStorage transport used by images, so the video/audio send-to helpers
 * below just hand off the URL string; the receiving panel calls this to get
 * a real File for its videoFile/audioFile/referenceAudioFile state (queue
 * items require a File, not a URL).
 */
export async function fetchUrlToFile(url: string, filename?: string): Promise<File> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch media (${response.status} ${response.statusText})`);
  }
  const blob = await response.blob();
  const name = filename || url.split("/").pop()?.split("?")[0] || "file";
  return new File([blob], name, { type: blob.type });
}

/**
 * Sends a video result to the Outpaint panel's outpaint_vid clip input.
 * The File is materialized before navigation so a missed same-tab event is
 * recovered from IndexedDB when the destination panel mounts.
 */
export async function sendVideoToOutpaint(videoUrl: string): Promise<void> {
  const file = await fetchUrlToFile(videoUrl);
  await saveMediaInput(OUTPAINT_VIDEO_INPUT_KEY, file);
  localStorage.removeItem("outpaint_input_video");
  localStorage.setItem(OUTPAINT_VIDEO_PENDING_KEY, "1");
  window.dispatchEvent(new Event("outpaint_input_video_updated"));
}

/**
 * Sends a video result/clip to the Inpaint panel's temporal inpaint clip
 * input. The File is persisted before the destination panel is notified.
 */
export async function sendVideoToInpaint(videoUrl: string): Promise<void> {
  const file = await fetchUrlToFile(videoUrl);
  await saveMediaInput(INPAINT_VIDEO_INPUT_KEY, file);
  localStorage.removeItem("inpaint_input_video");
  localStorage.setItem(INPAINT_VIDEO_PENDING_KEY, "1");
  window.dispatchEvent(new Event("inpaint_input_video_updated"));
}

/**
 * Sends an audio result to the Outpaint panel's outpaint_aud clip input.
 * Transport is the plain URL (not base64) -- see fetchUrlToFile.
 */
export function sendAudioToOutpaint(audioUrl: string): void {
  localStorage.setItem("outpaint_input_audio", audioUrl);
  window.dispatchEvent(new Event("outpaint_input_audio_updated"));
}

/**
 * Sends an audio result to the Img2Img panel's aud2aud reference clip input.
 * Transport is the plain URL (not base64) -- see fetchUrlToFile.
 */
export function sendAudioToImg2Img(audioUrl: string): void {
  localStorage.setItem("img2img_input_audio", audioUrl);
  window.dispatchEvent(new Event("img2img_input_audio_updated"));
}

/**
 * Sends a video result into the MiniMax-H3 ref2va reference track
 * (`h3References.videos`) -- whole-clip conditioning, not a placement anchor.
 * Both Txt2ImgPanel and Img2ImgPanel host the reference selector and listen
 * for this event; the clip is appended to whichever panel's list is mounted.
 * Transport is the plain URL (not base64) -- see fetchUrlToFile.
 */
export function sendVideoToReference(videoUrl: string): void {
  localStorage.setItem("h3_reference_video", videoUrl);
  window.dispatchEvent(new Event("h3_reference_video_updated"));
}
