import { saveTempImage, deleteTempImageRef } from "./tempImageStorage";
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
}

/**
 * Extended parameters with denoising strength for img2img/inpaint
 */
interface ExtendedSendParams extends BaseSendParams {
  denoising_strength?: number;
}

/**
 * Sends prompt to target panel's localStorage
 */
export function sendPromptToPanel(
  sourceParams: BaseSendParams,
  targetStorageKey: string
): void {
  const targetParams = JSON.parse(localStorage.getItem(targetStorageKey) || "{}");
  targetParams.prompt = sourceParams.prompt;
  targetParams.negative_prompt = sourceParams.negative_prompt;
  localStorage.setItem(targetStorageKey, JSON.stringify(targetParams));
}

/**
 * Sends parameters to target panel's localStorage
 */
export function sendParametersToPanel(
  sourceParams: ExtendedSendParams,
  targetStorageKey: string,
  includeDenoising: boolean = false
): void {
  const targetParams = JSON.parse(localStorage.getItem(targetStorageKey) || "{}");
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

  if (includeDenoising && sourceParams.denoising_strength !== undefined) {
    targetParams.denoising_strength = roundFloat(sourceParams.denoising_strength, 2);
  }

  localStorage.setItem(targetStorageKey, JSON.stringify(targetParams));
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
