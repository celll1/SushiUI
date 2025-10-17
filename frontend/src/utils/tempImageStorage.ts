/**
 * Temporary image storage utility
 * Manages image storage using backend temp directory instead of localStorage
 */

import { uploadTempImage, getTempImage, deleteTempImage } from "./api";

const TEMP_IMAGE_PREFIX = "temp_img://";

/**
 * Save an image to temp storage and return a reference string
 * @param imageBase64 - Base64 encoded image with data URL prefix
 * @returns Reference string to be stored in localStorage (e.g., "temp_img://123456_abc.png")
 */
export async function saveTempImage(imageBase64: string): Promise<string> {
  try {
    const imageId = await uploadTempImage(imageBase64);
    return `${TEMP_IMAGE_PREFIX}${imageId}`;
  } catch (error) {
    console.error("Failed to save temp image:", error);
    // Fallback to localStorage for small images
    if (imageBase64.length < 500000) { // ~500KB limit
      return imageBase64;
    }
    throw error;
  }
}

/**
 * Load an image from temp storage
 * @param reference - Reference string (either "temp_img://..." or direct base64)
 * @returns Base64 encoded image with data URL prefix
 */
export async function loadTempImage(reference: string): Promise<string> {
  if (!reference) {
    return "";
  }

  // If it's a temp reference, load from backend
  if (reference.startsWith(TEMP_IMAGE_PREFIX)) {
    const imageId = reference.substring(TEMP_IMAGE_PREFIX.length);
    try {
      return await getTempImage(imageId);
    } catch (error) {
      console.error("Failed to load temp image:", error);
      return "";
    }
  }

  // Otherwise, it's already a base64 string
  return reference;
}

/**
 * Delete a temp image
 * @param reference - Reference string
 */
export async function deleteTempImageRef(reference: string): Promise<void> {
  if (!reference || !reference.startsWith(TEMP_IMAGE_PREFIX)) {
    return;
  }

  const imageId = reference.substring(TEMP_IMAGE_PREFIX.length);
  try {
    await deleteTempImage(imageId);
  } catch (error) {
    console.error("Failed to delete temp image:", error);
  }
}

/**
 * Check if a reference is a temp image reference
 */
export function isTempImageRef(reference: string): boolean {
  return reference && reference.startsWith(TEMP_IMAGE_PREFIX);
}
