/**
 * Round floating point numbers to avoid precision issues
 * e.g., 0.6499999999999999 -> 0.65
 */
export function roundFloat(value: number, decimals: number = 2): number {
  return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
}

/**
 * Parse number from string/number and round to avoid floating point precision issues
 */
export function parseAndRoundFloat(value: any, decimals: number = 2): number {
  const num = typeof value === 'string' ? parseFloat(value) : value;
  if (isNaN(num)) return 0;
  return roundFloat(num, decimals);
}

/**
 * Fix floating point precision issues in params object
 * This handles common parameters that use decimals (cfg_scale, denoising_strength, etc.)
 */
export function fixFloatingPointParams<T extends Record<string, any>>(params: T): T {
  const fixed = { ...params };
  
  // List of parameters that are floating point numbers
  const floatParams = [
    'cfg_scale',
    'denoising_strength', 
    'inpaint_fill_strength',
    'inpaint_blur_strength',
    'scale',
  ];
  
  // Round each float parameter to 2 decimal places
  floatParams.forEach(key => {
    if (key in fixed && typeof fixed[key] === 'number') {
      fixed[key] = roundFloat(fixed[key], 2);
    }
  });
  
  return fixed;
}
