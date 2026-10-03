/**
 * Client-side evaluation of the training timestep distribution.
 *
 * Mirrors `backend/core/training/timestep_sampler.py`: the same versioned
 * sampler expressions the trainer publishes in
 * `GET /training/runs/{id}/timestep-distribution`, evaluated here only to DRAW
 * them. The backend remains the only thing that samples.
 *
 * Quantile mode is evaluated through the inverse CDF, which is what makes one
 * code path cover every sampler kind including a morph's interpolated law and a
 * flattened quantile table. Mixture mode has no quantile function, so its
 * density is mixed from the endpoints' instead.
 *
 * Endpoint conventions that must match the backend, not be re-invented here:
 * logit-normal and beta outputs are affine-scaled into
 * `[min_timestep, max_timestep]`, and a normal is conditioned on that range.
 */

export type SamplerConfig = {
  distribution: string;
  min_timestep?: number;
  max_timestep?: number;
  mean?: number;
  std?: number;
  alpha?: number;
  beta?: number;
  custom_weights?: number[];
};

export type SamplerExpr =
  | { version: number; kind: "config"; config: SamplerConfig }
  | {
      version: number;
      kind: "quantile_table";
      table: number[];
      min_timestep: number;
      max_timestep: number;
    }
  | {
      version: number;
      kind: "morph";
      source: SamplerExpr;
      target: SamplerExpr;
      steps: number;
      curve: string;
      interpolation: string;
      start_update: number;
      frozen_lam?: number | null;
    };

/** Inverse standard-normal CDF (Acklam's rational approximation, ~1e-9). */
function ndtri(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
             1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
             6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
             -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
             3.754408661907416];
  const pLow = 0.02425;
  const u = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
  if (u < pLow) {
    const q = Math.sqrt(-2 * Math.log(u));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (u > 1 - pLow) {
    const q = Math.sqrt(-2 * Math.log(1 - u));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = u - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

function normalSurvival(z: number): number {
  const x = Math.abs(z);
  const t = 1 / (1 + 0.2316419 * x);
  const tail = Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * t *
    (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return z >= 0 ? tail : 1 - tail;
}

function normalizeDistribution(name: string): string {
  const key = (name || "uniform").toLowerCase();
  if (key === "lognormal" || key === "logit-normal" || key === "logitnormal") {
    return "logit_normal";
  }
  return key;
}

/** lambda at a successful-optimizer-update position; matches `morph_lambda`. */
export function morphLambda(
  updateStep: number, startUpdate: number, steps: number, curve: string
): number {
  if (!steps || steps <= 0) return 1;
  const p = Math.min(1, Math.max(0, (updateStep - startUpdate) / steps));
  return curve === "linear" ? p : 0.5 * (1 - Math.cos(Math.PI * p));
}

function configIcdf(config: SamplerConfig, u: number): number | null {
  const min = config.min_timestep ?? 0;
  const max = config.max_timestep ?? 1;
  const scale = (t: number) => t * (max - min) + min;
  switch (normalizeDistribution(config.distribution)) {
    case "uniform":
      return scale(u);
    case "logit_normal": {
      const z = ndtri(u) * (config.std ?? 1) + (config.mean ?? 0);
      return scale(1 / (1 + Math.exp(-z)));
    }
    case "normal": {
      const mean = config.mean ?? 0.5;
      const std = config.std ?? 0.2;
      if (!(std > 0) || !(max > min)) return null;
      const lower = (min - mean) / std;
      const upper = (max - mean) / std;
      if (lower >= 0) {
        const start = normalSurvival(lower);
        const span = start - normalSurvival(upper);
        return span > 0 ? mean - std * ndtri(start - u * span) : null;
      }
      const start = normalSurvival(-lower);
      const span = normalSurvival(-upper) - start;
      return span > 0 ? mean + std * ndtri(start + u * span) : null;
    }
    case "custom": {
      const weights = config.custom_weights || [];
      const total = weights.reduce((sum, w) => sum + w, 0);
      if (!weights.length || total <= 0) return null;
      let acc = 0;
      for (let i = 0; i < weights.length; i++) {
        const next = acc + weights[i] / total;
        if (u <= next || i === weights.length - 1) {
          const width = (max - min) / weights.length;
          const frac = next > acc ? (u - acc) / (next - acc) : 0;
          return min + (i + Math.min(1, Math.max(0, frac))) * width;
        }
        acc = next;
      }
      return max;
    }
    default:
      // beta has no closed-form quantile; the backend falls back to mixture
      // whenever an endpoint is one, so this is only ever hit by a caller that
      // should be mixing densities instead.
      return null;
  }
}

/** Quantile function of a published sampler expression, or null if it has none. */
export function samplerIcdf(expr: SamplerExpr, u: number): number | null {
  if (!expr) return null;
  if (expr.kind === "config") return configIcdf(expr.config, u);
  if (expr.kind === "quantile_table") {
    const table = expr.table;
    if (!table || table.length < 2) return null;
    const pos = Math.min(1, Math.max(0, u)) * (table.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(table.length - 1, lo + 1);
    return table[lo] + (pos - lo) * (table[hi] - table[lo]);
  }
  if (expr.kind === "morph") {
    if (expr.interpolation !== "quantile") return null;
    const lam = expr.frozen_lam ?? 0;
    const source = samplerIcdf(expr.source, u);
    const target = samplerIcdf(expr.target, u);
    if (source === null || target === null) return null;
    return (1 - lam) * source + lam * target;
  }
  return null;
}

export type Curve = { x: number; y: number }[];

const GRID = 512;

// Lanczos approximation; sufficient for drawing a distribution, not sampling it.
function logGamma(value: number): number {
  const coefficients = [
    676.5203681218851, -1259.1392167224028, 771.3234287776531,
    -176.6150291621406, 12.507343278686905, -0.13857109526572012,
    9.984369578019572e-6, 1.5056327351493116e-7,
  ];
  if (value < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  }
  const z = value - 1;
  let sum = 0.9999999999998099;
  for (let i = 0; i < coefficients.length; i++) sum += coefficients[i] / (z + i + 1);
  const t = z + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(sum);
}

function betaDensity(config: SamplerConfig, bins: number): Curve | null {
  const min = config.min_timestep ?? 0;
  const max = config.max_timestep ?? 1;
  const alpha = config.alpha ?? 2;
  const beta = config.beta ?? 2;
  const width = max - min;
  if (!(alpha > 0) || !(beta > 0) || !(width > 0)) return null;
  const logNorm = logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
  return Array.from({ length: bins }, (_, i) => {
    const x = (i + 0.5) / bins;
    if (x <= min || x >= max) return { x, y: 0 };
    const unit = (x - min) / width;
    const logPdf = (alpha - 1) * Math.log(unit)
      + (beta - 1) * Math.log1p(-unit) - logNorm - Math.log(width);
    return { x, y: Math.exp(logPdf) };
  });
}

function normalDensity(config: SamplerConfig, bins: number): Curve | null {
  const min = config.min_timestep ?? 0;
  const max = config.max_timestep ?? 1;
  const mean = config.mean ?? 0.5;
  const std = config.std ?? 0.2;
  if (!(std > 0) || !(max > min)) return null;
  const lower = (min - mean) / std;
  const upper = (max - mean) / std;
  const mass = lower >= 0
    ? normalSurvival(lower) - normalSurvival(upper)
    : normalSurvival(-upper) - normalSurvival(-lower);
  if (!(mass > 0)) return null;
  return Array.from({ length: bins }, (_, i) => {
    const x = (i + 0.5) / bins;
    const z = (x - mean) / std;
    return { x, y: x >= min && x <= max
      ? Math.exp(-z * z / 2) / (std * Math.sqrt(2 * Math.PI) * mass)
      : 0 };
  });
}

/**
 * Density of a sampler expression, as a normalised histogram of its midpoint
 * quantiles. Deterministic — no sampling — and works for any expression with a
 * quantile function.
 */
export function densityFromIcdf(
  icdf: (u: number) => number | null, bins = 64
): Curve | null {
  const counts = new Array(bins).fill(0);
  for (let i = 0; i < GRID; i++) {
    const t = icdf((i + 0.5) / GRID);
    if (t === null || !isFinite(t)) return null;
    const bin = Math.min(bins - 1, Math.max(0, Math.floor(t * bins)));
    counts[bin] += 1;
  }
  return counts.map((count, i) => ({
    x: (i + 0.5) / bins,
    y: (count / GRID) * bins,
  }));
}

/** Density of one endpoint expression, for both interpolation modes. */
export function densityOf(expr: SamplerExpr, bins = 64): Curve | null {
  if (expr.kind === "config" && normalizeDistribution(expr.config.distribution) === "normal") {
    return normalDensity(expr.config, bins);
  }
  if (expr.kind === "config" && normalizeDistribution(expr.config.distribution) === "beta") {
    return betaDensity(expr.config, bins);
  }
  if (expr.kind === "morph" && expr.interpolation === "mixture") {
    const source = densityOf(expr.source, bins);
    const target = densityOf(expr.target, bins);
    if (!source || !target) return null;
    const lam = Math.min(1, Math.max(0, expr.frozen_lam ?? 0));
    return source.map((point, i) => ({
      x: point.x,
      y: (1 - lam) * point.y + lam * target[i].y,
    }));
  }
  return densityFromIcdf((u) => samplerIcdf(expr, u), bins);
}

/** The law actually in force at lambda, under either interpolation. */
export function morphDensity(
  source: SamplerExpr, target: SamplerExpr, lam: number,
  interpolation: string, bins = 64
): Curve | null {
  if (interpolation === "quantile") {
    return densityFromIcdf((u) => {
      const a = samplerIcdf(source, u);
      const b = samplerIcdf(target, u);
      if (a === null || b === null) return null;
      return (1 - lam) * a + lam * b;
    }, bins);
  }
  const a = densityOf(source, bins);
  const b = densityOf(target, bins);
  if (!a || !b) return null;
  return a.map((point, i) => ({
    x: point.x,
    y: (1 - lam) * point.y + lam * b[i].y,
  }));
}
