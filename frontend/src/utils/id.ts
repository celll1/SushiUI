// `crypto.randomUUID()` only exists in secure contexts (HTTPS or localhost).
// Accessing it over plain http:// on a LAN address (e.g. a phone reaching
// this dev server at http://192.168.x.x:3000) throws `TypeError:
// crypto.randomUUID is not a function` and crashes the caller synchronously
// -- including inside a `useState` initializer, which crashes the whole
// component tree before anything renders. Route every random ID through
// this helper instead of calling `crypto.randomUUID()` directly.
export function newId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    // RFC 4122 version 4 / variant bits.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  // Last-resort fallback (non-cryptographic): only reachable when the
  // `crypto` global itself is unavailable.
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}
