/** @type {import('next').NextConfig} */
const fs = require('fs');
const path = require('path');

// Read backend port from .port_info file
function getBackendPort() {
  const portInfoPath = path.join(__dirname, '..', 'backend', '.port_info');
  const defaultPort = 8000;

  try {
    if (fs.existsSync(portInfoPath)) {
      const portInfo = JSON.parse(fs.readFileSync(portInfoPath, 'utf8'));
      console.log(`[Next.js] Using backend port from .port_info: ${portInfo.port}`);
      return portInfo.port;
    }
  } catch (error) {
    console.log(`[Next.js] Could not read .port_info, using default port ${defaultPort}:`, error.message);
  }

  console.log(`[Next.js] Using default backend port: ${defaultPort}`);
  return defaultPort;
}

const backendPort = getBackendPort();
const backendUrl = `http://localhost:${backendPort}`;

// Suppress the ECONNREFUSED noise the proxy emits while the backend is still
// starting up.  ONLY that: the previous version dropped every line starting
// with 'Failed to proxy', which is also how Next reports a proxy ABORT -- and
// an abort is exactly what turns a still-running generation into a synthetic
// "500 Internal Server Error" in the browser with no trace on either side.
// That silence cost a full debugging session; keep real proxy failures visible.
const originalConsoleError = console.error;
const isConnectionRefused = (arg) =>
  arg?.code === 'ECONNREFUSED' || arg?.toString?.()?.includes?.('ECONNREFUSED');
console.error = (...args) => {
  if (args.some(isConnectionRefused)) {
    return;
  }
  // A proxy abort is reported by Next as `console.error('Failed to proxy <url>', err)`
  // (node_modules/next/dist/server/lib/router-utils/proxy-request.js), and it is
  // immediately followed by a synthesized `500 Internal Server Error` to the
  // browser even though the backend is still running the generation.  The bare
  // message does not say WHEN it happened or WHY, which is exactly the pair of
  // facts needed to tell one severed-socket cause from another (an idle-timeout
  // abort surfaces as ECONNRESET/ETIMEDOUT at a fixed wall-clock offset; a peer
  // close surfaces as ECONNRESET immediately).  Stamp both onto the line.
  const first = args[0];
  if (typeof first === 'string' && first.startsWith('Failed to proxy')) {
    const err = args.find((a) => a && typeof a === 'object' && ('code' in a || 'message' in a));
    originalConsoleError.call(
      console,
      `[proxy-abort ${new Date().toISOString()}] ${first}` +
        ` code=${err?.code ?? 'n/a'} syscall=${err?.syscall ?? 'n/a'} message=${err?.message ?? 'n/a'}`,
      ...args.slice(1)
    );
    return;
  }
  originalConsoleError.apply(console, args);
};

const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
      {
        source: '/ws/:path*',
        destination: `${backendUrl}/ws/:path*`,
      },
      {
        source: '/outputs/:path*',
        destination: `${backendUrl}/outputs/:path*`,
      },
      {
        source: '/thumbnails/:path*',
        destination: `${backendUrl}/thumbnails/:path*`,
      },
      {
        source: '/training/:path*',
        destination: `${backendUrl}/training/:path*`,
      },
    ]
  },
  // Inactivity timeout on the proxied socket, NOT a total request budget.
  //
  // Every generation endpoint answers on the POST that started it, and the
  // backend sends nothing at all until the image/video is finished -- so this
  // value is really "the longest single generation the UI can survive".  At the
  // old 10 minutes a 7-step MiniMax-H3 video (~150 s/step) blew past it mid-run:
  // http-proxy called socket.abort(), Next answered the browser with a
  // synthesized 500, and the backend -- which never saw an error -- ran to
  // completion and saved the video.  The UI reported a failure for a generation
  // that had succeeded.
  //
  // Unlimited would be `null`, but Next 14's config schema types this as
  // `z.number().gte(0)`, so a day is the practical stand-in.  Do not lower it
  // below the slowest supported model's wall-clock generation time.
  experimental: {
    proxyTimeout: 86400000, // 24 hours in milliseconds
  },
  // Increase server response timeout
  serverRuntimeConfig: {
    timeout: 86400000, // 24 hours
  },
}

module.exports = nextConfig
