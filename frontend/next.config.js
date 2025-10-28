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
    ]
  },
  // Increase timeout for proxied requests (experimental)
  experimental: {
    proxyTimeout: 600000, // 10 minutes in milliseconds
  },
  // Increase server response timeout
  serverRuntimeConfig: {
    timeout: 600000, // 10 minutes
  },
}

module.exports = nextConfig
