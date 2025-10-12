/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      {
        source: '/ws/:path*',
        destination: 'http://localhost:8000/ws/:path*',
      },
      {
        source: '/outputs/:path*',
        destination: 'http://localhost:8000/outputs/:path*',
      },
      {
        source: '/thumbnails/:path*',
        destination: 'http://localhost:8000/thumbnails/:path*',
      },
    ]
  },
}

module.exports = nextConfig
