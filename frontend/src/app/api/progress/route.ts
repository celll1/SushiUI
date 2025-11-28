import { NextRequest } from 'next/server';
import WebSocket from 'ws';

// This is a Next.js API Route that converts WebSocket messages to Server-Sent Events (SSE)
// This allows the frontend to receive real-time progress updates via HTTP, which works
// through Next.js proxy even when accessed from external networks

export async function GET(request: NextRequest) {
  // Create a readable stream for SSE
  const encoder = new TextEncoder();

  let ws: WebSocket | null = null;

  const stream = new ReadableStream({
    start(controller) {
      // Get backend port from environment or use default
      const backendPort = process.env.BACKEND_PORT || '8000';
      const backendHost = process.env.BACKEND_HOST || 'localhost';
      const wsUrl = `ws://${backendHost}:${backendPort}/api/v1/ws/progress`;

      console.log(`[SSE] Connecting to backend WebSocket: ${wsUrl}`);

      // Connect to backend WebSocket
      ws = new WebSocket(wsUrl);

      ws.on('open', () => {
        console.log('[SSE] Connected to backend WebSocket');
        // Send initial connection message
        const data = `data: ${JSON.stringify({ type: 'connected' })}\n\n`;
        controller.enqueue(encoder.encode(data));
      });

      ws.on('message', (message: Buffer) => {
        try {
          const messageStr = message.toString();

          // Log message without base64 data to avoid console spam
          try {
            const parsed = JSON.parse(messageStr);
            if (parsed.preview_image) {
              console.log('[SSE] Received from backend: progress with preview_image');
            } else {
              console.log('[SSE] Received from backend:', messageStr.substring(0, 200));
            }
          } catch {
            console.log('[SSE] Received from backend:', messageStr.substring(0, 200));
          }

          // Forward WebSocket message as SSE event
          const data = `data: ${messageStr}\n\n`;
          controller.enqueue(encoder.encode(data));
        } catch (error) {
          console.error('[SSE] Error processing message:', error);
        }
      });

      ws.on('error', (error) => {
        console.error('[SSE] WebSocket error:', error);
        const data = `data: ${JSON.stringify({ type: 'error', message: error.message })}\n\n`;
        controller.enqueue(encoder.encode(data));
      });

      ws.on('close', (code, reason) => {
        console.log(`[SSE] WebSocket closed (code: ${code}, reason: ${reason.toString()})`);
        const data = `data: ${JSON.stringify({ type: 'closed', code, reason: reason.toString() })}\n\n`;
        controller.enqueue(encoder.encode(data));
        controller.close();
      });
    },

    cancel() {
      console.log('[SSE] Client disconnected, closing WebSocket');
      if (ws) {
        ws.close();
        ws = null;
      }
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}
