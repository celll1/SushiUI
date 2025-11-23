"use client";

import type { Metadata } from "next";
import "./globals.css";
import { StartupProvider } from "@/contexts/StartupContext";
import { GenerationQueueProvider } from "@/contexts/GenerationQueueContext";
import { AuthProvider } from "@/contexts/AuthContext";
import { useEffect } from "react";

function LayoutContent({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    // Apply saved font size on initial load
    if (typeof window !== 'undefined') {
      const savedFontSize = localStorage.getItem('ui_font_size');
      if (savedFontSize) {
        const size = parseInt(savedFontSize);
        if (!isNaN(size) && size >= 50 && size <= 200) {
          document.documentElement.style.setProperty('--ui-font-size', `${size}%`);
        }
      }
    }
  }, []);

  return (
    <AuthProvider>
      <StartupProvider>
        <GenerationQueueProvider>{children}</GenerationQueueProvider>
      </StartupProvider>
    </AuthProvider>
  );
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja" className="dark">
      <body className="bg-gray-950 text-gray-100">
        <LayoutContent>{children}</LayoutContent>
      </body>
    </html>
  );
}
