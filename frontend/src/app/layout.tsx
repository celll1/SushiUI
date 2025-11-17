import type { Metadata } from "next";
import "./globals.css";
import { StartupProvider } from "@/contexts/StartupContext";
import { GenerationQueueProvider } from "@/contexts/GenerationQueueContext";
import { AuthProvider } from "@/contexts/AuthContext";

export const metadata: Metadata = {
  title: "SushiUI",
  description: "Stable Diffusion Web Interface - SushiUI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja" className="dark">
      <body className="bg-gray-950 text-gray-100">
        <AuthProvider>
          <StartupProvider>
            <GenerationQueueProvider>{children}</GenerationQueueProvider>
          </StartupProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
