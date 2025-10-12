import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SD WebUI",
  description: "Stable Diffusion Web Interface",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja" className="dark">
      <body className="bg-gray-950 text-gray-100">{children}</body>
    </html>
  );
}
