"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Image, ImagePlus, Settings, FileImage, Terminal } from "lucide-react";

const navigation = [
  { name: "Generate", href: "/generate", icon: ImagePlus },
  { name: "Gallery", href: "/gallery", icon: Image },
  { name: "Console", href: "/console", icon: Terminal },
  { name: "Settings", href: "/settings", icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="flex h-screen w-59 flex-col bg-gray-900 text-gray-100">
      <div className="flex items-center space-x-2 px-4 py-6">
        <span className="text-4xl">🍣</span>
        <h1 className="text-xl font-bold">SushiUI</h1>
      </div>

      <nav className="flex-1 space-y-1 px-2">
        {navigation.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.name}
              href={item.href}
              className={`flex items-center space-x-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-gray-800 text-white"
                  : "text-gray-400 hover:bg-gray-800 hover:text-white"
              }`}
            >
              <Icon className="h-5 w-5" />
              <span>{item.name}</span>
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-gray-800 p-4">
        <p className="text-xs text-gray-500">v0.1.0</p>
      </div>
    </div>
  );
}
