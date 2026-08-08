"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Clapperboard,
  Database,
  Image,
  ImagePlus,
  Menu,
  Settings,
  Tag,
  X,
  Zap,
} from "lucide-react";

const navigation = [
  { name: "Generate", href: "/generate", icon: ImagePlus },
  { name: "Studio", href: "/studio", icon: Clapperboard },
  { name: "Gallery", href: "/gallery", icon: Image },
  { name: "Dataset", href: "/dataset", icon: Database },
  { name: "Training", href: "/training", icon: Zap },
  { name: "Tagger", href: "/tagger", icon: Tag },
  { name: "Settings", href: "/settings", icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        className="fixed left-3 top-3 z-50 grid h-9 w-9 place-items-center rounded-md border border-gray-700 bg-gray-900/95 text-gray-200 shadow-lg backdrop-blur lg:hidden"
        aria-label="Toggle menu"
      >
        {isMobileMenuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
      </button>

      {isMobileMenuOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 lg:hidden"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex h-screen w-64 flex-col border-r border-gray-800 bg-gray-950 text-gray-100 transition-transform duration-200 ease-in-out lg:static lg:w-[68px] ${
          isMobileMenuOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        }`}
      >
        <div className="mt-12 flex h-14 items-center gap-3 border-b border-gray-800 px-4 lg:mt-0 lg:justify-center lg:px-0">
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 text-sm font-black text-white shadow-[0_0_18px_rgba(124,92,255,0.22)]">
            S
          </span>
          <div className="min-w-0 lg:hidden">
            <h1 className="truncate text-sm font-bold tracking-[0.12em] text-white">SUSHIUI</h1>
            <p className="text-[10px] uppercase tracking-[0.16em] text-gray-500">Creative Suite</p>
          </div>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto px-2 py-2">
          {navigation.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;

            return (
              <Link
                key={item.name}
                href={item.href}
                onClick={() => setIsMobileMenuOpen(false)}
                title={item.name}
                aria-current={isActive ? "page" : undefined}
                className={`group relative flex h-11 items-center gap-3 rounded-md px-3 text-sm font-medium transition-all lg:h-[50px] lg:flex-col lg:justify-center lg:gap-0.5 lg:px-0 ${
                  isActive
                    ? "bg-violet-500/15 text-violet-300 ring-1 ring-inset ring-violet-500/30"
                    : "text-gray-500 hover:bg-gray-900 hover:text-gray-100"
                }`}
              >
                {isActive && <span className="absolute inset-y-2 left-0 w-0.5 rounded-r bg-violet-400" />}
                <Icon className="h-[18px] w-[18px] shrink-0" />
                <span className="lg:max-w-[58px] lg:truncate lg:text-[9px] lg:font-medium">{item.name}</span>
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-gray-800 p-3 lg:px-1 lg:text-center">
          <p className="text-[10px] text-gray-600">v0.1.0</p>
        </div>
      </aside>
    </>
  );
}
