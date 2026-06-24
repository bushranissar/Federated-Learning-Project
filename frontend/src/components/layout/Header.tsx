"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Leaf, Activity, Upload, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Dashboard", href: "/", icon: BarChart3 },
  { name: "Predict", href: "/predict", icon: Upload },
];

export default function Header() {
  const pathname = usePathname();

  return (
    <header className="bg-gradient-navy text-white sticky top-0 z-50 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3 group">
            <div className="bg-purple-600 p-2 rounded-lg group-hover:bg-purple-500 transition-colors">
              <Leaf className="h-6 w-6" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-lg font-bold tracking-tight">
                Tomato Disease Detection
              </h1>
              <p className="text-xs text-purple-200">
                Federated Learning CNN-SVM Ensemble
              </p>
            </div>
          </Link>

          {/* Navigation */}
          <nav className="flex items-center gap-1">
            {navigation.map((item) => {
              const isActive = pathname === item.href;
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                    isActive
                      ? "bg-purple-600 text-white shadow-lg"
                      : "text-purple-200 hover:text-white hover:bg-white/10"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{item.name}</span>
                </Link>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Status Bar */}
      <div className="border-t border-white/10 bg-white/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-1.5">
          <div className="flex items-center justify-between text-xs text-purple-300">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <Activity className="h-3 w-3" />
                Federated Learning Active
              </span>
              <span className="hidden sm:inline">|</span>
              <span className="hidden sm:inline">FedAvg Aggregation</span>
              <span className="hidden sm:inline">|</span>
              <span className="hidden sm:inline">2 Clients Active</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span>System Online</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}