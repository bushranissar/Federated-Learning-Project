"use client";

import { Activity, Target, PieChart, TrendingUp } from "lucide-react";
import { formatPercentage } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { MetricsResponse } from "@/types";

interface MetricsCardsProps {
  metrics: MetricsResponse | null;
  loading: boolean;
}

export default function MetricsCards({ metrics, loading }: MetricsCardsProps) {
  const cards = [
    {
      label: "Accuracy",
      value: metrics?.accuracy ?? 0,
      icon: Activity,
      color: "text-emerald-600",
      bg: "bg-emerald-50",
    },
    {
      label: "Precision",
      value: metrics?.precision ?? 0,
      icon: Target,
      color: "text-blue-600",
      bg: "bg-blue-50",
    },
    {
      label: "Recall",
      value: metrics?.recall ?? 0,
      icon: PieChart,
      color: "text-violet-600",
      bg: "bg-violet-50",
    },
    {
      label: "F1 Score",
      value: metrics?.f1_score ?? 0,
      icon: TrendingUp,
      color: "text-purple-600",
      bg: "bg-purple-50",
    },
  ];

  if (loading) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="metric-card animate-pulse">
            <div className="h-4 bg-slate-200 rounded w-20 mb-3" />
            <div className="h-8 bg-slate-200 rounded w-24" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card) => {
        const Icon = card.icon;
        return (
          <div key={card.label} className="metric-card">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                {card.label}
              </span>
              <div className={cn("p-1.5 rounded-lg", card.bg)}>
                <Icon className={cn("h-4 w-4", card.color)} />
              </div>
            </div>
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-bold text-slate-800">
                {(card.value * 100).toFixed(1)}
              </span>
              <span className="text-sm text-slate-400">%</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}