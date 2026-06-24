"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { TrendingUp } from "lucide-react";
import type { FederatedStatus } from "@/types";

interface AccuracyVsRoundProps {
  data: FederatedStatus | null;
  loading: boolean;
}

export default function AccuracyVsRound({ data, loading }: AccuracyVsRoundProps) {
  if (loading) {
    return (
      <div className="card p-6">
        <div className="h-4 bg-slate-200 rounded w-40 mb-4 animate-pulse" />
        <div className="h-64 bg-slate-100 rounded animate-pulse" />
      </div>
    );
  }

  const chartData = data?.accuracy_progression?.map((item) => ({
    round: item.round,
    accuracy: parseFloat((item.accuracy * 100).toFixed(1)),
    loss: parseFloat((item.loss * 100).toFixed(1)),
  })) || [];

  return (
    <div className="card p-6">
      <div className="flex items-center gap-2 mb-6">
        <TrendingUp className="h-5 w-5 text-purple-600" />
        <h3 className="text-lg font-semibold text-slate-800">
          Accuracy & Loss vs Communication Round
        </h3>
      </div>

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              dataKey="round"
              stroke="#94a3b8"
              fontSize={12}
              tickLine={false}
              label={{ value: "Round", position: "insideBottom", offset: -5 }}
            />
            <YAxis
              stroke="#94a3b8"
              fontSize={12}
              tickLine={false}
              domain={[0, 100]}
              label={{ value: "Value (%)", angle: -90, position: "insideLeft" }}
            />
            <Tooltip
              contentStyle={{
                background: "white",
                border: "1px solid #e2e8f0",
                borderRadius: "8px",
                boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={false}
              name="Accuracy (%)"
            />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
              name="Loss"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}