"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { AlertCircle, CheckCircle2, FlaskConical, Brain, GitBranch, Info } from "lucide-react";
import type { PredictionResult as PredictionResultType } from "@/types";
import { cn } from "@/lib/utils";

interface PredictionResultProps {
  result: PredictionResultType | null;
  loading: boolean;
  error: string | null;
}

const COLORS = ["#8b5cf6", "#ef4444", "#f59e0b", "#10b981", "#3b82f6"];
const SEVERITY_COLORS: Record<string, string> = {
  "None": "bg-green-100 text-green-800",
  "Moderate": "bg-yellow-100 text-yellow-800",
  "Moderate to High": "bg-orange-100 text-orange-800",
  "High": "bg-red-100 text-red-800",
  "Very High": "bg-red-100 text-red-800",
};

export default function PredictionResultComponent({ result, loading, error }: PredictionResultProps) {
  if (loading) {
    return (
      <div className="card p-6 space-y-4">
        <div className="h-6 bg-slate-200 rounded w-48 animate-pulse" />
        <div className="h-12 bg-slate-200 rounded w-64 animate-pulse" />
        <div className="h-48 bg-slate-100 rounded animate-pulse" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-6">
        <div className="flex flex-col items-center gap-3 text-center">
          <AlertCircle className="h-12 w-12 text-red-400" />
          <h3 className="font-semibold text-slate-800">Prediction Failed</h3>
          <p className="text-sm text-slate-500">{error}</p>
        </div>
      </div>
    );
  }

  if (!result) return null;

  const top5Data = result.top5.map((item) => ({
    name: item.class.length > 15 ? item.class.substring(0, 15) + "..." : item.class,
    fullName: item.class,
    probability: parseFloat((item.probability * 100).toFixed(1)),
  }));

  const isDemo = result.demo_mode;
  const diseaseInfo = result.disease_info;

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Demo Mode Warning */}
      {isDemo && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-amber-500 flex-shrink-0" />
          <p className="text-sm text-amber-700">
            Running in Demo Mode — predictions are simulated for demonstration purposes.
          </p>
        </div>
      )}

      {/* Prediction Result */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-green-500" />
            <h3 className="text-lg font-semibold text-slate-800">Prediction Result</h3>
          </div>
          <span className={cn(
            "px-3 py-1 rounded-full text-xs font-medium",
            isDemo ? "bg-amber-100 text-amber-700" : "bg-green-100 text-green-700"
          )}>
            {isDemo ? "Demo" : "Trained Model"}
          </span>
        </div>

        <div className="flex items-center gap-4 mb-6">
          <div className="bg-gradient-purple p-4 rounded-2xl">
            <Brain className="h-8 w-8 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-slate-800">{result.prediction}</h2>
            <p className="text-sm text-slate-500">
              Confidence: <span className="font-semibold text-purple-600">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </p>
          </div>
        </div>

        {/* Model Predictions Comparison */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
          <div className="bg-purple-50 rounded-xl p-3 border border-purple-100">
            <div className="flex items-center gap-1.5 mb-1">
              <Brain className="h-3.5 w-3.5 text-purple-600" />
              <span className="text-xs font-medium text-purple-700">CNN</span>
            </div>
            <p className="text-sm font-semibold text-slate-800">{result.cnn_prediction}</p>
            <p className="text-xs text-slate-500">{(result.cnn_confidence * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-violet-50 rounded-xl p-3 border border-violet-100">
            <div className="flex items-center gap-1.5 mb-1">
              <FlaskConical className="h-3.5 w-3.5 text-violet-600" />
              <span className="text-xs font-medium text-violet-700">SVM</span>
            </div>
            <p className="text-sm font-semibold text-slate-800">{result.svm_prediction}</p>
            <p className="text-xs text-slate-500">{(result.svm_confidence * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-indigo-50 rounded-xl p-3 border border-indigo-100">
            <div className="flex items-center gap-1.5 mb-1">
              <GitBranch className="h-3.5 w-3.5 text-indigo-600" />
              <span className="text-xs font-medium text-indigo-700">Ensemble</span>
            </div>
            <p className="text-sm font-semibold text-slate-800">{result.ensemble_prediction}</p>
            <p className="text-xs text-slate-500">{(result.ensemble_confidence * 100).toFixed(1)}%</p>
          </div>
        </div>

        {/* Probability Chart */}
        <div>
          <h4 className="text-sm font-semibold text-slate-700 mb-3">Top-5 Class Probabilities</h4>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={top5Data} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis type="number" domain={[0, 100]} unit="%" stroke="#94a3b8" fontSize={12} />
                <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={11} width={120} />
                <Tooltip
                  formatter={(value: number) => [`${value.toFixed(1)}%`, "Probability"]}
                  labelFormatter={(label, payload) => payload?.[0]?.payload?.fullName || label}
                  contentStyle={{
                    background: "white",
                    border: "1px solid #e2e8f0",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="probability" radius={[0, 4, 4, 0]} barSize={20}>
                  {top5Data.map((_, index) => (
                    <Cell key={index} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Disease Info */}
      {diseaseInfo && (
        <div className="card p-6 animate-fade-in">
          <div className="flex items-center gap-2 mb-4">
            <Info className="h-5 w-5 text-purple-600" />
            <h3 className="text-lg font-semibold text-slate-800">Disease Information</h3>
          </div>

          <div className="space-y-4">
            <p className="text-sm text-slate-600 leading-relaxed">
              {diseaseInfo.description}
            </p>

            <div>
              <span className={cn(
                "inline-block px-3 py-1 rounded-full text-xs font-medium",
                SEVERITY_COLORS[diseaseInfo.severity] || "bg-slate-100 text-slate-700"
              )}>
                Severity: {diseaseInfo.severity}
              </span>
            </div>

            <div>
              <h4 className="text-sm font-semibold text-slate-700 mb-2">Symptoms</h4>
              <ul className="list-disc list-inside space-y-1">
                {diseaseInfo.symptoms.map((symptom, i) => (
                  <li key={i} className="text-sm text-slate-600">{symptom}</li>
                ))}
              </ul>
            </div>

            <div>
              <h4 className="text-sm font-semibold text-slate-700 mb-2">Treatment</h4>
              <p className="text-sm text-slate-600 leading-relaxed">{diseaseInfo.treatment}</p>
            </div>

            <div>
              <h4 className="text-sm font-semibold text-slate-700 mb-2">Prevention Tips</h4>
              <ul className="list-disc list-inside space-y-1">
                {diseaseInfo.prevention.map((tip, i) => (
                  <li key={i} className="text-sm text-slate-600">{tip}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}