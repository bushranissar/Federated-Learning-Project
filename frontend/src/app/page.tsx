"use client";

import { useEffect, useState } from "react";
import { Leaf, Cpu, Network, Brain, GitBranch, BarChart3, Server, Eye, Activity } from "lucide-react";
import SystemArchitecture from "@/components/dashboard/SystemArchitecture";
import MetricsCards from "@/components/metrics/MetricsCards";
import AccuracyVsRound from "@/components/fl-analytics/AccuracyVsRound";
import { getMetrics } from "@/api-services/metrics";
import { getFederatedStatus } from "@/api-services/federated";
import type { MetricsResponse, FederatedStatus } from "@/types";

const techStack = [
  { name: "PyTorch", description: "CNN Deep Learning", color: "bg-red-50 text-red-700 border-red-200" },
  { name: "Scikit-Learn", description: "SVM Classifier", color: "bg-blue-50 text-blue-700 border-blue-200" },
  { name: "Flower", description: "Federated Learning", color: "bg-green-50 text-green-700 border-green-200" },
  { name: "FastAPI", description: "Backend API", color: "bg-teal-50 text-teal-700 border-teal-200" },
  { name: "Next.js", description: "Frontend", color: "bg-purple-50 text-purple-700 border-purple-200" },
  { name: "TailwindCSS", description: "Styling", color: "bg-cyan-50 text-cyan-700 border-cyan-200" },
];

export default function HomePage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [flStatus, setFlStatus] = useState<FederatedStatus | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [flLoading, setFlLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [metricsData, flData] = await Promise.all([
          getMetrics(),
          getFederatedStatus(),
        ]);
        setMetrics(metricsData);
        setFlStatus(flData);
      } catch (err) {
        setError("Failed to load data from backend. Make sure the server is running.");
      } finally {
        setMetricsLoading(false);
        setFlLoading(false);
      }
    }
    fetchData();
  }, []);

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Error Banner */}
      {error && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-center gap-3">
          <Activity className="h-5 w-5 text-amber-500 flex-shrink-0" />
          <p className="text-sm text-amber-700">{error}</p>
        </div>
      )}

      {/* Hero Section */}
      <div className="card overflow-hidden">
        <div className="bg-gradient-navy p-8 text-white">
          <div className="flex items-center gap-3 mb-4">
            <Leaf className="h-8 w-8 text-purple-300" />
            <div>
              <h1 className="text-3xl font-bold tracking-tight">
                Federated Learning Tomato Leaf Disease Detection
              </h1>
              <p className="text-purple-200 mt-1 text-sm">
                CNN-SVM Ensemble with FedAvg Aggregation
              </p>
            </div>
          </div>
          <p className="text-purple-100 max-w-3xl text-sm leading-relaxed">
            A decentralized deep learning system for detecting tomato leaf diseases across 
            multiple clients using Federated Learning. The system employs a CNN feature extractor 
            with 128-dimensional embeddings and an SVM classifier, combined through weighted 
            ensemble (60% CNN + 40% SVM) for robust disease classification across 5 classes.
          </p>
        </div>
      </div>

      {/* Tech Stack */}
      <div>
        <h2 className="text-lg font-semibold text-slate-800 mb-4">Technologies Used</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {techStack.map((tech) => (
            <div key={tech.name} className={`rounded-xl p-3 border ${tech.color}`}>
              <p className="font-semibold text-sm">{tech.name}</p>
              <p className="text-xs opacity-75 mt-0.5">{tech.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* FL Overview */}
      <div className="card p-6">
        <div className="flex items-center gap-2 mb-4">
          <Network className="h-5 w-5 text-purple-600" />
          <h2 className="text-lg font-semibold text-slate-800">Federated Learning Overview</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-purple-50 to-purple-100/50 rounded-xl p-4 border border-purple-200">
            <p className="text-xs text-purple-600 font-medium uppercase tracking-wider">Clients</p>
            <p className="text-2xl font-bold text-purple-800 mt-1">{flStatus?.num_clients || 2}</p>
            <p className="text-xs text-purple-500 mt-1">Federated nodes</p>
          </div>
          <div className="bg-gradient-to-br from-blue-50 to-blue-100/50 rounded-xl p-4 border border-blue-200">
            <p className="text-xs text-blue-600 font-medium uppercase tracking-wider">Rounds Completed</p>
            <p className="text-2xl font-bold text-blue-800 mt-1">{flStatus?.rounds_completed || 40}</p>
            <p className="text-xs text-blue-500 mt-1">Communication rounds</p>
          </div>
          <div className="bg-gradient-to-br from-emerald-50 to-emerald-100/50 rounded-xl p-4 border border-emerald-200">
            <p className="text-xs text-emerald-600 font-medium uppercase tracking-wider">Aggregation</p>
            <p className="text-lg font-bold text-emerald-800 mt-1">FedAvg</p>
            <p className="text-xs text-emerald-500 mt-1">Federated Averaging</p>
          </div>
          <div className="bg-gradient-to-br from-amber-50 to-amber-100/50 rounded-xl p-4 border border-amber-200">
            <p className="text-xs text-amber-600 font-medium uppercase tracking-wider">Status</p>
            <p className="text-lg font-bold text-amber-800 mt-1 capitalize">
              {flStatus?.global_model_status || "Trained"}
            </p>
            <p className="text-xs text-amber-500 mt-1">Global model ready</p>
          </div>
        </div>
      </div>

      {/* System Architecture */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <SystemArchitecture />
        </div>
        <div className="space-y-4">
          {/* Model Status */}
          <div className="card p-6">
            <div className="flex items-center gap-2 mb-4">
              <Brain className="h-5 w-5 text-purple-600" />
              <h3 className="text-lg font-semibold text-slate-800">Model Status</h3>
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-purple-50 rounded-xl">
                <div className="flex items-center gap-2">
                  <Brain className="h-4 w-4 text-purple-600" />
                  <span className="text-sm font-medium text-slate-700">CNN</span>
                </div>
                <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">Active</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-violet-50 rounded-xl">
                <div className="flex items-center gap-2">
                  <GitBranch className="h-4 w-4 text-violet-600" />
                  <span className="text-sm font-medium text-slate-700">SVM</span>
                </div>
                <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">Active</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-indigo-50 rounded-xl">
                <div className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4 text-indigo-600" />
                  <span className="text-sm font-medium text-slate-700">Ensemble</span>
                </div>
                <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">Ready</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                <div className="flex items-center gap-2">
                  <Eye className="h-4 w-4 text-slate-600" />
                  <span className="text-sm font-medium text-slate-700">Classes</span>
                </div>
                <span className="text-xs font-medium text-slate-600">5</span>
              </div>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="card p-6">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="h-5 w-5 text-purple-600" />
              <h3 className="text-lg font-semibold text-slate-800">Quick Stats</h3>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-slate-500">Feature Dimension</span>
                <span className="font-semibold text-slate-800">128-D</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-500">Disease Classes</span>
                <span className="font-semibold text-slate-800">5</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-500">Ensemble Weights</span>
                <span className="font-semibold text-slate-800">0.6 CNN + 0.4 SVM</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Section */}
      <div>
        <h2 className="text-lg font-semibold text-slate-800 mb-4">Model Performance</h2>
        <MetricsCards metrics={metrics} loading={metricsLoading} />
      </div>

      {/* FL Analytics */}
      <div>
        <h2 className="text-lg font-semibold text-slate-800 mb-4">Federated Learning Analytics</h2>
        <AccuracyVsRound data={flStatus} loading={flLoading} />
      </div>
    </div>
  );
}