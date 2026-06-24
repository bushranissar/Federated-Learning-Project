"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { Database, Cpu, Network, Brain, BarChart3, Upload, Server, GitBranch } from "lucide-react";

interface ArchNode {
  id: string;
  label: string;
  icon: React.ReactNode;
  description: string;
  color: string;
}

const architectureSteps: ArchNode[] = [
  {
    id: "dataset",
    label: "Client Datasets",
    icon: <Database className="h-5 w-5" />,
    description: "Distributed tomato leaf data across 2 clients with 5 disease classes each",
    color: "from-blue-500 to-blue-600",
  },
  {
    id: "cnn",
    label: "CNN Feature Extractor",
    icon: <Brain className="h-5 w-5" />,
    description: "Custom CNN architecture extracting 128-dimensional feature vectors from leaf images",
    color: "from-purple-500 to-purple-600",
  },
  {
    id: "svm",
    label: "SVM Classifier",
    icon: <Cpu className="h-5 w-5" />,
    description: "Federated Linear SVM using SGDClassifier with log-loss for probability estimation",
    color: "from-violet-500 to-violet-600",
  },
  {
    id: "ensemble",
    label: "Ensemble (0.6 CNN + 0.4 SVM)",
    icon: <GitBranch className="h-5 w-5" />,
    description: "Weighted ensemble combining CNN and SVM probabilities for robust predictions",
    color: "from-indigo-500 to-indigo-600",
  },
  {
    id: "local",
    label: "Local Training",
    icon: <Cpu className="h-5 w-5" />,
    description: "Each client trains locally on private data using CNN-SVM pipeline",
    color: "from-cyan-500 to-cyan-600",
  },
  {
    id: "flower",
    label: "Flower Client",
    icon: <Network className="h-5 w-5" />,
    description: "Flower Framework clients send model parameters to federated server",
    color: "from-teal-500 to-teal-600",
  },
  {
    id: "fedavg",
    label: "FedAvg Server",
    icon: <Server className="h-5 w-5" />,
    description: "Federated Averaging aggregates client models into global model",
    color: "from-emerald-500 to-emerald-600",
  },
  {
    id: "global",
    label: "Global Model",
    icon: <Brain className="h-5 w-5" />,
    description: "Aggregated global model distributed back to clients for next training round",
    color: "from-green-500 to-green-600",
  },
  {
    id: "dashboard",
    label: "Prediction Dashboard",
    icon: <BarChart3 className="h-5 w-5" />,
    description: "Research dashboard for disease prediction, metrics visualization, and FL analytics",
    color: "from-purple-600 to-purple-700",
  },
];

export default function SystemArchitecture() {
  const [activeStep, setActiveStep] = useState<number | null>(null);

  return (
    <div className="card p-6">
      <div className="flex items-center gap-2 mb-6">
        <Server className="h-5 w-5 text-purple-600" />
        <h3 className="text-lg font-semibold text-slate-800">System Architecture</h3>
      </div>

      <div className="relative">
        {/* Vertical Flow */}
        <div className="space-y-3">
          {architectureSteps.map((step, index) => (
            <div key={step.id} className="relative">
              {/* Connector Line */}
              {index < architectureSteps.length - 1 && (
                <div className="absolute left-6 top-12 bottom-0 w-0.5 bg-gradient-to-b from-purple-400 to-purple-600 hidden md:block" />
              )}

              <div
                className={cn(
                  "relative flex items-start gap-4 p-4 rounded-xl cursor-pointer transition-all duration-300",
                  "hover:bg-slate-50 border border-transparent hover:border-purple-200",
                  activeStep === index && "bg-purple-50 border-purple-300 shadow-md"
                )}
                onMouseEnter={() => setActiveStep(index)}
                onMouseLeave={() => setActiveStep(null)}
              >
                {/* Icon */}
                <div
                  className={cn(
                    "flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center text-white shadow-lg",
                    "bg-gradient-to-br",
                    step.color
                  )}
                >
                  {step.icon}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-slate-800 text-sm">
                    {step.label}
                  </h4>
                  <p className={cn(
                    "text-xs text-slate-500 mt-1 transition-all duration-300",
                    activeStep === index ? "opacity-100" : "opacity-70"
                  )}>
                    {step.description}
                  </p>
                </div>

                {/* Step Number */}
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-slate-100 flex items-center justify-center">
                  <span className="text-xs font-semibold text-slate-500">{index + 1}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 pt-4 border-t border-slate-100">
        <div className="flex flex-wrap gap-4 text-xs text-slate-500">
          <span className="flex items-center gap-1">
            <Upload className="h-3 w-3" /> Data Flow (Input → Output)
          </span>
          <span className="flex items-center gap-1">
            <Network className="h-3 w-3" /> Federated Communication
          </span>
        </div>
      </div>
    </div>
  );
}