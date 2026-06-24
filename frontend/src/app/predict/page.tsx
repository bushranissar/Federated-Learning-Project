"use client";

import { useState, useCallback } from "react";
import { Upload, AlertCircle } from "lucide-react";
import ImageUploader from "@/components/predict/ImageUploader";
import PredictionResultComponent from "@/components/predict/PredictionResult";
import { predictImage } from "@/api-services/predictor";
import type { PredictionResult } from "@/types";

export default function PredictPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  const handleClear = useCallback(() => {
    setSelectedFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setResult(null);
    setError(null);
  }, [preview]);

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const prediction = await predictImage(selectedFile);
      setResult(prediction);
    } catch (err: any) {
      if (err.response) {
        setError(err.response.data?.detail || "Server error occurred");
      } else if (err.request) {
        setError("Cannot connect to server. Make sure the backend is running.");
      } else {
        setError(err.message || "An unexpected error occurred");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <Upload className="h-6 w-6 text-purple-600" />
          <h1 className="text-2xl font-bold text-slate-800">Disease Prediction</h1>
        </div>
        <p className="text-sm text-slate-500">
          Upload a tomato leaf image to detect diseases using the CNN-SVM ensemble model
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="space-y-4">
          <ImageUploader
            onImageSelect={handleImageSelect}
            onClear={handleClear}
            selectedImage={preview}
          />

          {preview && (
            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full py-3 px-6 bg-gradient-purple text-white rounded-xl font-medium
                         hover:opacity-90 transition-all duration-200 disabled:opacity-50
                         flex items-center justify-center gap-2 shadow-lg shadow-purple-200"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Upload className="h-5 w-5" />
                  Run Prediction
                </>
              )}
            </button>
          )}
        </div>

        {/* Results Section */}
        <div>
          <PredictionResultComponent
            result={result}
            loading={loading}
            error={error}
          />

          {!result && !loading && !error && (
            <div className="card p-8 text-center">
              <div className="flex flex-col items-center gap-3">
                <Upload className="h-12 w-12 text-slate-300" />
                <h3 className="font-semibold text-slate-600">No Image Selected</h3>
                <p className="text-sm text-slate-400 max-w-sm">
                  Upload a tomato leaf image to get disease predictions from the 
                  CNN-SVM ensemble model
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}