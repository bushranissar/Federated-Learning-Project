import apiClient from "./client";
import type { PredictionResult } from "@/types";

export async function predictImage(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post<PredictionResult>("/predict", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    timeout: 60000,
  });

  return response.data;
}