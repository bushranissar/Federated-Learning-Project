import apiClient from "./client";
import type { MetricsResponse } from "@/types";

export async function getMetrics(): Promise<MetricsResponse> {
  const response = await apiClient.get<MetricsResponse>("/metrics");
  return response.data;
}