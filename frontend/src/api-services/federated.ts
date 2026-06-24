import apiClient from "./client";
import type { FederatedStatus } from "@/types";

export async function getFederatedStatus(): Promise<FederatedStatus> {
  const response = await apiClient.get<FederatedStatus>("/federated-status");
  return response.data;
}