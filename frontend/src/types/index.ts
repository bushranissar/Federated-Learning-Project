export interface ProbabilityMap {
  [className: string]: number;
}

export interface Top5Item {
  class: string;
  probability: number;
}

export interface DiseaseInfo {
  name: string;
  description: string;
  symptoms: string[];
  severity: string;
  treatment: string;
  prevention: string[];
}

export interface PredictionResult {
  success: boolean;
  demo_mode: boolean;
  prediction: string;
  confidence: number;
  cnn_prediction: string;
  cnn_confidence: number;
  cnn_probabilities: ProbabilityMap;
  svm_prediction: string;
  svm_confidence: number;
  svm_probabilities: ProbabilityMap;
  ensemble_prediction: string;
  ensemble_confidence: number;
  ensemble_probabilities: ProbabilityMap;
  probabilities: ProbabilityMap;
  top5: Top5Item[];
  disease_info: DiseaseInfo | null;
  error?: string;
}

export interface MetricItem {
  [key: string]: number | string;
}

export interface ClassificationReport {
  [disease: string]: {
    precision: number;
    recall: number;
    f1_score: number;
    support: number;
  };
}

export interface MetricsResponse {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  loss: number;
  demo_mode: boolean;
  confusion_matrix: number[][];
  classification_report: ClassificationReport;
  rounds_data?: RoundData[];
  client_comparison?: ClientComparison[];
  total_rounds?: number;
}

export interface RoundData {
  round: number;
  cnn_accuracy: number;
  svm_accuracy: number;
  ensemble_accuracy: number;
  loss: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface ClientComparison {
  round: number;
  client1_accuracy: number;
  client2_accuracy: number;
  client1_loss: number;
  client2_loss: number;
}

export interface ClientStatus {
  name: string;
  status: string;
  accuracy: number;
  loss: number;
  samples: number;
}

export interface AccuracyProgression {
  round: number;
  accuracy: number;
  loss: number;
}

export interface FederatedStatus {
  num_clients: number;
  clients: string[];
  rounds_completed: number;
  aggregation_strategy: string;
  status: string;
  global_model_status: string;
  client_status: ClientStatus[];
  accuracy_progression: AccuracyProgression[];
  demo_mode: boolean;
}