/**
 * API client for Aesthetic Scorer backend
 */

import axios from "axios";

const API_BASE_URL = "http://localhost:8001/api";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// ============================================================
// Types
// ============================================================

export interface LatentRecord {
  id: number;
  filename: string;
  dataset_id: number;
  dataset_name: string;
  dataset_unique_id: string;
  image_path: string;
  caption: string;
  timestep: number;
  recon_loss: number;
  latent_shape: number[];
  scheduler_type: string;
  user_score: number | null;
  model_score: number | null;
  is_scored: boolean;
  true_latent_image_path: string | null;
  predicted_latent_image_path: string | null;
  created_at: string;
  updated_at: string;
}

export interface AestheticModel {
  id: number;
  name: string;
  version: string;
  architecture: string;
  parameters: Record<string, any>;
  training_config: Record<string, any>;
  num_scored_samples: number;
  num_epochs: number;
  train_loss: number | null;
  val_loss: number | null;
  model_path: string;
  is_active: boolean;
  created_at: string;
}

export interface LatentStats {
  total: number;
  scored: number;
  unscored: number;
  scored_percentage: number;
}

// ============================================================
// API Functions
// ============================================================

export const generateLatents = async (params: {
  dataset_id: number;
  model_path: string;
  num_samples?: number;
  timestep_range?: [number, number];
  shuffle?: boolean;
}) => {
  const response = await api.post("/generate_latents", params);
  return response.data;
};

export const getLatents = async (params: {
  skip?: number;
  limit?: number;
  scored_only?: boolean;
  unscored_only?: boolean;
}) => {
  const response = await api.get("/latents", { params });
  return response.data as { records: LatentRecord[]; total: number };
};

export const getLatent = async (recordId: number) => {
  const response = await api.get(`/latents/${recordId}`);
  return response.data as LatentRecord;
};

export const scoreLatent = async (recordId: number, score: number) => {
  const response = await api.post(`/latents/${recordId}/score`, { score });
  return response.data as LatentRecord;
};

export const getLatentStats = async () => {
  const response = await api.get("/latents/stats");
  return response.data as LatentStats;
};

export const decodeLatents = async (params: {
  record_ids: number[];
  vae_path: string;
}) => {
  const response = await api.post("/decode_latents", params);
  return response.data;
};

export const trainModel = async (params: {
  architecture?: string;
  learning_rate?: number;
  num_epochs?: number;
  batch_size?: number;
  val_split?: number;
  model_name?: string;
}) => {
  const response = await api.post("/train_model", params);
  return response.data;
};

export const getModels = async () => {
  const response = await api.get("/models");
  return response.data as { models: AestheticModel[] };
};

export const getModel = async (modelId: number) => {
  const response = await api.get(`/models/${modelId}`);
  return response.data as AestheticModel;
};

export const activateModel = async (modelId: number) => {
  const response = await api.post(`/models/${modelId}/activate`);
  return response.data;
};

export default api;
