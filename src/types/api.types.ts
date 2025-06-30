// API Types matching the backend models

export enum JobStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

export interface ProcessingJob {
  job_id: string;
  user_id: string;
  status: JobStatus;
  progress: number;
  target_platform: string;
  file_size_bytes: number;
  video_duration_seconds: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  processing_time_seconds?: number;
  retry_count: number;
  error_message?: string;
  final_video_url?: string;
}

export interface JobResult {
  transcript?: string;
  emphasis_points_count?: number;
  entities_count?: number;
  images_count?: number;
  final_video_url?: string;
  processing_time?: number;
}

export interface UploadResponse {
  job_id: string;
  status: JobStatus;
  message: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  progress: number;
  message?: string;
  result?: JobResult;
}

export interface ProcessingStage {
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  duration?: number;
}

export interface VideoPreferences {
  target_platform: 'tiktok' | 'instagram' | 'youtube';
  style: string;
  target_audience: string;
  duration_limit: number;
}

export interface HealthCheck {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  version: string;
  checks: {
    database: string;
    environment: string;
  };
} 