import axios from 'axios';
import type { AxiosInstance } from 'axios';
import type { 
  UploadResponse, 
  JobStatusResponse, 
  ProcessingJob, 
  HealthCheck,
  VideoPreferences 
} from '../types/api.types';

const API_BASE_URL = (import.meta.env as any).VITE_API_URL || 'http://localhost:8000/api/v1';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add auth interceptor
    this.client.interceptors.request.use((config) => {
      const apiKey = localStorage.getItem('api_key');
      if (apiKey) {
        config.headers.Authorization = `Bearer ${apiKey}`;
      }
      return config;
    });

    // Add response error interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized
          localStorage.removeItem('api_key');
          window.location.href = '/';
        }
        return Promise.reject(error);
      }
    );
  }

  // Health check
  async checkHealth(): Promise<HealthCheck> {
    const response = await this.client.get<HealthCheck>('/health');
    return response.data;
  }

  // Upload video
  async uploadVideo(
    file: File,
    preferences: VideoPreferences,
    onProgress?: (progress: number) => void
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_platform', preferences.target_platform);
    formData.append('user_id', 'user_' + Date.now()); // In production, get from auth

    const response = await this.client.post<UploadResponse>('/videos/upload', formData, {
      // Don't set Content-Type for FormData - axios will set it automatically with boundary
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });

    return response.data;
  }

  // Get job status
  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    const response = await this.client.get<JobStatusResponse>(`/jobs/${jobId}`);
    return response.data;
  }

  // Get job history
  async getJobHistory(limit = 10, offset = 0): Promise<{
    total_count: number;
    jobs: ProcessingJob[];
  }> {
    const response = await this.client.get('/jobs', {
      params: { limit, offset },
    });
    return response.data;
  }

  // Download video
  async getDownloadUrl(jobId: string): Promise<{ download_url: string }> {
    const response = await this.client.get(`/videos/${jobId}/download`);
    return response.data;
  }

  // Delete job
  async deleteJob(jobId: string): Promise<void> {
    await this.client.delete(`/jobs/${jobId}`);
  }

  // Retry failed job
  async retryJob(jobId: string): Promise<{ job_id: string; message: string }> {
    const response = await this.client.post(`/jobs/${jobId}/retry`);
    return response.data;
  }
}

export default new ApiService(); 