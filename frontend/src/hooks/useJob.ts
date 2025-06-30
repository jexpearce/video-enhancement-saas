import { useState, useEffect, useCallback } from 'react';
import apiService from '../services/api';
import { JobStatusResponse } from '../types/api.types';

export function useJob(jobId: string) {
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchJob = useCallback(async () => {
    if (!jobId) return;

    try {
      setLoading(true);
      const data = await apiService.getJobStatus(jobId);
      setJob(data);
      setError(null);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch job');
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    fetchJob();
  }, [fetchJob]);

  return {
    job,
    loading,
    error,
    refetch: fetchJob,
  };
} 