import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  LinearProgress, 
  Alert, 
  Button, 
  Card, 
  CardContent,
  Chip 
} from '@mui/material';
import { CheckCircle, Error, Download } from '@mui/icons-material';
import apiService from '../../services/api';
import type { JobStatusResponse, ProcessingJob } from '../../types/api.types';

// Use union type to handle different response formats
type JobData = JobStatusResponse | ProcessingJob;

const ProcessingDashboard: React.FC = () => {
  const { jobId } = useParams<{ jobId: string }>();
  const [job, setJob] = useState<JobData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Type guard to check if job has full details
  const hasFullDetails = (job: JobData): job is ProcessingJob => {
    return 'user_id' in job && 'target_platform' in job;
  };

  // Fetch job status
  const fetchJobStatus = async () => {
    if (!jobId) return;
    
    try {
      const jobData = await apiService.getJobStatus(jobId);
      setJob(jobData);
      setError(null);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch job status');
    } finally {
      setLoading(false);
    }
  };

  // Poll for updates if job is still processing
  useEffect(() => {
    if (!jobId) return;

    fetchJobStatus();

    // Set up polling for pending/processing jobs
    const interval = setInterval(() => {
      if (job?.status === 'pending' || job?.status === 'processing') {
        fetchJobStatus();
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [jobId, job?.status]);

  // Handle download
  const handleDownload = async () => {
    if (!jobId) return;
    
    try {
      // Try to get final video URL from job data
      const finalVideoUrl = hasFullDetails(job!) ? job.final_video_url : job?.result?.final_video_url;
      
      if (finalVideoUrl) {
        window.open(finalVideoUrl, '_blank');
      } else {
        alert('Video URL not available yet');
      }
    } catch (err) {
      console.error('Download failed:', err);
    }
  };

  if (loading) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h4" gutterBottom>
          Loading...
        </Typography>
        <LinearProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 4, maxWidth: 600, mx: 'auto' }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button onClick={fetchJobStatus} variant="contained">
          Retry
        </Button>
      </Box>
    );
  }

  if (!job) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h4">Job not found</Typography>
      </Box>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'processing': return 'primary';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle />;
      case 'failed': return <Error />;
      default: return undefined;
    }
  };

  const statusIcon = getStatusIcon(job.status);

  return (
    <Box sx={{ p: 4, maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom sx={{ textAlign: 'center', mb: 4 }}>
        Video Processing Status
      </Typography>

      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
            <Typography variant="h6">
              Job ID: {job.job_id}
            </Typography>
            <Chip 
              label={job.status.toUpperCase()} 
              color={getStatusColor(job.status) as any}
              {...(statusIcon && { icon: statusIcon })}
            />
          </Box>

          {job.status === 'processing' && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="body1" gutterBottom>
                Processing your video...
              </Typography>
              <LinearProgress variant="determinate" value={job.progress} />
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                {job.progress}% complete
              </Typography>
            </Box>
          )}

          {job.status === 'pending' && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="body1" gutterBottom>
                Your video is queued for processing...
              </Typography>
              <LinearProgress />
            </Box>
          )}

          {job.status === 'completed' && (
            <Box sx={{ mb: 3 }}>
              <Alert severity="success" sx={{ mb: 2 }}>
                🎉 Your video has been processed successfully!
              </Alert>
              
              {hasFullDetails(job) && job.processing_time_seconds && (
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Processing time: {job.processing_time_seconds.toFixed(2)}s
                </Typography>
              )}
              
              {!hasFullDetails(job) && job.result?.processing_time && (
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Processing time: {job.result.processing_time.toFixed(2)}s
                </Typography>
              )}
              
              <Button 
                variant="contained" 
                startIcon={<Download />}
                onClick={handleDownload}
                sx={{ mt: 2 }}
              >
                View Result
              </Button>
            </Box>
          )}

          {job.status === 'failed' && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Processing failed: {hasFullDetails(job) ? (job.error_message || 'Unknown error') : (job.message || 'Unknown error')}
            </Alert>
          )}

          {/* Job Details - only show if we have full details */}
          {hasFullDetails(job) && (
            <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
              <Typography variant="h6" gutterBottom>
                Job Details
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                <Typography variant="body2" color="text.secondary">Created:</Typography>
                <Typography variant="body2">{new Date(job.created_at).toLocaleString()}</Typography>
                
                <Typography variant="body2" color="text.secondary">Platform:</Typography>
                <Typography variant="body2">{job.target_platform}</Typography>
                
                <Typography variant="body2" color="text.secondary">File Size:</Typography>
                <Typography variant="body2">{(job.file_size_bytes / (1024 * 1024)).toFixed(2)} MB</Typography>
                
                {job.video_duration_seconds && (
                  <>
                    <Typography variant="body2" color="text.secondary">Duration:</Typography>
                    <Typography variant="body2">{job.video_duration_seconds}s</Typography>
                  </>
                )}
                
                {job.retry_count > 0 && (
                  <>
                    <Typography variant="body2" color="text.secondary">Retries:</Typography>
                    <Typography variant="body2">{job.retry_count}</Typography>
                  </>
                )}
              </Box>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default ProcessingDashboard; 