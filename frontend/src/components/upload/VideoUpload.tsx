import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  IconButton,
  Fade,
  Alert,
} from '@mui/material';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import {
  CloudUpload,
  VideoFile,
  Close,
  CheckCircle,
} from '@mui/icons-material';
import toast from 'react-hot-toast';

import apiService from '../../services/api';
import type { VideoPreferences } from '../../types/api.types';
import PreferencesPanel from './PreferencesPanel';

const ALLOWED_FORMATS = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

interface VideoUploadProps {
  onUploadStart?: () => void;
}

const VideoUpload: React.FC<VideoUploadProps> = ({ onUploadStart }) => {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [preferences, setPreferences] = useState<VideoPreferences>({
    target_platform: 'tiktok',
    style: 'energetic',
    target_audience: 'gen_z',
    duration_limit: 30,
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    
    if (!file) return;

    // Validate file
    if (!ALLOWED_FORMATS.includes(file.type)) {
      toast.error('Invalid file format. Please upload MP4, MOV, or AVI.');
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      toast.error('File too large. Maximum size is 100MB.');
      return;
    }

    setFile(file);
    
    // Create video preview
    const url = URL.createObjectURL(file);
    setVideoPreview(url);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi'],
    },
    maxFiles: 1,
    maxSize: MAX_FILE_SIZE,
  });

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    onUploadStart?.();

    try {
      // Ensure file has a name
      const fileWithName = new File([file], file.name || 'video.mp4', {
        type: file.type
      });

      const response = await apiService.uploadVideo(
        fileWithName,
        preferences,
        setUploadProgress
      );

      toast.success('Video uploaded successfully!');
      
      // Navigate to processing dashboard
      navigate(`/dashboard/${response.job_id}`);
    } catch (error: any) {
      console.error('Upload error:', error);
      
      // Extract error message safely
      let errorMessage = 'Upload failed';
      if (error.response?.data?.detail) {
        if (typeof error.response.data.detail === 'string') {
          errorMessage = error.response.data.detail;
        } else if (Array.isArray(error.response.data.detail)) {
          errorMessage = error.response.data.detail[0]?.msg || 'Upload failed';
        }
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      toast.error(errorMessage);
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setVideoPreview(null);
    setUploadProgress(0);
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Typography variant="h4" gutterBottom sx={{ textAlign: 'center', mb: 4 }}>
          Upload Your Video
        </Typography>

        <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
          {/* Upload Area */}
          <Box sx={{ flex: '1 1 600px' }}>
            <Paper
              {...getRootProps()}
              sx={{
                p: 4,
                textAlign: 'center',
                cursor: 'pointer',
                bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'divider',
                borderRadius: 3,
                transition: 'all 0.3s ease',
                minHeight: 400,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                '&:hover': {
                  borderColor: 'primary.main',
                  bgcolor: 'action.hover',
                },
              }}
            >
              <input {...getInputProps()} />
              
              {!file ? (
                <>
                  <CloudUpload sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    {isDragActive ? 'Drop your video here' : 'Drag your video here or click to upload'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Supported: MP4, MOV, AVI • Max: 100MB
                  </Typography>
                </>
              ) : (
                <Fade in>
                  <Box sx={{ width: '100%' }}>
                    {videoPreview && (
                      <Box sx={{ position: 'relative', mb: 2 }}>
                        <video
                          src={videoPreview}
                          controls
                          style={{
                            width: '100%',
                            maxHeight: 300,
                            borderRadius: 8,
                          }}
                        />
                        {!uploading && (
                          <IconButton
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRemoveFile();
                            }}
                            sx={{
                              position: 'absolute',
                              top: 8,
                              right: 8,
                              bgcolor: 'background.paper',
                              '&:hover': { bgcolor: 'error.main', color: 'white' },
                            }}
                          >
                            <Close />
                          </IconButton>
                        )}
                      </Box>
                    )}
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <VideoFile color="primary" />
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="body1">{file.name}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </Typography>
                      </Box>
                      {uploadProgress === 100 && (
                        <CheckCircle color="success" />
                      )}
                    </Box>

                    {uploading && (
                      <Box sx={{ mt: 2 }}>
                        <LinearProgress variant="determinate" value={uploadProgress} />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                          Uploading... {uploadProgress}%
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </Fade>
              )}
            </Paper>

            {file && uploadProgress === 100 && (
              <Alert severity="success" sx={{ mt: 2 }}>
                Upload complete! Processing will begin shortly...
              </Alert>
            )}
          </Box>

          {/* Preferences Panel */}
          <Box sx={{ flex: '0 0 300px' }}>
            <PreferencesPanel
              preferences={preferences}
              onChange={setPreferences}
              onProcess={handleUpload}
              disabled={!file || uploading}
              loading={uploading}
            />
          </Box>
        </Box>
      </motion.div>
    </Box>
  );
};

export default VideoUpload; 