import React from 'react';
import { Box, Typography, LinearProgress } from '@mui/material';

const ProcessingDashboard: React.FC = () => {
  return (
    <Box sx={{ p: 4, textAlign: 'center' }}>
      <Typography variant="h4" gutterBottom>
        Processing Your Video
      </Typography>
      <Typography variant="body1" sx={{ mb: 3 }}>
        Please wait while we enhance your video...
      </Typography>
      <LinearProgress />
    </Box>
  );
};

export default ProcessingDashboard; 