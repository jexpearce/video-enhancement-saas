import React from 'react';
import { Box, Typography } from '@mui/material';
import { useParams } from 'react-router-dom';
import ProcessingDashboard from '../components/processing/ProcessingDashboard';

const DashboardPage: React.FC = () => {
  const { jobId } = useParams<{ jobId: string }>();

  if (!jobId) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h5">Job ID not found</Typography>
      </Box>
    );
  }

  return <ProcessingDashboard />;
};

export default DashboardPage; 