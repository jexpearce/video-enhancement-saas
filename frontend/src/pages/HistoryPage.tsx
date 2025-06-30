import React from 'react';
import { Box, Typography } from '@mui/material';

const HistoryPage: React.FC = () => {
  return (
    <Box sx={{ p: 4, textAlign: 'center' }}>
      <Typography variant="h4">Processing History</Typography>
      <Typography variant="body1" sx={{ mt: 2 }}>
        Your video processing history will appear here.
      </Typography>
    </Box>
  );
};

export default HistoryPage; 