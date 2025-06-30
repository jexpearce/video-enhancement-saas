import React from 'react';
import { Box, Typography } from '@mui/material';

const ResultPage: React.FC = () => {
  return (
    <Box sx={{ p: 4, textAlign: 'center' }}>
      <Typography variant="h4">Video Results</Typography>
      <Typography variant="body1" sx={{ mt: 2 }}>
        Your enhanced video results will appear here.
      </Typography>
    </Box>
  );
};

export default ResultPage; 