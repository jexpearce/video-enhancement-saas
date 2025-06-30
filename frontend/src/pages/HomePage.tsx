import React from 'react';
import { Box, Typography, Container } from '@mui/material';
import { motion } from 'framer-motion';
import VideoUpload from '../components/upload/VideoUpload';

const HomePage: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 6 }}>
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Typography
            variant="h2"
            component="h1"
            textAlign="center"
            gutterBottom
            sx={{
              background: 'linear-gradient(135deg, #ff0050 0%, #ff7b00 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              fontWeight: 800,
            }}
          >
            AI-Powered Video Enhancement
          </Typography>
          <Typography
            variant="h5"
            textAlign="center"
            color="text.secondary"
            sx={{ mb: 6, maxWidth: 600, mx: 'auto' }}
          >
            Transform your talking-head videos with AI that detects emphasis
            and adds perfectly-timed visual context
          </Typography>
        </motion.div>

        <VideoUpload />
      </Box>
    </Container>
  );
};

export default HomePage; 