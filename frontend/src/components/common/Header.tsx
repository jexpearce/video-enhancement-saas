import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const Header: React.FC = () => {
  const navigate = useNavigate();

  return (
    <AppBar position="static" color="transparent" elevation={0}>
      <Toolbar>
        <Typography
          variant="h6"
          component="div"
          sx={{
            flexGrow: 1,
            fontWeight: 700,
            background: 'linear-gradient(135deg, #ff0050 0%, #ff7b00 100%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            cursor: 'pointer',
          }}
          onClick={() => navigate('/')}
        >
          🎬 VideoEnhance AI
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button onClick={() => navigate('/')}>Upload</Button>
          <Button onClick={() => navigate('/history')}>History</Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header; 