import React from 'react';
import { Card, CardContent, Typography, Button, Box } from '@mui/material';
import type { VideoPreferences } from '../../types/api.types';

interface PreferencesPanelProps {
  preferences: VideoPreferences;
  onChange: (preferences: VideoPreferences) => void;
  onProcess: () => void;
  disabled: boolean;
  loading: boolean;
}

const PreferencesPanel: React.FC<PreferencesPanelProps> = ({
  preferences,
  onChange,
  onProcess,
  disabled,
  loading,
}) => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Preferences
        </Typography>
        
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
          <Typography variant="body2">
            Platform: {preferences.target_platform}
          </Typography>
          <Typography variant="body2">
            Style: {preferences.style}
          </Typography>
          
          <Button
            variant="contained"
            fullWidth
            onClick={onProcess}
            disabled={disabled}
            sx={{ mt: 2 }}
          >
            {loading ? 'Processing...' : 'Process Video'}
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PreferencesPanel; 