import { Colors } from '@/constants/theme';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Switch,
  FormControlLabel,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
} from '@mui/material';
import React, { useState, useEffect } from 'react';
import { getSettings, saveSettings, AppSettings, clearRecordingHistory } from '@/services/storage';
import DeleteForeverIcon from '@mui/icons-material/DeleteForever';
import SaveIcon from '@mui/icons-material/Save';

export default function SettingsPage() {
  const themeColors = Colors.dark;
  const [settings, setSettings] = useState<AppSettings>({
    autoSaveRecordings: true,
    notificationsEnabled: true,
    theme: 'auto',
  });
  const [loading, setLoading] = useState(true);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const data = await getSettings();
      setSettings(data);
    } catch (error) {
      console.error('Failed to load settings:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      await saveSettings(settings);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (error) {
      console.error('Failed to save settings:', error);
    }
  };

  const handleClearHistory = async () => {
    if (confirm('Are you sure you want to clear all processed logs? This cannot be undone.')) {
      try {
        await clearRecordingHistory();
        alert('Processed logs cleared successfully');
      } catch (error) {
        console.error('Failed to clear history:', error);
        alert('Failed to clear history');
      }
    }
  };

  if (loading) {
    return (
      <Box
        sx={{
          minHeight: '100vh',
          background: `linear-gradient(-45deg, ${themeColors.background} 25%, ${themeColors.backgroundGradient})`,
          color: themeColors.text,
          p: 3,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Typography variant="body1" sx={{ color: themeColors.text }}>
          Loading settings...
        </Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: `linear-gradient(-45deg, ${themeColors.background} 25%, ${themeColors.backgroundGradient})`,
        color: themeColors.text,
        p: 3,
      }}
    >
      <Typography
        variant="h4"
        sx={{
          mb: 3,
          fontWeight: 700,
          color: themeColors.text,
          textAlign: 'center',
        }}
      >
        Settings
      </Typography>

      <Paper
        sx={{
          p: 3,
          mb: 3,
          background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
          borderRadius: '20px',
          boxShadow: `3px 3px 0 ${themeColors.text}`,
        }}
      >
        <Typography variant="h6" sx={{ mb: 2, color: themeColors.text, fontWeight: 600 }}>
          Patient Information
        </Typography>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mb: 3 }}>
          <TextField
            label="Patient ID"
            type="number"
            value={settings.patientId || ''}
            onChange={(e) =>
              setSettings({ ...settings, patientId: e.target.value ? parseInt(e.target.value) : undefined })
            }
            sx={{
              '& .MuiOutlinedInput-root': {
                color: themeColors.text,
                '& fieldset': {
                  borderColor: themeColors.text,
                },
                '&:hover fieldset': {
                  borderColor: themeColors.bright,
                },
              },
              '& .MuiInputLabel-root': {
                color: themeColors.text,
              },
            }}
          />

          <TextField
            label="Age"
            type="number"
            value={settings.age || ''}
            onChange={(e) =>
              setSettings({ ...settings, age: e.target.value ? parseInt(e.target.value) : undefined })
            }
            sx={{
              '& .MuiOutlinedInput-root': {
                color: themeColors.text,
                '& fieldset': {
                  borderColor: themeColors.text,
                },
                '&:hover fieldset': {
                  borderColor: themeColors.bright,
                },
              },
              '& .MuiInputLabel-root': {
                color: themeColors.text,
              },
            }}
          />

          <FormControl fullWidth>
            <InputLabel sx={{ color: themeColors.text }}>Sex</InputLabel>
            <Select
              value={settings.sex || ''}
              onChange={(e) => setSettings({ ...settings, sex: e.target.value || undefined })}
              label="Sex"
              sx={{
                color: themeColors.text,
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: themeColors.text,
                },
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: themeColors.bright,
                },
                '& .MuiSvgIcon-root': {
                  color: themeColors.text,
                },
              }}
            >
              <MenuItem value="">Not specified</MenuItem>
              <MenuItem value="M">Male</MenuItem>
              <MenuItem value="F">Female</MenuItem>
              <MenuItem value="Other">Other</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Paper>

      <Paper
        sx={{
          p: 3,
          mb: 3,
          background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
          borderRadius: '20px',
          boxShadow: `3px 3px 0 ${themeColors.text}`,
        }}
      >
        <Typography variant="h6" sx={{ mb: 2, color: themeColors.text, fontWeight: 600 }}>
          App Preferences
        </Typography>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={settings.autoSaveRecordings}
                onChange={(e) => setSettings({ ...settings, autoSaveRecordings: e.target.checked })}
                sx={{
                  '& .MuiSwitch-switchBase.Mui-checked': {
                    color: themeColors.bright,
                  },
                  '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                    backgroundColor: themeColors.bright,
                  },
                }}
              />
            }
            label="Auto-save processed logs"
            sx={{ color: themeColors.text }}
          />

          <FormControlLabel
            control={
              <Switch
                checked={settings.notificationsEnabled}
                onChange={(e) => setSettings({ ...settings, notificationsEnabled: e.target.checked })}
                sx={{
                  '& .MuiSwitch-switchBase.Mui-checked': {
                    color: themeColors.bright,
                  },
                  '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                    backgroundColor: themeColors.bright,
                  },
                }}
              />
            }
            label="Enable notifications"
            sx={{ color: themeColors.text }}
          />

          <FormControl fullWidth>
            <InputLabel sx={{ color: themeColors.text }}>Theme</InputLabel>
            <Select
              value={settings.theme}
              onChange={(e) => setSettings({ ...settings, theme: e.target.value as 'light' | 'dark' | 'auto' })}
              label="Theme"
              sx={{
                color: themeColors.text,
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: themeColors.text,
                },
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: themeColors.bright,
                },
                '& .MuiSvgIcon-root': {
                  color: themeColors.text,
                },
              }}
            >
              <MenuItem value="auto">Auto</MenuItem>
              <MenuItem value="light">Light</MenuItem>
              <MenuItem value="dark">Dark</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Paper>

      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Button
          variant="contained"
          startIcon={<SaveIcon />}
          onClick={handleSave}
          sx={{
            flex: 1,
            backgroundColor: themeColors.bright,
            color: themeColors.background,
            '&:hover': {
              backgroundColor: themeColors.text,
            },
          }}
        >
          {saved ? 'Saved!' : 'Save Settings'}
        </Button>
      </Box>

      <Divider sx={{ my: 3, borderColor: themeColors.text, opacity: 0.3 }} />

      <Paper
        sx={{
          p: 3,
          background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
          borderRadius: '20px',
          boxShadow: `3px 3px 0 ${themeColors.text}`,
        }}
      >
        <Typography variant="h6" sx={{ mb: 2, color: themeColors.text, fontWeight: 600 }}>
          Danger Zone
        </Typography>
        <Button
          variant="outlined"
          startIcon={<DeleteForeverIcon />}
          onClick={handleClearHistory}
          sx={{
            borderColor: '#ff4444',
            color: '#ff4444',
            '&:hover': {
              borderColor: '#ff6666',
              backgroundColor: 'rgba(255, 68, 68, 0.1)',
            },
          }}
        >
          Clear All Processed Logs
        </Button>
      </Paper>
    </Box>
  );
}

