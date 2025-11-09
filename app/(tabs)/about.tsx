import { Colors } from '@/constants/theme';
import { Box, Typography, Paper, Divider } from '@mui/material';
import React from 'react';
import InfoIcon from '@mui/icons-material/Info';
import ScienceIcon from '@mui/icons-material/Science';
import SecurityIcon from '@mui/icons-material/Security';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';

export default function AboutPage() {
  const themeColors = Colors.dark;

  const features = [
    {
      icon: <ScienceIcon />,
      title: 'Advanced AI Analysis',
      description:
        'Uses machine learning models to detect coughs and wheezing with high accuracy. Processes audio locally on your device for privacy and speed.',
    },
    {
      icon: <SecurityIcon />,
      title: 'Privacy First',
      description:
        'All audio processing happens on your device. Only summary data is sent to the backend for AI interpretation, keeping your audio recordings private.',
    },
    {
      icon: <HealthAndSafetyIcon />,
      title: 'Health Insights',
      description:
        'Get detailed analysis of your respiratory patterns, including cough frequency, wheeze detection, and AI-powered health interpretations.',
    },
  ];

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: `linear-gradient(-45deg, ${themeColors.background} 25%, ${themeColors.backgroundGradient})`,
        color: themeColors.text,
        p: 3,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <InfoIcon sx={{ fontSize: 40, color: themeColors.bright }} />
        <Typography variant="h4" sx={{ fontWeight: 700, color: themeColors.text }}>
          About BreathWatch
        </Typography>
      </Box>

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
          What is BreathWatch?
        </Typography>
        <Typography variant="body1" sx={{ color: themeColors.text, mb: 2 }}>
          BreathWatch is a comprehensive respiratory health monitoring application that uses advanced audio analysis
          technology to detect, track, and analyze coughs and wheezing patterns. Designed for both personal health
          monitoring and clinical use, BreathWatch provides detailed insights into respiratory health.
        </Typography>
        <Typography variant="body1" sx={{ color: themeColors.text }}>
          The app processes audio recordings in real-time, detecting respiratory events with high accuracy using
          machine learning models. All processing happens locally on your device, ensuring your privacy while
          providing powerful health insights.
        </Typography>
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
          Key Features
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {features.map((feature, index) => (
            <Box key={index}>
              <Box sx={{ display: 'flex', alignItems: 'start', gap: 2, mb: 1 }}>
                <Box sx={{ color: themeColors.bright, mt: 0.5 }}>{feature.icon}</Box>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600, color: themeColors.text, mb: 0.5 }}>
                    {feature.title}
                  </Typography>
                  <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.9 }}>
                    {feature.description}
                  </Typography>
                </Box>
              </Box>
              {index < features.length - 1 && (
                <Divider sx={{ mt: 2, borderColor: themeColors.text, opacity: 0.2 }} />
              )}
            </Box>
          ))}
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
          How It Works
        </Typography>
        <Box component="ol" sx={{ pl: 2, color: themeColors.text }}>
          <li>
            <Typography variant="body2" sx={{ color: themeColors.text, mb: 1 }}>
              <strong>Record:</strong> Start a recording session using the microphone button. The app will process
              audio in 10-minute chunks.
            </Typography>
          </li>
          <li>
            <Typography variant="body2" sx={{ color: themeColors.text, mb: 1 }}>
              <strong>Process:</strong> Audio is processed locally using machine learning models to detect coughs
              and wheezing in real-time.
            </Typography>
          </li>
          <li>
            <Typography variant="body2" sx={{ color: themeColors.text, mb: 1 }}>
              <strong>Analyze:</strong> View real-time charts and statistics as your recording progresses. See
              cough counts, wheeze detection, and quality metrics.
            </Typography>
          </li>
          <li>
            <Typography variant="body2" sx={{ color: themeColors.text }}>
              <strong>Interpret:</strong> Get AI-powered health interpretations and recommendations based on your
              respiratory patterns.
            </Typography>
          </li>
        </Box>
      </Paper>

      <Paper
        sx={{
          p: 3,
          background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
          borderRadius: '20px',
          boxShadow: `3px 3px 0 ${themeColors.text}`,
        }}
      >
        <Typography variant="h6" sx={{ mb: 2, color: themeColors.text, fontWeight: 600 }}>
          Version Information
        </Typography>
        <Typography variant="body2" sx={{ color: themeColors.text, mb: 1 }}>
          <strong>Version:</strong> 1.0.0
        </Typography>
        <Typography variant="body2" sx={{ color: themeColors.text, mb: 1 }}>
          <strong>Platform:</strong> Web, iOS, Android
        </Typography>
        <Typography variant="body2" sx={{ color: themeColors.text }}>
          <strong>Privacy:</strong> All audio processing is done locally on your device. Only summary data is
          transmitted for AI interpretation.
        </Typography>
      </Paper>
    </Box>
  );
}
