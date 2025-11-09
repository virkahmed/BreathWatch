import { CoughChart } from '@/components/CoughChart';
import { Paper, Typography, Box } from '@mui/material';
import React, { useMemo, useEffect } from 'react';
import { useRecording } from '@/contexts/RecordingContext';
import { useRouter } from 'expo-router';
import { Colors } from '@/constants/theme';
import HistoryIcon from '@mui/icons-material/History';
import BarChartIcon from '@mui/icons-material/BarChart';
import SettingsIcon from '@mui/icons-material/Settings';
import HelpIcon from '@mui/icons-material/Help';
import InfoIcon from '@mui/icons-material/Info';

export default function HomePage() {
  const themeColors = Colors.dark;
  const router = useRouter();
  const { chunkResponses, finalSummary, fetchFinalSummary, isFetchingSummary } = useRecording();

  // Default sample data (original styling) - all zeros initially
  const defaultData = {
    counts: [0, 0, 0, 0, 0, 0, 0],
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    breakdown: [
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
    ],
  };

  // Calculate data from chunk responses or final summary
  // Always show weekly view (Mon-Sun) with aggregated data
  const chartData = useMemo(() => {
    const weeklyLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const weeklyCounts = [0, 0, 0, 0, 0, 0, 0];
    const weeklyBreakdown = [
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
    ];

    // If we have a final summary, distribute data across the week
    if (finalSummary) {
      const hourlyData = finalSummary.hourly_breakdown || [];
      const totalCoughs = hourlyData.reduce((sum, hour) => sum + hour.cough_count, 0);
      
      // Distribute total coughs evenly across 7 days for visualization
      const coughsPerDay = Math.floor(totalCoughs / 7);
      const remainder = totalCoughs % 7;
      
      for (let i = 0; i < 7; i++) {
        weeklyCounts[i] = coughsPerDay + (i < remainder ? 1 : 0);
        
        // Calculate wet/dry breakdown based on attribute prevalence
        const wetEstimate = Math.round(
          (weeklyCounts[i] * finalSummary.attribute_prevalence.wet) / 100
        );
        weeklyBreakdown[i] = {
          wet: wetEstimate,
          dry: weeklyCounts[i] - wetEstimate,
        };
      }
      
      return { counts: weeklyCounts, labels: weeklyLabels, breakdown: weeklyBreakdown };
    }

    // If we have chunk responses, aggregate them
    if (chunkResponses.length > 0) {
      const totalCoughs = chunkResponses.reduce((sum, chunk) => sum + chunk.cough_count, 0);
      
      // Distribute total coughs across the week
      const coughsPerDay = Math.floor(totalCoughs / 7);
      const remainder = totalCoughs % 7;
      
      for (let i = 0; i < 7; i++) {
        weeklyCounts[i] = coughsPerDay + (i < remainder ? 1 : 0);
        
        // Estimate wet/dry (50/50 split as placeholder)
        const totalEvents = chunkResponses.reduce(
          (sum, chunk) => sum + (chunk.detected_events?.length || 0),
          0
        );
        const eventsPerDay = Math.floor(totalEvents / 7);
        weeklyBreakdown[i] = {
          wet: Math.floor(eventsPerDay * 0.5),
          dry: Math.ceil(eventsPerDay * 0.5),
        };
      }
      
      return { counts: weeklyCounts, labels: weeklyLabels, breakdown: weeklyBreakdown };
    }

    // Default: show weekly data with zeros
    return defaultData;
  }, [chunkResponses, finalSummary]);

  // Show real-time updates when chunks come in
  useEffect(() => {
    if (chunkResponses.length > 0) {
      const lastChunk = chunkResponses[chunkResponses.length - 1];
      console.log('New chunk processed:', {
        chunkIndex: lastChunk.chunk_index,
        coughCount: lastChunk.cough_count,
        wheezeWindows: lastChunk.wheeze_windows,
      });
    }
  }, [chunkResponses]);

  const navigationCards = [
    {
      title: 'History',
      description: 'Past processed logs',
      icon: <HistoryIcon />,
      route: '/history',
      color: themeColors.bright,
    },
    {
      title: 'Statistics',
      description: 'Track trends',
      icon: <BarChartIcon />,
      route: '/statistics',
      color: themeColors.bright,
    },
    {
      title: 'Settings',
      description: 'Preferences',
      icon: <SettingsIcon />,
      route: '/settings',
      color: themeColors.bright,
    },
    {
      title: 'Help',
      description: 'FAQ & docs',
      icon: <HelpIcon />,
      route: '/help',
      color: themeColors.bright,
    },
    {
      title: 'About',
      description: 'Learn more',
      icon: <InfoIcon />,
      route: '/about',
      color: themeColors.bright,
    },
  ];

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: `linear-gradient(135deg, ${themeColors.background} 0%, ${themeColors.backgroundGradient} 100%)`,
        display: 'flex',
        flexDirection: 'column',
        fontFamily: Colors.typography.fontFamily,
      }}
    >
      {/* Chart Section - Now at top */}
      <Box sx={{ flex: 1, overflowY: 'auto', pb: '80px' }}>
        <Paper sx={{ minHeight: '100%', background: 'transparent' }}>
        {isFetchingSummary && (
          <Box
            sx={{
              p: 3,
              textAlign: 'center',
              background: `linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(123, 44, 191, 0.1) 100%)`,
              borderRadius: '12px',
              mx: 2,
              mb: 2,
            }}
          >
            <Typography variant="body1" sx={{ color: themeColors.text, opacity: 0.8 }}>
              Generating final summary...
            </Typography>
          </Box>
        )}
      {chunkResponses.length > 0 && !finalSummary && !isFetchingSummary && (
        <Box
          sx={{
            p: 2,
            textAlign: 'center',
            background: `linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(123, 44, 191, 0.1) 100%)`,
            borderRadius: '12px',
            mx: 2,
            mb: 2,
            border: `1px solid rgba(0, 212, 255, 0.2)`,
          }}
        >
          <Typography variant="body2" sx={{ color: themeColors.text, fontWeight: 500 }}>
            {chunkResponses.length} chunk{chunkResponses.length !== 1 ? 's' : ''} processed •{' '}
            {chunkResponses.reduce((sum, chunk) => sum + chunk.cough_count, 0)} total coughs detected
          </Typography>
          <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7, mt: 1, display: 'block' }}>
            Chart shows real-time data from processed chunks
          </Typography>
        </Box>
      )}
      {finalSummary && (
        <Box
          sx={{
            p: 2,
            textAlign: 'center',
            background: `linear-gradient(135deg, rgba(6, 255, 165, 0.15) 0%, rgba(0, 212, 255, 0.1) 100%)`,
            borderRadius: '12px',
            mx: 2,
            mb: 2,
            border: `1px solid ${themeColors.success}`,
          }}
        >
          <Typography variant="body2" sx={{ color: themeColors.success, fontWeight: 600 }}>
            Analysis Complete!
          </Typography>
          <Typography variant="body2" sx={{ color: themeColors.text, mt: 0.5, opacity: 0.9 }}>
            {finalSummary.coughs_per_hour.toFixed(1)} coughs/hour •{' '}
            {finalSummary.wheeze_time_percent.toFixed(1)}% wheeze time
          </Typography>
        </Box>
      )}
      <CoughChart
        counts={chartData.counts}
        labels={chartData.labels}
        breakdown={chartData.breakdown}
      />
        </Paper>
      </Box>

      {/* Action Bar - Now at bottom */}
      <Box
        sx={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          background: `linear-gradient(to top, ${themeColors.background} 0%, ${themeColors.backgroundGradient} 100%)`,
          borderTop: `1px solid ${themeColors.secondary}`,
          backdropFilter: 'blur(10px)',
          zIndex: 1000,
          boxShadow: `0 -4px 20px rgba(0, 0, 0, 0.3)`,
        }}
      >
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-around',
            alignItems: 'center',
            px: 1,
            py: 1.5,
            maxWidth: '100%',
            overflowX: 'auto',
            '&::-webkit-scrollbar': {
              display: 'none',
            },
            scrollbarWidth: 'none',
          }}
        >
          {navigationCards.map((card, index) => (
            <Box
              key={index}
              onClick={() => router.push(card.route)}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minWidth: { xs: '60px', sm: '80px' },
                px: { xs: 1, sm: 2 },
                py: 1,
                cursor: 'pointer',
                borderRadius: '12px',
                transition: 'all 0.2s ease',
                position: 'relative',
                '&:hover': {
                  background: `linear-gradient(135deg, rgba(0, 212, 255, 0.15) 0%, rgba(123, 44, 191, 0.15) 100%)`,
                  transform: 'translateY(-2px)',
                },
                '&:active': {
                  transform: 'translateY(0)',
                },
              }}
            >
              <Box
                sx={{
                  color: card.color,
                  fontSize: { xs: '24px', sm: '28px' },
                  mb: 0.5,
                  filter: 'drop-shadow(0 2px 4px rgba(0, 212, 255, 0.3))',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {card.icon}
              </Box>
              <Typography
                variant="caption"
                sx={{
                  color: themeColors.text,
                  fontWeight: 500,
                  fontSize: { xs: '0.65rem', sm: '0.75rem' },
                  textAlign: 'center',
                  whiteSpace: 'nowrap',
                  opacity: 0.9,
                }}
              >
                {card.title}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
}
