import { CoughChart } from '@/components/CoughChart';
import { Paper, Typography, Box, CircularProgress, Chip, Divider } from '@mui/material';
import React, { useEffect, useMemo } from 'react';
import { useRecording } from '@/contexts/RecordingContext';
import { useRouter } from 'expo-router';
import { Colors } from '@/constants/theme';
import HistoryIcon from '@mui/icons-material/History';
import BarChartIcon from '@mui/icons-material/BarChart';
import SettingsIcon from '@mui/icons-material/Settings';
import HelpIcon from '@mui/icons-material/Help';
import InfoIcon from '@mui/icons-material/Info';
import { buildLocalInterpretation } from '@/utils/localInterpretation';
import HomeIcon from '@mui/icons-material/Home';
export default function HomePage() {
  const themeColors = Colors.dark;
  const router = useRouter();
  const { chunkResponses, finalSummary, isFetchingSummary, summaryProgress } = useRecording();

  const lastChunk = chunkResponses.length > 0 ? chunkResponses[chunkResponses.length - 1] : undefined;
  const timelineSource = finalSummary?.probability_timeline ?? lastChunk?.probability_timeline;
  const eventSummarySource = finalSummary?.event_summary ?? lastChunk?.event_summary;
  const summarySource = finalSummary ?? lastChunk;
  const totalCoughs =
    eventSummarySource?.num_events ??
    summarySource?.detected_events?.length ??
    summarySource?.cough_events?.length ??
    0;
  const attrSummary = useMemo(() => {
    if (finalSummary?.attribute_prevalence) {
      return finalSummary.attribute_prevalence;
    }
    if (timelineSource?.attr_series && timelineSource.p_cough.length > 0) {
      const { wet, wheezing, stridor, choking, congestion } = timelineSource.attr_series;
      const meanPct = (values: number[]) =>
        values.length ? (values.reduce((sum, val) => sum + val, 0) / values.length) * 100 : 0;
      return {
        wet: meanPct(wet),
        wheezing: meanPct(wheezing),
        stridor: meanPct(stridor),
        choking: meanPct(choking),
        congestion: meanPct(congestion),
      };
    }
    return null;
  }, [finalSummary?.attribute_prevalence, timelineSource]);
  const dedalus = finalSummary?.dedalus_interpretation;
  const fallbackInterpretation = useMemo(() => {
    if (dedalus) return null;
    if (!summarySource && !attrSummary) return null;
    return buildLocalInterpretation({
      coughCount: totalCoughs,
      wheezeCount: summarySource?.wheeze_windows ?? 0,
      wheezeProbability: ((finalSummary?.wheeze_time_percent ?? 0) / 100),
      attrWetPercent: attrSummary?.wet ?? 0,
      attrStridorPercent: attrSummary?.stridor ?? 0,
      attrChokingPercent: attrSummary?.choking ?? 0,
      attrCongestionPercent: attrSummary?.congestion ?? 0,
      attrWheezingPercent: attrSummary?.wheezing ?? 0,
    });
  }, [dedalus, summarySource, attrSummary, totalCoughs, finalSummary?.wheeze_time_percent]);
  const displayedInterpretation = dedalus ?? fallbackInterpretation;

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
      title: 'Home',
      description: 'Back to start',
      icon: <HomeIcon />,
      route: '/',
      color: themeColors.bright,
    },
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
      title: 'About',
      description: 'Learn more',
      icon: <InfoIcon />,
      route: '/about',
      color: themeColors.bright,
    },
    {
      title: 'Help',
      description: 'FAQ & docs',
      icon: <HelpIcon />,
      route: '/help',
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
      {/* Header */}
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography
          variant="h3"
          sx={{
            color: '#ffffff',
            fontWeight: 800,
            letterSpacing: '-0.03em',
            textShadow: '0 10px 30px rgba(0,0,0,0.35)',
          }}
        >
          BreathWatch
        </Typography>
        <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.7 }}>
          Nightly respiratory insights powered by real probabilities
        </Typography>
      </Box>

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
               border: `1px solid rgba(0, 212, 255, 0.3)`,
             }}
           >
             <CircularProgress size={24} sx={{ mb: 2, color: themeColors.bright }} />
             <Typography variant="body1" sx={{ color: themeColors.text, fontWeight: 600, mb: 1 }}>
               {summaryProgress || 'Generating final summary...'}
             </Typography>
             <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
               This may take up to 60 seconds depending on data size
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
            Timeline shows the latest tile-by-tile cough probabilities
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
          <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
            {finalSummary.event_summary?.num_events ?? finalSummary.cough_events?.length ?? 0} cough
            events detected
          </Typography>
        </Box>
      )}
      <CoughChart timeline={timelineSource} eventSummary={eventSummarySource} />
      {summarySource && (
        <Box
          sx={{
            background: 'rgba(0,0,0,0.2)',
            borderRadius: '16px',
            p: 3,
            mx: 2,
            mb: 3,
            border: `1px solid rgba(255,255,255,0.08)`,
          }}
        >
          <Typography variant="h6" sx={{ color: themeColors.text, fontWeight: 700, mb: 1 }}>
            Session Summary
          </Typography>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
              gap: 2,
              mb: 2,
            }}
          >
            <Box>
              <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                Total Coughs
              </Typography>
              <Typography variant="h4" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                {totalCoughs}
              </Typography>
            </Box>
            {finalSummary && (
              <Box>
                <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                  Coughs / Hour
                </Typography>
                <Typography variant="h4" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                  {finalSummary.coughs_per_hour.toFixed(1)}
                </Typography>
              </Box>
            )}
          </Box>
          {attrSummary && (
            <>
              <Divider sx={{ borderColor: 'rgba(255,255,255,0.15)', mb: 2 }} />
              <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                Average Attribute Probabilities
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                {[
                  { label: 'Wet', value: attrSummary.wet },
                  { label: 'Wheeze', value: attrSummary.wheezing },
                  { label: 'Stridor', value: attrSummary.stridor },
                  { label: 'Choking', value: attrSummary.choking },
                  { label: 'Congestion', value: attrSummary.congestion },
                ].map((attr) => (
                  <Chip
                    key={attr.label}
                    label={`${attr.label}: ${attr.value.toFixed(1)}%`}
                    sx={{
                      backgroundColor: 'rgba(255,255,255,0.08)',
                      color: themeColors.text,
                      fontWeight: 600,
                    }}
                  />
                ))}
              </Box>
            </>
          )}
          {displayedInterpretation ? (
            <>
              <Divider sx={{ borderColor: 'rgba(255,255,255,0.15)', mt: 2, mb: 1 }} />
              <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                AI Interpretation
              </Typography>
              {displayedInterpretation.severity && (
                <Chip
                  label={
                    fallbackInterpretation && !dedalus
                      ? `${displayedInterpretation.severity} (estimated)`
                      : displayedInterpretation.severity
                  }
                  size="small"
                  sx={{ mt: 1, backgroundColor: themeColors.bright, color: themeColors.background }}
                />
              )}
              <Typography variant="body2" sx={{ color: themeColors.text, mt: 1 }}>
                {displayedInterpretation.interpretation}
              </Typography>
              {displayedInterpretation.recommendations?.length ? (
                <Box component="ul" sx={{ mt: 1, color: themeColors.text, pl: 2 }}>
                  {displayedInterpretation.recommendations.map((rec, idx) => (
                    <li key={idx}>
                      <Typography variant="body2" sx={{ color: themeColors.text }}>
                        {rec}
                      </Typography>
                    </li>
                  ))}
                </Box>
              ) : null}
            </>
          ) : (
            <>
              <Divider sx={{ borderColor: 'rgba(255,255,255,0.15)', mt: 2, mb: 1 }} />
              <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                AI Interpretation
              </Typography>
              <Typography variant="body2" sx={{ color: themeColors.text, mt: 1, opacity: 0.8 }}>
                Dedalus is still running. The interpretation appears here once the final summary is ready.
              </Typography>
            </>
          )}
        </Box>
      )}
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
