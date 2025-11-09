import { Colors } from '@/constants/theme';
import { Box, Typography, Paper, Chip, Divider } from '@mui/material';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useState, useEffect, useCallback } from 'react';
import { getRecordingBySessionId, RecordingHistoryItem } from '@/services/storage';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import IconButton from '@mui/material/IconButton';
import { CoughChart } from '@/components/CoughChart';

export default function AnalysisPage() {
  const themeColors = Colors.dark;
  const router = useRouter();
  const { sessionId } = useLocalSearchParams<{ sessionId: string }>();
  const [recording, setRecording] = useState<RecordingHistoryItem | null>(null);
  const [loading, setLoading] = useState(true);

  const loadRecording = useCallback(async () => {
    try {
      const data = await getRecordingBySessionId(sessionId!);
      setRecording(data);
    } catch (error) {
      console.error('Failed to load recording:', error);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    if (sessionId) {
      loadRecording();
    }
  }, [sessionId, loadRecording]);

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
          Loading analysis...
        </Typography>
      </Box>
    );
  }

  if (!recording) {
    return (
      <Box
        sx={{
          minHeight: '100vh',
          background: `linear-gradient(-45deg, ${themeColors.background} 25%, ${themeColors.backgroundGradient})`,
          color: themeColors.text,
          p: 3,
        }}
      >
        <IconButton
          onClick={() => router.back()}
          sx={{ color: themeColors.text, mb: 2 }}
        >
          <ArrowBackIcon />
        </IconButton>
        <Typography variant="h6" sx={{ color: themeColors.text, textAlign: 'center' }}>
          Processed log not found
        </Typography>
      </Box>
    );
  }

  const summary = recording.summary;
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const timeline = summary.probability_timeline;
  const eventSummary = summary.event_summary;

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: `linear-gradient(-45deg, ${themeColors.background} 25%, ${themeColors.backgroundGradient})`,
        color: themeColors.text,
      }}
    >
      <Box sx={{ p: 3 }}>
        <IconButton
          onClick={() => router.back()}
          sx={{ color: themeColors.text, mb: 2 }}
        >
          <ArrowBackIcon />
        </IconButton>

        <Typography variant="h4" sx={{ mb: 1, fontWeight: 700, color: themeColors.text }}>
          Processed Log Analysis
        </Typography>
        <Typography variant="body2" sx={{ mb: 3, color: themeColors.text, opacity: 0.8 }}>
          {formatDate(recording.date)}
        </Typography>

        {/* Key Metrics */}
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
            Key Metrics
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 2 }}>
            <Box>
              <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                Coughs per Hour
              </Typography>
              <Typography variant="h5" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                {summary.coughs_per_hour.toFixed(1)}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                Cough Events
              </Typography>
              <Typography variant="h5" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                {eventSummary?.num_events ?? summary.cough_events?.length ?? 0}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                Total Duration
              </Typography>
              <Typography variant="h5" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                {Math.floor(summary.total_duration_minutes / 60)}h {Math.floor(summary.total_duration_minutes % 60)}m
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                Wheeze Time
              </Typography>
              <Typography variant="h5" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                {summary.wheeze_time_percent.toFixed(1)}%
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                Quality Score
              </Typography>
              <Typography variant="h5" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                {summary.quality_metrics?.quality_score.toFixed(0) || 'N/A'}
              </Typography>
            </Box>
          </Box>
        </Paper>

        {/* Charts */}
        <CoughChart timeline={timeline} eventSummary={eventSummary} />

        {/* Attribute Prevalence */}
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
            Cough Attributes
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {summary.attribute_prevalence.wet > 0 && (
              <Chip
                label={`Wet: ${summary.attribute_prevalence.wet.toFixed(1)}%`}
                sx={{ backgroundColor: themeColors.bright, color: themeColors.background }}
              />
            )}
            {summary.attribute_prevalence.stridor > 0 && (
              <Chip
                label={`Stridor: ${summary.attribute_prevalence.stridor.toFixed(1)}%`}
                sx={{ backgroundColor: themeColors.bright, color: themeColors.background }}
              />
            )}
            {summary.attribute_prevalence.choking > 0 && (
              <Chip
                label={`Choking: ${summary.attribute_prevalence.choking.toFixed(1)}%`}
                sx={{ backgroundColor: themeColors.bright, color: themeColors.background }}
              />
            )}
            {summary.attribute_prevalence.congestion > 0 && (
              <Chip
                label={`Congestion: ${summary.attribute_prevalence.congestion.toFixed(1)}%`}
                sx={{ backgroundColor: themeColors.bright, color: themeColors.background }}
              />
            )}
            {summary.attribute_prevalence.wheezing > 0 && (
              <Chip
                label={`Wheezing: ${summary.attribute_prevalence.wheezing.toFixed(1)}%`}
                sx={{ backgroundColor: themeColors.bright, color: themeColors.background }}
              />
            )}
          </Box>
        </Paper>

        {/* Dedalus Interpretation */}
        {summary.dedalus_interpretation && (
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
              AI Interpretation
            </Typography>
            {summary.dedalus_interpretation.severity && (
              <Chip
                label={summary.dedalus_interpretation.severity}
                sx={{
                  mb: 2,
                  backgroundColor: summary.display_strings?.severity_badge_color || themeColors.bright,
                  color: 'white',
                  fontWeight: 600,
                }}
              />
            )}
            <Typography variant="body1" sx={{ mb: 2, color: themeColors.text }}>
              {summary.dedalus_interpretation.interpretation}
            </Typography>
            {summary.dedalus_interpretation.recommendations && (
              <Box>
                <Typography variant="subtitle2" sx={{ mb: 1, color: themeColors.text, fontWeight: 600 }}>
                  Recommendations:
                </Typography>
                <Box component="ul" sx={{ pl: 2, color: themeColors.text }}>
                  {summary.dedalus_interpretation.recommendations.map((rec, idx) => (
                    <li key={idx}>
                      <Typography variant="body2" sx={{ color: themeColors.text }}>
                        {rec}
                      </Typography>
                    </li>
                  ))}
                </Box>
              </Box>
            )}
          </Paper>
        )}

        {/* Pattern Scores */}
        {summary.pattern_scores && summary.pattern_scores.length > 0 && (
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
              Pattern Analysis
            </Typography>
            {summary.pattern_scores.map((pattern, idx) => (
              <Box key={idx} sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="subtitle1" sx={{ color: themeColors.text, fontWeight: 600 }}>
                    {pattern.pattern_name}
                  </Typography>
                  <Typography variant="body2" sx={{ color: themeColors.bright, fontWeight: 600 }}>
                    {(pattern.score * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.8, fontSize: '0.85rem' }}>
                  {pattern.why}
                </Typography>
                <Divider sx={{ mt: 1, borderColor: themeColors.text, opacity: 0.2 }} />
              </Box>
            ))}
          </Paper>
        )}
      </Box>
    </Box>
  );
}
