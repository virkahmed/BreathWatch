import { Colors } from '@/constants/theme';
import { Box, Typography, Paper, Card, CardContent, Chip, Button, IconButton } from '@mui/material';
import { useRouter } from 'expo-router';
import React, { useState, useEffect } from 'react';
import { getRecordingHistory, RecordingHistoryItem, deleteRecording } from '@/services/storage';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

export default function HistoryPage() {
  const themeColors = Colors.dark;
  const router = useRouter();
  const [history, setHistory] = useState<RecordingHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const data = await getRecordingHistory();
      setHistory(data);
    } catch (error) {
      console.error('Failed to load history:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (sessionId: string) => {
    if (confirm('Are you sure you want to delete this processed log?')) {
      await deleteRecording(sessionId);
      await loadHistory();
    }
  };

  const handleViewDetails = (sessionId: string) => {
    router.push(`/analysis/${sessionId}`);
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatDuration = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = Math.floor(minutes % 60);
    if (hours > 0) {
      return `${hours}h ${mins}m`;
    }
    return `${mins}m`;
  };

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
        Past Processed Logs
      </Typography>

      {loading ? (
        <Typography variant="body1" align="center" sx={{ color: themeColors.text, mt: 4 }}>
          Loading history...
        </Typography>
      ) : history.length === 0 ? (
        <Paper
          sx={{
            p: 4,
            textAlign: 'center',
            background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
            borderRadius: '25px',
            boxShadow: `3px 3px 0 ${themeColors.text}`,
          }}
        >
          <Typography variant="h6" sx={{ mb: 2, color: themeColors.text }}>
            No Processed Logs Yet
          </Typography>
          <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.8 }}>
            Process audio to see your logs here
          </Typography>
        </Paper>
      ) : (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {history.map((item) => {
            const summary = item.summary;
            const severityColor =
              summary.display_strings?.severity_badge_color || '#666';
            
            return (
              <Card
                key={item.sessionId}
                sx={{
                  background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
                  borderRadius: '20px',
                  boxShadow: `3px 3px 0 ${themeColors.text}`,
                  border: `2px solid ${themeColors.text}`,
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
                    <Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <CalendarTodayIcon sx={{ fontSize: 18, color: themeColors.text }} />
                        <Typography variant="body2" sx={{ color: themeColors.text, fontWeight: 600 }}>
                          {formatDate(item.date)}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <AccessTimeIcon sx={{ fontSize: 18, color: themeColors.text }} />
                        <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.8 }}>
                          Duration: {formatDuration(summary.total_duration_minutes)}
                        </Typography>
                      </Box>
                    </Box>
                    <Chip
                      label={summary.display_strings?.severity_badge_color || 'Normal'}
                      size="small"
                      sx={{
                        backgroundColor: severityColor,
                        color: 'white',
                        fontWeight: 600,
                      }}
                    />
                  </Box>

                  <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
                    <Box>
                      <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                        Coughs/Hour
                      </Typography>
                      <Typography variant="h6" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                        {summary.coughs_per_hour.toFixed(1)}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                        Total Coughs
                      </Typography>
                      <Typography variant="h6" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                        {summary.cough_events?.length || 0}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                        Wheeze Time
                      </Typography>
                      <Typography variant="h6" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                        {summary.wheeze_time_percent.toFixed(1)}%
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                        Quality Score
                      </Typography>
                      <Typography variant="h6" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                        {summary.quality_metrics?.quality_score.toFixed(0) || 'N/A'}
                      </Typography>
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                    <Button
                      variant="contained"
                      startIcon={<VisibilityIcon />}
                      onClick={() => handleViewDetails(item.sessionId)}
                      sx={{
                        backgroundColor: themeColors.bright,
                        color: themeColors.background,
                        '&:hover': {
                          backgroundColor: themeColors.text,
                        },
                      }}
                    >
                      View Details
                    </Button>
                    <IconButton
                      onClick={() => handleDelete(item.sessionId)}
                      sx={{
                        color: '#ff4444',
                        '&:hover': {
                          backgroundColor: 'rgba(255, 68, 68, 0.1)',
                        },
                      }}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </CardContent>
              </Card>
            );
          })}
        </Box>
      )}
    </Box>
  );
}

