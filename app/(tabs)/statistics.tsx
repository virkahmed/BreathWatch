import { Colors } from '@/constants/theme';
import { Box, Typography, Paper, Card, CardContent } from '@mui/material';
import React, { useState, useEffect } from 'react';
import { getRecordingHistory, RecordingHistoryItem } from '@/services/storage';
import { LineChart } from '@mui/x-charts/LineChart';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

export default function StatisticsPage() {
  const themeColors = Colors.dark;
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

  // Calculate statistics
  const stats = {
    totalRecordings: history.length,
    avgCoughsPerHour: history.length > 0
      ? history.reduce((sum, item) => sum + item.summary.coughs_per_hour, 0) / history.length
      : 0,
    avgWheezeTime: history.length > 0
      ? history.reduce((sum, item) => sum + item.summary.wheeze_time_percent, 0) / history.length
      : 0,
    totalCoughs: history.reduce((sum, item) => sum + (item.summary.cough_events?.length || 0), 0),
    avgQualityScore: history.length > 0
      ? history.reduce((sum, item) => sum + (item.summary.quality_metrics?.quality_score || 0), 0) / history.length
      : 0,
  };

  // Prepare trend data (last 7 recordings)
  const recentHistory = history.slice(0, 7).reverse();
  const trendData = {
    dates: recentHistory.map((item) => {
      const date = new Date(item.date);
      return `${date.getMonth() + 1}/${date.getDate()}`;
    }),
    coughsPerHour: recentHistory.map((item) => item.summary.coughs_per_hour),
    wheezeTime: recentHistory.map((item) => item.summary.wheeze_time_percent),
  };

  // Calculate trend direction
  const getTrend = (values: number[]) => {
    if (values.length < 2) return null;
    const recent = values[values.length - 1];
    const previous = values[values.length - 2];
    return recent > previous ? 'up' : recent < previous ? 'down' : 'stable';
  };

  const coughTrend = getTrend(trendData.coughsPerHour);
  const wheezeTrend = getTrend(trendData.wheezeTime);

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
        Statistics & Trends
      </Typography>

      {loading ? (
        <Typography variant="body1" align="center" sx={{ color: themeColors.text, mt: 4 }}>
          Loading statistics...
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
            No Data Available
          </Typography>
          <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.8 }}>
            Process some audio to see statistics here
          </Typography>
        </Paper>
      ) : (
        <>
          {/* Summary Cards */}
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2, mb: 3 }}>
            <Card
              sx={{
                background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
                borderRadius: '20px',
                boxShadow: `3px 3px 0 ${themeColors.text}`,
              }}
            >
              <CardContent>
                <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                  Total Logs
                </Typography>
                <Typography variant="h4" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                  {stats.totalRecordings}
                </Typography>
              </CardContent>
            </Card>

            <Card
              sx={{
                background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
                borderRadius: '20px',
                boxShadow: `3px 3px 0 ${themeColors.text}`,
              }}
            >
              <CardContent>
                <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                  Avg Coughs/Hour
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="h4" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                    {stats.avgCoughsPerHour.toFixed(1)}
                  </Typography>
                  {coughTrend && (
                    <>
                      {coughTrend === 'up' ? (
                        <TrendingUpIcon sx={{ color: '#ff4444' }} />
                      ) : (
                        <TrendingDownIcon sx={{ color: '#4caf50' }} />
                      )}
                    </>
                  )}
                </Box>
              </CardContent>
            </Card>

            <Card
              sx={{
                background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
                borderRadius: '20px',
                boxShadow: `3px 3px 0 ${themeColors.text}`,
              }}
            >
              <CardContent>
                <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                  Avg Wheeze Time
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="h4" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                    {stats.avgWheezeTime.toFixed(1)}%
                  </Typography>
                  {wheezeTrend && (
                    <>
                      {wheezeTrend === 'up' ? (
                        <TrendingUpIcon sx={{ color: '#ff4444' }} />
                      ) : (
                        <TrendingDownIcon sx={{ color: '#4caf50' }} />
                      )}
                    </>
                  )}
                </Box>
              </CardContent>
            </Card>

            <Card
              sx={{
                background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
                borderRadius: '20px',
                boxShadow: `3px 3px 0 ${themeColors.text}`,
              }}
            >
              <CardContent>
                <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                  Total Coughs Detected
                </Typography>
                <Typography variant="h4" sx={{ color: themeColors.bright, fontWeight: 700 }}>
                  {stats.totalCoughs}
                </Typography>
              </CardContent>
            </Card>
          </Box>

          {/* Trend Charts */}
          {recentHistory.length >= 2 && (
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
                Recent Trends (Last 7 Logs)
              </Typography>
              <LineChart
                width={600}
                height={300}
                series={[
                  {
                    data: trendData.coughsPerHour,
                    label: 'Coughs/Hour',
                    color: themeColors.bright,
                  },
                ]}
                xAxis={[
                  {
                    scaleType: 'point',
                    data: trendData.dates,
                    tickLabelStyle: { fill: themeColors.text },
                  },
                ]}
                yAxis={[
                  {
                    tickLabelStyle: { fill: themeColors.text },
                  },
                ]}
                sx={{
                  '& .MuiChartsAxis-line': {
                    stroke: themeColors.text,
                  },
                  '& .MuiChartsAxis-tick': {
                    stroke: themeColors.text,
                  },
                }}
              />
            </Paper>
          )}

          {/* Quality Score Trend */}
          {recentHistory.length >= 2 && (
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
                Quality Score Trend
              </Typography>
              <LineChart
                width={600}
                height={300}
                series={[
                  {
                    data: recentHistory.map((item) => item.summary.quality_metrics?.quality_score || 0),
                    label: 'Quality Score',
                    color: '#4caf50',
                  },
                ]}
                xAxis={[
                  {
                    scaleType: 'point',
                    data: trendData.dates,
                    tickLabelStyle: { fill: themeColors.text },
                  },
                ]}
                yAxis={[
                  {
                    min: 0,
                    max: 100,
                    tickLabelStyle: { fill: themeColors.text },
                  },
                ]}
                sx={{
                  '& .MuiChartsAxis-line': {
                    stroke: themeColors.text,
                  },
                  '& .MuiChartsAxis-tick': {
                    stroke: themeColors.text,
                  },
                }}
              />
            </Paper>
          )}
        </>
      )}
    </Box>
  );
}

