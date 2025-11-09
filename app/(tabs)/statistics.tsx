import { Colors } from '@/constants/theme';
import { Box, Typography, Paper, Card, CardContent, Chip, IconButton } from '@mui/material';
import React, { useState, useEffect, useMemo } from 'react';
import { getRecordingHistory, RecordingHistoryItem } from '@/services/storage';
import { LineChart } from '@mui/x-charts/LineChart';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import { useRecording } from '@/contexts/RecordingContext';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { useRouter } from 'expo-router';
import { buildLocalInterpretation } from '@/utils/localInterpretation';

export default function StatisticsPage() {
  const themeColors = Colors.dark;
  const [history, setHistory] = useState<RecordingHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const { finalSummary, chunkResponses } = useRecording();
  const router = useRouter();

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

  const chunkPreview = useMemo(() => {
    if (chunkResponses.length === 0) {
      return null;
    }
    const lastChunk = chunkResponses[chunkResponses.length - 1];
    const events = chunkResponses.flatMap((chunk) => chunk.event_summary?.events || []);
    const timeline = lastChunk.probability_timeline;
    const mean = (values: number[] = []) =>
      values.length ? (values.reduce((sum, val) => sum + val, 0) / values.length) * 100 : 0;
    const attrSeries = timeline?.attr_series;
    const attributePrevalence = attrSeries
      ? {
          wet: mean(attrSeries.wet),
          wheezing: mean(attrSeries.wheezing),
          stridor: mean(attrSeries.stridor),
          choking: mean(attrSeries.choking),
          congestion: mean(attrSeries.congestion),
        }
      : undefined;
    const durationMinutes =
      timeline && timeline.times.length
        ? ((timeline.times[timeline.times.length - 1] ?? 0) + (timeline.tile_seconds ?? 1)) / 60
        : chunkResponses.length * 10;
    const coughsPerHour = durationMinutes > 0 ? (events.length / durationMinutes) * 60 : events.length;
    return {
      summary: {
        session_id: lastChunk.session_id ?? 'current',
        coughs_per_hour: coughsPerHour,
        wheeze_time_percent: 0,
        event_summary: { num_events: events.length, events },
        probability_timeline: timeline,
        attribute_prevalence:
          attributePrevalence ??
          { wet: 0, wheezing: 0, stridor: 0, choking: 0, congestion: 0 },
        quality_metrics: {
          avg_snr: 0,
          quality_score: 0,
          low_quality_periods_count: 0,
          high_confidence_events_count: 0,
          suppressed_events_count: 0,
        },
        display_strings: {
          sleep_duration_formatted: '',
          coughs_per_hour_formatted: '',
          severity_badge_color: '',
          overall_quality_score: 0,
        },
      },
      sessionId: lastChunk.session_id ?? 'current',
      date: new Date().toISOString(),
      live: true,
    };
  }, [chunkResponses]);

  const combinedSummaries = useMemo(() => {
    const stored = history.map((item) => ({
      summary: item.summary,
      sessionId: item.sessionId,
      date: item.date,
      live: false,
    }));
    const entries = [...stored];
    if (finalSummary) {
      entries.unshift({
        summary: finalSummary,
        sessionId: finalSummary.session_id ?? 'current',
        date: new Date().toISOString(),
        live: true,
      });
    } else if (chunkPreview) {
      entries.unshift(chunkPreview);
    }
    return entries;
  }, [history, finalSummary, chunkPreview]);

  const summaryPool = combinedSummaries.map((entry) => entry.summary);

  // Calculate statistics
  const stats = {
    totalRecordings: summaryPool.length,
    avgCoughsPerHour:
      summaryPool.length > 0
        ? summaryPool.reduce((sum, summary) => sum + summary.coughs_per_hour, 0) / summaryPool.length
        : 0,
    avgWheezeTime:
      summaryPool.length > 0
        ? summaryPool.reduce((sum, summary) => sum + summary.wheeze_time_percent, 0) / summaryPool.length
        : 0,
    totalCoughs: summaryPool.reduce((sum, summary) => {
      if (summary.event_summary?.num_events !== undefined) {
        return sum + summary.event_summary.num_events;
      }
      return sum + (summary.cough_events?.length || 0);
    }, 0),
    avgQualityScore:
      summaryPool.length > 0
        ? summaryPool.reduce((sum, summary) => sum + (summary.quality_metrics?.quality_score || 0), 0) /
          summaryPool.length
        : 0,
  };

  // Prepare trend data (last 7 recordings)
  const recentHistory = combinedSummaries.slice(0, 7).reverse();
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
  const primaryEntry = combinedSummaries[0];
  const primarySummary = primaryEntry?.summary;
  const attrSummary = primarySummary?.attribute_prevalence;
  const dedalus = primarySummary?.dedalus_interpretation;
  const fallbackInterpretation = useMemo(() => {
    if (!primarySummary || dedalus) return null;
    return buildLocalInterpretation({
      coughCount: primarySummary.event_summary?.num_events ?? primarySummary.cough_events?.length ?? 0,
      wheezeCount: 0,
      wheezeProbability: (primarySummary.wheeze_time_percent ?? 0) / 100,
      attrWetPercent: attrSummary?.wet ?? 0,
      attrStridorPercent: attrSummary?.stridor ?? 0,
      attrChokingPercent: attrSummary?.choking ?? 0,
      attrCongestionPercent: attrSummary?.congestion ?? 0,
      attrWheezingPercent: attrSummary?.wheezing ?? 0,
    });
  }, [primarySummary, attrSummary, dedalus]);
  const displayedInterpretation = dedalus ?? fallbackInterpretation;
  const latestEvents = primarySummary?.event_summary?.events || [];
  const allEventEntries = combinedSummaries.flatMap((entry) =>
    (entry.summary.event_summary?.events || []).map((event, idx) => ({
      event,
      sessionId: entry.sessionId,
      recordedAt: entry.live ? 'Current session' : new Date(entry.date).toLocaleString(),
      index: idx,
      live: entry.live,
    }))
  );

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: `linear-gradient(-45deg, ${themeColors.background} 25%, ${themeColors.backgroundGradient})`,
        color: themeColors.text,
        p: 3,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
        <IconButton onClick={() => router.replace('/')} sx={{ color: themeColors.text }}>
          <ArrowBackIcon />
        </IconButton>
        <Typography variant="h4" sx={{ fontWeight: 700, color: themeColors.text }}>
          Statistics & Trends
        </Typography>
      </Box>

      {loading ? (
        <Typography variant="body1" align="center" sx={{ color: themeColors.text, mt: 4 }}>
          Loading statistics...
        </Typography>
      ) : summaryPool.length === 0 ? (
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

          {primarySummary && (
            <Paper
              sx={{
                p: 3,
                mb: 3,
                background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
                borderRadius: '20px',
                boxShadow: `3px 3px 0 ${themeColors.text}`,
              }}
            >
              <Typography variant="h6" sx={{ color: themeColors.text, fontWeight: 600 }}>
                Session Overview
              </Typography>
              <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.75 }}>
                Session {primaryEntry?.sessionId} {primaryEntry?.live ? '(current)' : ''}
              </Typography>
              {attrSummary && (
                <>
                  <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7, mt: 2, display: 'block' }}>
                    Average attribute likelihoods
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
                          backgroundColor: 'rgba(0,0,0,0.25)',
                          color: themeColors.text,
                          fontWeight: 600,
                        }}
                      />
                    ))}
                  </Box>
                </>
              )}
              <Box sx={{ mt: 2 }}>
                <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                  AI Interpretation
                </Typography>
                {displayedInterpretation ? (
                  <>
                    {displayedInterpretation.severity && (
                      <Chip
                        label={
                          fallbackInterpretation && !dedalus
                            ? `${displayedInterpretation.severity} (estimated)`
                            : displayedInterpretation.severity
                        }
                        size="small"
                        sx={{
                          mt: 1,
                          backgroundColor: themeColors.bright,
                          color: themeColors.background,
                          fontWeight: 600,
                        }}
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
                  <Typography variant="body2" sx={{ color: themeColors.text, mt: 1, opacity: 0.8 }}>
                    Interpretation data will appear here once enough breathing data is available.
                  </Typography>
                )}
              </Box>
            </Paper>
          )}

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

          {primarySummary && (
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
                Most Recent Cough Events
              </Typography>
              {latestEvents.length === 0 ? (
                <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.8 }}>
                  No cough events recorded in the latest session.
                </Typography>
              ) : (
                latestEvents.slice(0, 6).map((event, idx) => (
                  <Box
                    key={`${event.start}-${idx}`}
                    sx={{
                      mb: 2,
                      p: 2,
                      borderRadius: '16px',
                      border: `1px solid rgba(255,255,255,0.15)`,
                      background: 'rgba(0,0,0,0.15)',
                    }}
                  >
                    <Typography variant="subtitle2" sx={{ color: themeColors.text, fontWeight: 600 }}>
                      Event {idx + 1}: {event.duration.toFixed(2)}s
                    </Typography>
                    <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.8 }}>
                      Tiles {event.tile_indices[0]}–{event.tile_indices[event.tile_indices.length - 1]}
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                      {[
                        { label: 'Wet', value: event.attr_probs.wet, flagged: event.attr_flags.wet },
                        { label: 'Wheeze', value: event.attr_probs.wheezing, flagged: event.attr_flags.wheezing },
                        { label: 'Stridor', value: event.attr_probs.stridor, flagged: event.attr_flags.stridor },
                        { label: 'Choking', value: event.attr_probs.choking, flagged: event.attr_flags.choking },
                        { label: 'Congestion', value: event.attr_probs.congestion, flagged: event.attr_flags.congestion },
                      ].map((attr) => (
                        <Chip
                          key={attr.label}
                          label={`${attr.label} ${(attr.value * 100).toFixed(0)}%`}
                          size="small"
                          sx={{
                            backgroundColor: attr.flagged ? themeColors.bright : 'rgba(255,255,255,0.08)',
                            color: attr.flagged ? themeColors.background : themeColors.text,
                            fontWeight: attr.flagged ? 700 : 500,
                          }}
                        />
                      ))}
                    </Box>
                  </Box>
                ))
              )}
              {latestEvents.length > 6 && (
                <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
                  Showing first 6 of {latestEvents.length} events. Open a recording for full details.
                </Typography>
              )}
            </Paper>
          )}

          {allEventEntries.length > 0 && (
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
                Event History ({allEventEntries.length})
              </Typography>
              <Box
                sx={{
                  maxHeight: 320,
                  overflowY: 'auto',
                  pr: 1,
                }}
              >
                {allEventEntries.map(({ event, recordedAt, sessionId }, idx) => (
                  <Box
                    key={`${sessionId}-${idx}`}
                    sx={{
                      mb: 2,
                      p: 2,
                      borderRadius: '16px',
                      border: `1px solid rgba(255,255,255,0.15)`,
                      background: 'rgba(0,0,0,0.15)',
                    }}
                  >
                    <Typography variant="subtitle2" sx={{ color: themeColors.text, fontWeight: 600 }}>
                      {recordedAt} • Session {sessionId}
                    </Typography>
                    <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.8 }}>
                      Length {event.duration.toFixed(2)}s • tiles {event.tile_indices[0]}–
                      {event.tile_indices[event.tile_indices.length - 1]}
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                      {[
                        { label: 'Wet', value: event.attr_probs.wet, flagged: event.attr_flags.wet },
                        { label: 'Wheeze', value: event.attr_probs.wheezing, flagged: event.attr_flags.wheezing },
                        { label: 'Stridor', value: event.attr_probs.stridor, flagged: event.attr_flags.stridor },
                        { label: 'Choking', value: event.attr_probs.choking, flagged: event.attr_flags.choking },
                        { label: 'Congestion', value: event.attr_probs.congestion, flagged: event.attr_flags.congestion },
                      ].map((attr) => (
                        <Chip
                          key={attr.label}
                          label={`${attr.label} ${(attr.value * 100).toFixed(0)}%`}
                          size="small"
                          sx={{
                            backgroundColor: attr.flagged ? themeColors.bright : 'rgba(255,255,255,0.08)',
                            color: attr.flagged ? themeColors.background : themeColors.text,
                            fontWeight: attr.flagged ? 700 : 500,
                          }}
                        />
                      ))}
                    </Box>
                  </Box>
                ))}
              </Box>
            </Paper>
          )}
        </>
      )}
    </Box>
  );
}
