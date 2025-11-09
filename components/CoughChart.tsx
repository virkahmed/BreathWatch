import React, { useMemo } from 'react';
import { Box, Typography, Chip } from '@mui/material';
import { LineChart } from '@mui/x-charts/LineChart';
import { Colors } from '@/constants/theme';
import { EventSummary, ProbabilityTimeline } from '@/services/api';

interface CoughChartProps {
  timeline?: ProbabilityTimeline;
  eventSummary?: EventSummary;
  threshold?: number;
}

export const CoughChart: React.FC<CoughChartProps> = ({
  timeline,
  eventSummary,
  threshold = 0.5,
}) => {
  const themeColors = Colors.dark;
  const events = eventSummary?.events ?? [];
  const hasTimeline = Boolean(timeline && timeline.times.length > 0);

  const { xData, yData, highlightData, thresholdLine } = useMemo(() => {
    if (!timeline || timeline.times.length === 0) {
      return {
        xData: [] as number[],
        yData: [] as number[],
        highlightData: [] as (number | null)[],
        thresholdLine: [] as number[],
      };
    }

    const highlight = timeline.p_cough.map((value) => (value >= threshold ? value : null));

    return {
      xData: timeline.times,
      yData: timeline.p_cough,
      highlightData: highlight,
      thresholdLine: timeline.p_cough.map(() => threshold),
    };
  }, [timeline, threshold]);

  return (
    <Box
      sx={{
        width: '100%',
        maxWidth: 'calc(100dvw - 40px)',
        marginLeft: '20px',
        my: '30px',
        background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
        borderRadius: '25px',
        boxShadow: `3px 3px 0 ${themeColors.text}`,
        padding: 3,
      }}
    >
      <Typography
        variant="h6"
        sx={{ color: themeColors.text, fontWeight: 700, mb: 1 }}
      >
        Cough Timeline
      </Typography>
      <Typography
        variant="body2"
        sx={{ color: themeColors.text, opacity: 0.75, mb: 2 }}
      >
        {hasTimeline
          ? `${timeline!.p_cough.length} tiles • stride ${timeline!.stride_seconds}s`
          : 'Timeline will appear once the first chunk finishes processing.'}
      </Typography>
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
        <ChipLegend color={themeColors.bright} label="Cough intensity" />
        <ChipLegend color="#ff6b9f" label="Cough tracking" />
        <ChipLegend color="rgba(255,255,255,0.6)" label="Cough threshold" outlined />
      </Box>

      {hasTimeline ? (
        <LineChart
          height={260}
          xAxis={[
            {
              data: xData,
              label: '',
              valueFormatter: (value) =>
                value >= 60 ? `${(value / 60).toFixed(1)}m` : `${value.toFixed(0)}s`,
              tickLabelStyle: { fill: '#ffffff' },
              labelStyle: { fill: '#ffffff' },
              sx: {
                '& .MuiChartsAxis-line': { stroke: '#ffffff' },
                '& .MuiChartsAxis-tick': { stroke: '#ffffff' },
              },
            },
          ]}
          yAxis={[
            {
              min: 0,
              max: 1,
              label: '',
              tickLabelStyle: { fill: '#ffffff', opacity: 0.2 },
              labelStyle: { fill: 'transparent' },
              sx: {
                '& .MuiChartsAxis-line': { stroke: 'rgba(255,255,255,0.2)' },
                '& .MuiChartsAxis-tick': { stroke: 'transparent' },
              },
            },
          ]}
          series={[
            {
              id: 'intensity',
              data: yData,
              label: 'Cough intensity',
              color: themeColors.bright,
              showMark: false,
              curve: 'linear',
            },
            {
              id: 'events',
              data: highlightData,
              label: 'Cough tracking',
              color: '#ff6b9f',
              showMark: false,
              curve: 'linear',
              lineStyle: { lineWidth: 3 },
            },
            {
              id: 'threshold',
              data: thresholdLine,
              label: 'Threshold',
              color: 'rgba(255,255,255,0.25)',
              showMark: false,
              curve: 'linear',
              lineStyle: { strokeDasharray: '6 6' },
            },
          ]}
          slotProps={{
            legend: { hidden: true },
            tooltip: { trigger: 'none' },
          }}
        />
      ) : (
        <Box
          sx={{
            height: 200,
            borderRadius: '16px',
            border: `1px dashed rgba(255,255,255,0.3)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.7 }}>
            Timeline will appear once the first chunk is processed.
          </Typography>
        </Box>
      )}

      <Box sx={{ mt: 3 }}>
        <Typography
          variant="subtitle1"
          sx={{ color: themeColors.text, fontWeight: 600, mb: 1 }}
        >
          Detected Cough Episodes ({eventSummary?.num_events ?? 0})
        </Typography>
        {events.length === 0 ? (
          <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.7 }}>
            No cough-like events exceeded the baseline yet. Keep recording to capture more data.
          </Typography>
        ) : (
          <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.7 }}>
            Tap the Statistics tab for detailed attributes per cough.
          </Typography>
        )}
      </Box>
    </Box>
  );
};

interface ChipLegendProps {
  color: string;
  label: string;
  outlined?: boolean;
}

const ChipLegend: React.FC<ChipLegendProps> = ({ color, label, outlined = false }) => (
  <Chip
    label={label}
    size="small"
    sx={{
      backgroundColor: outlined ? 'transparent' : color,
      border: outlined ? `1px solid ${color}` : 'none',
      '& .MuiChip-label': {
        color: '#ffffff',
        fontWeight: 600,
      },
    }}
  />
);
