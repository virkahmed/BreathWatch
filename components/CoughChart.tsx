import { Colors } from '@/constants/theme';
import { Box, Typography } from '@mui/material';
import { BarItemIdentifier } from '@mui/x-charts';
import { BarChart } from '@mui/x-charts/BarChart';
import { labelMarkClasses } from '@mui/x-charts/ChartsLabel';
import React from 'react';

interface CoughChartProps {
  counts: number[];
  labels: string[];
  breakdown?: { wet: number; dry: number }[]; // optional distribution per day
}

export const CoughChart: React.FC<CoughChartProps> = ({ counts, labels, breakdown }) => {
  const themeColors = Colors.dark;

  // Track selected day
  const [selectedIndex, setSelectedIndex] = React.useState<number | null>(null);

  const handleItemClick = (
    _event: React.MouseEvent<SVGElement, MouseEvent>,
    barItemIdentifier: BarItemIdentifier,
  ) => {
    console.log(1);
    setSelectedIndex(barItemIdentifier.dataIndex);
  };

  return (
    <Box
      sx={{
        width: '100%',
        minHeight: '100dvh',
        pb: '20px',
        mx: 'auto',
        my: 0,
        pt: '50px',
        background: `linear-gradient(-45deg, ${themeColors.background} 25%, ${themeColors.backgroundGradient})`,
        color: themeColors.text,
        fontFamily: Colors.typography.fontFamily,
        position: 'relative',
      }}
    >
      <Typography
        variant="h6"
        align="center"
        sx={{
          position: 'sticky',
          top: 0,
          zIndex: 10,
          py: 2,
          mb: 3,
          background: `linear-gradient(to bottom, ${themeColors.background} 50%, transparent)`,
          color: themeColors.text,
          fontWeight: 600,
        }}
      >
        Nightly Cough Count
      </Typography>

      {/* Total cough count chart */}
      <Box
        sx={{
          maxWidth: 'calc(100dvw - 40px)',
          marginLeft: '20px',
          my: '20px',
          background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
          borderRadius: '25px',
          boxShadow: `3px 3px 0 ${themeColors.text}`,
          padding: 0,
        }}
      >
        <BarChart
          sx={{
            height: '250px',
            maxWidth: 'calc(100dvw - 40px)',
            marginTop: '20px',
          }}
          xAxis={[
            {
              data: labels,
              scaleType: 'band',
              tickLabelStyle: { fill: themeColors.text },
              labelStyle: { fill: themeColors.text },
              sx: {
                '& .MuiChartsAxis-line': {
                  stroke: themeColors.text,
                },
                '& .MuiChartsAxis-tick': {
                  stroke: themeColors.text,
                },
              },
            },
          ]}
          yAxis={[
            {
              tickLabelStyle: { fill: themeColors.text },
              labelStyle: { fill: themeColors.text },
              sx: {
                '& .MuiChartsAxis-line': {
                  stroke: themeColors.text,
                },
                '& .MuiChartsAxis-tick': {
                  stroke: themeColors.text,
                },
              },
            },
          ]}
          series={[
            {
              data: counts,
              label: 'Coughs',
              color: themeColors.bright,
            },
          ]}
          spacing={0.3}
          borderRadius={4}
          onItemClick={handleItemClick}
          grid={{ vertical: true, horizontal: true }}
          slotProps={{
            tooltip: { trigger: 'none' },
            legend: {
              sx: {
                color: themeColors.text,
                [`.${labelMarkClasses.fill}`]: {
                  fill: themeColors.text,
                },
              },
            },
          }}
          layout="vertical"
        />
      </Box>
      <Box
        sx={{
          maxWidth: 'calc(100dvw - 40px)',
          marginLeft: '20px',
          my: '20px',
          background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
          borderRadius: '25px',
          boxShadow: `3px 3px 0 ${themeColors.text}`,
          padding: 0,
        }}
      >
        {/* Breakdown chart */}
        <Box>
          {selectedIndex === null ? (
            <Typography variant="body1" align="center" sx={{ py: 10, color: themeColors.text }}>
              Select a day
            </Typography>
          ) : breakdown && breakdown[selectedIndex] ? (
            <BarChart
              height={150}
              layout="horizontal"
              xAxis={[
                {
                  tickLabelStyle: { fill: themeColors.text },
                  labelStyle: { fill: themeColors.text },
                  sx: {
                    '& .MuiChartsAxis-line': {
                      stroke: themeColors.text,
                    },
                    '& .MuiChartsAxis-tick': {
                      stroke: themeColors.text,
                    },
                  },
                },
              ]}
              yAxis={[
                {
                  data: ['Wet', 'Dry'],
                  scaleType: 'band',
                  tickLabelStyle: { fill: themeColors.text },
                  labelStyle: { fill: themeColors.text },
                  sx: {
                    '& .MuiChartsAxis-line': {
                      stroke: themeColors.text,
                    },
                    '& .MuiChartsAxis-tick': {
                      stroke: themeColors.text,
                    },
                  },
                },
              ]}
              series={[
                {
                  data: [breakdown[selectedIndex].wet, breakdown[selectedIndex].dry],
                  label: 'Cough Distribution',
                  color: themeColors.text,
                },
              ]}
              spacing={0.3}
              grid={{ vertical: true, horizontal: true }}
              slotProps={{
                tooltip: { trigger: 'none' },
                legend: {
                  sx: {
                    color: themeColors.text,
                    [`.${labelMarkClasses.fill}`]: {
                      fill: themeColors.text,
                    },
                  },
                },
              }}
            />
          ) : (
            <Typography variant="body1" align="center" sx={{ py: 10, color: themeColors.text }}>
              No data available
            </Typography>
          )}
        </Box>
      </Box>
      <Box
        sx={{
          maxWidth: 'calc(100dvw - 40px)',
          marginLeft: '20px',
          my: '20px',
          background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
          borderRadius: '25px',
          boxShadow: `3px 3px 0 ${themeColors.text}`,
          padding: 0,
        }}
      >
        <Typography sx={{ m: '20px', pt: '20px', color: themeColors.text, fontWeight: 600 }}>
          You should know...
        </Typography>
        <Typography sx={{ m: '20px', color: themeColors.text }}>
          Wet coughs, also known as productive coughs, can indicate that the body is trying to clear
          mucus or phlegm from the airways, but they may also signal an underlying infection or
          respiratory issue. While an occasional wet cough can result from a mild cold, persistent
          or worsening symptoms could point to bronchitis, pneumonia, or other conditions that
          require medical attention. The presence of discolored or bloody mucus, chest pain, or
          shortness of breath can further increase concern. Ignoring a wet cough may allow
          infections to spread or worsen, so monitoring its duration and severity is important for
          protecting lung health.
        </Typography>
      </Box>
    </Box>
  );
};
