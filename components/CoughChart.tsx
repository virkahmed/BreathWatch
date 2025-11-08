import { Box, Typography, useTheme } from '@mui/material';
import { BarChart } from '@mui/x-charts/BarChart';
import React from 'react';

interface CoughChartProps {
  counts: number[];
  labels: string[];
  breakdown?: { wet: number; dry: number }[]; // optional distribution per day
}

export const CoughChart: React.FC<CoughChartProps> = ({ counts, labels, breakdown }) => {
  const theme = useTheme();

  // Track selected day
  const [selectedIndex, setSelectedIndex] = React.useState<number | null>(null);

  const handleItemClick = (
    _event: React.MouseEvent<SVGElement, MouseEvent>,
    barItemIdentifier: { seriesIndex: number; dataIndex: number },
  ) => {
    setSelectedIndex(barItemIdentifier.dataIndex);
  };

  return (
    <Box sx={{ width: '100%', maxWidth: 600, mx: 'auto', my: 4 }}>
      <Typography variant="h6" gutterBottom>
        Nightly Cough Count
      </Typography>

      {/* Vertical bar chart for total coughs */}
      <BarChart
        height={250}
        xAxis={[{ data: labels, scaleType: 'band' }]}
        series={[{ data: counts, label: 'Coughs', color: theme.palette.primary.main }]}
        spacing={0.3}
        borderRadius={4}
        onItemClick={handleItemClick}
        layout="vertical"
      />

      {/* Horizontal breakdown chart */}
      <Box sx={{ mt: 6 }}>
        {selectedIndex === null ? (
          <Typography
            variant="body1"
            align="center"
            sx={{ py: 10, color: theme.palette.text.secondary }}
          >
            Select a day
          </Typography>
        ) : breakdown && breakdown[selectedIndex] ? (
          <BarChart
            height={150}
            layout="horizontal" // bars go right
            yAxis={[{ data: ['Wet', 'Dry'], scaleType: 'band' }]} // categorical y-axis
            series={[
              {
                data: [breakdown[selectedIndex].wet, breakdown[selectedIndex].dry],
                label: 'Cough Distribution',
                color: theme.palette.secondary.main,
              },
            ]}
            spacing={0.3}
          />
        ) : (
          <Typography
            variant="body1"
            align="center"
            sx={{ py: 10, color: theme.palette.text.secondary }}
          >
            No data available
          </Typography>
        )}
      </Box>
    </Box>
  );
};
