import { Box, Typography, useTheme } from '@mui/material';
import { BarChart } from '@mui/x-charts/BarChart';
import React from 'react';

interface CoughChartProps {
  counts: number[];
  labels: string[];
}

export const CoughChart: React.FC<CoughChartProps> = ({ counts, labels }) => {
  const theme = useTheme();

  return (
    <Box sx={{ width: '100%', maxWidth: 600, mx: 'auto', my: 4 }}>
      <Typography variant="h6" gutterBottom>
        Nightly Cough Count
      </Typography>

      <BarChart
        height={250}
        // width="100%"
        xAxis={[{ data: labels, scaleType: 'band' }]}
        series={[{ data: counts, label: 'Coughs', color: theme.palette.primary.main }]}
        spacing={0.3} // small gap between bars
        borderRadius={4} // rounded bars like Apple Health
      />
    </Box>
  );
};
