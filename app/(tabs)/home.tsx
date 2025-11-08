import { CoughChart } from '@/components/CoughChart';
import React from 'react';

export default function HomePage() {
  // Example: nightly cough counts for a week
  const coughCounts = [5, 12, 7, 0, 3, 9, 6];
  const labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

  return <CoughChart counts={coughCounts} labels={labels} />;
}
