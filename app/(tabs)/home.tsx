import { CoughChart } from '@/components/CoughChart';
import React from 'react';

export default function HomePage() {
  const counts = [5, 12, 7, 0, 3, 9, 6];
  const labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

  const breakdown = [
    { wet: 2, dry: 3 },
    { wet: 7, dry: 5 },
    { wet: 3, dry: 4 },
    { wet: 0, dry: 0 },
    { wet: 1, dry: 2 },
    { wet: 4, dry: 5 },
    { wet: 2, dry: 4 },
  ];

  return <CoughChart counts={counts} labels={labels} breakdown={breakdown} />;
}
