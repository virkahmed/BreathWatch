export type InterpretationSeverity = 'normal' | 'mild' | 'moderate' | 'severe';

export interface InterpretationInput {
  coughCount: number;
  wheezeCount: number;
  wheezeProbability: number; // 0-1
  attrWetPercent: number;
  attrStridorPercent: number;
  attrChokingPercent: number;
  attrCongestionPercent: number;
  attrWheezingPercent: number;
}

export interface InterpretationResult {
  interpretation: string;
  severity: InterpretationSeverity;
  recommendations: string[];
  source: 'local';
}

export function buildLocalInterpretation({
  coughCount,
  wheezeCount,
  wheezeProbability,
  attrWetPercent,
  attrStridorPercent,
  attrChokingPercent,
  attrCongestionPercent,
  attrWheezingPercent,
}: InterpretationInput): InterpretationResult {
  let severity: InterpretationSeverity = 'normal';
  const interpretationParts: string[] = [];
  const recommendations: string[] = [];

  if (coughCount > 20) {
    interpretationParts.push(`Detected frequent coughing (${coughCount} events)`);
    recommendations.push('Consider consulting a healthcare provider if persistent');
    severity = 'moderate';
  } else if (coughCount > 10) {
    interpretationParts.push(`Detected moderate coughing (${coughCount} events)`);
    severity = severity === 'normal' ? 'mild' : severity;
  }

  if (wheezeCount > 0 || wheezeProbability > 0.7) {
    severity = severity === 'normal' ? 'moderate' : 'severe';
    interpretationParts.push('Significant wheeze patterns detected');
    recommendations.push('Wheezing may indicate airway inflammation or obstruction');
  } else if (wheezeProbability > 0.4) {
    interpretationParts.push('Mild wheeze patterns detected');
    severity = severity === 'normal' ? 'mild' : severity;
  }

  const attributeNotes: string[] = [];
  if (attrWetPercent > 30) {
    attributeNotes.push(`wet cough characteristics (${attrWetPercent.toFixed(1)}%)`);
    severity = severity === 'normal' ? 'mild' : severity;
    recommendations.push('Wet cough may indicate a productive respiratory condition');
  }
  if (attrStridorPercent > 20) {
    attributeNotes.push(`stridor-like sounds (${attrStridorPercent.toFixed(1)}%)`);
    severity = severity === 'normal' ? 'moderate' : severity;
    recommendations.push('Stridor may indicate upper airway obstruction—seek medical attention');
  }
  if (attrChokingPercent > 15) {
    attributeNotes.push(`choking episodes (${attrChokingPercent.toFixed(1)}%)`);
    severity = severity === 'normal' ? 'moderate' : 'severe';
    recommendations.push('Choking episodes require immediate medical evaluation');
  }
  if (attrCongestionPercent > 25) {
    attributeNotes.push(`congestion patterns (${attrCongestionPercent.toFixed(1)}%)`);
    if (!recommendations.some((r) => r.toLowerCase().includes('congestion'))) {
      recommendations.push('Nasal congestion may be contributing to respiratory symptoms');
    }
  }
  if (attrWheezingPercent > 20) {
    attributeNotes.push(`self-reported wheezing patterns (${attrWheezingPercent.toFixed(1)}%)`);
  }
  if (attributeNotes.length) {
    interpretationParts.push(`Cough attributes: ${attributeNotes.join(', ')}`);
  }

  let interpretation: string;
  if (interpretationParts.length) {
    interpretation = `${interpretationParts.join(' — ')}.`;
  } else {
    interpretation = 'No significant respiratory anomalies detected. Breathing patterns appear normal.';
    if (recommendations.length === 0) {
      recommendations.push('Monitor symptoms and consult a healthcare provider if they persist or worsen');
    }
  }

  return {
    interpretation,
    severity,
    recommendations,
    source: 'local',
  };
}
