import { Colors } from '@/constants/theme';
import { Box, Typography, Paper, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import HelpIcon from '@mui/icons-material/Help';
import InfoIcon from '@mui/icons-material/Info';
import QuestionAnswerIcon from '@mui/icons-material/QuestionAnswer';

export default function HelpPage() {
  const themeColors = Colors.dark;

  const faqs = [
    {
      question: 'How do I start a recording?',
      answer:
        'Click the microphone button (FAB) in the bottom right corner of the screen. The button will turn red and show a stop icon when recording. A red banner at the top will show your recording status and duration.',
    },
    {
      question: 'How long should I record?',
      answer:
        'Recordings are processed in 10-minute chunks. For best results, record for at least 30 minutes to get meaningful data. Longer recordings (several hours) provide more comprehensive analysis.',
    },
    {
      question: 'What does the cough chart show?',
      answer:
        'The cough chart displays the number of coughs detected over time. You can click on any bar to see a breakdown of wet vs dry coughs for that time period. The chart updates in real-time as your recording is processed.',
    },
    {
      question: 'What is a wheeze window?',
      answer:
        'A wheeze window is a 10-second period where wheezing was detected in the audio. The app tracks the percentage of time with wheezing, which can help identify respiratory conditions.',
    },
    {
      question: 'What do the quality scores mean?',
      answer:
        'Quality scores (0-100) indicate the reliability of the recording. Higher scores mean better audio quality and more accurate detection. Scores above 70 are considered good quality.',
    },
    {
      question: 'How do I view past processed logs?',
      answer:
        'Go to the History page to see all your past processed logs. You can view detailed analysis, delete old logs, or see statistics over time.',
    },
    {
      question: 'What is the AI interpretation?',
      answer:
        'The AI interpretation uses advanced pattern recognition to analyze your respiratory data and provide insights about potential health patterns. It includes severity assessment and recommendations.',
    },
    {
      question: 'Can I use this app offline?',
      answer:
        'Yes! Audio processing happens locally on your device. You only need an internet connection when fetching the final AI interpretation from the backend.',
    },
  ];

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: `linear-gradient(-45deg, ${themeColors.background} 25%, ${themeColors.backgroundGradient})`,
        color: themeColors.text,
        p: 3,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <HelpIcon sx={{ fontSize: 40, color: themeColors.bright }} />
        <Typography variant="h4" sx={{ fontWeight: 700, color: themeColors.text }}>
          Help & FAQ
        </Typography>
      </Box>

      <Paper
        sx={{
          p: 3,
          mb: 3,
          background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
          borderRadius: '20px',
          boxShadow: `3px 3px 0 ${themeColors.text}`,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <InfoIcon sx={{ color: themeColors.bright }} />
          <Typography variant="h6" sx={{ color: themeColors.text, fontWeight: 600 }}>
            About BreathWatch
          </Typography>
        </Box>
        <Typography variant="body1" sx={{ color: themeColors.text, mb: 2 }}>
          BreathWatch is a respiratory health monitoring app that uses advanced audio analysis to detect and track
          coughs and wheezing. The app processes audio locally on your device for privacy and provides detailed
          insights into your respiratory patterns.
        </Typography>
        <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.8 }}>
          Version 1.0.0
        </Typography>
      </Paper>

      <Paper
        sx={{
          p: 3,
          mb: 3,
          background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
          borderRadius: '20px',
          boxShadow: `3px 3px 0 ${themeColors.text}`,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <QuestionAnswerIcon sx={{ color: themeColors.bright }} />
          <Typography variant="h6" sx={{ color: themeColors.text, fontWeight: 600 }}>
            Frequently Asked Questions
          </Typography>
        </Box>

        <Box>
          {faqs.map((faq, index) => (
            <Accordion
              key={index}
              sx={{
                backgroundColor: 'transparent',
                boxShadow: 'none',
                border: `1px solid ${themeColors.text}`,
                borderOpacity: 0.2,
                mb: 1,
                '&:before': {
                  display: 'none',
                },
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon sx={{ color: themeColors.bright }} />}
                sx={{
                  color: themeColors.text,
                  '& .MuiAccordionSummary-content': {
                    my: 1.5,
                  },
                }}
              >
                <Typography variant="subtitle1" sx={{ fontWeight: 600, color: themeColors.text }}>
                  {faq.question}
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.9 }}>
                  {faq.answer}
                </Typography>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      </Paper>

      <Paper
        sx={{
          p: 3,
          background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
          borderRadius: '20px',
          boxShadow: `3px 3px 0 ${themeColors.text}`,
        }}
      >
        <Typography variant="h6" sx={{ mb: 2, color: themeColors.text, fontWeight: 600 }}>
          Getting Started
        </Typography>
        <Box component="ol" sx={{ pl: 2, color: themeColors.text }}>
          <li>
            <Typography variant="body2" sx={{ color: themeColors.text, mb: 1 }}>
              Start a recording by clicking the microphone button
            </Typography>
          </li>
          <li>
            <Typography variant="body2" sx={{ color: themeColors.text, mb: 1 }}>
              Let the app record for at least 30 minutes for meaningful data
            </Typography>
          </li>
          <li>
            <Typography variant="body2" sx={{ color: themeColors.text, mb: 1 }}>
              View real-time updates on the home screen
            </Typography>
          </li>
          <li>
            <Typography variant="body2" sx={{ color: themeColors.text, mb: 1 }}>
              After stopping, view detailed analysis and AI interpretation
            </Typography>
          </li>
          <li>
            <Typography variant="body2" sx={{ color: themeColors.text }}>
              Check your history and statistics to track trends over time
            </Typography>
          </li>
        </Box>
      </Paper>
    </Box>
  );
}

