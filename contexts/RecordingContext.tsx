/**
 * Recording context for sharing recording state across the app
 */

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { useAudioRecorder, ChunkProcessResponse } from '@/hooks/useAudioRecorder';
import { getFinalSummary, NightlySummary } from '@/services/api';
import { saveRecordingToHistory, getSettings } from '@/services/storage';

interface RecordingContextType {
  recordingState: ReturnType<typeof useAudioRecorder>['recordingState'];
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<void>;
  finalSummary: NightlySummary | null;
  isFetchingSummary: boolean;
  fetchFinalSummary: (symptomForm?: any) => Promise<void>;
  chunkResponses: ChunkProcessResponse[];
}

const RecordingContext = createContext<RecordingContextType | undefined>(undefined);

export function RecordingProvider({ children }: { children: ReactNode }) {
  const [finalSummary, setFinalSummary] = useState<NightlySummary | null>(null);
  const [isFetchingSummary, setIsFetchingSummary] = useState(false);
  const [chunkResponses, setChunkResponses] = useState<ChunkProcessResponse[]>([]);

  const handleChunkProcessed = useCallback((response: ChunkProcessResponse) => {
    setChunkResponses((prev) => [...prev, response]);
  }, []);

  const handleError = useCallback((error: string) => {
    console.error('Recording error:', error);
    // Could show a toast notification here
  }, []);

  const { recordingState, startRecording: startRec, stopRecording: stopRec, sendFinalChunk } =
    useAudioRecorder(handleChunkProcessed, handleError);

  const fetchFinalSummary = useCallback(
    async (symptomForm?: any) => {
      if (!recordingState.sessionId) {
        return;
      }

      setIsFetchingSummary(true);
      try {
        const summary = await getFinalSummary(recordingState.sessionId, symptomForm);
        setFinalSummary(summary);
        
        // Auto-save to history if enabled
        const settings = await getSettings();
        if (settings.autoSaveRecordings) {
          await saveRecordingToHistory(summary);
        }
      } catch (error) {
        console.error('Failed to fetch final summary:', error);
        handleError(error instanceof Error ? error.message : 'Failed to fetch summary');
      } finally {
        setIsFetchingSummary(false);
      }
    },
    [recordingState.sessionId, handleError]
  );

  const startRecording = useCallback(async () => {
    setChunkResponses([]);
    setFinalSummary(null);
    await startRec();
  }, [startRec]);

  const stopRecording = useCallback(async () => {
    const wasRecording = recordingState.isRecording;
    const sessionId = recordingState.sessionId;
    
    // Send final chunk if recording was active
    if (wasRecording && sessionId) {
      try {
        console.log('Sending final chunk...');
        await sendFinalChunk();
        console.log('Final chunk sent successfully');
        // Wait a moment for the chunk to be processed and added to chunkResponses
        await new Promise((resolve) => setTimeout(resolve, 500));
      } catch (error) {
        console.error('Error sending final chunk:', error);
        // Continue even if final chunk fails
      }
    }
    
    await stopRec();
    
    // Automatically fetch final summary if we have a session
    // Check chunkResponses again after final chunk might have been added
    if (wasRecording && sessionId) {
      // Wait a bit longer to ensure backend has processed everything
      setTimeout(async () => {
        try {
          console.log('Fetching final summary for session:', sessionId);
          console.log('Current chunk count:', chunkResponses.length);
          await fetchFinalSummary();
          console.log('Final summary fetched successfully');
        } catch (error) {
          console.error('Error fetching final summary:', error);
          // Show error to user
          handleError(`Failed to fetch summary: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      }, 2000); // Increased delay to 2 seconds
    }
  }, [stopRec, sendFinalChunk, recordingState.isRecording, recordingState.sessionId, chunkResponses.length, fetchFinalSummary, handleError]);

  return (
    <RecordingContext.Provider
      value={{
        recordingState,
        startRecording,
        stopRecording,
        finalSummary,
        isFetchingSummary,
        fetchFinalSummary,
        chunkResponses,
      }}
    >
      {children}
    </RecordingContext.Provider>
  );
}

export function useRecording() {
  const context = useContext(RecordingContext);
  if (context === undefined) {
    throw new Error('useRecording must be used within a RecordingProvider');
  }
  return context;
}

