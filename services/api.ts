/**
 * API service for backend communication
 */

import { API_ENDPOINTS } from '@/constants/api';

export interface AttributeVector {
  wet: number;
  wheezing: number;
  stridor: number;
  choking: number;
  congestion: number;
}

export interface AttributeFlags {
  wet: number;
  wheezing: number;
  stridor: number;
  choking: number;
  congestion: number;
}

export interface AttributeSeries {
  wet: number[];
  wheezing: number[];
  stridor: number[];
  choking: number[];
  congestion: number[];
}

export interface ProbabilityTimeline {
  tile_seconds: number;
  stride_seconds: number;
  indices: number[];
  times: number[];
  p_cough: number[];
  attr_series: AttributeSeries;
}

export interface ProbabilityEvent {
  start: number;
  end: number;
  duration: number;
  tile_indices: number[];
  p_cough_max: number;
  p_cough_mean: number;
  attr_probs: AttributeVector;
  attr_flags: AttributeFlags;
}

export interface EventSummary {
  num_events: number;
  events: ProbabilityEvent[];
}

export interface ChunkProcessResponse {
  chunk_index: number;
  session_id: string;
  cough_count: number;
  wheeze_windows: number;
  windows_processed: number;
  probability_timeline: ProbabilityTimeline;
  event_summary: EventSummary;
  detected_events: Array<{
    start_ms: number;
    end_ms: number;
    confidence: number;
    tags: string[];
    quality_flag?: string;
    window_indices: number[];
  }>;
}

export interface NightlySummary {
  session_id: string;
  patient_id?: number;
  age?: number;
  sex?: string;
  total_duration_minutes: number;
  coughs_per_hour: number;
  bout_count: number;
  bout_lengths: number[];
  inter_cough_intervals: number[];
  avg_bout_length_seconds?: number;
  avg_inter_cough_interval_seconds?: number;
  max_coughs_in_single_hour: number;
  wheeze_time_percent: number;
  longest_wheeze_duration_seconds?: number;
  wheeze_intensity_avg?: number;
  attribute_prevalence: {
    wet: number;
    wheezing: number;
    stridor: number;
    choking: number;
    congestion: number;
  };
  cough_events: Array<{
    start_ms: number;
    end_ms: number;
    confidence: number;
    tags: string[];
    quality_flag?: string;
    window_indices: number[];
  }>;
  event_summary: EventSummary;
  probability_timeline: ProbabilityTimeline;
  pattern_scores?: Array<{
    pattern_name: string;
    score: number;
    uncertainty_lower: number;
    uncertainty_upper: number;
    why: string;
  }>;
  hourly_breakdown: Array<{
    hour: number;
    cough_count: number;
    wheeze_percent: number;
    events: any[];
  }>;
  quality_metrics: {
    avg_snr: number;
    quality_score: number;
    low_quality_periods_count: number;
    high_confidence_events_count: number;
    suppressed_events_count: number;
  };
  display_strings: {
    sleep_duration_formatted: string;
    coughs_per_hour_formatted: string;
    severity_badge_color: string;
    overall_quality_score: number;
  };
  dedalus_interpretation?: {
    interpretation: string;
    severity?: string;
    recommendations?: string[];
  };
}

/**
 * Process a 10-minute audio chunk
 * 
 * Sends audio data as multipart/form-data to the backend.
 * The backend expects:
 * - audio_chunk: File (WAV format)
 * - chunk_index: integer
 * - session_id: string
 * - Optional: patient_id, age, sex
 * 
 * Fixed issues:
 * - Added proper error handling with detailed error messages
 * - Added validation for blob size
 * - Improved error parsing from backend response
 * - Added logging for debugging
 */
export async function processChunk(
  audioBlob: Blob,
  chunkIndex: number,
  sessionId: string,
  patientId?: number,
  age?: number,
  sex?: string
): Promise<ChunkProcessResponse> {
  // Validate inputs
  if (!audioBlob || audioBlob.size === 0) {
    throw new Error('Audio blob is empty or invalid');
  }
  
  if (chunkIndex < 0) {
    throw new Error(`Invalid chunk_index: ${chunkIndex}. Must be >= 0`);
  }
  
  if (!sessionId || !sessionId.trim()) {
    throw new Error('session_id is required and cannot be empty');
  }

  console.log(`📤 Sending chunk ${chunkIndex} to backend:`, {
    sessionId,
    blobSize: `${(audioBlob.size / 1024).toFixed(2)} KB`,
    blobType: audioBlob.type,
  });

  // Create FormData for multipart/form-data upload
  // Note: Do NOT set Content-Type header - browser will set it automatically with boundary
  const formData = new FormData();
  formData.append('audio_chunk', audioBlob, 'chunk.wav');
  formData.append('chunk_index', chunkIndex.toString());
  formData.append('session_id', sessionId);
  
  if (patientId !== undefined) {
    formData.append('patient_id', patientId.toString());
  }
  if (age !== undefined) {
    formData.append('age', age.toString());
  }
  if (sex !== undefined) {
    formData.append('sex', sex);
  }

  try {
    const response = await fetch(API_ENDPOINTS.PROCESS_CHUNK, {
      method: 'POST',
      // DO NOT set Content-Type header - browser will set it automatically
      // with the correct boundary for multipart/form-data
      body: formData,
    });

    // Parse error response
    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}`;
      let errorDetail = '';
      
      try {
        // Try to parse as JSON first (FastAPI returns JSON errors)
        const errorJson = await response.json();
        errorDetail = errorJson.detail || errorJson.message || JSON.stringify(errorJson);
        errorMessage = `Failed to process chunk: ${response.status} - ${errorDetail}`;
      } catch {
        // If not JSON, try as text
        try {
          errorDetail = await response.text();
          errorMessage = `Failed to process chunk: ${response.status} - ${errorDetail}`;
        } catch {
          errorMessage = `Failed to process chunk: ${response.status} ${response.statusText}`;
        }
      }
      
      console.error(`❌ Chunk upload failed: ${response.status}`, {
        errorDetail,
        chunkIndex,
        sessionId,
        blobSize: audioBlob.size,
      });
      
      throw new Error(errorMessage);
    }

    // Parse successful response
    try {
      const result = await response.json();
      console.log(`✅ Chunk ${chunkIndex} processed successfully:`, {
        coughCount: result.cough_count,
        wheezeWindows: result.wheeze_windows,
        windowsProcessed: result.windows_processed,
      });
      return result;
    } catch (parseError) {
      console.error('❌ Error parsing response JSON:', parseError);
      throw new Error(`Failed to parse response from backend: ${parseError instanceof Error ? parseError.message : 'Unknown error'}`);
    }
    
  } catch (error) {
    // Handle network errors, timeouts, etc.
    if (error instanceof TypeError && error.message.includes('fetch')) {
      console.error('❌ Network error:', error);
      throw new Error('Network error: Could not connect to backend. Ensure the server is running at ' + API_ENDPOINTS.PROCESS_CHUNK);
    }
    
    // Re-throw if it's already our custom error
    if (error instanceof Error) {
      throw error;
    }
    
    // Fallback for unknown errors
    console.error('❌ Unknown error in processChunk:', error);
    throw new Error(`Failed to process chunk: ${error}`);
  }
}

/**
 * Get final nightly summary with timeout
 */
export async function getFinalSummary(
  sessionId: string,
  symptomForm?: {
    fever: boolean;
    sore_throat: boolean;
    chest_tightness: boolean;
    duration: number;
    nocturnal_worsening: boolean;
    asthma_history: boolean;
    copd_history: boolean;
    age_band?: string;
    smoker: boolean;
  },
  onProgress?: (message: string) => void
): Promise<NightlySummary> {
  const TIMEOUT_MS = 60000; // 60 seconds timeout
  
  onProgress?.('Connecting to backend...');
  
  // Create abort controller for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, TIMEOUT_MS);

  try {
    onProgress?.('Sending request to backend...');
    
    const response = await fetch(API_ENDPOINTS.FINAL_SUMMARY(sessionId), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: symptomForm ? JSON.stringify({ symptom_form: symptomForm }) : undefined,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      onProgress?.(`Error: ${response.status}`);
      throw new Error(`Failed to get final summary: ${response.status} ${errorText}`);
    }

    onProgress?.('Processing response...');
    const result = await response.json();
    onProgress?.('Summary received!');
    
    return result;
  } catch (error) {
    clearTimeout(timeoutId);
    
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        onProgress?.('Request timed out after 60 seconds');
        throw new Error('Request timed out. The backend may be processing a large amount of data. Please try again.');
      }
      onProgress?.(`Error: ${error.message}`);
      throw error;
    }
    
    throw new Error('Unknown error occurred while fetching final summary');
  }
}

/**
 * Check backend health status
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(API_ENDPOINTS.HEALTH);
    return response.ok;
  } catch {
    return false;
  }
}
