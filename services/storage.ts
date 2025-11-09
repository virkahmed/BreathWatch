/**
 * Local storage service for saving recording history
 * Uses localStorage for web, will need AsyncStorage for native
 */

import { NightlySummary } from './api';

// Simple storage wrapper that works on web and can be extended for native
const storage = {
  getItem: (key: string): string | null => {
    if (typeof window !== 'undefined' && window.localStorage) {
      return window.localStorage.getItem(key);
    }
    return null;
  },
  setItem: (key: string, value: string): void => {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem(key, value);
    }
  },
  removeItem: (key: string): void => {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.removeItem(key);
    }
  },
};

export interface RecordingHistoryItem {
  sessionId: string;
  summary: NightlySummary;
  timestamp: number; // Unix timestamp
  date: string; // Formatted date string
}

const STORAGE_KEYS = {
  RECORDING_HISTORY: '@breathwatch:recording_history',
  SETTINGS: '@breathwatch:settings',
} as const;

/**
 * Save a recording summary to history
 */
export async function saveRecordingToHistory(summary: NightlySummary): Promise<void> {
  try {
    const history = await getRecordingHistory();
    const newItem: RecordingHistoryItem = {
      sessionId: summary.session_id,
      summary,
      timestamp: Date.now(),
      date: new Date().toISOString(),
    };
    
    // Add to beginning of array (most recent first)
    history.unshift(newItem);
    
    // Keep only last 50 recordings
    const trimmedHistory = history.slice(0, 50);
    
    await storage.setItem(STORAGE_KEYS.RECORDING_HISTORY, JSON.stringify(trimmedHistory));
  } catch (error) {
    console.error('Failed to save recording to history:', error);
  }
}

/**
 * Get all recording history
 */
export async function getRecordingHistory(): Promise<RecordingHistoryItem[]> {
  try {
    const data = storage.getItem(STORAGE_KEYS.RECORDING_HISTORY);
    if (!data) return [];
    return JSON.parse(data);
  } catch (error) {
    console.error('Failed to get recording history:', error);
    return [];
  }
}

/**
 * Get a specific recording by session ID
 */
export async function getRecordingBySessionId(sessionId: string): Promise<RecordingHistoryItem | null> {
  try {
    const history = await getRecordingHistory();
    return history.find(item => item.sessionId === sessionId) || null;
  } catch (error) {
    console.error('Failed to get recording by session ID:', error);
    return null;
  }
}

/**
 * Delete a recording from history
 */
export async function deleteRecording(sessionId: string): Promise<void> {
  try {
    const history = await getRecordingHistory();
    const filtered = history.filter(item => item.sessionId !== sessionId);
    await storage.setItem(STORAGE_KEYS.RECORDING_HISTORY, JSON.stringify(filtered));
  } catch (error) {
    console.error('Failed to delete recording:', error);
  }
}

/**
 * Clear all recording history
 */
export async function clearRecordingHistory(): Promise<void> {
  try {
    storage.removeItem(STORAGE_KEYS.RECORDING_HISTORY);
  } catch (error) {
    console.error('Failed to clear recording history:', error);
  }
}

/**
 * App settings interface
 */
export interface AppSettings {
  patientId?: number;
  age?: number;
  sex?: string;
  autoSaveRecordings: boolean;
  notificationsEnabled: boolean;
  theme: 'light' | 'dark' | 'auto';
}

const DEFAULT_SETTINGS: AppSettings = {
  autoSaveRecordings: true,
  notificationsEnabled: true,
  theme: 'auto',
};

/**
 * Get app settings
 */
export async function getSettings(): Promise<AppSettings> {
  try {
    const data = storage.getItem(STORAGE_KEYS.SETTINGS);
    if (!data) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...JSON.parse(data) };
  } catch (error) {
    console.error('Failed to get settings:', error);
    return DEFAULT_SETTINGS;
  }
}

/**
 * Save app settings
 */
export async function saveSettings(settings: Partial<AppSettings>): Promise<void> {
  try {
    const current = await getSettings();
    const updated = { ...current, ...settings };
    storage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(updated));
  } catch (error) {
    console.error('Failed to save settings:', error);
  }
}

