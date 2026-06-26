import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { kioskApi, type StartMeasurementPayload } from "../api/kioskApi";
import type {
  AssessmentResult,
  ConnectionState,
  Patient,
  ResultsPayload,
  RoomAssignment,
  TranscriptionResult,
  VisionResult,
  VitalHistoryPoint,
} from "../api/contracts";

interface KioskSessionState {
  sessionId: string | null;
  patient: Patient | null;
  latestVision: VisionResult | null;
  assessment: AssessmentResult | null;
  assignment: RoomAssignment | null;
  results: ResultsPayload | null;
  transcript: TranscriptionResult | null;
  connection: ConnectionState;
  loading: boolean;
  error: string | null;
  vitalHistory: {
    heartRate: VitalHistoryPoint[];
    spo2: VitalHistoryPoint[];
  };
}

interface KioskSessionContextValue {
  state: KioskSessionState;
  checkConnection: () => Promise<boolean>;
  startMeasurement: (payload: StartMeasurementPayload) => Promise<string>;
  processFrame: (imageBase64: string) => Promise<VisionResult | null>;
  stopMeasurement: () => Promise<void>;
  createAssessment: () => Promise<AssessmentResult | null>;
  createAssignment: () => Promise<RoomAssignment | null>;
  fetchResults: () => Promise<ResultsPayload | null>;
  transcribeSpeech: (language: "en" | "ja", audio: Blob) => Promise<TranscriptionResult | null>;
  clearError: () => void;
}

const KioskSessionContext = createContext<KioskSessionContextValue | null>(null);

const initialState: KioskSessionState = {
  sessionId: null,
  patient: null,
  latestVision: null,
  assessment: null,
  assignment: null,
  results: null,
  transcript: null,
  connection: "checking",
  loading: false,
  error: null,
  vitalHistory: {
    heartRate: [],
    spo2: [],
  },
};

function appendPoint(points: VitalHistoryPoint[], value: number) {
  if (!value || value <= 0) return points;
  const next = [...points, { t: points.length, v: Math.round(value * 10) / 10 }];
  return next.slice(-30);
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Unexpected API error";
}

export function KioskSessionProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<KioskSessionState>(initialState);

  const clearError = useCallback(() => {
    setState((current) => ({ ...current, error: null }));
  }, []);

  const checkConnection = useCallback(async () => {
    setState((current) => ({ ...current, connection: "checking" }));
    try {
      await kioskApi.health();
      setState((current) => ({ ...current, connection: "online" }));
      return true;
    } catch {
      setState((current) => ({ ...current, connection: "offline" }));
      return false;
    }
  }, []);

  const startMeasurement = useCallback(async (payload: StartMeasurementPayload) => {
    setState((current) => ({ ...current, loading: true, error: null }));
    try {
      const started = await kioskApi.startMeasurement(payload);
      setState((current) => ({
        ...current,
        sessionId: started.session_id,
        patient: started.patient,
        connection: "online",
        loading: false,
      }));
      return started.session_id;
    } catch (error) {
      setState((current) => ({
        ...current,
        connection: "offline",
        loading: false,
        error: errorMessage(error),
      }));
      throw error;
    }
  }, []);

  const processFrame = useCallback(async (imageBase64: string) => {
    if (!state.sessionId) return null;
    try {
      const result = await kioskApi.sendMeasurementFrame(state.sessionId, imageBase64);
      setState((current) => ({
        ...current,
        latestVision: result,
        connection: "online",
        vitalHistory: {
          heartRate: appendPoint(current.vitalHistory.heartRate, result.vitals.heart_rate),
          spo2: appendPoint(current.vitalHistory.spo2, result.vitals.spo2),
        },
      }));
      return result;
    } catch (error) {
      setState((current) => ({
        ...current,
        connection: "offline",
        error: errorMessage(error),
      }));
      return null;
    }
  }, [state.sessionId]);

  const stopMeasurement = useCallback(async () => {
    if (!state.sessionId) return;
    try {
      await kioskApi.stopMeasurement(state.sessionId);
    } catch (error) {
      setState((current) => ({ ...current, error: errorMessage(error) }));
    }
  }, [state.sessionId]);

  const createAssessment = useCallback(async () => {
    if (!state.sessionId) return null;
    setState((current) => ({ ...current, loading: true, error: null }));
    try {
      const assessment = await kioskApi.createAssessment(state.sessionId);
      setState((current) => ({ ...current, assessment, loading: false, connection: "online" }));
      return assessment;
    } catch (error) {
      setState((current) => ({ ...current, loading: false, error: errorMessage(error) }));
      return null;
    }
  }, [state.sessionId]);

  const createAssignment = useCallback(async () => {
    if (!state.sessionId) return null;
    setState((current) => ({ ...current, loading: true, error: null }));
    try {
      const assessment = state.assessment || (await kioskApi.createAssessment(state.sessionId));
      const assignment = await kioskApi.createAssignment(state.sessionId, assessment);
      setState((current) => ({
        ...current,
        assessment,
        assignment,
        loading: false,
        connection: "online",
      }));
      return assignment;
    } catch (error) {
      setState((current) => ({ ...current, loading: false, error: errorMessage(error) }));
      return null;
    }
  }, [state.assessment, state.sessionId]);

  const fetchResults = useCallback(async () => {
    if (!state.sessionId) return null;
    try {
      const results = await kioskApi.fetchResults(state.sessionId);
      setState((current) => ({ ...current, results, connection: "online" }));
      return results;
    } catch (error) {
      setState((current) => ({ ...current, error: errorMessage(error) }));
      return null;
    }
  }, [state.sessionId]);

  const transcribeSpeech = useCallback(async (language: "en" | "ja", audio: Blob) => {
    try {
      const transcript = await kioskApi.transcribeSpeech(language, audio, state.patient?.id);
      setState((current) => ({ ...current, transcript, connection: "online" }));
      return transcript;
    } catch (error) {
      setState((current) => ({ ...current, error: errorMessage(error) }));
      return null;
    }
  }, [state.patient?.id]);

  const value = useMemo<KioskSessionContextValue>(() => ({
    state,
    checkConnection,
    startMeasurement,
    processFrame,
    stopMeasurement,
    createAssessment,
    createAssignment,
    fetchResults,
    transcribeSpeech,
    clearError,
  }), [
    state,
    checkConnection,
    startMeasurement,
    processFrame,
    stopMeasurement,
    createAssessment,
    createAssignment,
    fetchResults,
    transcribeSpeech,
    clearError,
  ]);

  return <KioskSessionContext.Provider value={value}>{children}</KioskSessionContext.Provider>;
}

export function useKioskSession() {
  const context = useContext(KioskSessionContext);
  if (!context) {
    throw new Error("useKioskSession must be used inside KioskSessionProvider");
  }
  return context;
}
