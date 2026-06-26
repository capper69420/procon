import { apiRequest, checkHealth } from "./client";
import type {
  AssessmentResult,
  MeasurementStartResponse,
  Patient,
  ResultsPayload,
  RoomAssignment,
  TranscriptionResult,
  VisionResult,
} from "./contracts";

export interface StartMeasurementPayload {
  patient_id?: string;
  patient?: {
    name: string;
    name_ja?: string;
    age?: number;
    gender?: string;
    symptoms?: string;
    conditions?: string[];
  };
  language: string;
}

export const kioskApi = {
  health: checkHealth,

  startMeasurement(payload: StartMeasurementPayload) {
    return apiRequest<MeasurementStartResponse>("/api/measurements/start", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  sendMeasurementFrame(sessionId: string, imageBase64: string) {
    return apiRequest<VisionResult>(`/api/measurements/${sessionId}/frame`, {
      method: "POST",
      body: JSON.stringify({ image_base64: imageBase64 }),
    });
  },

  stopMeasurement(sessionId: string) {
    return apiRequest(`/api/measurements/${sessionId}/stop`, { method: "POST" });
  },

  createAssessment(sessionId: string) {
    return apiRequest<AssessmentResult>("/api/assessment", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    });
  },

  createAssignment(sessionId: string, assessment?: AssessmentResult) {
    return apiRequest<RoomAssignment>("/api/assignments", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, assessment: assessment?.triage }),
    });
  },

  fetchResults(sessionId: string) {
    return apiRequest<ResultsPayload>(`/api/results/${sessionId}`);
  },

  listPatients() {
    return apiRequest<{ patients: Patient[] }>("/api/patients");
  },

  transcribeSpeech(language: "en" | "ja", audio: Blob, patientId?: string) {
    const body = new FormData();
    body.append("audio_file", audio, `speech-${Date.now()}.webm`);
    if (patientId) body.append("patient_id", patientId);
    return apiRequest<TranscriptionResult>(`/api/speech/${language}/transcribe`, {
      method: "POST",
      body,
    });
  },
};
