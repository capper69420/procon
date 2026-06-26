export type TriageLevel = "A" | "B" | "C";
export type ConnectionState = "checking" | "online" | "offline";

export interface Patient {
  id: string;
  name: string;
  name_ja?: string;
  age?: number;
  gender?: string;
  sex?: string;
  blood_type?: string;
  symptoms?: string;
  conditions?: string[];
  current_triage_level: TriageLevel;
  urgency_score: number;
  created_at: string;
  updated_at: string;
}

export interface Vitals {
  spo2: number;
  heart_rate: number;
  signal_quality: number;
}

export interface Posture {
  status: string;
  confidence: number;
  fall_detected: boolean;
  immobile_seconds: number;
}

export interface VisionResult {
  patient_id: string;
  vitals: Vitals;
  posture: Posture;
  faces_detected: number;
  processed_at: string;
}

export interface AudioFeatures {
  breath_rate: number;
  cough_detected: boolean;
  distress_score: number;
  speech_clarity: number;
}

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  language: string;
}

export interface TranscriptionResult {
  patient_id?: string;
  language?: string;
  transcript: string;
  segments: TranscriptSegment[];
  processed_at: string;
}

export interface TriageResult {
  patient_id: string;
  triage_level: TriageLevel;
  urgency_score: number;
  reasons: string[];
  recommended_action: string;
  decided_at: string;
}

export interface MeasurementStartResponse {
  session_id: string;
  patient: Patient;
  status: "active" | "stopped";
  started_at: string;
}

export interface AssessmentResult {
  session_id?: string;
  patient_id: string;
  triage: TriageResult;
  findings: string[];
  summary: string[];
}

export interface RoomAssignment {
  session_id?: string;
  patient_id: string;
  department: string;
  department_ja: string;
  doctor: string;
  room: string;
  queue_position: string;
  est_wait_min: number;
  confidence_score: number;
  inputs_count: number;
  latency_sec: number;
  model: string;
  assigned_at: string;
}

export interface ResultsPayload {
  session: Record<string, unknown>;
  patient: Patient | null;
  latest_vision: VisionResult | null;
  latest_audio: AudioFeatures | null;
  latest_triage: TriageResult | null;
  assignment: RoomAssignment | null;
  generated_at: string;
}

export interface VitalHistoryPoint {
  t: number;
  v: number;
}
