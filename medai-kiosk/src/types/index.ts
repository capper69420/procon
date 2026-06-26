export type RiskLevel = "stable" | "moderate" | "critical";

export interface TranscriptLine {
  id: string;
  timestamp: string;
  ja: string;
  en: string;
}

export interface TriageFinding {
  id: string;
  label: string;
  tone: "neutral" | "warning" | "danger";
}

export interface VitalPoint {
  t: number;
  v: number;
}

export interface VitalMetric {
  key: "heartRate" | "respRate" | "spo2" | "temperature";
  label: string;
  value: string;
  unit: string;
  alert: boolean;
  color: string;
  history: VitalPoint[];
}

export interface PatientProfile {
  name: string;
  nameJa: string;
  age: number;
  gender: string;
  patientId: string;
  bloodType: string;
}

export interface RoutingStep {
  row: number;
  col: number;
}
