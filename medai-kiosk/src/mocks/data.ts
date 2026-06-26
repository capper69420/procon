import type { TranscriptLine, TriageFinding, VitalMetric, PatientProfile } from "../types";

export const transcriptScript: TranscriptLine[] = [
  {
    id: "t1",
    timestamp: "9:48",
    ja: "頭が痛くて、昨日から熱があります。",
    en: "I have a headache and fever since yesterday.",
  },
  {
    id: "t2",
    timestamp: "9:49",
    ja: "咳も出ています。少し息苦しいです。",
    en: "I also have a cough. I feel a little short of breath.",
  },
  {
    id: "t3",
    timestamp: "9:50",
    ja: "体温は38.5度でした。",
    en: "My temperature was 38.5°C.",
  },
];

export const triageFindings: TriageFinding[] = [
  { id: "f1", label: "Fever detected — 38.5°C (elevated)", tone: "danger" },
  { id: "f2", label: "Respiratory symptoms present", tone: "warning" },
  { id: "f3", label: "Elevated heart rate: 108 BPM", tone: "warning" },
  { id: "f4", label: "Cardiovascular risk: low", tone: "neutral" },
  { id: "f5", label: "Risk score updating — 78/100", tone: "neutral" },
];

export const quickSymptomTags = [
  { key: "fever" },
  { key: "headache" },
  { key: "cough" },
  { key: "fatigue" },
  { key: "dyspnea" },
];

function genHistory(base: number, variance: number, points = 24) {
  return Array.from({ length: points }, (_, i) => ({
    t: i,
    v: Math.round((base + (Math.sin(i / 2) * variance + (Math.random() - 0.5) * variance * 0.6)) * 10) / 10,
  }));
}

export const vitalMetrics: VitalMetric[] = [
  {
    key: "heartRate",
    label: "Heart Rate",
    value: "82",
    unit: "BPM",
    alert: false,
    color: "#dc2626",
    history: genHistory(82, 6),
  },
  {
    key: "respRate",
    label: "Resp. Rate",
    value: "18",
    unit: "breaths/min",
    alert: false,
    color: "#2f5fe0",
    history: genHistory(18, 2),
  },
  {
    key: "spo2",
    label: "SpO2",
    value: "94",
    unit: "%",
    alert: true,
    color: "#d97706",
    history: genHistory(94, 1.5),
  },
  {
    key: "temperature",
    label: "Temperature",
    value: "38.5",
    unit: "°C",
    alert: true,
    color: "#d97706",
    history: genHistory(38.5, 0.4),
  },
];

export const patientProfile: PatientProfile = {
  name: "Tanaka Hanako",
  nameJa: "田中 花子",
  age: 34,
  gender: "Female",
  patientId: "PT-2847",
  bloodType: "A+",
};

export const consultationSummary = [
  "Acute febrile illness, onset 24h",
  "Dry cough + mild dyspnea present",
  "SpO2 94% — below normal threshold",
  "Recommend CBC, CRP, chest X-ray",
  "Physician review within 30 minutes",
];

export const aiAssessmentBasis = [
  "Chief Complaint: Fever, Cough, Dyspnea",
  "Vitals: HR 108, SpO2 94%, Temp 38.5°C",
  "Duration: 24 hours, acute onset",
  "Past history: Asthma (2019)",
  "Current hospital load: 67%",
];

export const roomAssignment = {
  department: "Respiratory Medicine",
  departmentJa: "呼吸器内科",
  doctor: "Dr. Sato Kenji",
  room: "305 — 3F East Wing",
  queuePosition: "#7",
  estWaitMin: 8,
  confidenceScore: 92,
  inputsCount: 5,
  latencySec: 1.2,
  model: "GPT-4 Med v3.1",
  sessionId: "#2847",
};
