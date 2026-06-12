const STORAGE_KEY = 'medical_reception_sessions';

export function saveSession(session) {
  try {
    const sessions = loadSessions();
    sessions.unshift({ ...session, saved_at: new Date().toISOString() });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.slice(0, 50)));
  } catch {
    /* ignore quota errors */
  }
}

export function loadSessions() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

export function saveDraft(key, data) {
  try {
    localStorage.setItem(`kiosk_draft_${key}`, JSON.stringify(data));
  } catch {
    /* ignore */
  }
}

export function loadDraft(key) {
  try {
    const raw = localStorage.getItem(`kiosk_draft_${key}`);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

const SEED = [
  {
    id: '550e8400-e29b-41d4-a716-446655440003',
    name: 'Jane Doe',
    room: '103',
    age: 78,
    sex: 'F',
    symptoms: 'Shortness of breath, dizziness after fall',
    conditions: ['COPD', 'Hypertension'],
    emergency_contact: 'Mary Doe — 555-0103',
    current_triage_level: 'C',
    urgency_score: 0.91,
    vitals: { spo2: 87.5, heart_rate: 112, signal_quality: 0.72 },
    audio: { distress_score: 0.82, breath_rate: 28, cough_detected: true },
    posture: { fall_detected: true, immobile_seconds: 18, status: 'FALLEN' },
  },
  {
    id: '550e8400-e29b-41d4-a716-446655440002',
    name: 'John Smith',
    room: '102',
    age: 65,
    sex: 'M',
    symptoms: 'Persistent cough, mild fatigue',
    conditions: ['Diabetes'],
    emergency_contact: 'Lisa Smith — 555-0102',
    current_triage_level: 'B',
    urgency_score: 0.55,
    vitals: { spo2: 92.0, heart_rate: 88, signal_quality: 0.65 },
    audio: { distress_score: 0.48, breath_rate: 20, cough_detected: true },
    posture: { fall_detected: false, immobile_seconds: 0, status: 'SITTING' },
  },
  {
    id: '550e8400-e29b-41d4-a716-446655440001',
    name: 'Baby Alex',
    room: '101',
    age: 2,
    sex: 'M',
    symptoms: 'Routine check-in',
    conditions: [],
    emergency_contact: 'Parent Alex — 555-0101',
    current_triage_level: 'A',
    urgency_score: 0.08,
    vitals: { spo2: 98.5, heart_rate: 95, signal_quality: 0.88 },
    audio: { distress_score: 0.05, breath_rate: 24, cough_detected: false },
    posture: { fall_detected: false, immobile_seconds: 0, status: 'STANDING' },
  },
];

export function getSeedPatients() {
  return SEED.map((p) => ({ ...p }));
}

export function fluctuatePatient(patient) {
  const copy = { ...patient, vitals: { ...(patient.vitals || {}) }, audio: { ...(patient.audio || {}) } };
  const hr = copy.vitals.heart_rate || 72;
  const spo2 = copy.vitals.spo2 || 96;
  copy.vitals.heart_rate = Math.round(hr + (Math.random() * 4 - 2));
  copy.vitals.spo2 = Math.round((spo2 + (Math.random() * 2 - 1)) * 10) / 10;
  if (copy.audio.distress_score != null) {
    copy.audio.distress_score = Math.max(0, Math.min(1,
      Math.round((copy.audio.distress_score + (Math.random() * 0.06 - 0.03)) * 100) / 100,
    ));
  }
  copy.last_seen_at = new Date().toISOString();
  return copy;
}

export function sortPatients(patients) {
  const order = { C: 0, B: 1, A: 2 };
  return [...patients].sort((a, b) => {
    const la = order[a.current_triage_level] ?? 3;
    const lb = order[b.current_triage_level] ?? 3;
    if (la !== lb) return la - lb;
    return (b.urgency_score || 0) - (a.urgency_score || 0);
  });
}
