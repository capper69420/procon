const API_BASE = import.meta.env.VITE_API_URL || '';

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  try {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.error || `HTTP ${res.status}`);
    return data;
  } catch (err) {
    throw err;
  }
}

export async function checkHealth() {
  return request('/health');
}

export async function createPatient(patient) {
  return request('/api/patients', { method: 'POST', body: JSON.stringify(patient) });
}

export async function listPatients() {
  const data = await request('/api/patients');
  return data.patients || [];
}

export async function getPatient(id) {
  return request(`/api/patients/${id}`);
}

export async function postVision(patientId, imageBase64) {
  return request('/api/vision', {
    method: 'POST',
    body: JSON.stringify({ patient_id: patientId, image_base64: imageBase64 }),
  });
}

export async function postAudio(patientId, audioFeatures) {
  return request('/api/audio', {
    method: 'POST',
    body: JSON.stringify({ patient_id: patientId, audio_features: audioFeatures }),
  });
}

export async function postTriage(patientId, payload = {}) {
  return request('/api/triage', {
    method: 'POST',
    body: JSON.stringify({ patient_id: patientId, ...payload }),
  });
}

export async function isBackendOnline() {
  try {
    await checkHealth();
    return true;
  } catch {
    return false;
  }
}
