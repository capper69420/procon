import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useKiosk } from '../context/KioskContext';
import { t } from '../i18n';
import { createPatient } from '../api';

export default function PatientRegistration() {
  const navigate = useNavigate();
  const { lang, updateSession, setOffline } = useKiosk();
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [form, setForm] = useState({
    name: '',
    age: '',
    sex: '',
    symptoms: '',
    conditions: '',
    emergency_contact: '',
  });

  const validate = () => {
    const e = {};
    if (!form.name.trim()) e.name = t(lang, 'required');
    if (!form.age || Number(form.age) < 0) e.age = t(lang, 'required');
    if (!form.sex) e.sex = t(lang, 'required');
    if (!form.symptoms.trim()) e.symptoms = t(lang, 'required');
    if (!form.emergency_contact.trim()) e.emergency_contact = t(lang, 'required');
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSubmit = async (ev) => {
    ev.preventDefault();
    if (!validate()) return;
    setLoading(true);
    const payload = {
      name: form.name.trim(),
      age: Number(form.age),
      sex: form.sex,
      symptoms: form.symptoms.trim(),
      conditions: form.conditions
        ? form.conditions.split(',').map((c) => c.trim()).filter(Boolean)
        : [],
      emergency_contact: form.emergency_contact.trim(),
    };
    try {
      const patient = await createPatient(payload);
      updateSession({ patient });
      setOffline(false);
      navigate('/scan');
    } catch {
      setOffline(true);
      const localPatient = {
        id: crypto.randomUUID?.() || `local-${Date.now()}`,
        ...payload,
        current_triage_level: 'A',
        urgency_score: 0,
      };
      updateSession({ patient: localPatient });
      navigate('/scan');
    } finally {
      setLoading(false);
    }
  };

  const set = (field) => (ev) => setForm((f) => ({ ...f, [field]: ev.target.value }));

  return (
    <div className="screen">
      <div className="card">
        <h2 className="card-title">{t(lang, 'registration')}</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="name">{t(lang, 'name')} *</label>
            <input id="name" value={form.name} onChange={set('name')} autoComplete="name" />
            {errors.name && <div className="form-error">{errors.name}</div>}
          </div>
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="age">{t(lang, 'age')} *</label>
              <input id="age" type="number" min="0" max="150" value={form.age} onChange={set('age')} />
              {errors.age && <div className="form-error">{errors.age}</div>}
            </div>
            <div className="form-group">
              <label htmlFor="sex">{t(lang, 'sex')} *</label>
              <select id="sex" value={form.sex} onChange={set('sex')}>
                <option value="">—</option>
                <option value="M">{t(lang, 'male')}</option>
                <option value="F">{t(lang, 'female')}</option>
                <option value="O">{t(lang, 'other')}</option>
              </select>
              {errors.sex && <div className="form-error">{errors.sex}</div>}
            </div>
          </div>
          <div className="form-group">
            <label htmlFor="symptoms">{t(lang, 'symptoms')} *</label>
            <textarea id="symptoms" value={form.symptoms} onChange={set('symptoms')} />
            {errors.symptoms && <div className="form-error">{errors.symptoms}</div>}
          </div>
          <div className="form-group">
            <label htmlFor="conditions">{t(lang, 'conditions')}</label>
            <input id="conditions" value={form.conditions} onChange={set('conditions')} placeholder="COPD, Diabetes" />
          </div>
          <div className="form-group">
            <label htmlFor="contact">{t(lang, 'emergencyContact')} *</label>
            <input id="contact" value={form.emergency_contact} onChange={set('emergency_contact')} />
            {errors.emergency_contact && <div className="form-error">{errors.emergency_contact}</div>}
          </div>
          <div className="btn-row">
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/')}>
              {t(lang, 'back')}
            </button>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? t(lang, 'loading') : t(lang, 'continue')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
