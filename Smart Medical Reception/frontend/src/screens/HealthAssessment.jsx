import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useKiosk } from '../context/KioskContext';
import { t } from '../i18n';
import { postTriage } from '../api';

function triageLabel(lang, level) {
  if (level === 'C') return t(lang, 'levelC');
  if (level === 'B') return t(lang, 'levelB');
  return t(lang, 'levelA');
}

export default function HealthAssessment() {
  const navigate = useNavigate();
  const { lang, session, updateSession } = useKiosk();
  const [loading, setLoading] = useState(true);
  const [triage, setTriage] = useState(null);

  const vitals = session.vision?.vitals || {};
  const posture = session.vision?.posture || {};
  const spo2 = vitals.spo2 || 95;
  const hr = vitals.heart_rate || 72;
  const distress = session.audio?.distress_score ?? 0.15;
  const fallRisk = posture.fall_detected
    ? (posture.immobile_seconds >= 15 ? 'High' : 'Moderate')
    : 'Low';

  useEffect(() => {
    if (!session.patient?.id) {
      navigate('/register');
      return;
    }

    async function runTriage() {
      setLoading(true);
      try {
        const result = await postTriage(session.patient.id, {
          vision: session.vision,
          audio: session.audio,
          patient_context: {
            name: session.patient.name,
            age: session.patient.age,
            conditions: session.patient.conditions || [],
          },
        });
        setTriage(result);
        updateSession({ triage: result });
      } catch {
        const level = spo2 < 90 || hr < 40 || hr > 140 ? 'C'
          : spo2 < 94 || hr < 50 || hr > 120 ? 'B' : 'A';
        const fallback = {
          triage_level: level,
          urgency_score: level === 'C' ? 0.9 : level === 'B' ? 0.5 : 0.1,
          reasons: ['Local assessment — backend unavailable'],
          recommended_action: level === 'C' ? 'Immediate clinical review' : 'Continue monitoring',
        };
        setTriage(fallback);
        updateSession({ triage: fallback });
      } finally {
        setLoading(false);
      }
    }

    runTriage();
  }, [session.patient, session.vision, session.audio, navigate, updateSession, spo2, hr]);

  const level = triage?.triage_level || 'A';

  if (loading) {
    return (
      <div className="screen">
        <div className="card" style={{ textAlign: 'center' }}>
          <h2 className="card-title">{t(lang, 'assessment')}</h2>
          <div className="spinner" />
          <p>{t(lang, 'loading')}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="screen">
      <div className="card">
        <h2 className="card-title">{t(lang, 'assessment')}</h2>
        <div className="vitals-grid">
          <div className="vital-card">
            <div className="vital-value">{Math.round(hr)}</div>
            <div className="vital-label">{t(lang, 'heartRate')} ({t(lang, 'bpm')})</div>
          </div>
          <div className="vital-card">
            <div className={`vital-value ${spo2 < 90 ? 'triage-c' : spo2 < 94 ? 'triage-b' : 'triage-a'}`}>
              {spo2.toFixed(1)}{t(lang, 'percent')}
            </div>
            <div className="vital-label">{t(lang, 'spo2')}</div>
          </div>
          <div className="vital-card">
            <div className="vital-value">{(distress * 100).toFixed(0)}%</div>
            <div className="vital-label">{t(lang, 'distress')}</div>
          </div>
          <div className="vital-card">
            <div className={`vital-value ${fallRisk === 'High' ? 'triage-c' : fallRisk === 'Moderate' ? 'triage-b' : 'triage-a'}`}>
              {fallRisk}
            </div>
            <div className="vital-label">{t(lang, 'fallRisk')}</div>
          </div>
        </div>
        <div style={{ textAlign: 'center', margin: '24px 0' }}>
          <span className={`badge badge-${level.toLowerCase()}`} style={{ fontSize: '1.1rem', padding: '8px 20px' }}>
            {t(lang, 'triageLevel')}: {level} — {triageLabel(lang, level)}
          </span>
        </div>
        {triage?.reasons?.length > 0 && (
          <ul style={{ color: 'var(--text-muted)', marginBottom: 24, paddingLeft: 20 }}>
            {triage.reasons.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        )}
        <button type="button" className="btn btn-primary btn-large" onClick={() => navigate('/summary')}>
          {t(lang, 'proceed')}
        </button>
      </div>
    </div>
  );
}
