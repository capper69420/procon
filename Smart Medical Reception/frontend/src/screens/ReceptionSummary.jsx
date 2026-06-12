import React, { useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useKiosk } from '../context/KioskContext';
import { t } from '../i18n';
import { saveSession } from '../localStore';

export default function ReceptionSummary() {
  const navigate = useNavigate();
  const { lang, session, resetSession, offline } = useKiosk();
  const printRef = useRef(null);

  const patient = session.patient || {};
  const vitals = session.vision?.vitals || {};
  const triage = session.triage || {};
  const level = triage.triage_level || 'A';

  const handlePrint = () => window.print();

  const handleExportPdf = () => {
    handlePrint();
  };

  const handleSave = () => {
    saveSession({
      patient,
      vitals,
      triage,
      vision: session.vision,
      saved_at: new Date().toISOString(),
    });
    alert(lang === 'ja' ? 'ローカルに保存しました' : 'Saved locally');
  };

  const handleNew = () => {
    resetSession();
    navigate('/');
  };

  return (
    <div className="screen" ref={printRef}>
      <div className="card">
        <h2 className="card-title">{t(lang, 'summary')}</h2>
        {offline && <div className="offline-banner">{t(lang, 'offline')}</div>}

        <div className="summary-section">
          <h3>{t(lang, 'patientProfile')}</h3>
          <div className="summary-row"><span>{t(lang, 'name')}</span><strong>{patient.name || 'Unknown Patient'}</strong></div>
          <div className="summary-row"><span>{t(lang, 'age')}</span><span>{patient.age ?? '—'}</span></div>
          <div className="summary-row"><span>{t(lang, 'sex')}</span><span>{patient.sex || '—'}</span></div>
          <div className="summary-row"><span>{t(lang, 'symptoms')}</span><span>{patient.symptoms || '—'}</span></div>
          <div className="summary-row"><span>{t(lang, 'conditions')}</span><span>{(patient.conditions || []).join(', ') || '—'}</span></div>
          <div className="summary-row"><span>{t(lang, 'emergencyContact')}</span><span>{patient.emergency_contact || '—'}</span></div>
        </div>

        <div className="summary-section">
          <h3>{t(lang, 'vitalSigns')}</h3>
          <div className="summary-row"><span>{t(lang, 'heartRate')}</span><span>{Math.round(vitals.heart_rate || 72)} {t(lang, 'bpm')}</span></div>
          <div className="summary-row"><span>{t(lang, 'spo2')}</span><span>{(vitals.spo2 || 95).toFixed(1)}{t(lang, 'percent')}</span></div>
        </div>

        <div className="summary-section">
          <h3>{t(lang, 'triageLevel')}</h3>
          <div className="summary-row">
            <span>Level</span>
            <span className={`badge badge-${level.toLowerCase()}`}>{level}</span>
          </div>
          {triage.reasons?.map((r, i) => (
            <div key={i} className="summary-row"><span>•</span><span>{r}</span></div>
          ))}
        </div>

        <div className="summary-section">
          <h3>{t(lang, 'recommended')}</h3>
          <p>{triage.recommended_action || 'Continue routine monitoring'}</p>
        </div>

        <div className="btn-row">
          <button type="button" className="btn btn-secondary" onClick={handlePrint}>{t(lang, 'print')}</button>
          <button type="button" className="btn btn-secondary" onClick={handleExportPdf}>{t(lang, 'exportPdf')}</button>
          <button type="button" className="btn btn-secondary" onClick={handleSave}>{t(lang, 'saveLocal')}</button>
        </div>
        <button type="button" className="btn btn-primary btn-large" style={{ marginTop: 16 }} onClick={handleNew}>
          {t(lang, 'newPatient')}
        </button>
      </div>
    </div>
  );
}
