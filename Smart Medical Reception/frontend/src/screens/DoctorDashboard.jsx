import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useKiosk } from '../context/KioskContext';
import { t } from '../i18n';
import { listPatients, isBackendOnline } from '../api';
import { subscribePatients } from '../supabaseClient';
import { getSeedPatients, fluctuatePatient, sortPatients } from '../localStore';

function enrichPatient(p) {
  const vitals = p.latest_vitals?.vitals || p.vitals || {};
  const audio = p.latest_audio || p.audio || {};
  const posture = p.latest_vitals?.posture || p.posture || {};
  return {
    ...p,
    name: p.name || 'Unknown Patient',
    vitals: {
      spo2: vitals.spo2 || 95,
      heart_rate: vitals.heart_rate || 72,
      signal_quality: vitals.signal_quality || 0.5,
    },
    audio: {
      distress_score: audio.distress_score ?? 0,
      breath_rate: audio.breath_rate ?? 16,
      cough_detected: audio.cough_detected ?? false,
    },
    posture: {
      fall_detected: posture.fall_detected ?? false,
      immobile_seconds: posture.immobile_seconds ?? 0,
      status: posture.status || 'UNKNOWN',
    },
    current_triage_level: p.current_triage_level || 'A',
    urgency_score: p.urgency_score ?? 0,
  };
}

export default function DoctorDashboard() {
  const navigate = useNavigate();
  const { lang, demoMode } = useKiosk();
  const [patients, setPatients] = useState(() => getSeedPatients().map(enrichPatient));
  const [search, setSearch] = useState('');
  const [levelFilter, setLevelFilter] = useState('all');
  const [selected, setSelected] = useState(null);
  const [offline, setOffline] = useState(false);
  const [loading, setLoading] = useState(true);

  const fetchPatients = useCallback(async () => {
    try {
      const online = await isBackendOnline();
      if (!online) {
        setOffline(true);
        setPatients((prev) => sortPatients(prev.length ? prev : getSeedPatients().map(enrichPatient)));
        return;
      }
      const data = await listPatients();
      if (data.length > 0) {
        setPatients(sortPatients(data.map(enrichPatient)));
        setOffline(false);
      }
    } catch {
      setOffline(true);
      setPatients((prev) => (prev.length ? prev : getSeedPatients().map(enrichPatient)));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPatients();
    const unsub = subscribePatients(() => fetchPatients());
    return unsub;
  }, [fetchPatients]);

  useEffect(() => {
    if (!offline && !demoMode) return undefined;
    const interval = setInterval(() => {
      setPatients((prev) => sortPatients(prev.map(fluctuatePatient)));
    }, 4000);
    return () => clearInterval(interval);
  }, [offline, demoMode]);

  const filtered = useMemo(() => {
    let list = patients;
    if (levelFilter !== 'all') {
      list = list.filter((p) => p.current_triage_level === levelFilter);
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter((p) =>
        (p.name || '').toLowerCase().includes(q)
        || (p.room || '').includes(q)
        || (p.id || '').includes(q),
      );
    }
    return sortPatients(list);
  }, [patients, search, levelFilter]);

  return (
    <div className="screen screen-wide">
      <div style={{ width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>{t(lang, 'queue')}</h2>
          <button type="button" className="btn btn-ghost" onClick={() => navigate('/')}>
            {t(lang, 'back')}
          </button>
        </div>

        {offline && <div className="offline-banner">{t(lang, 'offline')}</div>}
        {(demoMode || offline) && (
          <div className="status-bar" style={{ marginBottom: 16 }}>
            <span className="status-dot warn" />
            {demoMode ? t(lang, 'demoMode') : t(lang, 'offline')}
          </div>
        )}

        <div className="dashboard-toolbar">
          <input
            type="search"
            placeholder={t(lang, 'search')}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select value={levelFilter} onChange={(e) => setLevelFilter(e.target.value)}>
            <option value="all">{t(lang, 'allLevels')}</option>
            <option value="C">Level C</option>
            <option value="B">Level B</option>
            <option value="A">Level A</option>
          </select>
          <button type="button" className="btn btn-secondary" onClick={fetchPatients}>
            {t(lang, 'retry')}
          </button>
        </div>

        {loading ? (
          <div className="spinner" />
        ) : filtered.length === 0 ? (
          <p style={{ textAlign: 'center', color: 'var(--text-muted)' }}>{t(lang, 'noPatients')}</p>
        ) : (
          <div className="dashboard-grid">
            {filtered.map((p) => (
              <div
                key={p.id}
                role="button"
                tabIndex={0}
                className={`patient-card level-${(p.current_triage_level || 'a').toLowerCase()}`}
                onClick={() => setSelected(p)}
                onKeyDown={(e) => e.key === 'Enter' && setSelected(p)}
              >
                <div className="patient-card-header">
                  <div>
                    <div className="patient-name">{p.name}</div>
                    <div className="patient-meta">
                      {p.room ? `Room ${p.room} · ` : ''}Age {p.age ?? '—'}
                    </div>
                  </div>
                  <span className={`badge badge-${(p.current_triage_level || 'a').toLowerCase()}`}>
                    {p.current_triage_level || 'A'}
                  </span>
                </div>
                <div className="vitals-row">
                  <span className="vitals-chip">SpO₂ {p.vitals.spo2}%</span>
                  <span className="vitals-chip">HR {Math.round(p.vitals.heart_rate)}</span>
                  <span className="vitals-chip">
                    Distress {((p.audio.distress_score || 0) * 100).toFixed(0)}%
                  </span>
                  {p.posture.fall_detected && (
                    <span className="vitals-chip" style={{ color: 'var(--red)' }}>FALL</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {selected && (
        <div className="modal-backdrop" onClick={() => setSelected(null)} role="presentation">
          <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog">
            <h3 style={{ marginBottom: 16 }}>{t(lang, 'details')}</h3>
            <div className="summary-row"><span>{t(lang, 'name')}</span><strong>{selected.name}</strong></div>
            <div className="summary-row"><span>ID</span><span style={{ fontSize: '0.75rem' }}>{selected.id}</span></div>
            <div className="summary-row"><span>{t(lang, 'age')}</span><span>{selected.age}</span></div>
            <div className="summary-row"><span>{t(lang, 'symptoms')}</span><span>{selected.symptoms || '—'}</span></div>
            <div className="summary-row"><span>{t(lang, 'triageLevel')}</span>
              <span className={`badge badge-${(selected.current_triage_level || 'a').toLowerCase()}`}>
                {selected.current_triage_level}
              </span>
            </div>
            <div className="summary-row"><span>{t(lang, 'spo2')}</span><span>{selected.vitals.spo2}%</span></div>
            <div className="summary-row"><span>{t(lang, 'heartRate')}</span><span>{Math.round(selected.vitals.heart_rate)} bpm</span></div>
            <div className="summary-row"><span>{t(lang, 'distress')}</span><span>{((selected.audio.distress_score || 0) * 100).toFixed(0)}%</span></div>
            <div className="summary-row"><span>{t(lang, 'fallRisk')}</span><span>{selected.posture.fall_detected ? 'Yes' : 'No'}</span></div>
            <button type="button" className="btn btn-primary" style={{ marginTop: 24, width: '100%' }} onClick={() => setSelected(null)}>
              {t(lang, 'close')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
