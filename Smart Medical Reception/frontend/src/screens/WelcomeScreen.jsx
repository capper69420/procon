import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useKiosk } from '../context/KioskContext';
import { t } from '../i18n';

export default function WelcomeScreen() {
  const navigate = useNavigate();
  const { lang, setLang, demoMode } = useKiosk();

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen?.().catch(() => {});
    } else {
      document.exitFullscreen?.().catch(() => {});
    }
  };

  return (
    <div className="screen">
      <div className="card welcome-hero">
        <div className="welcome-icon" aria-hidden>🏥</div>
        <h1 className="welcome-title">{t(lang, 'welcome')}</h1>
        <p className="card-subtitle">{t(lang, 'subtitle')}</p>
        {demoMode && (
          <div className="status-bar" style={{ justifyContent: 'center' }}>
            <span className="status-dot warn" />
            {t(lang, 'demoMode')}
          </div>
        )}
        <div className="welcome-actions">
          <button
            type="button"
            className="btn btn-primary btn-large"
            onClick={() => navigate('/register')}
          >
            {t(lang, 'startCheckIn')}
          </button>
          <button
            type="button"
            className="btn btn-secondary btn-large"
            onClick={() => navigate('/dashboard')}
          >
            {t(lang, 'doctorDashboard')}
          </button>
          <div className="btn-row">
            <button type="button" className="btn btn-ghost" onClick={() => setLang(lang === 'en' ? 'ja' : 'en')}>
              {t(lang, 'language')}
            </button>
            <button type="button" className="btn btn-ghost" onClick={toggleFullscreen}>
              {t(lang, 'fullscreen')}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
