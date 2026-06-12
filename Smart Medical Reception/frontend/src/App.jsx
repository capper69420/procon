import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import { KioskProvider, useKiosk } from './context/KioskContext';
import { isBackendOnline } from './api';
import WelcomeScreen from './screens/WelcomeScreen';
import PatientRegistration from './screens/PatientRegistration';
import CameraScan from './screens/CameraScan';
import HealthAssessment from './screens/HealthAssessment';
import ReceptionSummary from './screens/ReceptionSummary';
import DoctorDashboard from './screens/DoctorDashboard';
import { t } from './i18n';

function AppHeader() {
  const { lang, offline, demoMode } = useKiosk();
  const location = useLocation();
  const navigate = useNavigate();
  const isHome = location.pathname === '/';

  return (
    <header className="app-header">
      <h1
        style={{ cursor: 'pointer' }}
        onClick={() => navigate('/')}
        onKeyDown={(e) => e.key === 'Enter' && navigate('/')}
        role="button"
        tabIndex={0}
      >
        🏥 {t(lang, 'welcome')}
      </h1>
      <div className="header-actions">
        {offline && <span className="badge badge-b">{t(lang, 'offline')}</span>}
        {demoMode && <span className="badge badge-b">{t(lang, 'demoMode')}</span>}
        {!isHome && (
          <button type="button" className="btn btn-ghost" onClick={() => navigate('/dashboard')}>
            {t(lang, 'doctorDashboard')}
          </button>
        )}
      </div>
    </header>
  );
}

function AppRoutes() {
  const { setOffline } = useKiosk();

  useEffect(() => {
    isBackendOnline().then((ok) => setOffline(!ok)).catch(() => setOffline(true));
    const interval = setInterval(() => {
      isBackendOnline().then((ok) => setOffline(!ok)).catch(() => setOffline(true));
    }, 15000);
    return () => clearInterval(interval);
  }, [setOffline]);

  return (
    <>
      <AppHeader />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<WelcomeScreen />} />
          <Route path="/register" element={<PatientRegistration />} />
          <Route path="/scan" element={<CameraScan />} />
          <Route path="/assessment" element={<HealthAssessment />} />
          <Route path="/summary" element={<ReceptionSummary />} />
          <Route path="/dashboard" element={<DoctorDashboard />} />
          <Route path="*" element={<WelcomeScreen />} />
        </Routes>
      </main>
    </>
  );
}

export default function App() {
  const [lang, setLang] = useState(() => localStorage.getItem('kiosk_lang') || 'en');

  useEffect(() => {
    localStorage.setItem('kiosk_lang', lang);
  }, [lang]);

  return (
    <BrowserRouter>
      <KioskProvider lang={lang} setLang={setLang}>
        <div className="app-shell">
          <AppRoutes />
        </div>
      </KioskProvider>
    </BrowserRouter>
  );
}
