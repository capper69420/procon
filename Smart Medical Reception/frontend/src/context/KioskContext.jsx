import React, { createContext, useContext, useMemo, useState, useCallback } from 'react';

const KioskContext = createContext(null);

const initialSession = {
  patient: null,
  vision: null,
  audio: null,
  triage: null,
};

export function KioskProvider({ children, lang, setLang }) {
  const [session, setSession] = useState(initialSession);
  const [offline, setOffline] = useState(false);
  const demoMode = import.meta.env.VITE_DEMO_MODE === 'true'
    || import.meta.env.DEMO_MODE === 'true';

  const resetSession = useCallback(() => setSession(initialSession), []);

  const updateSession = useCallback((patch) => {
    setSession((prev) => ({ ...prev, ...patch }));
  }, []);

  const value = useMemo(() => ({
    session,
    updateSession,
    resetSession,
    offline,
    setOffline,
    lang,
    setLang,
    demoMode,
  }), [session, updateSession, resetSession, offline, lang, setLang, demoMode]);

  return (
    <KioskContext.Provider value={value}>
      {children}
    </KioskContext.Provider>
  );
}

export function useKiosk() {
  const ctx = useContext(KioskContext);
  if (!ctx) throw new Error('useKiosk must be used within KioskProvider');
  return ctx;
}
