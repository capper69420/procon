import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useKioskSession } from "../../state/KioskSessionContext";

export function ConnectionStatus() {
  const { t } = useTranslation();
  const { state, checkConnection } = useKioskSession();

  useEffect(() => {
    void checkConnection();
    const id = window.setInterval(() => {
      void checkConnection();
    }, 15000);
    return () => window.clearInterval(id);
  }, [checkConnection]);

  const online = state.connection === "online";
  const checking = state.connection === "checking";

  return (
    <>
      <span className="flex items-center gap-1.5">
        <span className={`w-2 h-2 rounded-full ${online ? "bg-success-500" : "bg-warning-500"} pulse-dot`} />
        {checking ? "Checking" : online ? t("status.online") : "Backend Offline"}
      </span>
      <span>{online ? t("status.connected") : "Retrying"}</span>
    </>
  );
}
