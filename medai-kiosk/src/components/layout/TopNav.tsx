import { NavLink } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Activity, Mic, MonitorDot, MapPin } from "lucide-react";
import { useEffect, useState } from "react";
import { ConnectionStatus } from "../system/ConnectionStatus";

const tabs = [
  { to: "/", key: "home", icon: Activity },
  { to: "/triage", key: "triage", icon: Mic },
  { to: "/monitor", key: "monitor", icon: MonitorDot },
  { to: "/assignment", key: "assignment", icon: MapPin },
];

export function TopNav() {
  const { t } = useTranslation();
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const timeStr = time.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  return (
    <header className="bg-white border-b border-slate-100 px-6 pt-3 pb-0 shrink-0">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center text-white font-bold text-sm">
            +
          </div>
          <span className="font-semibold text-ink-900">MedAI Kiosk</span>
          <span className="text-[10px] font-semibold text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">
            M.A.E
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs text-ink-600">
          <ConnectionStatus />
          <span>{t("status.battery")}</span>
          <span className="font-mono tabular-nums">{timeStr}</span>
        </div>
      </div>

      <nav className="flex gap-6 -mb-px">
        {tabs.map(({ to, key, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-1.5 text-sm font-medium pb-3 border-b-2 transition-colors ${
                isActive
                  ? "text-brand-600 border-brand-600"
                  : "text-ink-400 border-transparent hover:text-ink-600"
              }`
            }
          >
            <Icon size={15} />
            {t(`nav.${key}`)}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
