import { Fingerprint } from "lucide-react";
import { useTranslation } from "react-i18next";

export function FingerScan() {
  const { t } = useTranslation();
  return (
    <div className="bg-white rounded-2xl border border-slate-100 p-4 shadow-[0_2px_12px_rgba(16,24,40,0.06)] flex flex-col items-center text-center">
      <p className="text-xs font-semibold text-ink-900 mb-3">{t("monitor.fingerScanTitle")}</p>
      <div className="relative w-16 h-16 rounded-xl bg-brand-50 flex items-center justify-center overflow-hidden mb-2">
        <Fingerprint size={32} className="text-brand-500" />
        <div className="absolute left-0 right-0 h-[2px] bg-brand-400/80 animate-[scan_2.2s_ease-in-out_infinite]" />
      </div>
      <p className="text-[11px] text-ink-400">{t("monitor.fingerScanPrompt")}</p>
      <style>{`
        @keyframes scan {
          0% { top: 4px; opacity: 0.9; }
          50% { top: 60px; opacity: 0.9; }
          100% { top: 4px; opacity: 0.9; }
        }
      `}</style>
    </div>
  );
}
