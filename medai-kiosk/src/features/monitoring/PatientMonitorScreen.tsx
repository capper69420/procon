import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { Home, Activity, Settings, LayoutDashboard } from "lucide-react";
import { Card } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { ProgressBar } from "../../components/ui/ProgressBar";
import { CameraFeed } from "./CameraFeed";
import { VitalCard } from "./VitalCard";
import { FingerScan } from "./FingerScan";
import { vitalMetrics, patientProfile, consultationSummary } from "../../mocks/data";

export function PatientMonitorScreen() {
  const { t } = useTranslation();
  const navigate = useNavigate();

  return (
    <div className="h-full grid grid-cols-[56px_1fr_320px] gap-4 p-4 overflow-hidden">
      {/* mini sidebar */}
      <div className="flex flex-col items-center gap-2 pt-2">
        {[Home, LayoutDashboard, Activity, Settings].map((Icon, i) => (
          <button
            key={i}
            className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors ${
              i === 1 ? "bg-brand-600 text-white" : "text-ink-400 hover:bg-slate-100"
            }`}
          >
            <Icon size={18} />
          </button>
        ))}
      </div>

      {/* main column */}
      <div className="flex flex-col gap-4 overflow-y-auto pr-1">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="font-bold text-lg text-ink-900">{t("monitor.title")}</h1>
            <p className="text-xs text-ink-400">{t("monitor.subtitle")}</p>
          </div>
          <div className="flex items-center gap-2">
            <Badge tone="success" dot>
              {t("monitor.active")}
            </Badge>
            <Button variant="ghost" onClick={() => navigate("/")}>
              <Home size={14} />
              {t("monitor.home")}
            </Button>
          </div>
        </div>

        <Card padded={false} className="p-2">
          <CameraFeed />
        </Card>

        <div className="grid grid-cols-4 gap-3">
          {vitalMetrics.map((m) => (
            <VitalCard key={m.key} metric={m} />
          ))}
        </div>
      </div>

      {/* right info column */}
      <div className="flex flex-col gap-3 overflow-y-auto">
        <Card>
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-full bg-brand-50 text-brand-600 flex items-center justify-center font-bold">
              田
            </div>
            <div>
              <p className="font-semibold text-sm text-ink-900">{patientProfile.nameJa}</p>
              <p className="text-[11px] text-ink-400">
                {patientProfile.age} · {patientProfile.gender} · {patientProfile.bloodType}
              </p>
            </div>
          </div>
          <p className="text-[11px] text-ink-400 mb-2">{patientProfile.patientId}</p>
          <Badge tone="warning" className="w-full justify-center py-1.5">
            {t("monitor.moderateRisk")} — Awaiting Physician
          </Badge>
        </Card>

        <Card>
          <div className="flex items-center justify-between mb-2">
            <h2 className="font-semibold text-sm text-ink-900">{t("monitor.aiAssessment")}</h2>
            <Badge tone="warning">{t("monitor.moderateRisk")}</Badge>
          </div>
          <div className="flex items-center justify-between text-xs text-ink-400 mb-1">
            <span>{t("triage.confidence")}</span>
            <span className="font-bold text-warning-600">78%</span>
          </div>
          <ProgressBar value={78} color="#d97706" />
          <div className="flex items-center justify-between text-xs text-ink-400 mt-3">
            <span>{t("monitor.riskScore")}</span>
            <span className="font-bold text-ink-900">78/100</span>
          </div>
          <p className="text-[11px] text-ink-400 mt-1">
            {t("monitor.reviewIn")}: 30 {t("monitor.min")}
          </p>
        </Card>

        <Card className="flex-1">
          <h2 className="font-semibold text-sm text-ink-900 mb-3">{t("monitor.summaryTitle")}</h2>
          <ul className="space-y-2.5">
            {consultationSummary.map((line, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-ink-600">
                <span className="w-1 h-1 rounded-full bg-brand-400 mt-1.5 shrink-0" />
                {line}
              </li>
            ))}
          </ul>
          <p className="text-[10px] text-ink-400 mt-3">{t("monitor.updated")} 2:00 PM</p>
        </Card>

        <FingerScan />

        <Button variant="primary" size="lg" className="w-full" onClick={() => navigate("/assignment")}>
          {t("monitor.proceed")}
        </Button>
      </div>
    </div>
  );
}
