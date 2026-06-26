import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { CheckCircle2, Clock, Cpu, Footprints } from "lucide-react";
import { Card } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { ProgressBar } from "../../components/ui/ProgressBar";
import { HospitalMap } from "./HospitalMap";
import { aiAssessmentBasis, roomAssignment } from "../../mocks/data";

export function RoomAssignmentScreen() {
  const { t } = useTranslation();
  const navigate = useNavigate();

  return (
    <div className="h-full flex flex-col gap-3 p-4 overflow-hidden">
      {/* success banner */}
      <div className="shrink-0 bg-success-600 text-white rounded-xl px-5 py-3 flex items-center justify-between animate-fade-in-up">
        <div className="flex items-center gap-2.5">
          <CheckCircle2 size={20} />
          <div>
            <p className="font-semibold text-sm">{t("assignment.bannerTitle")}</p>
            <p className="text-[11px] text-white/80">{t("assignment.bannerSubtitle")}</p>
          </div>
        </div>
        <span className="text-[11px] font-mono text-white/80">
          {t("assignment.session")} {roomAssignment.sessionId}
        </span>
      </div>

      <div className="flex-1 grid grid-cols-[1fr_360px] gap-4 overflow-hidden">
        {/* left column */}
        <div className="flex flex-col gap-4 overflow-y-auto pr-1">
          <Card>
            <h2 className="font-semibold text-sm text-ink-900 mb-3">{t("assignment.assignedDept")}</h2>
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-slate-50 rounded-lg p-3">
                <p className="text-[11px] text-ink-400 mb-0.5">{t("assignment.department")}</p>
                <p className="font-semibold text-sm text-ink-900">{roomAssignment.department}</p>
              </div>
              <div className="bg-slate-50 rounded-lg p-3">
                <p className="text-[11px] text-ink-400 mb-0.5">{t("assignment.doctor")}</p>
                <p className="font-semibold text-sm text-ink-900">{roomAssignment.doctor}</p>
              </div>
              <div className="bg-slate-50 rounded-lg p-3">
                <p className="text-[11px] text-ink-400 mb-0.5">{t("assignment.room")}</p>
                <p className="font-semibold text-sm text-ink-900">{roomAssignment.room}</p>
              </div>
              <div className="bg-slate-50 rounded-lg p-3">
                <p className="text-[11px] text-ink-400 mb-0.5">{t("assignment.eta")}</p>
                <p className="font-semibold text-sm text-ink-900">
                  {roomAssignment.queuePosition} — Est. {roomAssignment.estWaitMin} min
                </p>
              </div>
            </div>
            <div className="bg-brand-50 rounded-xl p-4 flex items-center justify-center gap-3">
              <Clock className="text-brand-500" size={22} />
              <div className="text-center">
                <p className="text-[10px] tracking-wide text-brand-500 font-semibold uppercase">
                  {t("assignment.estWait")}
                </p>
                <p className="text-xl font-bold text-brand-700">
                  {roomAssignment.estWaitMin} {t("assignment.minutes")}
                </p>
              </div>
            </div>
          </Card>

          <Card className="flex-1">
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-semibold text-sm text-ink-900">{t("assignment.navTitle")}</h2>
              <span className="text-[11px] text-ink-400 flex items-center gap-1">
                <Footprints size={13} /> ~2 min {t("assignment.walkTime")}
              </span>
            </div>
            <HospitalMap />
          </Card>
        </div>

        {/* right column */}
        <div className="flex flex-col gap-3 overflow-y-auto">
          <Card>
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="font-semibold text-sm text-ink-900">{t("assignment.aiAssessmentTitle")}</h2>
                <p className="text-[11px] text-ink-400">{t("assignment.aiAssessmentSubtitle")}</p>
              </div>
            </div>
            <p className="text-[11px] font-semibold text-ink-400 tracking-wide mb-2">
              {t("assignment.basedOn")}:
            </p>
            <ul className="space-y-2 mb-4">
              {aiAssessmentBasis.map((line, i) => (
                <li key={i} className="flex items-start gap-2 text-xs text-ink-600">
                  <CheckCircle2 size={13} className="text-success-500 mt-0.5 shrink-0" />
                  {line}
                </li>
              ))}
            </ul>

            <div className="bg-brand-50 rounded-xl p-3.5 mb-3">
              <p className="text-[10px] font-semibold text-brand-500 tracking-wide uppercase mb-1">
                {t("assignment.recommendation")}
              </p>
              <p className="font-semibold text-sm text-ink-900">{roomAssignment.department}</p>
              <p className="text-xs text-ink-600 mb-2">{roomAssignment.room}</p>
              <div className="flex items-center justify-between text-[11px] text-ink-400 mb-1">
                <span>{t("assignment.confidenceScore")}</span>
                <span className="font-bold text-brand-600">{roomAssignment.confidenceScore}%</span>
              </div>
              <ProgressBar value={roomAssignment.confidenceScore} />
            </div>

            <div className="border border-slate-100 rounded-xl p-3">
              <p className="flex items-center gap-1.5 text-[11px] font-semibold text-ink-600 mb-2">
                <Cpu size={13} /> {t("assignment.routingEngine")}
              </p>
              <div className="grid grid-cols-3 gap-2 text-center mb-2">
                <div>
                  <p className="text-[10px] text-ink-400">{t("assignment.inputs")}</p>
                  <p className="text-xs font-semibold text-ink-900">
                    {roomAssignment.inputsCount} {t("assignment.signals")}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-ink-400">{t("assignment.latency")}</p>
                  <p className="text-xs font-semibold text-ink-900">{roomAssignment.latencySec}s</p>
                </div>
                <div>
                  <p className="text-[10px] text-ink-400">{t("assignment.model")}</p>
                  <p className="text-xs font-semibold text-ink-900">{roomAssignment.model}</p>
                </div>
              </div>
              <Badge tone="success" dot className="w-full justify-center py-1">
                {t("assignment.online")}
              </Badge>
            </div>
          </Card>

          <div className="grid grid-cols-1 gap-2">
            <Button variant="primary" size="lg" onClick={() => navigate("/")}>
              {t("assignment.proceedRoom")}
            </Button>
            <Button variant="success-outline" size="md">
              {t("assignment.ticketPrinted")}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
