import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, AlertTriangle, AlertCircle, Info } from "lucide-react";
import { useEffect, useState } from "react";
import { Card } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { ProgressBar } from "../../components/ui/ProgressBar";
import { Waveform } from "./Waveform";
import { RiskCard } from "./RiskCard";
import { transcriptScript, triageFindings, quickSymptomTags } from "../../mocks/data";

const findingIcon = {
  neutral: Info,
  warning: AlertTriangle,
  danger: AlertCircle,
};
const findingColor = {
  neutral: "text-ink-400",
  warning: "text-warning-500",
  danger: "text-danger-500",
};

export function VoiceTriageScreen() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [visibleLines, setVisibleLines] = useState(0);
  const [progress, setProgress] = useState(15);
  const [recording, setRecording] = useState(true);

  useEffect(() => {
    if (visibleLines >= transcriptScript.length) return;
    const id = setTimeout(() => setVisibleLines((v) => v + 1), 1400);
    return () => clearTimeout(id);
  }, [visibleLines]);

  useEffect(() => {
    const id = setInterval(() => setProgress((p) => (p < 85 ? p + 1 : 85)), 90);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="h-full grid grid-cols-[1fr_360px] gap-4 p-4 overflow-hidden">
      {/* Left column */}
      <div className="flex flex-col gap-4 overflow-hidden">
        <Card padded={false} className="p-4 shrink-0">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h1 className="font-bold text-lg text-ink-900">{t("triage.title")}</h1>
              <p className="text-xs text-ink-400">{t("triage.subtitle")}</p>
            </div>
            <div className="flex items-center gap-2">
              <Badge tone="danger" dot>
                {t("triage.live")}
              </Badge>
              <Button variant="ghost" size="md" onClick={() => navigate("/")}>
                <ArrowLeft size={15} />
                {t("triage.back")}
              </Button>
            </div>
          </div>
          <div className="flex items-center gap-3 text-xs text-ink-600 mb-2">
            <span className="font-medium text-brand-600">① {t("triage.step1")}</span>
            <span>→</span>
            <span className="font-medium text-brand-600">② {t("triage.step2")}</span>
            <span>→</span>
            <span className="text-ink-400">{t("triage.step3")}</span>
          </div>
          <ProgressBar value={progress} />
        </Card>

        <Card className="flex-1 flex flex-col overflow-hidden">
          <div className="flex items-center justify-between mb-3 shrink-0">
            <h2 className="font-semibold text-sm text-ink-900">{t("triage.transcriptTitle")}</h2>
            <Button
              variant={recording ? "outline" : "primary"}
              size="md"
              onClick={() => setRecording((r) => !r)}
              className="!text-danger-600 !border-danger-200"
            >
              {t("triage.stop")}
            </Button>
          </div>

          <div className="bg-slate-50 rounded-xl p-3 mb-4 shrink-0">
            <Waveform active={recording} />
          </div>

          <div className="flex-1 overflow-y-auto space-y-3 pr-1">
            {transcriptScript.slice(0, visibleLines).map((line) => (
              <div key={line.id} className="animate-fade-in-up">
                <div className="flex items-baseline gap-2 mb-0.5">
                  <span className="text-[11px] font-mono text-ink-400">{line.timestamp}</span>
                  <span className="text-sm text-ink-900">{line.ja}</span>
                </div>
                <p className="text-xs text-ink-400 pl-9">{line.en}</p>
              </div>
            ))}
            {recording && (
              <div className="flex items-center gap-1.5 text-xs text-brand-500 font-medium pl-1">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-500 pulse-dot" />
                {t("triage.listening")}
              </div>
            )}
          </div>

          <div className="shrink-0 pt-4 mt-2 border-t border-slate-100">
            <p className="text-xs font-medium text-ink-400 mb-2">{t("triage.quickTags")}</p>
            <div className="flex flex-wrap gap-2">
              {quickSymptomTags.map((tag) => (
                <span
                  key={tag.key}
                  className="text-xs font-medium text-brand-600 bg-brand-50 px-3 py-1.5 rounded-full"
                >
                  {t(`triage.tag.${tag.key}`)}
                </span>
              ))}
            </div>
          </div>
        </Card>
      </div>

      {/* Right column */}
      <div className="flex flex-col gap-3 overflow-y-auto">
        <Card>
          <h2 className="font-semibold text-sm text-ink-900 mb-3">{t("triage.resultsTitle")}</h2>
          <div className="space-y-2.5">
            <RiskCard letter="A" label={t("triage.stable")} sublabel="安定" confidence={18} tone="success" />
            <RiskCard
              letter="B"
              label={t("triage.moderate")}
              sublabel="中等度"
              confidence={78}
              tone="warning"
              active
            />
            <RiskCard letter="C" label={t("triage.critical")} sublabel="重症" confidence={12} tone="danger" />
          </div>
        </Card>

        <Card className="flex-1">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold text-sm text-ink-900">{t("triage.findingsTitle")}</h2>
            <Badge tone="warning">live</Badge>
          </div>
          <ul className="space-y-3">
            {triageFindings.map((f) => {
              const Icon = findingIcon[f.tone];
              return (
                <li key={f.id} className="flex items-start gap-2 text-sm text-ink-600">
                  <Icon size={15} className={`mt-0.5 shrink-0 ${findingColor[f.tone]}`} />
                  {f.label}
                </li>
              );
            })}
          </ul>
        </Card>

        <Button variant="primary" size="lg" className="w-full" onClick={() => navigate("/monitor")}>
          {t("triage.viewDashboard")}
        </Button>
      </div>
    </div>
  );
}
