interface RiskCardProps {
  letter: string;
  label: string;
  sublabel: string;
  confidence: number;
  tone: "success" | "warning" | "danger";
  active?: boolean;
}

const toneStyles = {
  success: { bg: "bg-success-50", text: "text-success-600", bar: "#16a34a", ring: "" },
  warning: { bg: "bg-warning-50", text: "text-warning-600", bar: "#d97706", ring: "ring-2 ring-warning-500/40" },
  danger: { bg: "bg-danger-50", text: "text-danger-600", bar: "#dc2626", ring: "" },
};

export function RiskCard({ letter, label, sublabel, confidence, tone, active }: RiskCardProps) {
  const s = toneStyles[tone];
  return (
    <div
      className={`flex items-center justify-between rounded-xl border border-slate-100 px-4 py-3 ${
        active ? `${s.ring} bg-warning-50/40` : ""
      }`}
    >
      <div className="flex items-center gap-3">
        <div className={`w-9 h-9 rounded-lg ${s.bg} ${s.text} flex items-center justify-center font-bold`}>
          {letter}
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm text-ink-900">{label}</span>
            {active && (
              <span className="text-[10px] font-bold text-warning-600 bg-warning-50 px-1.5 py-0.5 rounded">
                ACTIVE
              </span>
            )}
          </div>
          <span className="text-xs text-ink-400">{sublabel}</span>
        </div>
      </div>
      <div className="text-right">
        <div className="text-xs text-ink-400 mb-0.5">Confidence</div>
        <div className="font-bold text-sm" style={{ color: s.bar }}>
          {confidence}%
        </div>
      </div>
    </div>
  );
}
