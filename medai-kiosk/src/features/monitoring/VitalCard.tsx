import { Line, LineChart, ResponsiveContainer, YAxis } from "recharts";
import { useTranslation } from "react-i18next";
import type { VitalMetric } from "../../types";

export function VitalCard({ metric }: { metric: VitalMetric }) {
  const { t } = useTranslation();
  return (
    <div className="bg-white rounded-2xl border border-slate-100 p-4 shadow-[0_2px_12px_rgba(16,24,40,0.06)]">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium text-ink-400">{t(`monitor.${metric.key}`)}</span>
        {metric.alert && (
          <span className="text-[10px] font-bold text-danger-600 bg-danger-50 px-1.5 py-0.5 rounded">
            {t("monitor.alert")}
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-1 mb-2">
        <span className="text-2xl font-bold text-ink-900">{metric.value}</span>
        <span className="text-xs text-ink-400">{metric.unit}</span>
      </div>
      <div className="h-10 -mx-1">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={metric.history}>
            <YAxis hide domain={["dataMin - 2", "dataMax + 2"]} />
            <Line type="monotone" dataKey="v" stroke={metric.color} strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <span className="text-[10px] text-ink-400">{t("monitor.vsPrior")}</span>
    </div>
  );
}
