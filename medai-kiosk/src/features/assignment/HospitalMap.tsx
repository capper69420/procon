import { useTranslation } from "react-i18next";

const ROWS = 5;
const COLS = 7;
const START = { r: 4, c: 0 };
const END = { r: 1, c: 5 };

// simple L-shaped path from start to end
const path: { r: number; c: number }[] = [];
for (let c = 0; c <= END.c; c++) path.push({ r: START.r, c });
for (let r = START.r; r >= END.r; r--) path.push({ r, c: END.c });

const roomLabels: Record<string, string> = {
  "0-0": "Reception",
  "0-2": "Triage",
  "0-5": "Emergency",
  "2-0": "Cardiology",
  "2-2": "Radiology",
  "4-0": "Pharmacy",
  "1-5": "Room 305",
};

export function HospitalMap() {
  const { t } = useTranslation();
  const cellPct = 100 / COLS;
  const rowPct = 100 / ROWS;

  return (
    <div className="w-full">
      <div
        className="relative w-full aspect-[7/5] bg-slate-50 rounded-xl border border-slate-100 overflow-hidden"
      >
        {/* grid */}
        {Array.from({ length: ROWS }).map((_, r) =>
          Array.from({ length: COLS }).map((_, c) => (
            <div
              key={`${r}-${c}`}
              className="absolute border border-slate-200/70 flex items-center justify-center text-[8px] text-ink-400 font-medium px-0.5 text-center"
              style={{
                left: `${c * cellPct}%`,
                top: `${r * rowPct}%`,
                width: `${cellPct}%`,
                height: `${rowPct}%`,
              }}
            >
              {roomLabels[`${r}-${c}`]}
            </div>
          ))
        )}

        {/* destination highlight */}
        <div
          className="absolute bg-brand-100/60 border-2 border-brand-500 rounded"
          style={{
            left: `${END.c * cellPct}%`,
            top: `${END.r * rowPct}%`,
            width: `${cellPct}%`,
            height: `${rowPct}%`,
          }}
        />

        {/* route line */}
        <svg className="absolute inset-0 w-full h-full" viewBox={`0 0 ${COLS} ${ROWS}`} preserveAspectRatio="none">
          <polyline
            points={path.map((p) => `${p.c + 0.5},${p.r + 0.5}`).join(" ")}
            fill="none"
            stroke="#2f5fe0"
            strokeWidth={0.12}
            strokeDasharray="0.25 0.18"
            strokeLinecap="round"
          />
        </svg>

        {/* start marker */}
        <div
          className="absolute w-2.5 h-2.5 rounded-full bg-danger-500 ring-2 ring-white -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${(START.c + 0.5) * cellPct}%`, top: `${(START.r + 0.5) * rowPct}%` }}
        />
        {/* destination marker */}
        <div
          className="absolute w-2.5 h-2.5 rounded-full bg-brand-600 ring-2 ring-white -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${(END.c + 0.5) * cellPct}%`, top: `${(END.r + 0.5) * rowPct}%` }}
        />
      </div>

      <div className="flex items-center gap-4 mt-3 text-[11px] text-ink-400">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-0.5 bg-brand-500" /> {t("assignment.legendRoute")}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-danger-500" /> {t("assignment.legendHere")}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-brand-600" /> {t("assignment.legendDest")}
        </span>
      </div>
    </div>
  );
}
