import { useTranslation } from "react-i18next";

export function CameraFeed() {
  const { t } = useTranslation();

  return (
    <div className="relative w-full aspect-video rounded-xl overflow-hidden bg-[#0b1220]">
      {/* simulated camera gradient background */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_40%,#1a2740_0%,#0b1220_70%)]" />

      {/* scanlines texture */}
      <div
        className="absolute inset-0 opacity-[0.08]"
        style={{
          backgroundImage:
            "repeating-linear-gradient(0deg, transparent, transparent 2px, white 2px, white 3px)",
        }}
      />

      {/* top-left LIVE badge */}
      <div className="absolute top-3 left-3 flex items-center gap-1.5 text-white text-xs font-medium bg-black/40 backdrop-blur px-2.5 py-1 rounded-md">
        <span className="w-1.5 h-1.5 rounded-full bg-danger-500 pulse-dot" />
        CAM-01 · {t("monitor.live")}
      </div>

      {/* top-right resolution */}
      <div className="absolute top-3 right-3 text-[11px] text-white/60 font-mono">1920×1080</div>

      {/* simulated person silhouette with YOLO bounding box */}
      <div className="absolute left-1/2 top-[58%] -translate-x-1/2 -translate-y-1/2">
        <div className="relative w-40 h-52">
          <div className="absolute inset-0 rounded-[40%] bg-white/[0.06] blur-sm" />
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 130">
            <ellipse cx="50" cy="22" rx="16" ry="20" fill="white" opacity="0.12" />
            <path d="M20 130 Q20 60 50 60 Q80 60 80 130 Z" fill="white" opacity="0.10" />
          </svg>
          {/* bounding box */}
          <div className="absolute -inset-2 border-2 border-brand-400/80 rounded-lg">
            <span className="absolute -top-5 left-0 text-[10px] font-mono text-brand-300 bg-brand-600/80 px-1.5 py-0.5 rounded">
              ID: 96.4%
            </span>
          </div>
          {/* facial landmark dots */}
          {[
            [50, 12],
            [44, 18],
            [56, 18],
            [50, 25],
          ].map(([x, y], i) => (
            <span
              key={i}
              className="absolute w-1 h-1 rounded-full bg-success-500"
              style={{ left: `${x}%`, top: `${y}%` }}
            />
          ))}
        </div>
      </div>

      {/* floating vital bubbles */}
      <div className="absolute left-[28%] top-[30%] flex flex-col items-center">
        <div className="w-12 h-12 rounded-full bg-danger-500/15 border border-danger-500/40 flex items-center justify-center text-danger-300 text-xs font-bold">
          82
        </div>
        <span className="text-[9px] text-white/50 mt-1">HR</span>
      </div>
      <div className="absolute right-[24%] top-[26%] flex flex-col items-center">
        <div className="w-12 h-12 rounded-full bg-warning-500/15 border border-warning-500/40 flex items-center justify-center text-warning-300 text-xs font-bold">
          94%
        </div>
        <span className="text-[9px] text-white/50 mt-1">SpO2</span>
      </div>

      {/* bottom info bar */}
      <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between text-[11px] text-white/70 font-mono">
        <span>{t("monitor.yolo")}</span>
        <span>2:00 / 6:00</span>
      </div>
    </div>
  );
}
