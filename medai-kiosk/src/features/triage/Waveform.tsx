import { useEffect, useState } from "react";

const BAR_COUNT = 48;

export function Waveform({ active = true, color = "#2f5fe0" }: { active?: boolean; color?: string }) {
  const [bars, setBars] = useState<number[]>(Array.from({ length: BAR_COUNT }, () => 20));

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => {
      setBars((prev) =>
        prev.map((v, i) => {
          const center = BAR_COUNT / 2;
          const dist = Math.abs(i - center) / center;
          const envelope = 1 - dist * 0.6;
          const target = (8 + Math.random() * 85) * envelope;
          return v + (target - v) * 0.5;
        })
      );
    }, 110);
    return () => clearInterval(id);
  }, [active]);

  return (
    <div className="flex items-center gap-[2px] h-12 w-full">
      {bars.map((h, i) => (
        <div
          key={i}
          className="flex-1 rounded-full transition-all duration-100"
          style={{ height: `${Math.max(6, h)}%`, background: color, opacity: active ? 1 : 0.3 }}
        />
      ))}
    </div>
  );
}
