export function ProgressBar({
  value,
  color = "#2f5fe0",
  trackClassName = "",
  height = 8,
}: {
  value: number;
  color?: string;
  trackClassName?: string;
  height?: number;
}) {
  return (
    <div
      className={`w-full rounded-full bg-slate-100 overflow-hidden ${trackClassName}`}
      style={{ height }}
    >
      <div
        className="h-full rounded-full transition-[width] duration-700 ease-out"
        style={{ width: `${Math.min(100, Math.max(0, value))}%`, background: color }}
      />
    </div>
  );
}
