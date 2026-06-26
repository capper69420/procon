import type { ButtonHTMLAttributes, ReactNode } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: "primary" | "outline" | "ghost" | "success-outline";
  size?: "md" | "lg";
}

const base =
  "inline-flex items-center justify-center gap-2 font-semibold rounded-xl transition-all duration-150 active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none";

const variants: Record<string, string> = {
  primary:
    "bg-brand-600 text-white shadow-[0_8px_20px_rgba(47,95,224,0.35)] hover:bg-brand-700",
  outline: "border-2 border-slate-200 text-ink-900 bg-white hover:border-brand-300 hover:bg-brand-50",
  ghost: "text-ink-600 hover:bg-slate-100",
  "success-outline": "border-2 border-success-500/30 text-success-600 bg-success-50 hover:bg-success-50",
};

const sizes: Record<string, string> = {
  md: "px-4 py-2.5 text-sm",
  lg: "px-6 py-4 text-base",
};

export function Button({
  children,
  variant = "primary",
  size = "md",
  className = "",
  ...rest
}: ButtonProps) {
  return (
    <button className={`${base} ${variants[variant]} ${sizes[size]} ${className}`} {...rest}>
      {children}
    </button>
  );
}
