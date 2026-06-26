import type { HTMLAttributes, ReactNode } from "react";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  padded?: boolean;
}

export function Card({ children, className = "", padded = true, ...rest }: CardProps) {
  return (
    <div
      className={`bg-white rounded-2xl border border-slate-100 shadow-[0_2px_12px_rgba(16,24,40,0.06)] ${
        padded ? "p-5" : ""
      } ${className}`}
      {...rest}
    >
      {children}
    </div>
  );
}
