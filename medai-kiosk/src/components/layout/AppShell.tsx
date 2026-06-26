import type { ReactNode } from "react";
import { TopNav } from "./TopNav";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="kiosk-shell h-screen w-screen flex flex-col bg-surface">
      <TopNav />
      <main className="flex-1 overflow-hidden">{children}</main>
    </div>
  );
}
