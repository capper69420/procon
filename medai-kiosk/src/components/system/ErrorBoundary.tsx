import { Component, type ErrorInfo, type ReactNode } from "react";
import { Button } from "../ui/Button";

interface ErrorBoundaryState {
  message: string | null;
}

export class ErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { message: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { message: error.message || "Unexpected application error" };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[MedAI ErrorBoundary]", error, info);
  }

  render() {
    if (!this.state.message) return this.props.children;

    return (
      <div className="h-screen w-screen bg-surface flex items-center justify-center p-6">
        <div className="bg-white border border-slate-100 rounded-2xl shadow-[0_2px_12px_rgba(16,24,40,0.06)] p-6 max-w-md text-center">
          <h1 className="text-lg font-bold text-ink-900 mb-2">Something went wrong</h1>
          <p className="text-sm text-ink-400 mb-5">{this.state.message}</p>
          <Button onClick={() => window.location.assign("/")}>Return Home</Button>
        </div>
      </div>
    );
  }
}
