import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { ErrorBoundary } from "./components/system/ErrorBoundary";
import { KioskSessionProvider } from "./state/KioskSessionContext";
import "./i18n";
import "./index.css";
import App from "./App.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <BrowserRouter>
        <KioskSessionProvider>
          <App />
        </KioskSessionProvider>
      </BrowserRouter>
    </ErrorBoundary>
  </StrictMode>
);
