import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { ChevronRight, Globe, Settings, Plus } from "lucide-react";
import { Button } from "../../components/ui/Button";

export function HomeScreen() {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();

  const toggleLanguage = () => i18n.changeLanguage(i18n.language === "ja" ? "en" : "ja");

  return (
    <div className="h-full flex items-center justify-center px-6">
      <div className="w-full max-w-md text-center animate-fade-in-up">
        <div className="w-16 h-16 mx-auto rounded-2xl bg-brand-600 flex items-center justify-center shadow-[0_10px_30px_rgba(47,95,224,0.35)] mb-5">
          <Plus className="text-white" size={30} strokeWidth={2.5} />
        </div>

        <p className="text-[11px] font-semibold tracking-widest text-brand-500 uppercase mb-2">
          {t("home.eyebrow")}
        </p>
        <h1 className="text-2xl font-bold text-ink-900 mb-1">{t("home.hospitalName")}</h1>
        <p className="text-sm text-ink-400 mb-8">Tokyo Medical Center</p>

        <h2 className="text-5xl font-extrabold text-ink-900 tracking-tight mb-2">
          {t("home.welcomeTitle")}
        </h2>
        <p className="text-lg text-ink-400 mb-6">Welcome</p>

        <div className="inline-flex items-center gap-1.5 bg-brand-50 text-brand-600 text-xs font-medium px-3 py-1.5 rounded-full mb-1.5">
          <span className="text-sm">✦</span>
          {t("home.pill")}
        </div>
        <p className="text-xs text-ink-400 mb-10">AI Healthcare Consultation System</p>

        <Button
          variant="primary"
          size="lg"
          className="w-full mb-4 text-lg"
          onClick={() => navigate("/triage")}
        >
          {t("home.bookButton")}
          <ChevronRight size={20} />
        </Button>

        <div className="grid grid-cols-2 gap-3">
          <Button variant="outline" onClick={toggleLanguage}>
            <Globe size={16} />
            {t("home.language")}
          </Button>
          <Button variant="outline">
            <Settings size={16} />
            {t("home.settings")}
          </Button>
        </div>
      </div>
    </div>
  );
}
