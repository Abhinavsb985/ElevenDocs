// This file was removed as part of removing the language translation feature.

// The LanguageContext and related functionality have been removed from the project.
import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

type SupportedLanguage = "en" | "es" | "hi";

interface LanguageContextType {
  language: SupportedLanguage;
  setLanguage: (lang: SupportedLanguage) => void;
  t: (key: string, englishFallback?: string) => string;
}

const STORAGE_KEY = "site_language";

// Minimal dictionary example. Add keys over time as needed.
const DICTIONARY: Record<SupportedLanguage, Record<string, string>> = {
  en: {
    // Keys intentionally empty; use englishFallback parameter as default.
  },
  es: {},
  hi: {},
};

const SUPPORTED: SupportedLanguage[] = ["en", "es", "hi"];

function normalizeLanguage(lang: string | null | undefined): SupportedLanguage {
  if (!lang) return "en";
  const lower = lang.toLowerCase();
  if (SUPPORTED.includes(lower as SupportedLanguage)) return lower as SupportedLanguage;
  const base = lower.split("-")[0];
  return (SUPPORTED.includes(base as SupportedLanguage) ? base : "en") as SupportedLanguage;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const useLanguage = (): LanguageContextType => {
  const ctx = useContext(LanguageContext);
  if (!ctx) throw new Error("useLanguage must be used within LanguageProvider");
  return ctx;
};

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [language, setLanguageState] = useState<SupportedLanguage>("en");

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    const initial = normalizeLanguage(saved || navigator.language);
    setLanguageState(initial);
  }, []);

  const setLanguage = (lang: SupportedLanguage) => {
    const normalized = normalizeLanguage(lang);
    localStorage.setItem(STORAGE_KEY, normalized);
    setLanguageState(normalized);
    document.documentElement.lang = normalized;
  };

  const t = useMemo(() => {
    return (key: string, englishFallback?: string) => {
      const dict = DICTIONARY[language] || {};
      if (key in dict) return dict[key];
      const fallbackDict = DICTIONARY["en"] || {};
      if (key in fallbackDict) return fallbackDict[key];
      return englishFallback ?? key;
    };
  }, [language]);

  const value: LanguageContextType = { language, setLanguage, t };

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
};


