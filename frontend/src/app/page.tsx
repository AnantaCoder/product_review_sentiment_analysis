"use client";
import { useState, useEffect } from "react";
import Header from "../components/Header";
import AnalyzeTab from "../components/AnalyzeTab";
import HistoryTab from "../components/HistoryTab";
import { ReviewEntry } from "../types";

const STORAGE_KEY = "sentiment_reviews";

export default function MovieReviewPage() {
  const [history, setHistory] = useState<ReviewEntry[]>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });
  const [activeTab, setActiveTab] = useState<"analyze" | "history">("analyze");

  const saveToHistory = (entry: ReviewEntry) => {
    setHistory((prev) => {
      const updated = [entry, ...prev];
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      } catch { }
      return updated;
    });
  };

  const clearHistory = () => {
    setHistory([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch { }
  };

  const totalReviews = history.length;

  return (
    <div className="min-h-screen bg-[#0a0a0a] flex flex-col items-center px-4 py-12 relative overflow-hidden">

      {/* Ambient glows */}
      <div className="pointer-events-none fixed top-0 left-1/2 -translate-x-1/2 w-[700px] h-[320px] bg-red-700/10 blur-[100px] rounded-full z-0" />
      <div className="pointer-events-none fixed bottom-0 right-0 w-[400px] h-[400px] bg-orange-500/5 blur-[120px] rounded-full z-0" />

      <div className="w-full max-w-5xl relative z-10 transition-all duration-300">

        <Header />

        {/* Tabs */}
        <div className="flex gap-1 bg-zinc-900 border border-zinc-800 rounded-xl p-1 mb-6 max-w-2xl mx-auto">
          {(["analyze", "history"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex-1 py-2 rounded-lg text-sm font-semibold uppercase tracking-widest transition-all duration-200 ${activeTab === tab
                ? "bg-gradient-to-r from-red-600 to-orange-500 text-white shadow-[0_2px_12px_rgba(220,38,38,0.3)]"
                : "text-zinc-500 hover:text-zinc-300"
                }`}
            >
              {tab === "history" ? `History (${totalReviews})` : tab}
            </button>
          ))}
        </div>

        {/* ── ANALYZE TAB ── */}
        <div style={{ display: activeTab === "analyze" ? "block" : "none" }}>
          <AnalyzeTab onAnalyzed={saveToHistory} />
        </div>

        {/* ── HISTORY TAB ── */}
        <div style={{ display: activeTab === "history" ? "block" : "none" }}>
          <HistoryTab history={history} onClearHistory={clearHistory} />
        </div>

      </div>

      <style>{`
        @keyframes shimmer {
          0%   { transform: translateX(-100%); }
          100% { transform: translateX(300%); }
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(14px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}