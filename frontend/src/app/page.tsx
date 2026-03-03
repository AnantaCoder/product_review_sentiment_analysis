"use client";
import { useState, useEffect } from "react";

interface ReviewEntry {
  id: string;
  review: string;
  sentiment: string;
  confidence: number;
  timestamp: string;
}

const STORAGE_KEY = "sentiment_reviews";

const sentimentColor = (s: string) => {
  if (s?.toLowerCase() === "positive") return "text-emerald-400";
  if (s?.toLowerCase() === "negative") return "text-red-400";
  return "text-yellow-400";
};

const sentimentBg = (s: string) => {
  if (s?.toLowerCase() === "positive") return "bg-emerald-500";
  if (s?.toLowerCase() === "negative") return "bg-red-500";
  return "bg-yellow-400";
};

const sentimentBorder = (s: string) => {
  if (s?.toLowerCase() === "positive") return "border-emerald-500/30";
  if (s?.toLowerCase() === "negative") return "border-red-500/30";
  return "border-yellow-400/30";
};

export default function MovieReviewPage() {
  const [review, setReview] = useState("");
  const [loading, setLoading] = useState(false);
  const [responseData, setResponseData] = useState<any>(null);
  const [error, setError] = useState("");
  const [focused, setFocused] = useState(false);
  const [history, setHistory] = useState<ReviewEntry[]>([]);
  const [activeTab, setActiveTab] = useState<"analyze" | "history">("analyze");

  // Load history from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) setHistory(JSON.parse(stored));
    } catch {}
  }, []);

  const saveToHistory = (entry: ReviewEntry) => {
    setHistory((prev) => {
      const updated = [entry, ...prev];
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      } catch {}
      return updated;
    });
  };

  const clearHistory = () => {
    setHistory([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {}
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!review.trim()) {
      setError("Please enter a review.");
      return;
    }
    try {
      setLoading(true);
      setError("");
      const res = await fetch(process.env.NEXT_PUBLIC_BACKEND_URI!, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review }),
      });
      const data = await res.json();
      setResponseData(data);

      // Save to localStorage history
      const entry: ReviewEntry = {
        id: Date.now().toString(),
        review: review.trim(),
        sentiment: data.sentiment,
        confidence: data.confidence,
        timestamp: new Date().toLocaleString(),
      };
      saveToHistory(entry);
      setReview("");
    } catch (err) {
      setError("Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  };

  // Stats derived from history
  const totalReviews = history.length;
  const positiveCount = history.filter((h) => h.sentiment?.toLowerCase() === "positive").length;
  const negativeCount = history.filter((h) => h.sentiment?.toLowerCase() === "negative").length;
  const neutralCount = history.filter((h) => h.sentiment?.toLowerCase() === "neutral").length;
  const avgConfidence =
    history.length > 0
      ? Math.round(history.reduce((a, b) => a + b.confidence, 0) / history.length)
      : 0;

  return (
    <div className="min-h-screen bg-[#0a0a0a] flex flex-col items-center px-4 py-12 relative overflow-hidden">

      {/* Ambient glows */}
      <div className="pointer-events-none fixed top-0 left-1/2 -translate-x-1/2 w-[700px] h-[320px] bg-red-700/10 blur-[100px] rounded-full z-0" />
      <div className="pointer-events-none fixed bottom-0 right-0 w-[400px] h-[400px] bg-orange-500/5 blur-[120px] rounded-full z-0" />

      <div className="w-full max-w-2xl relative z-10">

        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-3">
            <span className="block w-8 h-px bg-gradient-to-r from-transparent to-orange-500" />
            <span className="text-orange-500 text-[11px] font-semibold tracking-[0.22em] uppercase">
              AI Powered Analysis
            </span>
            <span className="block w-8 h-px bg-gradient-to-l from-transparent to-orange-500" />
          </div>
          <h1 className="text-6xl sm:text-7xl font-black uppercase tracking-tight leading-none bg-gradient-to-br from-white via-yellow-300 to-red-600 bg-clip-text text-transparent mb-3">
            Sentiment<br />Analysis
          </h1>
          <p className="text-zinc-600 text-sm tracking-widest font-light">
            Submit a product review to uncover its sentiment
          </p>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 bg-zinc-900 border border-zinc-800 rounded-xl p-1 mb-6">
          {(["analyze", "history"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex-1 py-2 rounded-lg text-sm font-semibold uppercase tracking-widest transition-all duration-200 ${
                activeTab === tab
                  ? "bg-gradient-to-r from-red-600 to-orange-500 text-white shadow-[0_2px_12px_rgba(220,38,38,0.3)]"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {tab === "history" ? `History (${totalReviews})` : tab}
            </button>
          ))}
        </div>

        {/* ── ANALYZE TAB ── */}
        {activeTab === "analyze" && (
          <>
            {/* Form Card */}
            <div className="bg-zinc-900/80 border border-zinc-800 rounded-2xl p-7 relative overflow-hidden shadow-2xl mb-6">
              <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-orange-500/40 to-transparent" />

              <form onSubmit={handleSubmit} className="flex flex-col gap-5">
                <div className="flex flex-col gap-2">
                  <label className="text-[11px] uppercase tracking-[0.18em] text-zinc-500 font-medium">
                    Your Review
                  </label>
                  <textarea
                    value={review}
                    onChange={(e) => setReview(e.target.value)}
                    onFocus={() => setFocused(true)}
                    onBlur={() => setFocused(false)}
                    placeholder="Write your product review here..."
                    rows={5}
                    className={`w-full bg-[#0d0d0d] rounded-xl px-4 py-3 text-zinc-200 text-sm font-light leading-relaxed placeholder:text-zinc-700 resize-none outline-none border transition-all duration-200 caret-orange-400 ${
                      focused
                        ? "border-orange-500 shadow-[0_0_0_3px_rgba(249,115,22,0.1)]"
                        : "border-zinc-800 hover:border-zinc-700"
                    }`}
                  />
                  <span className="text-right text-[11px] text-zinc-700">{review.length} characters</span>
                </div>

                {error && (
                  <div className="flex items-center gap-2 text-red-400 text-sm bg-red-500/5 border border-red-500/15 rounded-lg px-4 py-2.5">
                    <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="12" y1="8" x2="12" y2="12" />
                      <line x1="12" y1="16" x2="12.01" y2="16" />
                    </svg>
                    {error}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className={`relative w-full py-3.5 rounded-xl font-black text-lg uppercase tracking-widest overflow-hidden transition-all duration-200 ${
                    loading
                      ? "bg-zinc-800 text-zinc-600 cursor-not-allowed"
                      : "bg-gradient-to-r from-red-600 to-orange-500 text-white shadow-[0_4px_24px_rgba(220,38,38,0.35)] hover:-translate-y-0.5 hover:shadow-[0_8px_32px_rgba(220,38,38,0.5)] active:translate-y-0"
                  }`}
                >
                  {!loading && (
                    <span className="absolute inset-0 -translate-x-full animate-[shimmer_2s_infinite] bg-gradient-to-r from-transparent via-white/10 to-transparent" />
                  )}
                  {loading ? (
                    <span className="flex items-center justify-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-zinc-500 animate-bounce [animation-delay:0ms]" />
                      <span className="w-2 h-2 rounded-full bg-zinc-500 animate-bounce [animation-delay:150ms]" />
                      <span className="w-2 h-2 rounded-full bg-zinc-500 animate-bounce [animation-delay:300ms]" />
                    </span>
                  ) : (
                    "Analyze Review"
                  )}
                </button>
              </form>
            </div>

            {/* Result + Confidence Graph */}
            {responseData && (
              <div className="animate-[fadeUp_0.4s_ease] space-y-4">

                {/* Sentiment Badge */}
                <div className={`bg-zinc-900 border ${sentimentBorder(responseData.sentiment)} rounded-2xl p-6 relative overflow-hidden`}>
                  <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-red-600 via-orange-400 to-yellow-300 opacity-60" />
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div>
                      <p className="text-[10px] uppercase tracking-[0.18em] text-zinc-500 font-semibold mb-1">Detected Sentiment</p>
                      <p className={`text-3xl font-black uppercase tracking-wide ${sentimentColor(responseData.sentiment)}`}>
                        {responseData.sentiment}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] uppercase tracking-[0.18em] text-zinc-500 font-semibold mb-1">Confidence</p>
                      <p className="text-3xl font-black text-white">{responseData.confidence}%</p>
                    </div>
                  </div>

                  {/* Confidence bar */}
                  <div className="mt-5">
                    <div className="flex justify-between text-[10px] text-zinc-600 mb-1.5 uppercase tracking-widest">
                      <span>0%</span>
                      <span>Confidence Score</span>
                      <span>100%</span>
                    </div>
                    <div className="w-full h-3 bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-700 ease-out ${sentimentBg(responseData.sentiment)}`}
                        style={{ width: `${responseData.confidence}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] text-zinc-600 mt-1">
                      <span>Low confidence</span>
                      <span>High confidence</span>
                    </div>
                  </div>
                </div>

                {/* Confidence Gauge Chart */}
                <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500 font-semibold mb-4">Confidence Breakdown</p>

                  {/* Stacked segment bar */}
                  <div className="w-full h-8 bg-zinc-800 rounded-lg overflow-hidden flex mb-3">
                    <div
                      className="h-full bg-gradient-to-r from-red-700 to-red-500 transition-all duration-700"
                      style={{ width: "33.33%" }}
                    />
                    <div
                      className="h-full bg-gradient-to-r from-yellow-600 to-yellow-400 transition-all duration-700"
                      style={{ width: "33.33%" }}
                    />
                    <div
                      className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 transition-all duration-700"
                      style={{ width: "33.33%" }}
                    />
                    {/* Confidence marker */}
                    <div
                      className="absolute h-8 w-0.5 bg-white shadow-[0_0_8px_white] rounded transition-all duration-700"
                      style={{ marginLeft: `calc(${responseData.confidence}% - 1px)`, position: "relative" }}
                    />
                  </div>

                  {/* Zones legend */}
                  <div className="grid grid-cols-3 gap-2 mt-2">
                    {[
                      { label: "Low", range: "0–33%", color: "bg-red-500" },
                      { label: "Medium", range: "34–66%", color: "bg-yellow-400" },
                      { label: "High", range: "67–100%", color: "bg-emerald-500" },
                    ].map(({ label, range, color }) => (
                      <div key={label} className="flex items-center gap-2">
                        <span className={`w-2.5 h-2.5 rounded-sm shrink-0 ${color}`} />
                        <div>
                          <p className="text-[10px] text-zinc-400 font-medium">{label}</p>
                          <p className="text-[9px] text-zinc-600">{range}</p>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Score pointer row */}
                  <div className="mt-4 flex items-center gap-3 bg-zinc-800/60 rounded-lg px-4 py-3">
                    <div className={`w-3 h-3 rounded-full shrink-0 ${sentimentBg(responseData.sentiment)}`} />
                    <p className="text-zinc-300 text-sm">
                      Your review scored{" "}
                      <span className={`font-bold ${sentimentColor(responseData.sentiment)}`}>
                        {responseData.confidence}%
                      </span>{" "}
                      confidence as{" "}
                      <span className={`font-bold ${sentimentColor(responseData.sentiment)}`}>
                        {responseData.sentiment}
                      </span>
                    </p>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* ── HISTORY TAB ── */}
        {activeTab === "history" && (
          <div className="animate-[fadeUp_0.3s_ease]">

            {history.length === 0 ? (
              <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-12 text-center">
                <p className="text-4xl mb-3">📭</p>
                <p className="text-zinc-500 text-sm tracking-wide">No reviews analyzed yet.</p>
                <p className="text-zinc-700 text-xs mt-1">Submit a review to see history here.</p>
              </div>
            ) : (
              <>
                {/* Stats row */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
                  {[
                    { label: "Total", value: totalReviews, color: "text-orange-400" },
                    { label: "Positive", value: positiveCount, color: "text-emerald-400" },
                    { label: "Negative", value: negativeCount, color: "text-red-400" },
                    { label: "Avg Conf.", value: `${avgConfidence}%`, color: "text-yellow-300" },
                  ].map(({ label, value, color }) => (
                    <div key={label} className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 text-center">
                      <p className={`text-2xl font-black ${color}`}>{value}</p>
                      <p className="text-[10px] uppercase tracking-widest text-zinc-600 mt-0.5">{label}</p>
                    </div>
                  ))}
                </div>

                {/* Sentiment distribution bar */}
                {totalReviews > 0 && (
                  <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5 mb-5">
                    <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500 font-semibold mb-3">Sentiment Distribution</p>
                    <div className="w-full h-5 rounded-full overflow-hidden flex gap-0.5">
                      {positiveCount > 0 && (
                        <div
                          className="h-full bg-emerald-500 rounded-l-full transition-all duration-500"
                          style={{ width: `${(positiveCount / totalReviews) * 100}%` }}
                          title={`Positive: ${positiveCount}`}
                        />
                      )}
                      {neutralCount > 0 && (
                        <div
                          className="h-full bg-yellow-400 transition-all duration-500"
                          style={{ width: `${(neutralCount / totalReviews) * 100}%` }}
                          title={`Neutral: ${neutralCount}`}
                        />
                      )}
                      {negativeCount > 0 && (
                        <div
                          className="h-full bg-red-500 rounded-r-full transition-all duration-500"
                          style={{ width: `${(negativeCount / totalReviews) * 100}%` }}
                          title={`Negative: ${negativeCount}`}
                        />
                      )}
                    </div>
                    <div className="flex gap-4 mt-2">
                      {[
                        { label: `Positive (${positiveCount})`, color: "bg-emerald-500" },
                        { label: `Neutral (${neutralCount})`, color: "bg-yellow-400" },
                        { label: `Negative (${negativeCount})`, color: "bg-red-500" },
                      ].map(({ label, color }) => (
                        <div key={label} className="flex items-center gap-1.5">
                          <span className={`w-2 h-2 rounded-full ${color}`} />
                          <span className="text-[10px] text-zinc-500">{label}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Review list */}
                <div className="space-y-3 mb-4">
                  {history.map((entry, i) => (
                    <div
                      key={entry.id}
                      className={`group bg-zinc-900 border ${sentimentBorder(entry.sentiment)} rounded-xl p-5 relative overflow-hidden transition-all duration-200 hover:-translate-y-0.5`}
                    >
                      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-red-600 via-orange-400 to-yellow-300 opacity-40 group-hover:opacity-80 transition-opacity duration-200" />

                      <div className="flex items-start justify-between gap-3 mb-2">
                        <p className="text-zinc-300 text-sm font-light leading-relaxed line-clamp-2 flex-1">
                          {entry.review}
                        </p>
                        <span className={`shrink-0 text-[10px] font-bold uppercase px-2 py-1 rounded-md border ${sentimentBorder(entry.sentiment)} ${sentimentColor(entry.sentiment)} bg-zinc-800`}>
                          {entry.sentiment}
                        </span>
                      </div>

                      {/* Mini confidence bar */}
                      <div className="flex items-center gap-3 mt-3">
                        <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${sentimentBg(entry.sentiment)}`}
                            style={{ width: `${entry.confidence}%` }}
                          />
                        </div>
                        <span className="text-[11px] text-zinc-500 shrink-0">{entry.confidence}%</span>
                        <span className="text-[10px] text-zinc-700 shrink-0">{entry.timestamp}</span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Clear button */}
                <button
                  onClick={clearHistory}
                  className="w-full py-3 rounded-xl border border-zinc-800 text-zinc-500 text-sm uppercase tracking-widest font-semibold hover:border-red-500/40 hover:text-red-400 transition-all duration-200"
                >
                  Clear History
                </button>
              </>
            )}
          </div>
        )}

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