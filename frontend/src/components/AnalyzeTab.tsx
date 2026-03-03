"use client";
import { useState, useEffect } from "react";
import { ResponseData, ReviewEntry } from "../types";
import { sentimentColor, sentimentBg, sentimentBorder } from "../utils/sentiment";

interface AnalyzeTabProps {
    onAnalyzed: (entry: ReviewEntry) => void;
}

export default function AnalyzeTab({ onAnalyzed }: AnalyzeTabProps) {
    const [review, setReview] = useState("");
    const [loading, setLoading] = useState(false);
    const [responseData, setResponseData] = useState<ResponseData | null>(null);
    const [error, setError] = useState("");
    const [focused, setFocused] = useState(false);

    // Persist draft review so it survives tab switching or reloads
    useEffect(() => {
        const saved = sessionStorage.getItem("draft_review");
        if (saved) setReview(saved);

        const savedResponse = sessionStorage.getItem("draft_response");
        if (savedResponse) setResponseData(JSON.parse(savedResponse));
    }, []);

    useEffect(() => {
        sessionStorage.setItem("draft_review", review);
    }, [review]);

    useEffect(() => {
        if (responseData) {
            sessionStorage.setItem("draft_response", JSON.stringify(responseData));
        } else {
            sessionStorage.removeItem("draft_response");
        }
    }, [responseData]);

    const handleClear = () => {
        setReview("");
        setResponseData(null);
        setError("");
        sessionStorage.removeItem("draft_review");
        sessionStorage.removeItem("draft_response");
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

            const entry: ReviewEntry = {
                id: Date.now().toString(),
                review: review.trim(),
                sentiment: data.sentiment,
                confidence: data.confidence,
                timestamp: new Date().toLocaleString(),
            };
            onAnalyzed(entry);
            // review stays in the box after analysis
        } catch (err) {
            setError("Something went wrong. Try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className={`grid grid-cols-1 ${responseData ? 'lg:grid-cols-2' : ''} gap-6 items-start`}>
            {/* LEFT COLUMN: Input Area */}
            <div className="bg-zinc-900/80 border border-zinc-800 rounded-2xl p-7 relative overflow-hidden shadow-2xl">
                <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-orange-500/40 to-transparent" />

                <form onSubmit={handleSubmit} className="flex flex-col gap-5">
                    <div className="flex flex-col gap-2">
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-[11px] uppercase tracking-[0.18em] text-zinc-500 font-medium">
                                Your Review
                            </label>
                            {review && (
                                <button
                                    type="button"
                                    onClick={handleClear}
                                    className="text-[10px] uppercase tracking-wider text-red-400 hover:text-red-300 transition-colors bg-red-500/10 hover:bg-red-500/20 px-2 py-1 rounded"
                                >
                                    Clear
                                </button>
                            )}
                        </div>
                        <textarea
                            value={review}
                            onChange={(e) => setReview(e.target.value)}
                            onFocus={() => setFocused(true)}
                            onBlur={() => setFocused(false)}
                            placeholder="Write your product review here..."
                            rows={responseData ? 8 : 5}
                            className={`w-full bg-[#0d0d0d] rounded-xl px-4 py-3 text-zinc-200 text-sm font-light leading-relaxed placeholder:text-zinc-700 resize-none outline-none border transition-all duration-200 caret-orange-400 ${focused
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
                        className={`relative w-full py-3.5 rounded-xl font-black text-lg uppercase tracking-widest overflow-hidden transition-all duration-200 ${loading
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

            {/* RIGHT COLUMN: Results Array */}
            {responseData && (
                <div className="animate-[fadeUp_0.4s_ease] space-y-4">
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

                    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6">
                        <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500 font-semibold mb-4">Probability Breakdown</p>

                        <div className="w-full h-8 bg-zinc-800 rounded-lg overflow-hidden flex mb-3 relative">
                            <div
                                className="h-full bg-gradient-to-r from-red-600 to-red-500 transition-all duration-700"
                                style={{ width: `${responseData.insights?.raw_probabilities_percent?.Negative ?? 33.33}%` }}
                                title="Negative"
                            />
                            <div
                                className="h-full bg-gradient-to-r from-yellow-500 to-yellow-400 transition-all duration-700"
                                style={{ width: `${responseData.insights?.raw_probabilities_percent?.Neutral ?? 33.33}%` }}
                                title="Neutral"
                            />
                            <div
                                className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-700"
                                style={{ width: `${responseData.insights?.raw_probabilities_percent?.Positive ?? 33.34}%` }}
                                title="Positive"
                            />
                        </div>

                        <div className="grid grid-cols-3 gap-2 mt-2">
                            {[
                                { label: "Negative", value: responseData.insights?.raw_probabilities_percent?.Negative ?? 33.33, color: "bg-red-500" },
                                { label: "Neutral", value: responseData.insights?.raw_probabilities_percent?.Neutral ?? 33.33, color: "bg-yellow-400" },
                                { label: "Positive", value: responseData.insights?.raw_probabilities_percent?.Positive ?? 33.34, color: "bg-emerald-500" },
                            ].map(({ label, value, color }) => (
                                <div key={label} className="flex items-center gap-2">
                                    <span className={`w-2.5 h-2.5 rounded-sm shrink-0 ${color}`} />
                                    <div>
                                        <p className="text-[10px] text-zinc-400 font-medium">{label}</p>
                                        <p className="text-[9px] text-zinc-600">{value}%</p>
                                    </div>
                                </div>
                            ))}
                        </div>

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

                    {responseData.insights && (
                        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-4">
                                <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500 font-semibold">Model Insights</p>
                                <div className="text-[11px] font-medium text-zinc-400 bg-zinc-800 px-2 py-1 rounded">
                                    {responseData.insights.word_count} words
                                </div>
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                <div className="bg-zinc-800/40 rounded-xl p-4 border border-zinc-800">
                                    <p className="text-[10px] uppercase tracking-widest text-zinc-500 font-semibold mb-3">Raw Probabilities</p>
                                    <div className="space-y-2.5">
                                        {Object.entries(responseData.insights.raw_probabilities_percent).map(([key, val]) => (
                                            <div key={key}>
                                                <div className="flex justify-between text-xs mb-1">
                                                    <span className={sentimentColor(key)}>{key}</span>
                                                    <span className="text-white font-medium">{val as number}%</span>
                                                </div>
                                                <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                                                    <div
                                                        className={`h-full ${sentimentBg(key)} opacity-80`}
                                                        style={{ width: `${val}%` }}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div className="bg-zinc-800/40 rounded-xl p-4 border border-zinc-800">
                                    <p className="text-[10px] uppercase tracking-widest text-zinc-500 font-semibold mb-3">Calibrated Score</p>
                                    <div className="space-y-2.5">
                                        {Object.entries(responseData.insights.calibrated_scores).map(([key, val]) => (
                                            <div key={key}>
                                                <div className="flex justify-between text-xs mb-1">
                                                    <span className={sentimentColor(key)}>{key}</span>
                                                    <span className="text-white font-medium">{val as number}</span>
                                                </div>
                                                <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                                                    <div
                                                        className={`h-full ${sentimentBg(key)} opacity-60`}
                                                        style={{ width: `${Math.min(100, (val as number) * 20)}%` }}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
