"use client";
import { ReviewEntry } from "../types";
import { sentimentColor, sentimentBg, sentimentBorder } from "../utils/sentiment";

interface HistoryTabProps {
    history: ReviewEntry[];
    onClearHistory: () => void;
}

export default function HistoryTab({ history, onClearHistory }: HistoryTabProps) {
    const totalReviews = history.length;
    const positiveCount = history.filter((h) => h.sentiment?.toLowerCase() === "positive").length;
    const negativeCount = history.filter((h) => h.sentiment?.toLowerCase() === "negative").length;
    const neutralCount = history.filter((h) => h.sentiment?.toLowerCase() === "neutral").length;
    const avgConfidence =
        history.length > 0
            ? Math.round(history.reduce((a, b) => a + b.confidence, 0) / history.length)
            : 0;

    if (history.length === 0) {
        return (
            <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-12 text-center">
                <p className="text-4xl mb-3">📭</p>
                <p className="text-zinc-400 text-base tracking-wide">No reviews analyzed yet.</p>
                <p className="text-zinc-500 text-sm mt-1">Submit a review to see history here.</p>
            </div>
        );
    }

    return (
        <div className="animate-[fadeUp_0.3s_ease]">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
                {[
                    { label: "Total", value: totalReviews, color: "text-orange-400" },
                    { label: "Positive", value: positiveCount, color: "text-emerald-400" },
                    { label: "Negative", value: negativeCount, color: "text-red-400" },
                    { label: "Avg Conf.", value: `${avgConfidence}%`, color: "text-yellow-300" },
                ].map(({ label, value, color }) => (
                    <div key={label} className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 text-center">
                        <p className={`text-3xl font-black ${color}`}>{value}</p>
                        <p className="text-xs uppercase tracking-widest text-zinc-500 mt-1">{label}</p>
                    </div>
                ))}
            </div>

            <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5 mb-5">
                <p className="text-xs uppercase tracking-[0.18em] text-zinc-400 font-semibold mb-3">Sentiment Distribution</p>
                <div className="w-full h-6 rounded-full overflow-hidden flex gap-0.5">
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
                <div className="flex gap-4 mt-3">
                    {[
                        { label: `Positive (${positiveCount})`, color: "bg-emerald-500" },
                        { label: `Neutral (${neutralCount})`, color: "bg-yellow-400" },
                        { label: `Negative (${negativeCount})`, color: "bg-red-500" },
                    ].map(({ label, color }) => (
                        <div key={label} className="flex items-center gap-2">
                            <span className={`w-2.5 h-2.5 rounded-full ${color}`} />
                            <span className="text-xs text-zinc-400">{label}</span>
                        </div>
                    ))}
                </div>
            </div>

            <div className="space-y-4 mb-4">
                {history.map((entry) => (
                    <div
                        key={entry.id}
                        className={`group bg-zinc-900 border ${sentimentBorder(entry.sentiment)} rounded-xl p-6 relative overflow-hidden transition-all duration-200 hover:-translate-y-0.5 shadow-lg`}
                    >
                        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-red-600 via-orange-400 to-yellow-300 opacity-40 group-hover:opacity-80 transition-opacity duration-200" />
                        <div className="flex items-start justify-between gap-4 mb-3">
                            <p className="text-zinc-200 text-base font-normal leading-relaxed line-clamp-3 w-full">
                                {entry.review}
                            </p>
                            <span className={`shrink-0 text-[11px] font-bold uppercase px-2.5 py-1.5 rounded-md border ${sentimentBorder(entry.sentiment)} ${sentimentColor(entry.sentiment)} bg-zinc-800`}>
                                {entry.sentiment}
                            </span>
                        </div>
                        <div className="flex items-center gap-3 mt-4">
                            <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                                <div
                                    className={`h-full rounded-full ${sentimentBg(entry.sentiment)}`}
                                    style={{ width: `${entry.confidence}%` }}
                                />
                            </div>
                            <span className="text-xs font-medium text-zinc-400 shrink-0">{entry.confidence}% confidence</span>
                            <span className="text-xs text-zinc-600 shrink-0 ml-2">{entry.timestamp}</span>
                        </div>
                    </div>
                ))}
            </div>

            <button
                onClick={onClearHistory}
                className="w-full py-3 rounded-xl border border-zinc-800 text-zinc-500 text-sm uppercase tracking-widest font-semibold hover:border-red-500/40 hover:text-red-400 transition-all duration-200"
            >
                Clear History
            </button>
        </div>
    );
}
