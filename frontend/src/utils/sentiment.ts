export const sentimentColor = (s: string) => {
  if (s?.toLowerCase() === "positive") return "text-emerald-400";
  if (s?.toLowerCase() === "negative") return "text-red-400";
  return "text-yellow-400";
};

export const sentimentBg = (s: string) => {
  if (s?.toLowerCase() === "positive") return "bg-emerald-500";
  if (s?.toLowerCase() === "negative") return "bg-red-500";
  return "bg-yellow-400";
};

export const sentimentBorder = (s: string) => {
  if (s?.toLowerCase() === "positive") return "border-emerald-500/30";
  if (s?.toLowerCase() === "negative") return "border-red-500/30";
  return "border-yellow-400/30";
};
