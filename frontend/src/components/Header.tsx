export default function Header() {
  return (
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
  );
}
