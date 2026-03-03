"use client";
import { useState } from "react";

export default function MovieReviewPage() {
  const [review, setReview] = useState("");
  const [loading, setLoading] = useState(false);
  const [responseData, setResponseData] = useState<any>(null);
  const [error, setError] = useState("");

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
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ review }),
      });

      const data = await res.json();
      setResponseData(data);
      setReview("");
    } catch (err) {
      setError("Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-6">
      <div className="w-full max-w-2xl bg-white shadow-lg rounded-xl p-6">
        <h1 className="text-2xl font-bold mb-4 text-center">
          Movie Review Analyzer
        </h1>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <textarea
            value={review}
            onChange={(e) => setReview(e.target.value)}
            placeholder="Write your movie review here..."
            className="border rounded-lg p-3 resize-none h-32 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />

          {error && (
            <p className="text-red-500 text-sm">{error}</p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition disabled:bg-gray-400"
          >
            {loading ? "Submitting..." : "Submit Review"}
          </button>
        </form>
      </div>

      {/* Response Grid */}
      {responseData && (
        <div className="w-full max-w-4xl mt-8 bg-white shadow-lg rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4 text-center">
            Server Response
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(responseData).map(([key, value]) => (
              <div
                key={key}
                className="border rounded-lg p-4 bg-gray-50 shadow-sm"
              >
                <p className="font-medium text-gray-700 capitalize">
                  {key}
                </p>
                <p className="text-gray-900 mt-1">
                  {String(value)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}