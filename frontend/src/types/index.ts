export interface ReviewEntry {
  id: string;
  review: string;
  sentiment: string;
  confidence: number;
  timestamp: string;
}

export interface ResponseData {
  sentiment: string;
  confidence: number;
  insights?: {
    word_count: number;
    raw_probabilities_percent: Record<string, number>;
    calibrated_scores: Record<string, number>;
  };
}
