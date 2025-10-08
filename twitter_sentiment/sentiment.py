from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SentimentResult:
    label: str
    scores: Dict[str, float]
    confidence: Optional[float] = None


_vader_analyzer = None
_hf_pipeline = None


def analyze_vader(text: str) -> SentimentResult:
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _vader_analyzer = SentimentIntensityAnalyzer()
    scores = _vader_analyzer.polarity_scores(text or "")
    compound = scores.get("compound", 0.0)
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return SentimentResult(label=label, scores=scores, confidence=abs(compound))


def analyze_transformer(
    text: str,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
) -> SentimentResult:
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _hf_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,  # return all labels with scores in some models; but we take top
        )
    outputs = _hf_pipeline(text or "", truncation=True)
    # Pipeline may return a list of dicts; ensure we pick top result
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        best = max(outputs, key=lambda x: x.get("score", 0.0))
    elif isinstance(outputs, list) and outputs and isinstance(outputs[0], list):
        # In case of return_all_scores style
        best = max(outputs[0], key=lambda x: x.get("score", 0.0))
    else:
        best = outputs
    label = (best.get("label") or "").lower()
    confidence = float(best.get("score", 0.0))
    scores = {label: confidence}
    return SentimentResult(label=label, scores=scores, confidence=confidence)

