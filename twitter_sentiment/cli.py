from __future__ import annotations

import io
import sys
from typing import Iterable, Optional

import click
from . import jsonio

from .scrape import run_snscrape_search
from .sentiment import analyze_transformer, analyze_vader
from .exporter import export_records


def _iter_jsonl_from_stream(stream: io.TextIOBase) -> Iterable[dict]:
    for line in stream:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        yield jsonio.loads(line_stripped)


def _write_jsonl_to_stdout(records: Iterable[dict]) -> None:
    out = sys.stdout.buffer
    for record in records:
        out.write(jsonio.dumps(record))
        out.write(b"\n")
    out.flush()


@click.group()
def cli() -> None:
    """Twitter Sentiment Analysis CLI."""


@cli.command()
@click.option("--query", required=True, help="Search query for snscrape")
@click.option("--limit", default=100, type=int, show_default=True)
@click.option("--lang", default=None, help="Language code filter (e.g., en)")
@click.option("--since", default=None, help="Since date YYYY-MM-DD")
@click.option("--until", default=None, help="Until date YYYY-MM-DD")
@click.option("--include-retweets/--exclude-retweets", default=False, show_default=True)
def fetch(query: str, limit: int, lang: Optional[str], since: Optional[str], until: Optional[str], include_retweets: bool) -> None:
    """Fetch tweets via snscrape and output JSONL to stdout."""
    tweets = run_snscrape_search(
        query=query,
        limit=limit,
        lang=lang,
        since=since,
        until=until,
        exclude_retweets=not include_retweets,
        raw=False,
    )
    _write_jsonl_to_stdout(tweets)


@cli.command()
@click.option("--analyzer", type=click.Choice(["vader", "hf"], case_sensitive=False), default="vader", show_default=True)
@click.option("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest", help="HF model name when analyzer=hf")
@click.option("--input", "input_path", default=None, help="Optional JSONL input path; otherwise read stdin")
def analyze(analyzer: str, model: str, input_path: Optional[str]) -> None:
    """Read JSONL of tweets, attach sentiment fields, write JSONL to stdout."""
    if input_path:
        stream = open(input_path, "r", encoding="utf-8")
    else:
        stream = sys.stdin

    def _gen():
        for obj in _iter_jsonl_from_stream(stream):
            text = obj.get("content") or obj.get("text") or ""
            if analyzer.lower() == "vader":
                res = analyze_vader(text)
            else:
                res = analyze_transformer(text, model_name=model)
            obj["sentiment"] = res.label
            obj["sentiment_scores"] = res.scores
            if res.confidence is not None:
                obj["sentiment_confidence"] = res.confidence
            yield obj

    _write_jsonl_to_stdout(_gen())


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["csv", "parquet", "jsonl"], case_sensitive=False), required=True)
@click.option("--output", required=True, help="Output file path")
@click.option("--input", "input_path", default=None, help="Optional JSONL input path; otherwise read stdin")
def export(fmt: str, output: str, input_path: Optional[str]) -> None:
    """Export tweets from JSONL to CSV/Parquet/JSONL."""
    if input_path:
        with open(input_path, "r", encoding="utf-8") as f:
            records = list(_iter_jsonl_from_stream(f))
    else:
        records = list(_iter_jsonl_from_stream(sys.stdin))
    export_records(records, fmt=fmt, output_path=output)


if __name__ == "__main__":
    cli()

