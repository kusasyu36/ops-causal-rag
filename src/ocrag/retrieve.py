"""検索: グラフ探索＋キーワードスコアのハイブリッド。埋め込み不使用。

設計判断: この規模（文書10本・ノード18個）では、ベクトル検索より
「質問中のノード→依存グラフで因果経路を展開→経路上のノードに触れる文書を集める」
の方が、なぜその文書が選ばれたかを完全に説明できる。検索の説明可能性を優先した。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .graph import LABELS, impact_chain, nodes_in_question, root_causes_of

# 文書ID -> 関連ノード（文書を書いた人間が付与するタグ。自動抽出に頼らない）
DOC_NODES: dict[str, list[str]] = {
    "rb01": ["launchd-scheduler", "auth-keychain", "llm-cli", "user-session", "asakai-report"],
    "rb02": ["mac-power", "launchd-scheduler", "asakai-report", "llm-cli"],
    "rb03": ["auth-keychain", "llm-cli", "user-session", "subscription-quota"],
    "rb04": ["slack-bot", "socket-conn", "slack-api", "llm-cli", "launchd-scheduler"],
    "rb05": ["network", "socket-conn", "slack-api", "llm-cli", "gas-runtime", "vercel-hosting"],
    "rb06": ["sheet-webhook", "gas-runtime", "google-quota", "record-pipeline", "diagnosis-page"],
    "rb07": ["diagnosis-page", "vercel-hosting", "network", "record-pipeline", "gas-runtime"],
    "rb08": ["llm-cli"],
    "rb09": ["notifier", "slack-api", "network"],
    "rb10": ["llm-cli"],
}


@dataclass
class Retrieved:
    doc_id: str
    score: float
    why: str          # なぜ選ばれたか（グラフ経路 or キーワード）
    text: str


def _load_corpus(corpus_dir: Path) -> dict[str, str]:
    docs = {}
    for p in sorted(corpus_dir.glob("rb*.md")):
        docs[p.stem.split("_")[0]] = p.read_text()
    return docs


def _keyword_score(query: str, text: str) -> float:
    """文字2-gram一致率。空白で切れない日本語質問文に対応する。

    v0.1の失敗: 空白区切りだと日本語の一文が丸ごと1語になり、一致ゼロ→検索0件だった。
    """
    grams = {query[i:i + 2] for i in range(len(query) - 1)
             if not re.match(r"[\s、。・()（）?？]", query[i]) and not re.match(r"[\s、。・()（）?？]", query[i + 1])}
    if not grams:
        return 0.0
    hits = sum(1 for g in grams if g in text)
    return hits / len(grams)


def expand_nodes(query: str, direction: str = "auto") -> tuple[list[str], list[list[str]]]:
    """質問中のノードを因果方向に展開する。

    direction: "impact"=波及先 / "cause"=原因側 / "auto"=質問文から推定
    """
    seeds = nodes_in_question(query)
    if direction == "auto":
        direction = "impact" if any(k in query for k in ("影響", "波及", "落ちると", "止まると")) else "cause"
    chains: list[list[str]] = []
    for s in seeds:
        chains += impact_chain(s) if direction == "impact" else root_causes_of(s)
    nodes = list(dict.fromkeys([n for c in chains for n in c]))  # 順序保持で重複除去
    return nodes, chains


def retrieve(query: str, corpus_dir: Path, k: int = 4, use_graph: bool = True) -> list[Retrieved]:
    """上位k文書を返す。use_graph=False はキーワードのみ（比較実験用）。"""
    docs = _load_corpus(corpus_dir)
    graph_nodes, chains = expand_nodes(query) if use_graph else ([], [])
    out = []
    for doc_id, text in docs.items():
        kw = _keyword_score(query, text)
        overlap = [n for n in DOC_NODES.get(doc_id, []) if n in graph_nodes]
        # グラフ票は2票で頭打ち: 「経路に触れているか」が大事で、触れている数を
        # 増やすほど良いわけではない（v0.1で、ノード数の多い文書が
        # キーワード的に正しい文書を押し出す失敗があった）
        g = float(min(len(overlap), 2))
        score = g + 3.0 * kw
        if score <= 0:
            continue
        why = []
        if overlap:
            why.append("グラフ経路上のノードに言及: " + ", ".join(LABELS[n] for n in overlap[:4]))
        if kw > 0:
            why.append(f"キーワード一致 {kw:.2f}")
        out.append(Retrieved(doc_id=doc_id, score=score, why=" / ".join(why), text=text))
    out.sort(key=lambda r: (-r.score, r.doc_id))
    return out[:k]
