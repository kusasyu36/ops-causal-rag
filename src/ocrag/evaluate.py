"""評価: 検索と回答を分けて測る。

検索評価（決定的・LLM不要）:
- doc recall: 正解文書が上位kに入った割合
- node recall: 正解ノードがグラフ展開に含まれた割合
- 比較: グラフあり vs キーワードのみ（自作の主張を自分で検証する）

回答評価（LLM使用時のみ）:
- 引用妥当率: 引用が全て提供文書内か（validateの結果）
"""
from __future__ import annotations

import json
from pathlib import Path

from .qa import answer
from .retrieve import expand_nodes, retrieve


def eval_retrieval(corpus_dir: Path, eval_path: Path, k: int = 4) -> dict:
    cases = json.loads(eval_path.read_text())
    rows = []
    for mode in (True, False):  # use_graph
        doc_hit = 0
        doc_total = 0
        node_hit = 0
        node_total = 0
        node_hit_auto = 0
        for c in cases:
            hits = retrieve(c["q"], corpus_dir, k=k, use_graph=mode)
            got = {h.doc_id for h in hits}
            doc_hit += len(set(c["gold_docs"]) & got)
            doc_total += len(c["gold_docs"])
            if mode:
                # oracle: 評価セットの正解方向を与えた場合（グラフ探索単体の性能）
                nodes, _ = expand_nodes(c["q"], c.get("direction", "auto"))
                node_hit += len(set(c["gold_nodes"]) & set(nodes))
                # auto: 実運用どおり方向を自動判定した場合（フルパイプラインの性能）
                nodes_auto, _ = expand_nodes(c["q"], "auto")
                node_hit_auto += len(set(c["gold_nodes"]) & set(nodes_auto))
                node_total += len(c["gold_nodes"])
        rows.append({
            "mode": "graph+keyword" if mode else "keyword-only",
            "doc_recall": round(doc_hit / doc_total, 3),
            "node_recall_oracle_direction": round(node_hit / node_total, 3) if mode else None,
            "node_recall_auto_direction": round(node_hit_auto / node_total, 3) if mode else None,
        })
    return {"k": k, "n_cases": len(cases), "results": rows}


def eval_answers(corpus_dir: Path, eval_path: Path, llm=None) -> dict:
    from .qa import _claude
    cases = json.loads(eval_path.read_text())
    llm = llm or _claude
    valid = 0
    details = []
    for c in cases:
        ans, hits = answer(c["q"], corpus_dir, use_graph=True, llm=llm)
        valid += int(ans.valid)
        details.append({"q": c["q"], "valid": ans.valid, "citations": ans.citations,
                        "reason": ans.reason, "answer": ans.text[:600]})
    return {"n_cases": len(cases), "citation_valid_rate": round(valid / len(cases), 3),
            "details": details}
