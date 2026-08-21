"""コマンドライン入口: 質問→因果経路＋引用つき回答。"""
from __future__ import annotations

import sys
from pathlib import Path

from .graph import LABELS
from .qa import answer
from .retrieve import expand_nodes


def main():
    if len(sys.argv) < 2:
        print("usage: python -m ocrag.cli <質問>")
        sys.exit(1)
    query = sys.argv[1]
    corpus = Path(__file__).parent.parent.parent / "corpus"

    _, chains = expand_nodes(query)
    if chains:
        print("[因果経路]")
        for c in chains[:6]:
            print("  " + " → ".join(LABELS[n] for n in c))

    ans, hits = answer(query, corpus)
    print("\n[検索文書]")
    for h in hits:
        print(f"  {h.doc_id}: {h.why}")
    print("\n[回答]" + ("" if ans.valid else f"（⚠️検証不合格: {ans.reason}）"))
    print(ans.text.strip())


if __name__ == "__main__":
    main()
