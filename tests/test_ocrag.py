"""グラフ探索・検索・引用検証・評価の検証。LLM不要（決定的）。"""
from pathlib import Path

import pytest

from ocrag.evaluate import eval_retrieval
from ocrag.graph import (DEPENDS_ON, dependents_of, impact_chain,
                         nodes_in_question, root_causes_of)
from ocrag.qa import Answer, build_prompt, validate
from ocrag.retrieve import DOC_NODES, Retrieved, expand_nodes, retrieve

CORPUS = Path(__file__).parent.parent / "corpus"
EVAL = Path(__file__).parent.parent / "eval" / "eval_set.json"


class TestGraph:
    def test_graph_is_closed(self):
        # 依存先は全てノードとして定義されている（参照切れなし）
        for node, deps in DEPENDS_ON.items():
            for d in deps:
                assert d in DEPENDS_ON, f"{node} -> {d} が未定義"

    def test_dependents_inverse_of_depends(self):
        assert "slack-bot" in dependents_of("llm-cli")
        assert "llm-cli" in DEPENDS_ON["slack-bot"]

    def test_impact_chain_network_reaches_bot(self):
        chains = impact_chain("network")
        flat = {tuple(c) for c in chains}
        # network → socket-conn → slack-bot の波及経路が見つかる
        assert any(c[0] == "network" and c[-1] == "slack-bot" for c in chains), flat

    def test_root_causes_of_asakai(self):
        chains = root_causes_of("asakai-report")
        ends = {c[-1] for c in chains}
        # 朝会レポートの根本原因候補に「Mac電源」「利用枠」が含まれる
        assert "mac-power" in ends
        assert "subscription-quota" in ends

    def test_deterministic(self):
        assert impact_chain("network") == impact_chain("network")
        assert root_causes_of("slack-bot") == root_causes_of("slack-bot")

    def test_unknown_node_raises(self):
        with pytest.raises(KeyError):
            impact_chain("no-such-node")

    def test_nodes_in_question_by_label(self):
        found = nodes_in_question("Slack常駐ボットが無応答です")
        assert "slack-bot" in found


class TestRetrieve:
    def test_doc_tags_reference_real_nodes(self):
        for doc, nodes in DOC_NODES.items():
            for n in nodes:
                assert n in DEPENDS_ON, f"{doc}: unknown node {n}"

    def test_graph_mode_finds_causal_doc(self):
        # 「朝会レポート未生成」の原因検索で、スリープのrunbook(rb02)が上位に来る
        hits = retrieve("朝会レポート自動生成が今朝動いていない。原因は？", CORPUS)
        ids = [h.doc_id for h in hits]
        assert "rb02" in ids

    def test_graph_mode_explains_with_chains(self):
        # グラフありは「なぜその文書か」を因果ノードで説明できる（キーワードのみは不可）
        hits_g = retrieve("朝会レポートが今朝動いていない。原因は？", CORPUS)
        assert any("グラフ経路上" in h.why for h in hits_g)
        hits_k = retrieve("朝会レポートが今朝動いていない。原因は？", CORPUS, use_graph=False)
        assert all("グラフ経路上" not in h.why for h in hits_k)

    def test_every_hit_has_reason(self):
        hits = retrieve("ネットワークが落ちると何に影響が出る？", CORPUS)
        assert hits and all(h.why for h in hits)

    def test_direction_auto(self):
        nodes_i, _ = expand_nodes("ネットワークが落ちると何に影響が出る？")
        nodes_c, _ = expand_nodes("Slack常駐ボットが無応答。原因は？")
        assert "slack-bot" in nodes_i     # 波及方向
        assert "network" in nodes_c       # 原因方向


class TestCitationValidation:
    HITS = [Retrieved(doc_id="rb01", score=1, why="", text=""),
            Retrieved(doc_id="rb02", score=1, why="", text="")]

    def test_valid_citation(self):
        a = validate("スリープが原因の可能性が高い [rb02]。", self.HITS)
        assert a.valid and a.citations == ["rb02"]

    def test_missing_citation_rejected(self):
        a = validate("スリープが原因です。", self.HITS)
        assert not a.valid and a.reason == "引用なし"

    def test_hallucinated_citation_rejected(self):
        a = validate("原因はDNSです [rb99]。", self.HITS)
        assert not a.valid and "rb99" in a.reason

    def test_prompt_contains_only_provided_docs(self):
        p = build_prompt("q", self.HITS, [["network", "slack-bot"]])
        assert "[rb01]" in p and "[rb99]" not in p


class TestEvaluation:
    def test_eval_metrics(self):
        r = eval_retrieval(CORPUS, EVAL)
        modes = {x["mode"]: x for x in r["results"]}
        # 文書再現率はどちらのモードも高水準（この規模ではキーワードも強い）
        assert modes["graph+keyword"]["doc_recall"] >= 0.9
        assert modes["keyword-only"]["doc_recall"] >= 0.9
        # 因果ノードの再現はグラフモードだけが提供できる
        # oracle=正解方向を与えたグラフ探索単体 / auto=方向自動判定込みの実運用条件
        assert modes["graph+keyword"]["node_recall_oracle_direction"] >= 0.9
        assert modes["graph+keyword"]["node_recall_auto_direction"] >= 0.7
        assert modes["graph+keyword"]["node_recall_auto_direction"] <= modes["graph+keyword"]["node_recall_oracle_direction"]
        assert modes["keyword-only"]["node_recall_oracle_direction"] is None
