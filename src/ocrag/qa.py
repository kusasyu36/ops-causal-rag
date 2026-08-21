"""質問応答: 検索結果＋因果経路をLLMに渡し、回答の引用を検証する。

ハーネスの安全弁:
- 回答は必ず引用タグ [rbXX] を含むこと。実在しない文書IDの引用は検証で弾く
- 引用ゼロ・不正引用の回答は「根拠不足」として差し戻し（1回だけ再試行）
- LLMには検索で渡した文書以外の知識で断定しないよう指示する
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .graph import LABELS
from .retrieve import Retrieved, expand_nodes, retrieve

CITE_RE = re.compile(r"\[(rb\d{2})\]")


@dataclass
class Answer:
    text: str
    citations: list[str]
    valid: bool
    reason: str = ""


def _claude(prompt: str, model: str = "claude-sonnet-5", timeout: int = 180) -> str:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CLAUDE_", "ANTHROPIC_"))}
    proc = subprocess.Popen(
        ["claude", "-p", "--model", model, "--tools", ""],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, start_new_session=True,
    )
    try:
        out, err = proc.communicate(input=prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        raise RuntimeError(f"claude timeout ({timeout}s)")
    if proc.returncode != 0:
        raise RuntimeError(f"claude failed: {err[:300]}")
    return out


def build_prompt(query: str, hits: list[Retrieved], chains: list[list[str]]) -> str:
    chain_text = "\n".join(
        " → ".join(LABELS[n] for n in c) for c in chains[:8]) or "(該当なし)"
    docs_text = "\n\n".join(f"[{h.doc_id}] (選定理由: {h.why})\n{h.text}" for h in hits)
    return f"""あなたは小さな自動化基盤の運用担当です。以下の資料と依存関係だけを根拠に質問に答えてください。

[依存グラフから導いた因果経路]
{chain_text}

[検索された運用資料]
{docs_text}

[質問]
{query}

回答の条件:
- 3〜6文で簡潔に。因果の順番（何が何に波及するか）を明示する
- 根拠にした資料を文中に [rb01] の形式で必ず引用する
- 資料にないことは「資料からは不明」と言う。推測で断定しない"""


def validate(text: str, hits: list[Retrieved]) -> Answer:
    cites = CITE_RE.findall(text)
    provided = {h.doc_id for h in hits}
    if not cites:
        return Answer(text=text, citations=[], valid=False, reason="引用なし")
    bad = [c for c in cites if c not in provided]
    if bad:
        return Answer(text=text, citations=cites, valid=False,
                      reason=f"未提供文書の引用: {bad}")
    return Answer(text=text, citations=sorted(set(cites)), valid=True)


def answer(query: str, corpus_dir: Path, use_graph: bool = True,
           llm=_claude) -> tuple[Answer, list[Retrieved]]:
    hits = retrieve(query, corpus_dir, use_graph=use_graph)
    _, chains = expand_nodes(query) if use_graph else ([], [])
    prompt = build_prompt(query, hits, chains)
    out = llm(prompt)
    ans = validate(out, hits)
    if not ans.valid:
        retry_prompt = prompt + f"\n\n前回の回答は不正でした（{ans.reason}）。提供された資料IDだけを [rbXX] 形式で引用し直してください。"
        out = llm(retry_prompt)
        ans = validate(out, hits)
    return ans, hits
