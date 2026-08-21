"""小さな自動化基盤の依存グラフ。

ノード=サービス/リソース、辺=「AはBに依存する」。
障害の波及は依存の逆向きに伝わる（Bが落ちるとAに影響が出る）。
純Python実装（外部依存なし）。探索は決定的（訪問順を固定）。
"""
from __future__ import annotations

# service -> 依存先のリスト（この順で探索する。決定性のため辞書順に整列済み）
DEPENDS_ON: dict[str, list[str]] = {
    "asakai-report":   ["launchd-scheduler", "llm-cli"],
    "slack-bot":       ["llm-cli", "slack-api", "socket-conn"],
    "socket-conn":     ["network"],
    "llm-cli":         ["auth-keychain", "network", "subscription-quota"],
    "launchd-scheduler": ["mac-power", "user-session"],
    "auth-keychain":   ["user-session"],
    "sheet-webhook":   ["gas-runtime", "network"],
    "diagnosis-page":  ["vercel-hosting"],
    "record-pipeline": ["diagnosis-page", "sheet-webhook"],
    "notifier":        ["slack-api", "network"],
    "slack-api":       ["network"],
    "gas-runtime":     ["google-quota"],
    "vercel-hosting":  ["network"],
    # 末端リソース
    "network": [], "mac-power": [], "user-session": [],
    "subscription-quota": [], "google-quota": [],
}

LABELS = {
    "asakai-report": "朝会レポート自動生成",
    "slack-bot": "Slack常駐ボット",
    "socket-conn": "Slackソケット接続",
    "llm-cli": "LLM CLI(claude -p)",
    "launchd-scheduler": "launchdスケジューラ",
    "auth-keychain": "Keychain認証情報",
    "sheet-webhook": "スプレッドシート受け口(GAS)",
    "diagnosis-page": "Web診断ページ",
    "record-pipeline": "回答記録パイプライン",
    "notifier": "完了通知",
    "slack-api": "Slack API",
    "gas-runtime": "GASランタイム",
    "vercel-hosting": "Vercelホスティング",
    "network": "ネットワーク",
    "mac-power": "Mac電源/スリープ状態",
    "user-session": "ユーザーログインセッション",
    "subscription-quota": "サブスクリプション利用枠",
    "google-quota": "Google実行割当",
}


def dependents_of(node: str) -> list[str]:
    """nodeに依存しているサービス（=nodeが落ちたとき影響を受ける側）。"""
    return sorted(s for s, deps in DEPENDS_ON.items() if node in deps)


def impact_chain(root: str, max_depth: int = 5) -> list[list[str]]:
    """rootの障害が波及する経路を列挙する（root→…→末端サービス）。幅優先・決定的。"""
    if root not in DEPENDS_ON:
        raise KeyError(f"unknown node: {root}")
    chains: list[list[str]] = []
    frontier: list[list[str]] = [[root]]
    for _ in range(max_depth):
        nxt: list[list[str]] = []
        for path in frontier:
            ups = dependents_of(path[-1])
            ups = [u for u in ups if u not in path]  # 循環防止
            if not ups:
                chains.append(path)
            for u in ups:
                nxt.append(path + [u])
        if not nxt:
            break
        frontier = nxt
    chains.extend(frontier if frontier and frontier[0][-1] != root else [])
    # 重複を除き、経路長→辞書順で安定ソート
    uniq = sorted({tuple(c) for c in chains if len(c) > 1})
    return [list(c) for c in sorted(uniq, key=lambda c: (len(c), c))]


def root_causes_of(service: str, max_depth: int = 5) -> list[list[str]]:
    """serviceの不調の原因候補経路（service→依存先→…→末端リソース）。深さ優先・決定的。"""
    if service not in DEPENDS_ON:
        raise KeyError(f"unknown node: {service}")
    chains: list[list[str]] = []

    def dfs(path: list[str], depth: int):
        deps = DEPENDS_ON[path[-1]]
        if not deps or depth == 0:
            if len(path) > 1:
                chains.append(list(path))
            return
        for d in sorted(deps):
            if d in path:
                continue
            dfs(path + [d], depth - 1)

    dfs([service], max_depth)
    return chains


def nodes_in_question(text: str) -> list[str]:
    """質問文に登場するノードを、日本語ラベル/英名の両方から拾う（出現順・重複なし）。"""
    found = []
    for node, label in LABELS.items():
        if node in text or label in text:
            found.append((min(text.find(node) % 10**6 if node in text else 10**6,
                              text.find(label) % 10**6 if label in text else 10**6), node))
    return [n for _, n in sorted(found)]
