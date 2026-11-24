"""
reasoning_engine.py — Linguistic signal-based reasoning engine.

Provides structured explanations for model predictions using:
- Sensationalist language detection
- Credibility signal analysis
- Hedge/uncertainty language detection
- Source attribution heuristics
"""
import re

SENSATIONALIST_TERMS = {
    "hoax", "conspiracy", "fake", "fraud", "exposed", "shocking",
    "scandal", "coverup", "cover-up", "bombshell", "crisis", "panic",
    "secret", "leaked", "banned", "corrupt", "breaking", "breaking:",
    "urgent", "warning", "illegal", "criminal", "attacked", "betrayal",
    "explosion", "catastrophe", "disaster", "extreme", "outrage",
    "unbelievable", "incredible", "stunning", "destroy", "collapse",
    "destroy", "revolution", "epic", "massive", "dangerous", "threat",
    "alarming", "disturbing", "chaos", "riot", "martial", "arrest",
    "death", "died", "killed", "murder", "violence", "terror",
}

CREDIBILITY_SIGNALS = [
    r"\baccording to\b",
    r"\bstudies? (show|suggest|indicate|find)\b",
    r"\bresearch(ers?)?\b",
    r"\bpeer[- ]reviewed\b",
    r"\bpublished in\b",
    r"\bdata (shows?|indicates?)\b",
    r"\bstatistics?\b",
    r"\bofficial(ly)?\b",
    r"\b(confirmed|verified|confirmed by)\b",
    r"\buniversity of\b",
    r"\bgovernment report\b",
    r"\bwho\b",  # World Health Organization
    r"\bcdc\b",
    r"\bfda\b",
    r"\bcongressional\b",
    r"\bsenate\b",
    r"\bwhite house\b",
]

HEDGE_TERMS = {
    "allegedly", "reportedly", "claimed", "claims", "possibly", "perhaps",
    "may", "might", "could", "seems", "appears", "supposedly",
    "unconfirmed", "rumored", "rumour", "believed", "thought",
    "sources say", "anonymous", "insiders claim",
}


def extract_linguistic_signals(text: str) -> dict:
    """
    Returns a dict of identified linguistic risk signals.
    """
    lowered = text.lower()
    words   = set(re.findall(r"\b\w+\b", lowered))

    sensational_hits = list(words & SENSATIONALIST_TERMS)
    hedge_hits       = list(words & HEDGE_TERMS)
    credible_hits    = [
        pat for pat in CREDIBILITY_SIGNALS
        if re.search(pat, lowered)
    ]

    # Punctuation excess
    exclamation_count = text.count("!")
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    # Source citation heuristic
    has_url = bool(re.search(r"https?://", lowered))
    has_attribution = bool(re.search(r"\bsaid\b|\bstated\b|\baccording to\b", lowered))

    return {
        "sensationalist_words": sensational_hits,
        "hedge_words":          hedge_hits,
        "credibility_signals":  credible_hits,
        "exclamation_marks":    exclamation_count,
        "caps_ratio":           round(caps_ratio, 3),
        "has_url":              has_url,
        "has_attribution":      has_attribution,
    }


def build_reasoning(
    text: str,
    verdict: str,
    fake_prob: float,
    real_prob: float,
    confidence: float,
) -> str:
    """
    Returns a human-readable reasoning paragraph given model outputs.
    """
    sigs  = extract_linguistic_signals(text)
    lines = []

    # Verdict summary
    is_fake = fake_prob > 0.5
    level   = (
        "high" if confidence > 85 else
        "moderate" if confidence > 65 else
        "low"
    )
    lines.append(
        f"**{'⚠️ MISINFORMATION DETECTED' if is_fake else '✅ CREDIBLE CONTENT'}** "
        f"(confidence: {confidence:.1f}%)"
    )
    lines.append(
        f"The model classifies this text as **{'likely misinformation' if is_fake else 'likely credible'}** "
        f"with {level} confidence (Fake: {fake_prob * 100:.1f}%, Real: {real_prob * 100:.1f}%)."
    )
    lines.append("")

    # Reasoning signals
    reasons = []
    if sigs["sensationalist_words"]:
        reasons.append(
            f"🔴 **Sensationalist language detected**: "
            f"`{', '.join(sigs['sensationalist_words'][:6])}` — "
            "emotionally charged words commonly associated with misleading content."
        )
    if sigs["hedge_words"]:
        reasons.append(
            f"🟡 **Hedging & unverified claims**: "
            f"`{', '.join(sigs['hedge_words'][:5])}` — "
            "language that avoids direct accountability or factual grounding."
        )
    if sigs["credibility_signals"]:
        reasons.append(
            f"🟢 **Credibility markers present**: "
            f"Phrases like {', '.join(repr(p) for p in sigs['credibility_signals'][:3])} "
            "indicate a degree of sourcing or academic framing."
        )
    if sigs["exclamation_marks"] > 1:
        reasons.append(
            f"🔴 **Excessive punctuation**: {sigs['exclamation_marks']} exclamation marks "
            "found — typical of attention-seeking misinformation headlines."
        )
    if sigs["caps_ratio"] > 0.15:
        reasons.append(
            f"🔴 **Excessive capitalization**: {sigs['caps_ratio']*100:.0f}% of characters "
            "are uppercase — a stylistic pattern common in sensationalist writing."
        )
    if not sigs["has_attribution"] and is_fake:
        reasons.append(
            "🔴 **No source attribution**: The text makes no reference to "
            "named sources, studies, or official bodies."
        )
    if sigs["has_url"]:
        reasons.append(
            "🟢 **URL present**: A link is included, which may indicate sourcing "
            "(but does not guarantee accuracy)."
        )

    if not reasons:
        reasons.append(
            "ℹ️ No strong linguistic signals found. "
            "The model's prediction relies primarily on learned contextual patterns."
        )

    lines.extend(reasons)
    lines.append("")
    lines.append(
        "> **Disclaimer**: This tool is an AI assistant and should be used alongside "
        "human fact-checking. It cannot replace expert journalism verification."
    )

    return "\n\n".join(lines)
