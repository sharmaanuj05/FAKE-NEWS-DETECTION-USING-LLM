"""
src/red_flags.py — Transparent, rule-based red flag checks for news text.

All flags are strictly ADVISORY and do not alter model probabilities or verdicts.
Each check returns a name and a concise reason if triggered.

Rules:
1. excessive_caps: Unusually high frequency of ALL CAPS words.
2. clickbait_phrases: Common viral/clickbait phrase patterns.
3. excessive_punctuation: Multiple consecutive or high counts of exclamation/question marks.
4. emotional_absolutist: Sensationalist, conspiratorial, or absolutist assertions.
5. vague_authority: Unsubstantiated anonymous authority claims ("sources claim", "experts warn").
6. no_named_attribution: Complete lack of journalistic attribution or quotes.
"""
import re
from typing import List, Dict, Optional

# Whitelist common acronyms that appear in legitimate news
ALLOWED_ACRONYMS = {
    "US", "USA", "UK", "EU", "UN", "NATO", "FBI", "CIA", "DOJ", "DHS",
    "GOP", "DNC", "CDC", "FDA", "NASA", "EPA", "SEC", "FCC", "IRS",
    "WHO", "IMF", "WTO", "OPEC", "ISIS", "ISIL", "AP", "AFP", "BBC",
    "CNN", "NBC", "ABC", "CBS", "NPR", "PBS", "CEO", "CFO", "CTO",
    "COVID", "SARS", "HIV", "AIDS", "GDP", "CPI", "AI", "LLM", "EV",
    "POTUS", "FLOTUS", "SCOTUS", "SENATE", "HOUSE", "CONGRESS"
}

# Clickbait regex patterns
CLICKBAIT_PATTERNS = [
    r"\byou won'?t believe\b",
    r"\bshocking (?:proof|truth|discovery|video|revelation)\b",
    r"\bbombshell\b",
    r"\bwhat happened next\b",
    r"\bwhat they don'?t want you to know\b",
    r"\bsecret (?:they|media|government|elites) (?:is|are) hiding\b",
    r"\bexposed:\b",
    r"\bbreaking:?\s*explosive\b",
    r"\bspread this everywhere\b",
    r"\bshare before it'?s taken down\b",
    r"\bwake up (?:america|people|sheeple)\b",
    r"\bmainstream media won'?t (?:show|tell|report)\b",
    r"\bproof that\b",
]

# Emotional & absolutist language patterns
EMOTIONAL_PATTERNS = [
    r"\b100% (?:proven|guaranteed|true|false)\b",
    r"\bundeniable (?:proof|evidence)\b",
    r"\bindisputable (?:truth|facts)\b",
    r"\babsolute (?:treason|disgrace|evil|insanity|corruption)\b",
    r"\bdisgusting (?:truth|display|act|agenda)\b",
    r"\bmonstrous (?:crime|plot|lie)\b",
    r"\bevil (?:cabal|globalists|agenda|deep state)\b",
    r"\bmiracle (?:cure|solution|discovery)\b",
    r"\bcatastrophic (?:hoax|conspiracy|disaster)\b",
    r"\bunforgivable (?:betrayal|crime)\b",
]

# Vague authority without named source
VAGUE_AUTHORITY_PATTERNS = [
    r"\b(?:many|some|certain) experts say\b",
    r"\bsources (?:close to|claim|whisper|revealed)\b",
    r"\binsiders (?:confirm|warn|say|leak)\b",
    r"\bofficials whisper\b",
    r"\banonymous sources say\b",
    r"\bpeople are saying\b",
    r"\beveryone knows that\b",
    r"\bit is widely known that\b",
    r"\bscientists secretly admit\b",
]

# Common journalistic attribution markers
ATTRIBUTION_MARKERS = [
    r"\bsaid\b",
    r"\btold\b",
    r"\bstated\b",
    r"\breported\b",
    r"\bannounced\b",
    r"\baccording to\b",
    r"\bspokesperson\b",
    r"\bconfirmed\b",
    r"\btestified\b",
    r"\bciting\b",
    r"\bquoted\b",
    r"\bpress release\b",
    r"\bspokesman\b",
    r"\bspokeswoman\b",
    r"[\"“][^\"“”]{10,}[\"”]",  # direct quote of >= 10 chars
]


def check_excessive_caps(text: str) -> Optional[Dict[str, str]]:
    """Flags text with unusually high count or proportion of ALL-CAPS words."""
    words = re.findall(r"\b[A-Za-z]+\b", text)
    if len(words) < 6:
        return None
    caps_words = [
        w for w in words
        if w.isupper() and len(w) >= 3 and w not in ALLOWED_ACRONYMS
    ]
    # Trigger if >= 3 long all-caps words, or > 8% of all words are all-caps
    if len(caps_words) >= 3 or (len(caps_words) >= 2 and len(caps_words) / len(words) > 0.08):
        sample = ", ".join(caps_words[:3])
        return {
            "name": "Excessive Capitalization",
            "reason": f"Contains multiple ALL-CAPS words ({sample}) often used for sensationalist emphasis."
        }
    return None


def check_clickbait_phrases(text: str) -> Optional[Dict[str, str]]:
    """Flags characteristic clickbait, urgency, or viral hook phrases."""
    lower = text.lower()
    matches = []
    for pattern in CLICKBAIT_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            matches.append(m.group(0))
    if matches:
        return {
            "name": "Clickbait Phrasing",
            "reason": f"Uses sensational hook phrasing ({matches[0]!r}) typical of viral or deceptive headlines."
        }
    return None


def check_excessive_punctuation(text: str) -> Optional[Dict[str, str]]:
    """Flags repeated exclamation marks, interrobangs, or excessive exclamation."""
    # Check repeated marks: !!, ???, !?!, etc.
    repeated = re.findall(r"[!?]{2,}", text)
    total_exclamations = text.count("!")
    if repeated or total_exclamations >= 3:
        return {
            "name": "Excessive Punctuation",
            "reason": f"Contains heightened punctuation ({repeated[0] if repeated else 'multiple exclamation marks'}) signaling non-neutral emotion."
        }
    return None


def check_emotional_absolutist(text: str) -> Optional[Dict[str, str]]:
    """Flags absolutist certainty claims and highly charged moralizing language."""
    lower = text.lower()
    matches = []
    for pattern in EMOTIONAL_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            matches.append(m.group(0))
    if matches:
        return {
            "name": "Absolutist / Emotional Framing",
            "reason": f"Uses hyper-charged or absolutist assertions ({matches[0]!r}) rather than measured reporting."
        }
    return None


def check_vague_authority(text: str) -> Optional[Dict[str, str]]:
    """Flags anonymous or unverified authority appeals without attribution."""
    lower = text.lower()
    matches = []
    for pattern in VAGUE_AUTHORITY_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            matches.append(m.group(0))
    if matches:
        return {
            "name": "Vague Authority",
            "reason": f"Attributes claims to unverified anonymous entities ({matches[0]!r}) without named institutional sourcing."
        }
    return None


def check_no_named_attribution(text: str) -> Optional[Dict[str, str]]:
    """
    Flags articles of substantial length (>= 40 words) that lack any attribution verbs or quotes.
    Shorter headlines (< 40 words) are excluded because headlines naturally lack quotes.
    """
    words = text.split()
    if len(words) < 40:
        return None

    lower = text.lower()
    for marker in ATTRIBUTION_MARKERS:
        if re.search(marker, lower):
            return None

    return {
        "name": "Missing Source Attribution",
        "reason": "Article text contains no quoted statements or standard attribution verbs ('said', 'reported', 'according to')."
    }


ALL_RULES = [
    ("excessive_caps", check_excessive_caps),
    ("clickbait_phrases", check_clickbait_phrases),
    ("excessive_punctuation", check_excessive_punctuation),
    ("emotional_absolutist", check_emotional_absolutist),
    ("vague_authority", check_vague_authority),
    ("no_named_attribution", check_no_named_attribution),
]


def detect_red_flags(text: str, active_rules: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Runs configured red-flag checks on the input text.
    Returns list of triggered flags with their name and explanatory reason.
    """
    if not text or not text.strip():
        return []

    flags = []
    for rule_id, rule_fn in ALL_RULES:
        if active_rules is not None and rule_id not in active_rules:
            continue
        res = rule_fn(text)
        if res:
            flags.append({
                "rule_id": rule_id,
                "name": res["name"],
                "reason": res["reason"],
            })
    return flags
