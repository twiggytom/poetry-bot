#web scraper to scrape poems from The Adroit Journal's issue archive.

# https://theadroitjournal.org/issue-archives/

# output: data/poems.json (not in repo)


from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import requests
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
from dotenv import load_dotenv
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0        # make language detection reproducible


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "adroit")

BASE = "https://theadroitjournal.org"
ARCHIVE_URL = f"{BASE}/issue-archives/"


load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
CONTACT = os.environ.get("SCRAPER_CONTACT_EMAIL", "").strip()
USER_AGENT = "adroit-poem-scraper/1.0 (+personal research" + \
             (f"; contact: {CONTACT})" if CONTACT else ")")


POETRY_HEADING = re.compile(r"^(?:poetry|poems)$|priz\w*\s+for\s+poetry$", re.I)

HEADING_TAG = re.compile(r"^h[1-4]$")
BYLINE = re.compile(r"^by\s+(.{2,90})$", re.I)
NAV = re.compile(r"^(next\s*\(|<\s*previous|previous\s*\(|back to issue)", re.I)
INDENT_PX = re.compile(r"(?:margin|padding)-left:\s*(\d+)", re.I)

BIO_PHRASES = re.compile(
    r"\b(is the author|is the recipient|is a poet|is a writer|poems? (?:have|has) appeared"
    r"|work has appeared|holds an? MFA|received (?:her|his|their|an|a)|lives in"
    r"|currently (?:lives|teaches)|is pursuing|teaches at)\b",
    re.I,
)

WORD_JUNK = (
    re.compile(r"\{cke_protected\}\S*"),
    re.compile(r"</?xml>", re.I),
    re.compile(r"\[if\s+gte\s+mso.*?\[endif\]", re.S | re.I),
    re.compile(r"\[if\s+(?:!|gte|gt|lte|lt)[^\]]*\]", re.I),
    re.compile(r"<!?\[endif\]-?-?>?|\[endif\]", re.I),
    re.compile(r"<!--\[if.*?\[endif\]-->", re.S | re.I),
    re.compile(r"</?[owmvxst]:[^>]*>", re.I),
    re.compile(r"/\* Style Definitions \*/.*?(?=\n\n|\Z)", re.S),
    re.compile(r"\bmso-[a-z-]+\s*:[^;{}]*;?", re.I),
)

# min characters for poem
MIN_POEM_CHARS = 40


TYPOGRAPHY = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "—": "--", "―": "--", "‒": "-", "–": "-", "‐": "-",
    "‑": "-", "…": "...", " ": " ", "−": "-",
}

# drops ambiguous characters
AMBIGUOUS_MIN_COUNT = 4      # fewer than this is scrubbed in place
AMBIGUOUS_MIN_RATIO = 0.001  # ...and it must also be this dense to drop

# drops poems with too many non-English letters
MAX_NON_ENGLISH = 0.30

PARA_TAGS = {"p", "blockquote"}
LINE_TAGS = {"div", "li", "tr", "pre", "section", "article", "figcaption",
             "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt"}

# markup drops
STRIP_TAGS = ("script", "style", "audio", "video", "iframe", "img", "figure",
              "noscript", "form", "svg", "source")



# HTTP


class Fetcher:
    """Polite, retrying, thread-safe HTTP getter with an on-disk HTML cache."""

    def __init__(self, use_cache: bool = True, delay: float = 0.4, timeout: int = 30):
        self.use_cache = use_cache
        self.delay = delay
        self.timeout = timeout
        self._local = threading.local()
        self._throttle = threading.Lock()
        self._next_ok = 0.0
        os.makedirs(CACHE_DIR, exist_ok=True)

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers["User-Agent"] = USER_AGENT
            self._local.session = s
        return s

    def _cache_path(self, url: str) -> str:
        return os.path.join(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest() + ".html")

    def _wait(self) -> None:
        with self._throttle:
            now = time.monotonic()
            if now < self._next_ok:
                time.sleep(self._next_ok - now)
            self._next_ok = max(now, self._next_ok) + self.delay

    def get(self, url: str) -> str | None:
        path = self._cache_path(url)
        if self.use_cache and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return fh.read()

        last_error = None
        for attempt in range(4):
            self._wait()
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                resp.encoding = resp.encoding or "utf-8"
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(resp.text)
                return resp.text
            except Exception as exc:  # network hiccup, 5xx, rate limit
                last_error = exc
                time.sleep((2 ** attempt) + random.random())
        print(f"  ! giving up on {url}: {last_error}", file=sys.stderr)
        return None



# Helpers


def normalize_url(href: str, base: str = BASE) -> str:
    """Canonicalize the archive's mix of relative / http / www / stray-space links."""
    href = href.strip().replace("%20", "").replace(" ", "")
    href = re.sub(r"^http://(www\.)?", "https://", href)
    href = re.sub(r"^https://www\.", "https://", href)
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = base + href
    href = href.split("#")[0].split("?")[0]
    return href.rstrip("/")


def norm_name(value: str) -> str:
    """Letters only, lowercased, accents folded -- for comparing names."""
    folded = unicodedata.normalize("NFKD", value or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", folded.lower())


def smart_title(name: str) -> str:
    """Title-case an ALL-CAPS byline without mangling Mc-/Mac-/O'/hyphenated names."""
    def fix_word(word: str) -> str:
        if not word:
            return word
        for sep in ("-", "'", "’", "."):
            if sep in word:
                return sep.join(fix_word(p) for p in word.split(sep))
        low = word.lower()
        if low.startswith("mc") and len(low) > 2:
            return "Mc" + low[2:].capitalize()
        if low.startswith("mac") and len(low) > 4:
            return "Mac" + low[3:].capitalize()
        return word.capitalize()

    return " ".join(fix_word(w) for w in name.split())


def is_ambiguous(ch: str) -> bool:
    """True for invisible, control, private-use and look-alike-space characters.

    Zero-width spaces, soft hyphens, bidi marks, narrow no-break spaces and the
    U+FFFD replacement character all survive a copy-paste invisibly and corrupt
    tokenization later.
    """
    if ch in "\n\t":
        return False
    category = unicodedata.category(ch)
    if category in ("Cf", "Cc", "Co", "Cs"):
        return True
    if category == "Zs" and ch != " ":
        return True
    return ch == "�"


def script_of(ch: str) -> str:
    """Unicode script family of a character, as the first word of its name."""
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return "?"


def mixed_script_runs(text: str) -> int:
    """Characters sitting in words that mix alphabets -- the classic homoglyph
    attack, e.g. a Cyrillic 'о' hiding inside an otherwise Latin word."""
    total = 0
    for word in re.findall(r"[^\W\d_]+", text, re.UNICODE):
        if len({script_of(c) for c in word if c.isalpha()}) > 1:
            total += len(word)
    return total


def ambiguous_count(text: str) -> int:
    return sum(1 for ch in text if is_ambiguous(ch)) + mixed_script_runs(text)


def scrub_ambiguous(text: str) -> str:
    """Clean the stray invisible characters out of a poem.

    Space-like ones stand in for a real word break -- the journal uses narrow
    no-break and zero-width spaces as ordinary spaces -- so they become a plain
    space. Deleting them would weld the neighbouring words together. The rest
    are intra-word marks or controls and go entirely.
    """
    out = []
    for ch in text:
        if not is_ambiguous(ch):
            out.append(ch)
        elif ch == "​" or unicodedata.category(ch) == "Zs":
            out.append(" ")
    return "".join(out)


def normalize_typography(text: str) -> str:
    """Fold confusable punctuation onto its ASCII equivalent."""
    return "".join(TYPOGRAPHY.get(ch, ch) for ch in text)


ENGLISH_FUNCTION_WORDS = frozenset("""
a an the and or but nor so yet if then than that this these those there here
of to in on at by for with from into onto over under about after before
is are was were be been being am do does did have has had will would can
could shall should may might must not no never all some any each every
i you he she it we they me him her us them my your his its our their
what when where who whom which how why
""".split())

# langdetect lazily loads its language profiles on first use and is not
# documented as thread-safe, so detection is serialised behind a lock.
_DETECT_LOCK = threading.Lock()


def english_word_ratio(segment: str) -> float:
    """Share of word tokens that are English function words. Even odd poetry
    scores well above zero here, while foreign text scores near it."""
    words = re.findall(r"[a-z']+", segment.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS) / len(words)


def _segments(text: str, min_letters: int) -> list[str]:
    """Lines, with short ones folded into the next, so each piece is long enough
    to identify but a bilingual poem still splits along its language seam."""
    out: list[str] = []
    buffer = ""
    for line in text.split("\n"):
        candidate = f"{buffer} {line}".strip() if buffer else line.strip()
        if sum(1 for c in candidate if c.isalpha()) >= min_letters:
            out.append(candidate)
            buffer = ""
        else:
            buffer = candidate
    if buffer:
        out.append(buffer)
    return out


def non_english_share(text: str, min_letters: int = 40, min_prob: float = 0.90,
                      guard: float = 0.25) -> float:
    """Fraction of a poem's letters that are not English.

    Letters outside the Latin alphabet count as foreign outright. The Latin part
    is judged line by line, so a poem that alternates Spanish with its English
    translation is scored on both halves rather than averaged into English.
    Segments that read as English are never sent to the detector, which is
    otherwise prone to calling a short, sparse English line Danish.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return 0.0
    non_latin = sum(1 for c in letters if script_of(c) != "LATIN")
    latin_only = "".join(c for c in text if not c.isalpha() or script_of(c) == "LATIN")

    judged = foreign = 0
    for segment in _segments(latin_only, min_letters):
        count = sum(1 for c in segment if c.isalpha())
        if count < min_letters:
            continue
        judged += count
        if english_word_ratio(segment) >= guard:
            continue
        try:
            with _DETECT_LOCK:
                top = detect_langs(segment)[0]
        except LangDetectException:
            continue
        if top.lang != "en" and top.prob > min_prob:
            foreign += count

    denominator = non_latin + judged
    return (non_latin + foreign) / denominator if denominator else 0.0


def attr(tag: Tag, name: str) -> str:
    """One tag attribute as a string. bs4 hands back a list for the attributes
    it treats as multi-valued, so everything is flattened to text here."""
    value = tag.get(name)
    if isinstance(value, (list, tuple)):
        return " ".join(value)
    return value or ""


def indent_width(tag: Tag) -> int:
    """Convert a CSS left offset into leading spaces (~25px per 4-space step)."""
    match = INDENT_PX.search(attr(tag, "style"))
    if not match:
        return 0
    return max(1, round(int(match.group(1)) / 25)) * 4


def element_text(node: Tag) -> str:
    """Plain text of a block: <br> and block-level tags break lines, CSS left
    offsets become leading spaces.

    Follows HTML whitespace rules -- newlines in the source are ordinary
    whitespace, only markup breaks a line. Non-breaking spaces are preserved,
    because the journal uses runs of them to indent lines inside a stanza.
    """
    out: list[str] = []
    pending = 0          # newlines owed before the next piece of text

    def emit(raw: str) -> None:
        nonlocal pending
        chunk = re.sub(r"[ \t\r\n]+", " ", raw)
        if not chunk.strip(" \t\r\n"):          # whitespace-only text node
            if out and pending == 0 and not out[-1].endswith(" "):
                out.append(" ")
            return
        if out and pending:
            out.append("\n" * min(pending, 2))
        pending = 0
        if out and out[-1].endswith("\n"):
            chunk = chunk.lstrip(" ")
        if chunk:
            out.append(chunk)

    def walk(current: Tag) -> None:
        nonlocal pending
        for child in current.children:
            # Every child is either a Tag or a string; testing for Tag first
            # lets the type checker see that `.name` and walk() are safe below.
            if not isinstance(child, Tag):
                if isinstance(child, NavigableString):
                    emit(str(child))
            elif child.name == "br":
                pending = min(pending + 1, 2)
            elif child.name in STRIP_TAGS:
                continue
            else:
                breaks = 2 if child.name in PARA_TAGS else (
                    1 if child.name in LINE_TAGS else 0)
                if breaks:
                    pending = max(pending, breaks)
                mark = len(out)
                walk(child)
                pad = indent_width(child)
                if pad:
                    inner = "".join(out[mark:])
                    if inner.strip():
                        del out[mark:]
                        # The indent belongs after any line break, not before it.
                        lead = inner[: len(inner) - len(inner.lstrip("\n"))]
                        out.append(lead + " " * pad + inner[len(lead):].lstrip(" "))
                if breaks and len(out) > mark:
                    pending = max(pending, breaks)

    walk(node)
    text = "".join(out).replace("\xa0", " ")
    # Some pages stored the Word markup double-escaped, so the brackets survive
    # as numeric entities and the patterns below would otherwise miss them.
    text = re.sub(r"&#0?91;", "[", text)
    text = re.sub(r"&#0?93;", "]", text)
    for pattern in WORD_JUNK:
        text = pattern.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    pad = indent_width(node)
    if pad and text.strip():
        text = " " * pad + text.lstrip(" ")
    return text.strip("\n").rstrip()


def content_blocks(soup: BeautifulSoup) -> list[Tag]:
    """Top-level blocks of the page body, in document order."""
    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()
    columns = soup.select("div.wpb_text_column") or soup.select(
        "div.entry-content, div.post-content"
    )
    blocks: list[Tag] = []
    for column in columns:
        wrapper = column.find("div", class_="wpb_wrapper") or column
        blocks.extend(child for child in wrapper.children if isinstance(child, Tag))
    return blocks


def is_embed_credit(block: Tag) -> bool:
    """SoundCloud leaves behind a caption div once the <iframe> is stripped."""
    return any("soundcloud.com" in attr(a, "href") for a in block.find_all("a"))


def leading_bold(block: Tag) -> str:
    first = next((c for c in block.children if isinstance(c, Tag)), None)
    return first.get_text(" ", strip=True) if first is not None and first.name in ("b", "strong") else ""


def bio_name(text: str, poet: str) -> str:
    """If `text` opens with the poet's name, return that name as written."""
    target = norm_name(poet)
    if len(target) < 5:
        return ""
    words = text.split()
    for i in range(min(len(words), 6)):
        candidate = " ".join(words[: i + 1])
        candidate = re.sub(r"[\u2019']s$", "", candidate).strip(" ,.;:\u2014\u2013-")
        if not candidate:
            continue
        if norm_name(candidate) == target:
            return candidate
        if not target.startswith(norm_name(candidate)):
            break
    return ""


# issue archive -> issue pages -> poem links

def find_issue_urls(fetcher: Fetcher) -> list[str]:
    html = fetcher.get(ARCHIVE_URL)
    if html is None:
        raise SystemExit("could not load the issue archive")
    soup = BeautifulSoup(html, "html.parser")

    urls: list[str] = []
    for anchor in soup.select("a[href]"):
        url = normalize_url(attr(anchor, "href"))
        if not url.startswith(BASE + "/"):
            continue
        slug = url[len(BASE) + 1:]
        if "/" in slug:
            continue
        # Regular issues, plus the Adroit Prizes pages that stand in for issues.
        if re.match(r"issue-(?!archives)", slug) or re.search(r"adroit-prizes$", slug):
            if url not in urls:
                urls.append(url)

    # The archive links Issue Twelve to a URL that 404s; the live page is elsewhere.
    urls = [u for u in urls if not u.endswith("/issue-twelve-toc")]
    fixed = f"{BASE}/issue-twelve-contents"
    if fixed not in urls:
        urls.append(fixed)
    return urls


def poem_links(html: str, issue_url: str) -> list[tuple[str, str]]:
    """(url, link text) for every entry under an issue's Poetry heading."""
    soup = BeautifulSoup(html, "html.parser")
    # Drop the site chrome first: when Poetry is the last heading on the page,
    # the link scan below would otherwise run on into the footer and sidebar.
    for chrome in soup.select("header, footer, nav, aside, .widget, .sidebar, "
                              "#footer, .main-navigation, .thb-navigation"):
        chrome.decompose()
    headings = [h for h in soup.find_all(HEADING_TAG) if h.get_text(strip=True)]

    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, heading in enumerate(headings):
        if not POETRY_HEADING.search(heading.get_text(" ", strip=True)):
            continue
        nxt = headings[index + 1] if index + 1 < len(headings) else None
        stop_line = nxt.sourceline if nxt is not None and nxt.sourceline else float("inf")

        for anchor in heading.find_all_next("a", href=True):
            if anchor.sourceline and anchor.sourceline >= stop_line:
                break
            url = normalize_url(attr(anchor, "href"))
            text = anchor.get_text(" ", strip=True)
            if not url.startswith(BASE + "/") or url == issue_url or url in seen:
                continue
            path = url[len(BASE) + 1:]
            if "/" not in path and not re.match(r"(issue|20\d\d)[-\d]", path):
                continue    # site furniture: /our-supporters, /contributors, ...
            if re.search(r"(editors?-note|a-conversation-with|a-review-of"
                         r"|table-of-contents|^contents|-contents$|-toc$"
                         r"|adroit-prizes$|^about/|^issue-archives)", path):
                continue
            seen.add(url)
            results.append((url, text))
    return results


# poem page -> poems


def parse_poem_page(html: str, url: str, toc_text: str = "") -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    blocks = content_blocks(soup)
    tail_start = max(0, len(blocks) - 8)

    poems: list[dict] = []
    current: dict | None = None

    for index, block in enumerate(blocks):
        if HEADING_TAG.match(block.name or ""):
            title = re.sub(r"\s+", " ", element_text(block)).strip()
            if title:
                current = {"title": title, "byline": "", "pretty": "", "body": []}
                poems.append(current)
            continue

        if is_embed_credit(block):
            continue

        text = element_text(block)
        if not text.strip():
            continue
        flat = re.sub(r"\s+", " ", text).strip()

        if NAV.match(flat):
            current = None
            continue
        if current is None:
            continue

        first_line = re.sub(r"\s+", " ", text.split("\n")[0]).strip()
        byline = BYLINE.match(first_line)
        if byline and not current["byline"] and not current["body"]:
            current["byline"] = byline.group(1).strip()
            continue

        # Contributor bio: ends the poem, and supplies a properly cased name.
        if current["byline"] and index >= tail_start:
            candidate = bio_name(leading_bold(block), current["byline"]) or \
                        bio_name(flat, current["byline"])
            if candidate:
                current["pretty"] = candidate
                current = None
                continue
            if ("\n" not in text and len(flat) > 150 and BIO_PHRASES.search(flat)
                    and index >= len(blocks) - 3):
                current = None
                continue

        current["body"].append(text)

    return [row for row in (_finalize(p, url, toc_text) for p in poems) if row]


def _finalize(poem: dict, url: str, toc_text: str) -> dict | None:
    body = "\n\n".join(part for part in poem["body"] if part.strip())
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if not body:
        return None
    # Poems published only as an image or PDF leave behind a "click here" stub,
    # or nothing but an epigraph. Neither is usable text.
    if re.match(r"click\s+here\s+to\s+(read|view|see)", poem["title"], re.I):
        return None
    body = re.sub(r"(?im)^\s*click\s+here\s+to\s+(read|view|see)\b.*$", "", body).strip()
    if len(body) < MIN_POEM_CHARS:
        return None
    # A lone epigraph is all that survives when the poem is a scanned image.
    if "\n" not in body and len(body) < 150 and re.match(r"(after|for)\s", body, re.I):
        return None

    byline = poem["byline"]
    if byline and norm_name(toc_text) == norm_name(byline):
        poet = toc_text                      # TOC spells it properly, accents and all
    elif poem["pretty"]:
        poet = poem["pretty"]                # bio spells it properly
    elif byline:
        poet = smart_title(byline) if byline.isupper() else byline
    elif norm_name(toc_text) and norm_name(toc_text) != norm_name(poem["title"]):
        poet = toc_text                      # no byline on the page; trust the TOC
    else:
        poet = ""

    # Titles are kept verbatim; the journal's own capitalization is part of the
    # poem. The one exception is an ALL-CAPS title the TOC spells out properly.
    title = poem["title"]
    if title.isupper() and norm_name(toc_text) == norm_name(title):
        title = toc_text

    # Ambiguous Unicode: a handful of invisibles is noise to scrub, but a poem
    # dense with them came through some lossy copy-paste and is not worth keeping.
    sample = f"{title}\n{body}"
    ambiguous = ambiguous_count(sample)
    if ambiguous >= AMBIGUOUS_MIN_COUNT and ambiguous / len(sample) >= AMBIGUOUS_MIN_RATIO:
        return None
    title, body, poet = (scrub_ambiguous(x) for x in (title, body, poet))
    title, body, poet = (normalize_typography(x) for x in (title, body, poet))
    body = "\n".join(line.rstrip() for line in body.split("\n")).strip()
    if len(body) < MIN_POEM_CHARS:
        return None

    # English only: a quoted phrase or a foreign title is fine, a bilingual poem
    # or a reprinted original is not.
    if non_english_share(f"{title}\n{body}") > MAX_NON_ENGLISH:
        return None

    return {"title": title.strip(), "poet": poet.strip(), "text": body, "url": url}



# driver


def scrape(limit: int | None, workers: int, use_cache: bool, delay: float,
           only: list[str] | None = None, max_pages: int | None = None) -> list[dict]:
    fetcher = Fetcher(use_cache=use_cache, delay=delay)

    targets: dict[str, str] = {}
    if only:
        # Scrape named contributor pages directly, skipping the archive walk.
        for raw in only:
            targets[normalize_url(raw)] = ""
        print(f"Scraping {len(targets)} named contributor pages.")
    else:
        issue_urls = find_issue_urls(fetcher)
        if limit:
            issue_urls = issue_urls[:limit]
        print(f"Found {len(issue_urls)} issues.")

        def links_for(issue_url: str) -> list[tuple[str, str]]:
            html = fetcher.get(issue_url)
            if html is None:
                print(f"  ! missing issue page {issue_url}", file=sys.stderr)
                return []
            found = poem_links(html, issue_url)
            print(f"  {issue_url.rsplit('/', 1)[-1]}: {len(found)} poetry entries")
            return found

        with ThreadPoolExecutor(workers) as pool:
            per_issue = list(pool.map(links_for, issue_urls))

        for found in per_issue:
            for url, text in found:
                targets.setdefault(url, text)
        if max_pages:
            targets = dict(list(targets.items())[:max_pages])
        print(f"\nFetching {len(targets)} contributor pages...")

    def scrape_one(item: tuple[str, str]) -> list[dict]:
        url, toc_text = item
        html = fetcher.get(url)
        if html is None:
            return []
        try:
            return parse_poem_page(html, url, toc_text)
        except Exception as exc:
            print(f"  ! parse failed for {url}: {exc}", file=sys.stderr)
            return []

    poems: list[dict] = []
    with ThreadPoolExecutor(workers) as pool:
        for done, batch in enumerate(pool.map(scrape_one, targets.items()), 1):
            poems.extend(batch)
            if done % 100 == 0:
                print(f"  {done}/{len(targets)} pages -> {len(poems)} poems")

    return dedupe(poems)


def dedupe(poems: Iterable[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for poem in poems:
        key = (norm_name(poem["poet"]), re.sub(r"\s+", " ", poem["text"]).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(poem)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only scrape the first N issues")
    parser.add_argument("--workers", type=int, default=6, help="concurrent requests (default 6)")
    parser.add_argument("--delay", type=float, default=0.4,
                        help="seconds between requests, globally (default 0.4)")
    parser.add_argument("--no-cache", action="store_true", help="ignore the on-disk HTML cache")
    parser.add_argument("--max-pages", type=int,
                        help="stop after N contributor pages (quick test run)")
    parser.add_argument("--url", action="append", metavar="URL",
                        help="scrape only these contributor pages; repeatable")
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "poems.json"))
    args = parser.parse_args()

    poems = scrape(args.limit, args.workers, not args.no_cache, args.delay,
                   only=args.url, max_pages=args.max_pages)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(poems, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    missing_poet = sum(1 for p in poems if not p["poet"])
    print(f"\nWrote {len(poems)} poems to {args.out}")
    print(f"  distinct poets: {len({norm_name(p['poet']) for p in poems if p['poet']})}")
    if missing_poet:
        print(f"  poems with no poet attributed: {missing_poet}")


if __name__ == "__main__":
    main()
