#!/usr/bin/env python3
"""Notizie di Formula 1 per il Muretto.

Legge i feed RSS/Atom dei siti principali e scrive news.json con solo titolo, fonte, data e link
(niente testi degli articoli). Un feed che non risponde viene saltato senza fermare gli altri.
Il file viene riscritto solo se l'elenco delle notizie è cambiato.

Uso: python news_fetch.py --out news.json
"""
import argparse
import email.utils
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

UTC = timezone.utc
MAX_AGE_DAYS = 7
MAX_ITEMS = 30
PER_FEED = 12

# (fonte mostrata, indirizzo del feed, lingua, solo_f1)
# solo_f1=False: il feed mescola altri sport, quindi si tengono solo i titoli che parlano di F1
FEEDS = [
    ("Sky Sport", "https://news.google.com/rss/search?q=site:sport.sky.it+%22formula+1%22+when:7d&hl=it&gl=IT&ceid=IT:it", "it", False),
    ("Motorsport.com", "https://it.motorsport.com/rss/f1/news/", "it", True),
    ("FormulaPassion", "https://www.formulapassion.it/feed", "it", False),
    ("Sky Sports F1", "https://www.skysports.com/rss/12433", "en", True),
    ("Formula1.com", "https://www.formula1.com/en/latest/all.xml", "en", True),
]

F1_WORDS = re.compile(
    r"\b(f1|formula ?1|formula uno|gran premio|gp|ferrari|leclerc|hamilton|verstappen|norris|piastri|russell|"
    r"antonelli|mclaren|red bull|mercedes|aston martin|alpine|williams|haas|sauber|audi|cadillac|racing bulls|"
    r"alonso|sainz|bearman|hadjar|lawson|ocon|gasly|stroll|albon|hulkenberg|bortoleto|colapinto|bottas|perez|"
    r"pirelli|fia|pole|qualifiche|qualifying|sprint)\b", re.I)

UA = "Mozilla/5.0 (compatible; muretto/1.0; +https://github.com/)"


def log(msg):
    print(msg, flush=True)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read()


def local(tag):
    return tag.rsplit("}", 1)[-1].lower()


def child(el, *names):
    for c in el:
        if local(c.tag) in names:
            return c
    return None


def text_of(el):
    return "" if el is None else "".join(el.itertext()).strip()


def parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        d = email.utils.parsedate_to_datetime(s)
        if d is not None:
            return d if d.tzinfo else d.replace(tzinfo=UTC)
    except (TypeError, ValueError, IndexError):
        pass
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=UTC)
    except ValueError:
        return None


def clean_title(t, source):
    t = html.unescape(re.sub(r"<[^>]+>", "", t or "")).strip()
    t = re.sub(r"\s+", " ", t)
    if source == "Sky Sport":  # Google News aggiunge " - Sky Sport" in coda
        t = re.sub(r"\s+[-–]\s+Sky Sport.*$", "", t)
    return t


def parse_feed(raw, source, lang, only_f1):
    root = ET.fromstring(raw)
    items = []
    kind = local(root.tag)
    if kind == "rss" or kind == "rdf":
        nodes = [e for e in root.iter() if local(e.tag) == "item"]
    else:  # Atom
        nodes = [e for e in root.iter() if local(e.tag) == "entry"]
    for n in nodes:
        title = clean_title(text_of(child(n, "title")), source)
        link = ""
        le = child(n, "link")
        if le is not None:
            link = (le.get("href") or text_of(le)).strip()
        if not link:
            g = child(n, "guid")
            if g is not None and text_of(g).startswith("http"):
                link = text_of(g)
        date = parse_date(text_of(child(n, "pubdate", "published", "updated", "date")))
        if not title or not link.startswith("http"):
            continue
        if not only_f1 and not F1_WORDS.search(title):
            continue
        items.append({"title": title, "link": link, "source": source, "lang": lang,
                      "ts": int(date.timestamp()) if date else None})
    return items[:PER_FEED]


def norm_key(title):
    return re.sub(r"[^a-z0-9àèéìòù]+", " ", title.lower()).strip()[:70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="news.json")
    args = ap.parse_args()
    now = time.time()
    got, ok = [], []
    for source, url, lang, only_f1 in FEEDS:
        try:
            items = parse_feed(fetch(url), source, lang, only_f1)
            log(f"{source}: {len(items)} notizie")
            got.extend(items)
            if items:
                ok.append(source)
        except Exception as e:  # noqa: BLE001
            log(f"::warning::{source}: feed non disponibile ({type(e).__name__}: {e})")
    fresh = [x for x in got if x["ts"] is None or now - x["ts"] <= MAX_AGE_DAYS * 86400]
    fresh = [x for x in fresh if x["ts"] is None or x["ts"] <= now + 3600]
    fresh.sort(key=lambda x: x["ts"] or 0, reverse=True)
    seen, out = set(), []
    for x in fresh:
        k1, k2 = x["link"], norm_key(x["title"])
        if k1 in seen or k2 in seen:
            continue
        seen.update((k1, k2))
        out.append(x)
        if len(out) >= MAX_ITEMS:
            break
    if not out:
        log("Nessuna notizia raccolta: news.json resta com'è.")
        return 0
    try:
        with open(args.out, encoding="utf-8") as f:
            prev = json.load(f)
    except (OSError, ValueError):
        prev = {}
    if [x["link"] for x in prev.get("items", [])] == [x["link"] for x in out]:
        log("Notizie invariate.")
        return 0
    data = {"updated": datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "sources": ok, "items": out}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    log(f"Scritte {len(out)} notizie in {args.out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
