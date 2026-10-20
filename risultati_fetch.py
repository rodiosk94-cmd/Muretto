#!/usr/bin/env python3
"""Muretto: scarica da OpenF1 solo i risultati (primi 3) di FP2, FP3, qualifica e gara.

Serve per le statistiche "migliore in libera -> vittoria/pole" su stagioni intere,
senza scaricare giri e telemetria. Scrive storico/risultati_sessioni.json.

Uso:
    python risultati_fetch.py --anni 2023,2024,2025,2026
Solo libreria standard.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

BASE = os.environ.get("OPENF1_BASE", "https://api.openf1.org/v1")
GAP = float(os.environ.get("OPENF1_GAP", "2.1"))
SESSIONI = ("Practice 2", "Practice 3", "Qualifying", "Race")
_last = [0.0]


def get(endpoint, **params):
    url = BASE + "/" + endpoint
    if params:
        url += "?" + "&".join(f"{k}={str(v).replace(' ', '%20')}" for k, v in params.items())
    for attesa in (5, 15, 30, 60, 90, None):
        pausa = GAP - (time.time() - _last[0])
        if pausa > 0:
            time.sleep(pausa)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "muretto-risultati/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            if attesa is None:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attesa is None:
                raise
        time.sleep(attesa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anni", default="2023,2024,2025,2026")
    ap.add_argument("--out", default="storico/risultati_sessioni.json")
    a = ap.parse_args()

    out = {"aggiornato": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "weekend": []}
    for anno in [int(x) for x in a.anni.split(",") if x.strip()]:
        sessioni = get("sessions", year=anno)
        meeting = {}
        for s in sessioni:
            if s.get("is_cancelled"):
                continue
            meeting.setdefault(s["meeting_key"], []).append(s)
        for mk, ss in sorted(meeting.items(), key=lambda kv: min(x["date_start"] for x in kv[1])):
            nomi = {s["session_name"] for s in ss}
            if "Race" not in nomi:
                continue  # test pre-stagionali
            piloti = {d["driver_number"]: d.get("name_acronym") for d in get("drivers", meeting_key=mk)}
            w = {"anno": anno, "meeting_key": mk, "luogo": ss[0].get("location"),
                 "sprint": "Sprint" in nomi, "sessioni": {}}
            for s in ss:
                if s["session_name"] not in SESSIONI:
                    continue
                res = get("session_result", session_key=s["session_key"])
                top = sorted([r for r in res if isinstance(r.get("position"), int)],
                             key=lambda r: r["position"])[:3]
                w["sessioni"][s["session_name"]] = [
                    {"pos": r["position"], "num": r["driver_number"],
                     "pilota": piloti.get(r["driver_number"])} for r in top]
            out["weekend"].append(w)
            print(anno, w["luogo"], {k: [x["pilota"] for x in v] for k, v in w["sessioni"].items()}, flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Scritto {a.out}: {len(out['weekend'])} weekend")


if __name__ == "__main__":
    main()
