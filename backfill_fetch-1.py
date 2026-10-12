#!/usr/bin/env python3
"""Muretto: scarica da OpenF1 l'archivio delle gare già corse, per tarare il modello.

Per ogni weekend concluso scrive un file storico/<anno>/<NN>-<gp>.json.gz con tutte le sessioni
(libere, qualifiche, Sprint, gara): piloti, risultati, griglia, giri, stint, soste, posizioni,
meteo, direzione gara e, per gara e Sprint, la telemetria dei primi minuti (partenze).
Scarica anche il weekend in corso, con le sole sessioni già finite (file "parziale", riscaricato
al lancio successivo): serve al modello di gara dopo la qualifica.
Scarica anche le edizioni passate di uno o più circuiti (di default Austin), senza telemetria.

Uso:
    python backfill_fetch.py                                  # 2026 + Austin 2023-2025
    python backfill_fetch.py --anni 2026 --circuito "Austin,Mexico,Brazil" --anni-circuito 2024,2025

Riprende da dove si era fermato: i weekend già scaricati senza errori vengono saltati
(quelli parziali no).
Solo libreria standard. I dati storici di OpenF1 sono gratuiti e senza chiave.
"""
import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

BASE = os.environ.get("OPENF1_BASE", "https://api.openf1.org/v1")
UTC = timezone.utc
SESSIONS = ("Practice 1", "Practice 2", "Practice 3", "Sprint Qualifying", "Sprint Shootout",
            "Sprint", "Qualifying", "Race")
RACES = ("Sprint", "Race")
COMMON = ("drivers", "session_result", "laps", "stints", "weather", "race_control")
RACE_ONLY = ("pit", "position", "starting_grid", "overtakes")
CAR_FIELDS = ("date", "speed", "rpm", "n_gear", "throttle", "brake", "drs")
START_BEFORE = 360     # secondi di telemetria prima dell'inizio del giro 2
START_AFTER = 10       # secondi dopo
MIN_REQ_GAP = float(os.environ.get("OPENF1_GAP", "2.1"))  # circa 28 richieste al minuto


class Denied(Exception):
    """Accesso negato (di solito durante una sessione in diretta): si riprova più tardi."""


_last = [0.0]
_count = [0]


def get(endpoint, *conds):
    url = BASE + "/" + endpoint
    if conds:
        url += "?" + "&".join(c.replace(" ", "%20") for c in conds)
    delays = (5, 15, 30, 60, 90)
    for attempt in range(len(delays) + 1):
        wait = MIN_REQ_GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        _count[0] += 1
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "muretto-storico/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            if e.code in (401, 403):
                raise Denied(f"{endpoint}: accesso negato ({e.code})")
            if e.code in (422,):
                raise ValueError(f"{endpoint}: richiesta non valida ({e.code})")
            if e.code not in (429, 500, 502, 503, 504) or attempt == len(delays):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            if attempt == len(delays):
                raise
        time.sleep(delays[attempt])
    raise RuntimeError(f"{endpoint}: troppi tentativi falliti")


def ts(s):
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    return d.timestamp()


def iso(epoch):
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def log(msg):
    print(msg, flush=True)


def warn(msg):
    print(f"::warning::{msg}" if os.environ.get("GITHUB_ACTIONS") else f"ATTENZIONE: {msg}", flush=True)


def slug(text):
    t = (text or "gp").lower()
    t = re.sub(r"[àáâä]", "a", t)
    t = re.sub(r"[èéêë]", "e", t)
    t = re.sub(r"[ìíîï]", "i", t)
    t = re.sub(r"[òóôö]", "o", t)
    t = re.sub(r"[ùúûü]", "u", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "gp"


def strip(rows, drop=("meeting_key", "session_key")):
    out = []
    for r in rows or []:
        out.append({k: v for k, v in r.items() if k not in drop and not k.startswith("segments_")})
    return out


def meetings_of(year):
    ms = [m for m in get("meetings", f"year={year}") if not m.get("is_cancelled")]
    ms = [m for m in ms if "testing" not in ((m.get("meeting_name") or "")
                                             + (m.get("meeting_official_name") or "")).lower()]
    return sorted(ms, key=lambda m: m["date_start"])


def start_window(laps):
    """Finestra della partenza: fino all'inizio del giro 2 del primo pilota."""
    t2 = [ts(l["date_start"]) for l in laps if l.get("lap_number") == 2 and l.get("date_start")]
    if not t2:
        return None
    end = min(t2)
    return end - START_BEFORE, end + START_AFTER


def start_telemetry(key, drivers, win):
    out = {}
    for d in drivers:
        num = int(d["driver_number"])
        rows = get("car_data", f"session_key={key}", f"driver_number={num}",
                   f"date>={iso(win[0])}", f"date<={iso(win[1])}")
        rows.sort(key=lambda r: r.get("date") or "")
        out[str(num)] = {f: [r.get(f) for r in rows] for f in CAR_FIELDS}
    return out


def fetch_session(s, with_car):
    key = s["session_key"]
    rec = {"session": {k: s.get(k) for k in ("session_key", "session_name", "session_type", "date_start",
                                              "date_end", "gmt_offset")},
           "errori": []}
    for ep in COMMON + (RACE_ONLY if s["session_name"] in RACES else ()):
        try:
            rec[ep] = strip(get(ep, f"session_key={key}"))
        except Denied:
            raise
        except Exception as e:  # noqa: BLE001
            rec[ep] = []
            rec["errori"].append(f"{ep}: {type(e).__name__}: {e}")
            warn(f"  {s['session_name']} {ep}: {e}")
    if s["session_name"] == "Qualifying":
        try:  # in alcune stagioni la griglia è legata alla sessione di qualifica
            g = strip(get("starting_grid", f"session_key={key}"))
            if g:
                rec["starting_grid"] = g
        except Denied:
            raise
        except Exception:  # noqa: BLE001
            pass
    if with_car and s["session_name"] in RACES:
        win = start_window(rec.get("laps") or [])
        if win and rec.get("drivers"):
            try:
                rec["partenza"] = {"da": iso(win[0]), "a": iso(win[1]),
                                   "car_data": start_telemetry(key, rec["drivers"], win)}
            except Denied:
                raise
            except Exception as e:  # noqa: BLE001
                rec["errori"].append(f"car_data partenza: {type(e).__name__}: {e}")
                warn(f"  {s['session_name']} car_data partenza: {e}")
        else:
            rec["errori"].append("car_data partenza: giro 2 non trovato")
    return rec


def done_ok(path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("parziale"):
            return False  # weekend in corso: si riscarica
        return not d.get("errori") and not any(s.get("errori") for s in d.get("sessioni", []))
    except Exception:  # noqa: BLE001
        return False


def fetch_meeting(m, rnd, year, out_dir, with_car, parziale=False, now=None):
    name = m.get("meeting_name") or m.get("location") or "gp"
    path = os.path.join(out_dir, str(year), f"{rnd:02d}-{slug(name)}.json.gz")
    if os.path.exists(path) and done_ok(path):
        log(f"{year} {rnd:02d} {name}: già scaricato")
        return path, None
    sessions = [s for s in get("sessions", f"meeting_key={m['meeting_key']}")
                if not s.get("is_cancelled") and s.get("session_name") in SESSIONS]
    if parziale:
        # weekend in corso: solo le sessioni finite da almeno 20 minuti
        sessions = [s for s in sessions if s.get("date_end") and ts(s["date_end"]) + 20 * 60 < now]
        if not sessions:
            log(f"{year} {rnd:02d} {name}: in corso, nessuna sessione ancora finita")
            return None, None
    log(f"{year} {rnd:02d} {name}: scarico" + (" (in corso: " + ", ".join(s["session_name"] for s in sessions) + ")"
                                               if parziale else ""))
    sessions.sort(key=lambda s: s["date_start"])
    data = {"anno": year, "round": rnd,
            "meeting": {k: m.get(k) for k in ("meeting_key", "meeting_name", "meeting_official_name", "location",
                                              "country_name", "circuit_key", "circuit_short_name",
                                              "date_start", "date_end", "gmt_offset")},
            "sessioni": [], "errori": [],
            "scaricato": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if parziale:
        data["parziale"] = True
    for s in sessions:
        t0, n0 = time.time(), _count[0]
        rec = fetch_session(s, with_car)
        data["sessioni"].append(rec)
        log(f"  {s['session_name']}: {len(rec.get('laps') or [])} giri, {_count[0] - n0} richieste, "
            f"{time.time() - t0:.0f} s{' — ' + str(len(rec['errori'])) + ' errori' if rec['errori'] else ''}")
    if not sessions:
        data["errori"].append("nessuna sessione trovata")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, path)
    return path, data


def summary_of(path, data):
    if data is None:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    m = data["meeting"]
    errs = list(data.get("errori") or [])
    for s in data["sessioni"]:
        errs += [f"{s['session']['session_name']}: {e}" for e in s.get("errori") or []]
    return {"file": path.replace(os.sep, "/"), "anno": data["anno"], "round": data["round"],
            "gp": m.get("meeting_name"), "luogo": m.get("location"),
            "sessioni": [s["session"]["session_name"] for s in data["sessioni"]],
            "giri_gara": next((len(s.get("laps") or []) for s in data["sessioni"]
                               if s["session"]["session_name"] == "Race"), 0),
            "partenze": sum(1 for s in data["sessioni"] if s.get("partenza")),
            "errori": errs, "scaricato": data.get("scaricato"), "parziale": bool(data.get("parziale")),
            "kb": round(os.path.getsize(path) / 1024)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="storico")
    ap.add_argument("--anni", default="2026", help="stagioni complete da scaricare, es. 2026")
    ap.add_argument("--circuito", default="Austin",
                    help="circuiti di cui scaricare le edizioni passate, separati da virgola "
                         "(es. Austin,Mexico,Brazil,Las Vegas,Qatar,Abu Dhabi)")
    ap.add_argument("--senza-in-corso", action="store_true", help="non scaricare il weekend in corso")
    ap.add_argument("--anni-circuito", default="2023,2024,2025")
    ap.add_argument("--max-minuti", type=float, default=230, help="si ferma in modo pulito dopo questo tempo")
    ap.add_argument("--now", help=argparse.SUPPRESS)
    a = ap.parse_args()
    now = ts(a.now) if a.now else time.time()
    t_start = time.time()
    idx_path = os.path.join(a.out_dir, "indice.json")
    index = {}
    if os.path.exists(idx_path):
        try:
            index = {r["file"]: r for r in json.load(open(idx_path, encoding="utf-8"))["file"]}
        except Exception:  # noqa: BLE001
            index = {}

    jobs = []  # (meeting, round, anno, telemetria partenze, weekend in corso)
    try:
        for y in [int(x) for x in a.anni.split(",") if x.strip()]:
            ms = meetings_of(y)
            for i, m in enumerate(ms, 1):
                if ts(m["date_end"]) + 10 * 3600 < now:
                    jobs.append((m, i, y, True, False))
                elif not a.senza_in_corso and ts(m["date_start"]) < now:
                    jobs.append((m, i, y, True, True))
        circs = [c.strip().lower() for c in (a.circuito or "").split(",") if c.strip()]
        if circs:
            for y in [int(x) for x in a.anni_circuito.split(",") if x.strip()]:
                ms = meetings_of(y)
                for i, m in enumerate(ms, 1):
                    names = " ".join(str(m.get(k) or "") for k in ("location", "circuit_short_name",
                                                                   "meeting_name", "country_name")).lower()
                    if any(c in names for c in circs) and ts(m["date_end"]) + 10 * 3600 < now:
                        jobs.append((m, i, y, False, False))
    except Denied as e:
        warn(f"{e}. Probabilmente c'è una sessione in diretta: rilancia più tardi.")
        return 0
    log(f"Weekend da controllare: {len(jobs)}")

    stopped = None
    for m, rnd, y, with_car, parziale in jobs:
        if (time.time() - t_start) / 60 > a.max_minuti:
            stopped = "tempo massimo raggiunto: rilancia il workflow per finire"
            break
        try:
            path, data = fetch_meeting(m, rnd, y, a.out_dir, with_car, parziale, now)
            if path is None:
                continue
        except Denied as e:
            stopped = f"{e}. Probabilmente c'è una sessione in diretta: rilancia più tardi."
            break
        except Exception as e:  # noqa: BLE001
            warn(f"{y} {rnd:02d} {m.get('meeting_name')}: errore {type(e).__name__}: {e}")
            continue
        s = summary_of(path, data)
        index[s["file"]] = s

    rows = sorted(index.values(), key=lambda r: (r["anno"], r["round"]))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({"aggiornato": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "fermato": stopped, "richieste": _count[0], "file": rows}, f, ensure_ascii=False, indent=1)
    bad = [r for r in rows if r["errori"]]
    log(f"Fatto: {len(rows)} weekend in archivio, {len(bad)} con errori, {_count[0]} richieste, "
        f"{(time.time() - t_start) / 60:.0f} minuti.")
    if stopped:
        warn(stopped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
