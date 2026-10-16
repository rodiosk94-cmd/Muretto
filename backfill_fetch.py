#!/usr/bin/env python3
"""Muretto: scarica da OpenF1 l'archivio delle gare già corse, per tarare il modello.

Per ogni weekend concluso scrive un file storico/<anno>/<NN>-<gp>.json.gz con tutte le sessioni
(libere, qualifiche, Sprint, gara): piloti, risultati, griglia, giri, stint, soste, posizioni,
meteo, direzione gara e, per gara e Sprint, la telemetria dei primi minuti (partenze).
Scarica anche il weekend in corso, con le sole sessioni già finite (file "parziale", riscaricato
al lancio successivo): serve al modello di gara dopo la qualifica.
Scarica anche le edizioni passate di uno o più circuiti (di default Austin), senza telemetria.
Telemetria dei giri per il modello energetico (chiave 'energia' di ogni sessione della stagione):
giro migliore di libere e qualifiche, un long run di FP1 e FP2, due blocchi di 4 giri in gara e uno
nella Sprint (velocità, gas, freno, giri motore, marcia). Si aggiunge anche ai weekend già scaricati.
Per ogni circuito del calendario, il giro migliore dei primi 6 nella qualifica delle edizioni 2024 e
2025 (storico/circuiti/): misura rettilinei e frenate con le stesse auto per tutte le piste.

Uso:
    python backfill_fetch.py                                  # 2026 + Austin 2023-2025
    python backfill_fetch.py --anni 2026 --circuito "Austin,Mexico,Brazil" --anni-circuito 2024,2025

Riprende da dove si era fermato: i weekend già scaricati senza errori vengono saltati
(quelli parziali no); a quelli senza telemetria dei giri si aggiunge solo quella.
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
ENERGIA_V = 1
# piano della telemetria dei giri per sessione: (giro migliore, long run, blocchi di gara da 4 giri)
PIANO_ENERGIA = {"Practice 1": (True, True, 0), "Practice 2": (True, True, 0), "Practice 3": (True, False, 0),
                 "Sprint Qualifying": (True, False, 0), "Sprint Shootout": (True, False, 0),
                 "Qualifying": (True, False, 0), "Sprint": (False, False, 1), "Race": (False, False, 2)}
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


def _neutralizzazioni(rec):
    """Intervalli con SC, VSC o bandiera rossa (secondi epoch), più un margine dopo la fine: i giri
    subito dopo una ripartenza hanno batteria e gomme fuori dal regime normale."""
    gara = rec["session"]["session_name"] in RACES
    iv, aperto = [], {}
    for m in sorted(rec.get("race_control") or [], key=lambda x: x.get("date") or ""):
        msg = (m.get("message") or "").upper()
        if not m.get("date"):
            continue
        t = ts(m["date"])
        if msg == "SAFETY CAR DEPLOYED":
            aperto.setdefault("SC", t)
        elif msg.startswith("SAFETY CAR IN THIS LAP") and "SC" in aperto:
            iv.append((aperto.pop("SC"), t + 250))
        elif msg in ("VSC DEPLOYED", "VIRTUAL SAFETY CAR DEPLOYED"):
            aperto.setdefault("VSC", t)
        elif msg in ("VSC ENDING", "VIRTUAL SAFETY CAR ENDING") and "VSC" in aperto:
            iv.append((aperto.pop("VSC"), t + 100))
        elif m.get("flag") == "RED" or msg.startswith("RED FLAG"):
            for k in ("SC", "VSC"):  # la bandiera rossa chiude SC e VSC
                if k in aperto:
                    iv.append((aperto.pop(k), t))
            aperto.setdefault("RED", t)
        elif m.get("category") == "SessionStatus" and msg == "SESSION STARTED" and "RED" in aperto:
            iv.append((aperto.pop("RED"), t + (400 if gara else 0)))
    for t in aperto.values():
        iv.append((t, t + 1e5))
    return iv


def blocchi_energia(rec, primi=None):
    """Giri di cui scaricare la telemetria: lista di (pilota, tipo, [giri], t0, t1, [(inizio, durata)]).
    primi=N: solo il giro migliore dei primi N (classifica della sessione o, se manca, giro migliore)."""
    piano = PIANO_ENERGIA.get(rec["session"]["session_name"])
    if not piano:
        return []
    migliore, long_run, n_blocchi = piano
    if primi:
        migliore, long_run, n_blocchi = True, False, 0
    iv = _neutralizzazioni(rec)
    pit = {(p.get("driver_number"), p.get("lap_number")) for p in rec.get("pit") or []}
    by = {}
    for l in rec.get("laps") or []:
        if l.get("lap_duration") and l.get("date_start") and l.get("lap_number") and l.get("driver_number"):
            by.setdefault(l["driver_number"], []).append(l)
    if not by:
        return []

    def pulito(l, dn):
        if l.get("is_pit_out_lap") or l["lap_number"] <= 1 or (dn, l["lap_number"]) in pit:
            return False
        t0 = ts(l["date_start"])
        t1 = t0 + l["lap_duration"]
        return not any(a <= t1 and t0 <= b for a, b in iv)

    def finestra(gl):
        t0 = ts(gl[0]["date_start"]) - 1.0
        t1 = ts(gl[-1]["date_start"]) + gl[-1]["lap_duration"] + 1.0
        return t0, t1, [(ts(x["date_start"]), x["lap_duration"]) for x in gl]

    puliti = {dn: [l for l in sorted(v, key=lambda l: l["lap_number"]) if pulito(l, dn)] for dn, v in by.items()}
    piloti = sorted(by)
    if primi:
        pos = {r.get("driver_number"): r.get("position") for r in rec.get("session_result") or []}
        if any(isinstance(p, int) for p in pos.values()):
            piloti = sorted([d for d in piloti if isinstance(pos.get(d), int)], key=lambda d: pos[d])
        else:
            piloti = sorted([d for d in piloti if puliti[d]], key=lambda d: min(l["lap_duration"] for l in puliti[d]))
        piloti = [d for d in piloti if puliti[d]][:primi]
    n_tot = max(l["lap_number"] for v in by.values() for l in v)

    out = []
    for dn in piloti:
        gl = sorted(by[dn], key=lambda l: l["lap_number"])
        ok = puliti[dn]
        if not ok:
            continue
        if migliore:
            b = min(ok, key=lambda l: l["lap_duration"])
            out.append((dn, "migliore", [b["lap_number"]]) + finestra([b]))
        if long_run:
            # long run: 4 giri consecutivi entro il 2,5% della loro mediana, nel tratto di giri puliti
            # consecutivi con più giri a quel ritmo (i tentativi di qualifica alternano giri lanciati e lenti)
            num = {l["lap_number"] for l in ok}
            tratti, cur = [], []
            for l in gl:
                if l["lap_number"] in num and cur and l["lap_number"] == cur[-1]["lap_number"] + 1:
                    cur.append(l)
                else:
                    if len(cur) >= 4:
                        tratti.append(cur)
                    cur = [l] if l["lap_number"] in num else []
            if len(cur) >= 4:
                tratti.append(cur)
            cand = []
            for tr in tratti:
                for i in range(len(tr) - 3):
                    w = tr[i:i + 4]
                    med = sorted(x["lap_duration"] for x in w)[1:3]
                    med = (med[0] + med[1]) / 2
                    if all(abs(x["lap_duration"] / med - 1) <= 0.025 for x in w):
                        n_ritmo = sum(1 for x in tr if abs(x["lap_duration"] / med - 1) <= 0.03)
                        cand.append((n_ritmo, i >= 1, -i, w))
            if cand:
                n_ritmo, _, _, w = max(cand, key=lambda c: c[:3])
                if n_ritmo >= 5:
                    out.append((dn, "long_run", [x["lap_number"] for x in w]) + finestra(w))
        if n_blocchi:
            # blocchi di 4 giri verdi consecutivi intorno al 30% e al 70% della gara (50% nella Sprint),
            # gli stessi giri per tutti quando possibile
            mediana = sorted(l["lap_duration"] for l in ok)[len(ok) // 2]
            buoni = {l["lap_number"]: l for l in ok if l["lap_duration"] <= 1.07 * mediana}
            presi = set()
            for f in ([0.5] if n_blocchi == 1 else [0.3, 0.7]):
                centro = int(round(f * n_tot))
                trovato = False
                for spost in range(0, 12):
                    for s0 in (centro + spost, centro - spost):
                        giri = list(range(s0, s0 + 4))
                        if all(g in buoni for g in giri) and not presi & set(giri):
                            out.append((dn, "gara", giri) + finestra([buoni[g] for g in giri]))
                            presi |= set(giri)
                            trovato = True
                            break
                    if trovato:
                        break
    return out


def compatta(rows, t0):
    rows = sorted(rows, key=lambda r: r.get("date") or "")
    return {"t": [int(round((ts(r["date"]) - t0) * 100)) for r in rows],
            "v": [r.get("speed") for r in rows],
            "gas": [r.get("throttle") for r in rows],
            "freno": [1 if (r.get("brake") or 0) > 0 else 0 for r in rows],
            "rpm": [r.get("rpm") for r in rows],
            "marcia": [r.get("n_gear") for r in rows]}


def scarica_energia(rec, primi=None, scadenza=None, vecchi=None):
    """Telemetria dei giri scelti della sessione, in rec['energia']. Riusa i blocchi già scaricati
    (vecchi: il dict 'energia' precedente) e non riprova quelli senza dati.
    Blocco: pilota, tipo (migliore, long_run, gara), giri, da (inizio della finestra), inizi e durate
    dei giri (centesimi dall'inizio, secondi), t (centesimi), v, gas, freno (0/1), rpm, marcia."""
    key = rec["session"]["session_key"]
    out = {"v": ENERGIA_V, "blocchi": [], "vuoti": [], "errori": []}
    gia, vuoti = {}, set()
    if vecchi and vecchi.get("v") == ENERGIA_V:
        gia = {(b["pilota"], b["tipo"], tuple(b["giri"])): b for b in vecchi.get("blocchi") or []}
        vuoti = {(x[0], x[1], tuple(x[2])) for x in vecchi.get("vuoti") or []}
    rec["energia"] = out
    mancanti, negato = 0, None
    for dn, tipo, giri, t0, t1, info in blocchi_energia(rec, primi):
        k = (dn, tipo, tuple(giri))
        if k in gia:
            out["blocchi"].append(gia[k])
            continue
        if k in vuoti:
            out["vuoti"].append([dn, tipo, giri])
            continue
        if negato or (scadenza and time.time() > scadenza):
            mancanti += 1
            continue
        try:
            rows = get("car_data", f"session_key={key}", f"driver_number={dn}",
                       f"date>={iso(t0)}", f"date<={iso(t1)}")
        except Denied as e:
            negato = e
            mancanti += 1
            continue
        except Exception as e:  # noqa: BLE001
            out["errori"].append(f"{dn} {tipo} {giri}: {type(e).__name__}: {e}")
            continue
        rows = [r for r in rows or [] if r.get("date")]
        if not rows:
            out["vuoti"].append([dn, tipo, giri])
            continue
        b = {"pilota": dn, "tipo": tipo, "giri": giri, "da": iso(t0),
             "inizi": [int(round((s - t0) * 100)) for s, _ in info], "durate": [d for _, d in info]}
        b.update(compatta(rows, t0))
        out["blocchi"].append(b)
    if mancanti:
        out["errori"].append(f"{'accesso negato' if negato else 'tempo massimo raggiunto'}: "
                             f"{mancanti} blocchi da scaricare")
    if negato:
        raise negato
    return out


def manca_energia(data):
    """Sessioni del weekend senza telemetria dei giri, con una versione vecchia o con errori."""
    out = []
    for rec in data.get("sessioni", []):
        if rec["session"]["session_name"] not in PIANO_ENERGIA:
            continue
        e = rec.get("energia")
        if not e or e.get("v") != ENERGIA_V or e.get("errori"):
            out.append(rec)
    return out


def salva(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, path)


def aggiungi_energia(path, data, scadenza=None):
    """Aggiunge la telemetria dei giri alle sessioni che non l'hanno e salva il file (anche se si ferma)."""
    mancanti = manca_energia(data)
    if not mancanti or (scadenza and time.time() > scadenza):
        return False
    try:
        for rec in mancanti:
            t0, n0 = time.time(), _count[0]
            try:
                scarica_energia(rec, scadenza=scadenza, vecchi=rec.get("energia"))
            finally:
                e = rec["energia"]
                log(f"  energia {rec['session']['session_name']}: {len(e['blocchi'])} blocchi, "
                    f"{_count[0] - n0} richieste, {time.time() - t0:.0f} s"
                    + (f" — {'; '.join(e['errori'][:3])}" if e["errori"] else ""))
    finally:
        salva(path, data)
    return True


def energia_vecchia(path):
    """Telemetria dei giri di un file che si sta riscaricando, per non riscaricarla (session_key -> dict).
    Se il file era parziale non si tengono i blocchi senza dati: si riprovano."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            old = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for s in old.get("sessioni", []):
        e = s.get("energia")
        if e and e.get("v") == ENERGIA_V:
            out[s["session"]["session_key"]] = {"v": ENERGIA_V, "blocchi": e.get("blocchi") or [],
                                                "vuoti": [] if old.get("parziale") else e.get("vuoti") or []}
    return out


def scarica_circuito(m, year, out_dir, scadenza=None, primi=6):
    """Giro migliore in qualifica dei primi 6 di un'edizione passata di un circuito del calendario: serve a
    misurare il circuito (rettilinei, frenate) con le stesse auto per tutte le piste.
    File storico/circuiti/<anno>-<circuito>.json.gz. Ritorna (path, data) oppure (None, None)."""
    nome = m.get("circuit_short_name") or m.get("location") or "circuito"
    path = os.path.join(out_dir, "circuiti", f"{year}-{slug(nome)}.json.gz")
    vecchi = None
    if os.path.exists(path):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                old = json.load(f)
            e = old.get("energia") or {}
            # senza nessun giro (OpenF1 non aveva i giri al primo scaricamento) si riprova
            if not old.get("errori") and e.get("v") == ENERGIA_V and not e.get("errori") and e.get("blocchi"):
                return path, None
            vecchi = e
        except Exception:  # noqa: BLE001
            pass
    if scadenza and time.time() > scadenza:
        return None, None
    qs = [s for s in get("sessions", f"meeting_key={m['meeting_key']}")
          if s.get("session_name") == "Qualifying" and not s.get("is_cancelled")]
    if not qs:
        log(f"  {year} {nome}: qualifica non trovata")
        return None, None
    s = qs[0]
    key = s["session_key"]
    rec = {"session": {k: s.get(k) for k in ("session_key", "session_name", "session_type", "date_start",
                                              "date_end", "gmt_offset")}, "errori": []}
    for ep in ("session_result", "laps", "race_control", "weather"):
        try:
            rec[ep] = strip(get(ep, f"session_key={key}"))
        except Denied:
            raise
        except Exception as e:  # noqa: BLE001
            rec[ep] = []
            rec["errori"].append(f"{ep}: {type(e).__name__}: {e}")
    t0, n0 = time.time(), _count[0]
    scarica_energia(rec, primi=primi, scadenza=scadenza, vecchi=vecchi)
    piloti = {b["pilota"] for b in rec["energia"]["blocchi"]}
    data = {"anno": year,
            "meeting": {k: m.get(k) for k in ("meeting_key", "meeting_name", "location", "country_name",
                                              "circuit_key", "circuit_short_name", "date_start", "gmt_offset")},
            "sessione": rec["session"],
            "session_result": rec["session_result"],
            "laps": [l for l in rec["laps"] if l.get("driver_number") in piloti],
            "weather": rec["weather"], "energia": rec["energia"], "errori": rec["errori"],
            "scaricato": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    salva(path, data)
    e = rec["energia"]
    log(f"  {year} {nome}: {len(e['blocchi'])} giri, {_count[0] - n0} richieste, {time.time() - t0:.0f} s"
        + (f" — {'; '.join((rec['errori'] + e['errori'])[:3])}" if rec["errori"] or e["errori"] else ""))
    return path, data


def done_ok(path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("parziale"):
            return False  # weekend in corso: si riscarica
        return not d.get("errori") and not any(s.get("errori") for s in d.get("sessioni", []))
    except Exception:  # noqa: BLE001
        return False


def fetch_meeting(m, rnd, year, out_dir, with_car, parziale=False, now=None, energia=True, scadenza=None):
    name = m.get("meeting_name") or m.get("location") or "gp"
    path = os.path.join(out_dir, str(year), f"{rnd:02d}-{slug(name)}.json.gz")
    energia = energia and with_car   # edizioni passate: niente telemetria dei giri (c'è storico/circuiti)
    if os.path.exists(path) and done_ok(path):
        if energia:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            if manca_energia(data) and not (scadenza and time.time() > scadenza):
                log(f"{year} {rnd:02d} {name}: aggiungo la telemetria dei giri")
                aggiungi_energia(path, data, scadenza)
                return path, data
        log(f"{year} {rnd:02d} {name}: già scaricato")
        return path, None
    vecchi = energia_vecchia(path) if energia and os.path.exists(path) else {}
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
        if energia and s["session_name"] in PIANO_ENERGIA:
            scarica_energia(rec, scadenza=scadenza, vecchi=vecchi.get(s["session_key"]))
        data["sessioni"].append(rec)
        e = rec.get("energia") or {}
        log(f"  {s['session_name']}: {len(rec.get('laps') or [])} giri, {_count[0] - n0} richieste, "
            f"{time.time() - t0:.0f} s{' — ' + str(len(rec['errori'])) + ' errori' if rec['errori'] else ''}"
            + (f", telemetria di {len(e['blocchi'])} blocchi" if e else "")
            + (f" ({'; '.join(e['errori'][:2])})" if e.get("errori") else ""))
    if not sessions:
        data["errori"].append("nessuna sessione trovata")
    salva(path, data)
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
            "energia": sum(len((s.get("energia") or {}).get("blocchi") or []) for s in data["sessioni"]),
            "energia_errori": [f"{s['session']['session_name']}: {e}" for s in data["sessioni"]
                               for e in (s.get("energia") or {}).get("errori") or []],
            "errori": errs, "scaricato": data.get("scaricato"), "parziale": bool(data.get("parziale")),
            "kb": round(os.path.getsize(path) / 1024)}


_meetings = {}


def meetings_cached(year):
    if year not in _meetings:
        _meetings[year] = meetings_of(year)
    return _meetings[year]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="storico")
    ap.add_argument("--anni", default="2026", help="stagioni complete da scaricare, es. 2026")
    ap.add_argument("--circuito", default="Austin",
                    help="circuiti di cui scaricare le edizioni passate, separati da virgola "
                         "(es. Austin,Mexico,Brazil,Las Vegas,Qatar,Abu Dhabi)")
    ap.add_argument("--senza-in-corso", action="store_true", help="non scaricare il weekend in corso")
    ap.add_argument("--senza-energia", action="store_true", help="non scaricare la telemetria dei giri")
    ap.add_argument("--anni-circuito", default="2023,2024,2025")
    ap.add_argument("--anni-riferimento", default="2024,2025",
                    help="edizioni dei circuiti del calendario di cui scaricare il giro migliore dei primi 6 "
                         "in qualifica (profilo energetico dei circuiti); vuoto = nessuna")
    ap.add_argument("--max-minuti", type=float, default=230, help="si ferma in modo pulito dopo questo tempo")
    ap.add_argument("--now", help=argparse.SUPPRESS)
    a = ap.parse_args()
    now = ts(a.now) if a.now else time.time()
    t_start = time.time()
    scadenza = t_start + a.max_minuti * 60
    idx_path = os.path.join(a.out_dir, "indice.json")
    index, index_c = {}, {}
    if os.path.exists(idx_path):
        try:
            old = json.load(open(idx_path, encoding="utf-8"))
            index = {r["file"]: r for r in old["file"]}
            index_c = {r["file"]: r for r in old.get("circuiti") or []}
        except Exception:  # noqa: BLE001
            index, index_c = {}, {}

    jobs = []  # (meeting, round, anno, telemetria partenze, weekend in corso)
    rif = []   # (meeting, anno): circuiti del calendario, edizioni di riferimento
    try:
        anni = [int(x) for x in a.anni.split(",") if x.strip()]
        for y in anni:
            ms = meetings_cached(y)
            for i, m in enumerate(ms, 1):
                if ts(m["date_end"]) + 10 * 3600 < now:
                    jobs.append((m, i, y, True, False))
                elif not a.senza_in_corso and ts(m["date_start"]) < now:
                    jobs.append((m, i, y, True, True))
        circs = [c.strip().lower() for c in (a.circuito or "").split(",") if c.strip()]
        if circs:
            for y in [int(x) for x in a.anni_circuito.split(",") if x.strip()]:
                ms = meetings_cached(y)
                for i, m in enumerate(ms, 1):
                    names = " ".join(str(m.get(k) or "") for k in ("location", "circuit_short_name",
                                                                   "meeting_name", "country_name")).lower()
                    if any(c in names for c in circs) and ts(m["date_end"]) + 10 * 3600 < now:
                        jobs.append((m, i, y, False, False))
        if not a.senza_energia:
            chiavi = {m.get("circuit_key") for y in anni for m in meetings_cached(y)} - {None}
            for y in [int(x) for x in (a.anni_riferimento or "").split(",") if x.strip()]:
                for m in meetings_cached(y):
                    if m.get("circuit_key") in chiavi and ts(m["date_end"]) + 10 * 3600 < now:
                        rif.append((m, y))
    except Denied as e:
        warn(f"{e}. Probabilmente c'è una sessione in diretta: rilancia più tardi.")
        return 0
    log(f"Weekend da controllare: {len(jobs)}; circuiti di riferimento: {len(rif)}")

    stopped = None
    if rif:
        log("Circuiti di riferimento (giro migliore dei primi 6 in qualifica):")
    for m, y in rif:
        if time.time() > scadenza:
            stopped = "tempo massimo raggiunto: rilancia il workflow per finire"
            break
        try:
            path, data = scarica_circuito(m, y, a.out_dir, scadenza)
        except Denied as e:
            stopped = f"{e}. Probabilmente c'è una sessione in diretta: rilancia più tardi."
            break
        except Exception as e:  # noqa: BLE001
            warn(f"{y} {m.get('circuit_short_name')}: errore {type(e).__name__}: {e}")
            continue
        if path is None:
            continue
        if data is None:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        f_ = path.replace(os.sep, "/")
        index_c[f_] = {"file": f_, "anno": y, "circuito": data["meeting"].get("circuit_short_name"),
                       "circuit_key": data["meeting"].get("circuit_key"),
                       "giri": len(data["energia"]["blocchi"]), "errori": data["errori"] + data["energia"]["errori"]}

    for m, rnd, y, with_car, parziale in ([] if stopped else jobs):
        if time.time() > scadenza:
            stopped = "tempo massimo raggiunto: rilancia il workflow per finire"
            break
        try:
            path, data = fetch_meeting(m, rnd, y, a.out_dir, with_car, parziale, now,
                                       energia=not a.senza_energia, scadenza=scadenza)
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
    if not stopped and any((index.get(k) or {}).get("energia_errori") for k in index):
        stopped = "telemetria dei giri incompleta: rilancia il workflow per finire"

    rows = sorted(index.values(), key=lambda r: (r["anno"], r["round"]))
    rows_c = sorted(index_c.values(), key=lambda r: (r["anno"], r["circuito"] or ""))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({"aggiornato": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "fermato": stopped, "richieste": _count[0], "file": rows, "circuiti": rows_c},
                  f, ensure_ascii=False, indent=1)
    bad = [r for r in rows if r["errori"]]
    log(f"Fatto: {len(rows)} weekend in archivio, {len(bad)} con errori, {len(rows_c)} circuiti di riferimento, "
        f"{_count[0]} richieste, {(time.time() - t_start) / 60:.0f} minuti.")
    if stopped:
        warn(stopped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
