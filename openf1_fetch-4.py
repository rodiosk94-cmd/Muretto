#!/usr/bin/env python3
"""Muretto: scarica da OpenF1 la telemetria del giro migliore dei quattro top team
(Mercedes, McLaren, Red Bull, Ferrari) per ogni sessione finita e la scrive in telemetry.json.

Uso:
    python openf1_fetch.py                   # elabora le sessioni finite non ancora presenti
    python openf1_fetch.py --session 9999    # forza una sessione (session_key OpenF1)

Solo libreria standard + numpy. I dati storici di OpenF1 sono gratuiti e senza chiave.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

BASE = os.environ.get("OPENF1_BASE", "https://api.openf1.org/v1")
TOP_TEAMS = ("mercedes", "mclaren", "red bull", "ferrari")
LABELS = {
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
    "Sprint Qualifying": "Sprint Q", "Sprint Shootout": "Sprint Q",
    "Sprint": "Sprint", "Qualifying": "Qualifica", "Race": "Gara",
}
# Sessioni in cui il tempo ufficiale migliore (session_result) identifica il giro da usare
BEST_BY_RESULT = ("Practice 1", "Practice 2", "Practice 3", "Qualifying", "Sprint Qualifying", "Sprint Shootout")
# Lunghezza ufficiale (m). Se il circuito manca, la lunghezza si stima dalla telemetria.
KNOWN_LENGTHS = {  # metri, chiave = circuit_short_name o location di OpenF1 in minuscolo
    "melbourne": 5278, "shanghai": 5451, "suzuka": 5807, "sakhir": 5412, "jeddah": 6174, "miami": 5412,
    "miami gardens": 5412, "montreal": 4361, "montréal": 4361, "monte carlo": 3337, "monaco": 3337,
    "catalunya": 4657, "barcelona": 4657, "spielberg": 4318, "silverstone": 5891, "spa-francorchamps": 7004,
    "hungaroring": 4381, "budapest": 4381, "zandvoort": 4259, "monza": 5793, "baku": 6003, "singapore": 4927,
    "marina bay": 4927, "austin": 5513, "mexico city": 4304, "interlagos": 4309, "são paulo": 4309,
    "sao paulo": 4309, "las vegas": 6201, "lusail": 5419, "yas marina": 5281, "yas marina circuit": 5281,
    "yas island": 5281, "kuala lumpur": 5543, "sepang": 5543,
}
STEP_M = 10.0          # passo della griglia in metri
USE_GPS = False        # distanza da coordinate GPS: provata su dati di prova, senza vantaggio netto, quindi spenta
MARGIN_MIN = 20        # attesa dopo la fine ufficiale della sessione
MIN_REQ_GAP = 0.5      # secondi tra due richieste
UTC = timezone.utc


class Pending(Exception):
    """Dati non ancora disponibili (accesso negato o non pubblicati): si riprova al prossimo giro."""


_last_req = [0.0]


def get(endpoint, *conds):
    url = BASE + "/" + endpoint
    if conds:
        url += "?" + "&".join(c.replace(" ", "%20") for c in conds)
    for attempt in range(5):
        wait = MIN_REQ_GAP - (time.time() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "muretto/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            if e.code in (401, 403):
                raise Pending(f"{endpoint}: accesso negato ({e.code})")
            if e.code in (429, 500, 502, 503, 504):
                time.sleep((2, 5, 10, 20, 30)[attempt])
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep((2, 5, 10, 20, 30)[attempt])
    raise RuntimeError(f"{endpoint}: troppi tentativi falliti")


def ts(s):
    d = datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    return d.timestamp()


def iso(epoch):
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def log(msg):
    print(msg, flush=True)


def warn(msg):
    print(f"::warning::{msg}" if os.environ.get("GITHUB_ACTIONS") else f"ATTENZIONE: {msg}", flush=True)


# ---------- elaborazione di un giro ----------

def nearest(tsrc, vsrc, tq):
    i = np.clip(np.searchsorted(tsrc, tq), 1, len(tsrc) - 1)
    pick = np.where(np.abs(tq - tsrc[i - 1]) <= np.abs(tsrc[i] - tq), i - 1, i)
    return vsrc[pick]


def load_lap(key, num, lap):
    """Campioni del giro su una griglia temporale fine, con distanza percorsa integrata dalla velocità."""
    t0 = ts(lap["date_start"])
    T = float(lap["lap_duration"])
    rows = get("car_data", f"session_key={key}", f"driver_number={num}",
               f"date>={iso(t0 - 1.5)}", f"date<={iso(t0 + T + 1.5)}")
    if len(rows) < 0.6 * T * 3.7:
        raise ValueError(f"campioni insufficienti ({len(rows)})")
    rows.sort(key=lambda r: r["date"])
    t = np.array([ts(r["date"]) - t0 for r in rows])
    keep = np.concatenate(([True], np.diff(t) > 1e-6))
    t = t[keep]
    arr = lambda k: np.array([r[k] for r in rows], float)[keep]
    spd, thr, brk, gear = arr("speed"), arr("throttle"), arr("brake"), arr("n_gear")
    tg = np.arange(0.0, T, 0.05)
    v = np.interp(tg, t, spd)
    th = np.interp(tg, t, thr)
    br = (nearest(t, brk, tg) > 50).astype(int)
    g = nearest(t, gear, tg).astype(int)
    d = np.concatenate(([0.0], np.cumsum((v[1:] + v[:-1]) / 2 / 3.6 * 0.05)))
    return {"t0": t0, "T": T, "tg": tg, "v": v, "th": th, "br": br, "g": g, "d": d, "D": float(d[-1]),
            "st": t, "sv": spd, "sth": thr, "sbr": brk, "sg": gear}


def resample(raw, L, n, shift_m=0.0):
    """Porta il giro su n punti equidistanti lungo i metri della pista (con eventuale spostamento in metri)."""
    d = raw["d"] * (L / raw["D"])
    d = d + np.arange(len(d)) * 1e-9
    grid = np.clip(np.linspace(0.0, L, n, endpoint=False) + shift_m, 0.0, L)
    tq = np.interp(grid, d, raw["tg"])
    return {
        "v": np.interp(tq, raw["tg"], raw["v"]),
        "th": np.interp(tq, raw["tg"], raw["th"]),
        "br": nearest(raw["tg"], raw["br"], tq),
        "g": nearest(raw["tg"], raw["g"], tq),
        "scaled_d": d,
    }


def align_to(ref_v, other_v, max_shift=10):
    """Spostamento (in punti di griglia, frazionario) che allinea `other_v` a `ref_v`.
    date_start è approssimativo e cambia da pilota a pilota: senza questo, il distacco mostra differenze finte."""
    a = ref_v - ref_v.mean()
    b = other_v - other_v.mean()
    cs = [float(np.dot(a, np.roll(b, sh))) for sh in range(-max_shift, max_shift + 1)]
    k = int(np.argmax(cs))
    sh = float(k - max_shift)
    if 0 < k < len(cs) - 1:
        den = cs[k - 1] - 2 * cs[k] + cs[k + 1]
        if den != 0:
            sh += 0.5 * (cs[k - 1] - cs[k + 1]) / den
    return sh


def dist_at(raw, L, t):
    d = raw["d"] * (L / raw["D"])
    return float(np.interp(t, raw["tg"], d))


# ---------- distanza da GPS (location) ----------
# La distanza integrata dalla velocità accumula errori di 10-20 m lungo il giro: nei punti lenti
# (110 km/h) valgono 0,3-0,5 s di distacco finto. Le coordinate GPS danno la posizione vera.

def load_location(key, num, t0, T):
    rows = get("location", f"session_key={key}", f"driver_number={num}",
               f"date>={iso(t0 - 1.5)}", f"date<={iso(t0 + T + 1.5)}")
    if len(rows) < 0.6 * T * 3.7:
        raise ValueError(f"posizioni insufficienti ({len(rows)})")
    rows.sort(key=lambda r: r["date"])
    t = np.array([ts(r["date"]) - t0 for r in rows])
    keep = np.concatenate(([True], np.diff(t) > 1e-6))
    return t[keep], np.array([r["x"] for r in rows], float)[keep], np.array([r["y"] for r in rows], float)[keep]


def build_path(lt, lx, ly, T, L):
    """Percorso di riferimento (un giro del pilota più veloce), ricampionato ogni ~1 m."""
    m = (lt >= 0) & (lt <= T)
    x, y = lx[m], ly[m]
    s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    N = int(round(L))
    ss = np.linspace(0.0, s[-1], N, endpoint=False)
    return {"x": np.interp(ss, s, x), "y": np.interp(ss, s, y), "N": N, "step": L / N, "L": L}


def project_distance(path, lt, lx, ly, prior):
    """Distanza (m) lungo il percorso per ogni campione GPS, vicino al valore atteso `prior` (±200 m)."""
    N, step, px, py = path["N"], path["step"], path["x"], path["y"]
    w = int(200 / step)
    out = np.empty(len(lt))
    for k in range(len(lt)):
        lo = int(round(prior[k] / step)) - w
        idx = np.arange(lo, lo + 2 * w + 1)
        j = idx % N
        j_best = int(np.argmin((px[j] - lx[k]) ** 2 + (py[j] - ly[k]) ** 2))
        out[k] = idx[j_best] * step
    return out


def gps_lap(raw, path, loc, L, n):
    """Tempo trascorso a ogni punto della griglia di distanza, ricavato dalle posizioni GPS."""
    lt, lx, ly = loc
    # distanza attesa dall'integrale della velocità (solo come guida per non confondere tratti vicini)
    sd = np.concatenate(([0.0], np.cumsum((raw["sv"][1:] + raw["sv"][:-1]) / 2 / 3.6 * np.diff(raw["st"]))))
    sd = sd - np.interp(0.0, raw["st"], sd)
    prior = np.interp(lt, raw["st"], sd) * (L / max(raw["D"], 1.0))
    d = project_distance(path, lt, lx, ly, prior)
    d = np.maximum.accumulate(d) + np.arange(len(d)) * 1e-6
    if d[0] > 0.0 or d[-1] < L:
        raise ValueError("il GPS non copre il giro intero")
    grid = np.linspace(0.0, L, n, endpoint=False)
    t_grid = np.interp(np.append(grid, L), d, lt)
    t_start, measured = t_grid[0], t_grid[-1] - t_grid[0]
    if abs(measured - raw["T"]) > 0.35:
        raise ValueError(f"durata da GPS {measured:.2f} s lontana da {raw['T']:.2f} s")
    elapsed = (t_grid - t_start) * (raw["T"] / measured)
    tq = t_grid[:-1]
    return {
        "t": elapsed,
        "v": np.interp(tq, raw["st"], raw["sv"]),
        "th": np.interp(tq, raw["st"], raw["sth"]),
        "br": (nearest(raw["st"], raw["sbr"], tq) > 50).astype(int),
        "g": nearest(raw["st"], raw["sg"], tq).astype(int),
        "measured": measured,
    }


def corners_on_path(meeting, path):
    try:
        info = fetch_circuit_info((meeting or {}).get("circuit_info_url"))
        marks = [c for c in info.get("corners", []) if c.get("trackPosition")]
        if not marks:
            return None
        out, gaps = [], []
        for c in marks:
            cx, cy = c["trackPosition"]["x"], c["trackPosition"]["y"]
            dd = np.hypot(path["x"] - cx, path["y"] - cy)
            j = int(np.argmin(dd))
            gaps.append(float(dd[j]))
            out.append({"n": f"{c.get('number', '')}{c.get('letter', '') or ''}", "d": round(j * path["step"], 0)})
        if float(np.median(gaps)) > 600:
            warn("Curve: coordinate non allineate, uso il rilevamento automatico")
            return None
        out.sort(key=lambda c: c["d"])
        return out
    except Exception as e:  # noqa: BLE001
        warn(f"Curve da circuito non disponibili ({e}); uso il rilevamento automatico")
        return None


# ---------- curve ----------

def fetch_circuit_info(url):
    """Mappa curve di MultiViewer. Se per l'anno corrente non c'è (404), prova gli anni precedenti."""
    import re
    m = re.match(r"^(.*/)(\d{4})$", url or "")
    urls = [url] if not m else [f"{m.group(1)}{y}" for y in range(int(m.group(2)), int(m.group(2)) - 5, -1)]
    last = None
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "muretto/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code != 404:
                raise
    raise last if last else RuntimeError("circuit_info_url mancante")


def corners_from_circuit(meeting, key, ref_num, ref_lap, ref_raw, L):
    url = (meeting or {}).get("circuit_info_url")
    if not url:
        return None
    try:
        info = fetch_circuit_info(url)
        marks = [c for c in info.get("corners", []) if c.get("trackPosition")]
        if not marks:
            return None
        t0, T = ref_raw["t0"], ref_raw["T"]
        loc = get("location", f"session_key={key}", f"driver_number={ref_num}",
                  f"date>={iso(t0)}", f"date<={iso(t0 + T)}")
        if len(loc) < 100:
            return None
        loc.sort(key=lambda r: r["date"])
        lt = np.array([ts(r["date"]) - t0 for r in loc])
        lx = np.array([r["x"] for r in loc], float)
        ly = np.array([r["y"] for r in loc], float)
        ld = np.interp(lt, ref_raw["tg"], ref_raw["d"] * (L / ref_raw["D"]))
        out, gaps = [], []
        for c in marks:
            cx, cy = c["trackPosition"]["x"], c["trackPosition"]["y"]
            dd = np.hypot(lx - cx, ly - cy)
            i = int(np.argmin(dd))
            gaps.append(float(dd[i]))
            out.append({"n": f"{c.get('number', '')}{c.get('letter', '') or ''}", "d": round(float(ld[i]), 0)})
        if float(np.median(gaps)) > 600:  # 60 m: sistemi di coordinate non allineati
            warn("Curve: coordinate non allineate, uso il rilevamento automatico")
            return None
        out.sort(key=lambda c: c["d"])
        return out
    except Exception as e:  # noqa: BLE001
        warn(f"Curve da circuito non disponibili ({e}); uso il rilevamento automatico")
        return None


def corners_auto(v_med, L):
    n = len(v_med)
    v = np.convolve(np.r_[v_med[-2:], v_med, v_med[:2]], np.ones(5) / 5, mode="valid")
    found = []
    for i in range(n):
        win = [v[(i + k) % n] for k in range(-12, 13)]
        if v[i] > min(win) + 1e-9:
            continue
        left = max(v[(i - k) % n] for k in range(0, 40))
        right = max(v[(i + k) % n] for k in range(0, 40))
        if min(left, right) - v[i] >= 25 and (not found or i - found[-1] > 12):
            found.append(i)
    step = L / n
    return [{"n": str(k + 1), "d": round(i * step, 0)} for k, i in enumerate(found)]


def summarize_weather(w):
    """Sintesi meteo di sessione. OpenF1 contiene letture a 0 (sensore senza dato): vanno scartate."""
    if not w:
        return None
    good = lambda k: [x[k] for x in w if isinstance(x.get(k), (int, float)) and x[k] > 0]
    tt, at = good("track_temperature"), good("air_temperature")
    if not tt:
        return None
    return {"track": [round(min(tt), 1), round(sum(tt) / len(tt), 1), round(max(tt), 1)],
            "air": round(sum(at) / len(at), 1) if at else None,
            "rain": any((x.get("rainfall") or 0) > 0 for x in w)}


# ---------- una sessione ----------

def process_session(sess, meeting, forced_length=None):
    key, name = sess["session_key"], sess["session_name"]
    log(f"Sessione {key}: {name}")
    drv = {}
    for r in get("drivers", f"session_key={key}"):
        team = (r.get("team_name") or "").lower()
        if any(t in team for t in TOP_TEAMS):
            drv[int(r["driver_number"])] = {"code": r["name_acronym"], "team": r["team_name"]}
    if not drv:
        raise Pending("piloti non ancora disponibili")

    laps_all = get("laps", f"session_key={key}")
    by_driver = {}
    for l in laps_all:
        if int(l["driver_number"]) in drv:
            by_driver.setdefault(int(l["driver_number"]), []).append(l)
    if not by_driver:
        raise Pending("giri non ancora disponibili")

    official = {}
    if name in BEST_BY_RESULT:
        for r in get("session_result", f"session_key={key}"):
            dur = r.get("duration")
            vals = [x for x in (dur if isinstance(dur, list) else [dur]) if isinstance(x, (int, float))]
            if vals:
                official[int(r["driver_number"])] = min(vals)

    stints = {}
    for s in get("stints", f"session_key={key}"):
        stints.setdefault(int(s["driver_number"]), []).append(s)

    def compound(num, lap_no):
        for s in stints.get(num, []):
            if s.get("lap_start") and s.get("lap_end") and s["lap_start"] <= lap_no <= s["lap_end"]:
                return (s.get("compound") or "?")[:1].upper()
        return None

    chosen = {}
    for num, laps in by_driver.items():
        ok = [l for l in laps if l.get("lap_duration") and not l.get("is_pit_out_lap") and l.get("date_start")]
        if not ok:
            continue
        pick = None
        if num in official:
            close = [l for l in ok if abs(l["lap_duration"] - official[num]) <= 0.0015]
            pick = close[0] if close else None
        chosen[num] = pick or min(ok, key=lambda l: l["lap_duration"])

    raws = {}
    for num, lap in chosen.items():
        try:
            raws[num] = load_lap(key, num, lap)
        except Exception as e:  # noqa: BLE001
            warn(f"{drv[num]['code']}: giro saltato ({e})")
    if len(raws) < 2:
        raise Pending("telemetria non ancora disponibile")

    ref = min(raws, key=lambda n: raws[n]["T"])
    meeting_name = (meeting or {}).get("circuit_short_name") or (meeting or {}).get("location") or ""
    known = KNOWN_LENGTHS.get(meeting_name.lower()) or KNOWN_LENGTHS.get(((meeting or {}).get("location") or "").lower())
    est = float(np.median([r["D"] for r in raws.values()]))
    L = float(forced_length or known or round(est))
    if abs(est / L - 1) > 0.04:
        warn(f"Distanza integrata {est:.0f} m lontana dalla lunghezza {L:.0f} m: controlla l'allineamento dei giri")
    n = int(round(L / STEP_M))
    log(f"  lunghezza {L:.0f} m ({'ufficiale' if (forced_length or known) else 'stimata'}), {n} punti, {len(raws)} piloti")

    rs_all = {num: resample(raw, L, n) for num, raw in raws.items()}
    step = L / n
    shifts = {num: (0.0 if num == ref else align_to(rs_all[ref]["v"], rs_all[num]["v"])) for num in raws}
    for num, sh in shifts.items():
        if abs(sh) > 1e-6:
            rs_all[num] = resample(raws[num], L, n, shift_m=-sh * step)
    log("  allineamento (m): " + ", ".join(f"{drv[k]['code']} {v * step:+.0f}" for k, v in shifts.items()))

    # --- distanza e tempo da GPS (preferiti): la velocità integrata accumula errori di 10-20 m ---
    gps, path = {}, None
    try:
        if not USE_GPS:
            raise RuntimeError("GPS disattivato")
        loc_ref = load_location(key, ref, raws[ref]["t0"], raws[ref]["T"])
        path = build_path(*loc_ref, raws[ref]["T"], L)
        for num, raw in raws.items():
            try:
                loc = loc_ref if num == ref else load_location(key, num, raw["t0"], raw["T"])
                gps[num] = gps_lap(raw, path, loc, L, n)
            except Exception as e:  # noqa: BLE001
                warn(f"{drv[num]['code']}: GPS non utilizzabile ({e}); uso la velocità")
    except Exception as e:  # noqa: BLE001
        if USE_GPS:
            warn(f"GPS di riferimento non disponibile ({e}); distanza ricavata dalla velocità")
    grid_ext = np.append(np.linspace(0.0, L, n, endpoint=False), L)
    if USE_GPS:
        log(f"  distanza da GPS: {len(gps)} piloti su {len(raws)}")

    res, sec_d1, sec_d2 = {}, [], []
    for num, raw in raws.items():
        lap = chosen[num]
        rs = gps.get(num) or rs_all[num]
        code = drv[num]["code"]
        d = {
            "lap": round(raw["T"], 3), "lapNo": lap.get("lap_number"),
            "v": [round(float(x), 1) for x in rs["v"]],
            "th": [int(round(float(x))) for x in rs["th"]],
            "br": [int(x) for x in rs["br"]],
            "g": [int(x) for x in rs["g"]],
        }
        if num in gps:
            d["t"] = [round(float(x), 3) for x in gps[num]["t"]]
        s1, s2, s3 = (lap.get("duration_sector_1"), lap.get("duration_sector_2"), lap.get("duration_sector_3"))
        if all(isinstance(x, (int, float)) for x in (s1, s2, s3)):
            d["sec"] = [round(s1, 3), round(s2, 3), round(s3, 3)]
            if num in gps:
                sec_d1.append(float(np.interp(s1, gps[num]["t"], grid_ext)))
                sec_d2.append(float(np.interp(s1 + s2, gps[num]["t"], grid_ext)))
            else:
                sec_d1.append(dist_at(raw, L, s1))
                sec_d2.append(dist_at(raw, L, s1 + s2))
        tyre = compound(num, lap.get("lap_number") or -1)
        if tyre:
            d["tyre"] = tyre
        runs = []
        for l in sorted(by_driver[num], key=lambda x: x.get("lap_number") or 0):
            if l.get("lap_duration"):
                runs.append([l["lap_number"], round(l["lap_duration"], 3), compound(num, l["lap_number"]),
                             1 if l.get("is_pit_out_lap") else 0])
        d["laps"] = runs
        res[code] = d

    sectors = [0.0, round(float(np.median(sec_d1)), 0), round(float(np.median(sec_d2)), 0), L] if sec_d1 else \
        [0.0, round(L / 3), round(2 * L / 3), L]

    # controllo di coerenza: il tempo da GPS ai confini dei settori deve coincidere con i tempi ufficiali
    errs = []
    for num in gps:
        dd = res[drv[num]["code"]]
        if "sec" in dd:
            tt = gps[num]["t"]
            errs.append(abs(float(np.interp(sectors[1], grid_ext, tt)) - dd["sec"][0]))
            errs.append(abs(float(np.interp(sectors[2], grid_ext, tt)) - (dd["sec"][0] + dd["sec"][1])))
    if errs:
        log(f"  coerenza GPS/settori ufficiali: errore medio {np.mean(errs):.3f} s, massimo {np.max(errs):.3f} s")

    corners = corners_on_path(meeting, path) if path else corners_from_circuit(meeting, key, ref, chosen[ref], raws[ref], L)
    auto = False
    if not corners:
        vm = np.median([r["v"] for r in (gps.get(k) or rs_all[k] for k in raws)], axis=0)
        corners, auto = corners_auto(vm, L), True

    weather = summarize_weather(get("weather", f"session_key={key}"))

    gp = ((meeting or {}).get("meeting_name") or "").replace(" Grand Prix", "") or meeting_name
    label = LABELS.get(name, name)
    return {
        "id": f"{sess['meeting_key']}-{key}", "session_key": key, "name": name, "short": label,
        "label": f"{gp}, {label}", "gp": gp, "date": sess["date_start"], "simulated": False,
        "length": L, "n": n, "sectors": sectors, "corners": corners, "cornersAuto": auto,
        "weather": weather, "drivers": res,
    }


# ---------- riepilogo compatto (si legge in testa al file, prima delle tracce) ----------
SUMMARY_V = 1


def clip_zones(v, th, ds):
    """Tratti a gas pieno in cui la velocità non sale più (stima del super clipping): [inizio m, lunghezza m, perdita km/h]."""
    n = len(v)
    raw, start = [], -1
    for i in range(n):
        j = (i + 1) % n
        if th[i] >= 98 and th[j] >= 98 and v[i] >= 230 and (v[j] - v[i]) <= 0.35:
            if start < 0:
                start = i
        elif start >= 0:
            raw.append([start, i])
            start = -1
    if start >= 0:
        raw.append([start, n - 1])
    merged = []
    for z in raw:
        if merged and z[0] - merged[-1][1] <= 3:
            merged[-1][1] = z[1]
        else:
            merged.append(list(z))
    out = []
    for a, b in merged:
        loss = max(v[a:b + 1]) - v[b]
        length = (b - a) * ds
        k, run = a, 0.0
        while run < 400 and th[(k - 1) % n] >= 98:
            k -= 1
            run += ds
        if length >= 40 and loss >= 1.5 and run >= 150:
            out.append([int(round(a * ds)), int(round(length)), round(loss, 1)])
    return out


def build_summary(sessions):
    out = []
    for S in sessions:
        ds = S["length"] / S["n"]
        drivers = {}
        for code, d in sorted(S["drivers"].items(), key=lambda kv: kv[1]["lap"]):
            v, th, br = d["v"], d["th"], d["br"]
            clips = clip_zones(v, th, ds)
            best = d["lap"]
            laps = [l for l in d.get("laps", []) if l[1] <= min(2.5 * best, 400)]
            drivers[code] = {
                "lap": d["lap"], "lapNo": d.get("lapNo"), "sec": d.get("sec"), "tyre": d.get("tyre"),
                "vmax": round(max(v)), "vmin": round(min(v)),
                "gasPieno%": round(100 * sum(1 for x in th if x >= 98) / len(th)),
                "freno%": round(100 * sum(br) / len(br)),
                "clip": clips, "clipTot_m": sum(c[1] for c in clips), "laps": laps,
            }
        secs = [d["sec"] for d in S["drivers"].values() if d.get("sec")]
        ideal = round(sum(min(x[k] for x in secs) for k in range(3)), 3) if secs else None
        out.append({"id": S["id"], "short": S.get("short"), "label": S["label"], "date": S["date"],
                    "weather": S.get("weather"), "length": S["length"], "sectors": S["sectors"],
                    "cornersAuto": S.get("cornersAuto"), "idealLap": ideal, "drivers": drivers})
    return out


def ordered(data):
    """Chiavi in ordine: prima ciò che serve a chi legge, per ultime le tracce pesanti."""
    head = ["updated", "meeting_key", "gp", "info", "summaryV", "summary", "sessions"]
    out = {k: data[k] for k in head if k in data}
    out.update({k: v for k, v in data.items() if k not in out})
    return out


# ---------- main ----------

# ---------- informazioni del weekend: calendario, classifiche, ultimo GP, circuito ----------
IT_MONTHS = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre",
             "ottobre", "novembre", "dicembre"]
SESSION_META = {  # nome OpenF1 -> (id, etichetta breve, nome, mercato)
    "Practice 1": ("FP1", "FP1", "Prove libere 1", "Miglior tempo FP1"),
    "Practice 2": ("FP2", "FP2", "Prove libere 2", "Miglior tempo FP2"),
    "Practice 3": ("FP3", "FP3", "Prove libere 3", "Miglior tempo FP3"),
    "Sprint Qualifying": ("SQ", "Sprint Q", "Sprint Qualifying", "Pole sprint"),
    "Sprint Shootout": ("SQ", "Sprint Q", "Sprint Qualifying", "Pole sprint"),
    "Sprint": ("SPR", "Sprint", "Sprint", "Vincitore sprint"),
    "Qualifying": ("Q", "Qualifica", "Qualifica", "Pole"),
    "Race": ("R", "Gara", "Gara", "Vincitore gara"),
}
CIRCUIT_TYPES = {"Permanent": "permanente", "Temporary - Street": "cittadino", "Temporary - Road": "stradale temporaneo"}
HISTORY_YEARS = 3


def slug(text):
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def parse_offset(s):
    s = s or "00:00:00"
    sign = -1 if s.startswith("-") else 1
    parts = s.lstrip("+-").split(":")
    return sign * timedelta(hours=int(parts[0]), minutes=int(parts[1]) if len(parts) > 1 else 0)


def local_dt(iso_s, off):
    return datetime.fromtimestamp(ts(iso_s), UTC) + off


def it_num(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def fmt_lap(sec):
    m = int(sec // 60)
    return f"{m}:{sec - m * 60:06.3f}"


def nice_name(s):
    return " ".join(w.capitalize() if w.isupper() and len(w) > 1 else w for w in (s or "").split())


def driver_map(session_key):
    out = {}
    for r in get("drivers", f"session_key={session_key}"):
        last = r.get("last_name") or r.get("name_acronym")
        out[int(r["driver_number"])] = {
            "code": r["name_acronym"], "name": nice_name(r.get("full_name") or last),
            "short": nice_name(last), "team": r.get("team_name") or "", "colour": r.get("team_colour"),
        }
    return out


def season_meetings(year):
    ms = [m for m in get("meetings", f"year={year}") if not m.get("is_cancelled")]
    ms = [m for m in ms if "testing" not in ((m.get("meeting_name") or "") + (m.get("meeting_official_name") or "")).lower()]
    return sorted(ms, key=lambda m: m["date_start"])


def pick_current(ms, now):
    """Il weekend 'corrente' è il primo non ancora concluso (10 ore di margine dopo la fine)."""
    for i, m in enumerate(ms):
        if ts(m["date_end"]) + 10 * 3600 > now:
            return i, m
    return len(ms) - 1, ms[-1]


def last_finished_race(year, now):
    races = [s for s in get("sessions", f"year={year}", "session_name=Race")
             if not s.get("is_cancelled") and ts(s["date_end"]) + MARGIN_MIN * 60 < now]
    return max(races, key=lambda s: s["date_start"]) if races else None


def count_safety_cars(session_key):
    rows = get("race_control", f"session_key={session_key}", "category=SafetyCar")
    sc = vsc = 0
    for r in rows:
        m = (r.get("message") or "").upper()
        if "DEPLOYED" in m:
            if "VIRTUAL" in m:
                vsc += 1
            else:
                sc += 1
    return sc, vsc


def lap_metrics(raw):
    th, br, v = raw["th"], raw["br"], raw["v"]
    zones, run = 0, 0
    for x in br:
        if x:
            run += 1
        else:
            zones += 1 if run >= 6 else 0
            run = 0
    zones += 1 if run >= 6 else 0
    return [["Gas a fondo", f"{float(np.mean(th >= 98)) * 100:.0f}% del giro"],
            ["Frenata", f"{float(np.mean(br)) * 100:.0f}% del giro, {zones} zone"],
            ["Velocità massima", f"{float(v.max()):.0f} km/h"],
            ["Velocità minima", f"{float(v.min()):.0f} km/h"]]


def history_year(ckey, y):
    """Vincitore, pole, giri, Safety Car, rimonta e meteo di un'edizione passata."""
    ss = [s for s in get("sessions", f"circuit_key={ckey}", f"year={y}") if not s.get("is_cancelled")]
    race = next((s for s in ss if s["session_name"] == "Race"), None)
    if not race:
        return None
    quali = next((s for s in ss if s["session_name"] == "Qualifying"), None)
    sprint = next((s for s in ss if s["session_name"] == "Sprint"), None)
    dm = driver_map(race["session_key"])
    name_of = lambda n: (dm.get(int(n)) or {}).get("name") or f"#{n}"
    rr = get("session_result", f"session_key={race['session_key']}")
    win = next((r for r in rr if r.get("position") == 1), None)
    out = {"year": y, "winner": name_of(win["driver_number"]) if win else None, "pole": None,
           "laps": (win or {}).get("number_of_laps"), "mobility": None, "sprint_laps": None,
           "pole_lap": None}
    qpos = {}
    if quali:
        qr = get("session_result", f"session_key={quali['session_key']}")
        qpos = {int(r["driver_number"]): r.get("position") for r in qr}
        p1 = next((r for r in qr if r.get("position") == 1), None)
        if p1:
            out["pole"] = name_of(p1["driver_number"])
            dur = p1.get("duration")
            vals = [x for x in (dur if isinstance(dur, list) else [dur]) if isinstance(x, (int, float))]
            out["pole_lap"] = {"session_key": quali["session_key"], "driver": int(p1["driver_number"]),
                               "time": min(vals) if vals else None}
    moves = [abs(r["position"] - qpos[int(r["driver_number"])]) for r in rr
             if r.get("position") and not (r.get("dnf") or r.get("dns") or r.get("dsq"))
             and qpos.get(int(r["driver_number"]))]
    if moves:
        out["mobility"] = sum(moves) / len(moves)
    if sprint:
        sr = get("session_result", f"session_key={sprint['session_key']}", "position=1")
        out["sprint_laps"] = (sr[0].get("number_of_laps") if sr else None)
    out["sc"], out["vsc"] = count_safety_cars(race["session_key"])
    wx = []
    for s in ss:
        meta = SESSION_META.get(s["session_name"])
        if not meta:
            continue
        w = summarize_weather(get("weather", f"session_key={s['session_key']}"))
        if w:
            wx.append([meta[1], w["track"][1], w["air"], bool(w["rain"])])
    out["weather"] = wx
    return out


def build_circuit_base(meeting, now, length_hint):
    """Parte del circuito che cambia solo da un anno all'altro: si calcola una volta e si conserva."""
    ckey = meeting["circuit_key"]
    year = meeting["year"]
    hist = []
    for y in range(year - 1, max(2022, year - 1 - HISTORY_YEARS), -1):
        try:
            h = history_year(ckey, y)
        except Pending:
            raise
        except Exception as e:  # noqa: BLE001
            warn(f"Storico {y}: {type(e).__name__}: {e}")
            h = None
        if h:
            hist.append(h)
    base = {"key": ckey, "builtYear": year, "name": meeting.get("circuit_short_name") or meeting.get("location"),
            "location": meeting.get("location"), "country": meeting.get("country_name"),
            "type": CIRCUIT_TYPES.get(meeting.get("circuit_type") or "", None)}
    name_keys = [(meeting.get("circuit_short_name") or "").lower(), (meeting.get("location") or "").lower()]
    length = next((KNOWN_LENGTHS[k] for k in name_keys if k in KNOWN_LENGTHS), None) or length_hint
    base["length"] = length
    base["lengthKnown"] = bool(next((k for k in name_keys if k in KNOWN_LENGTHS), None))
    laps = next((h["laps"] for h in hist if h.get("laps")), None)
    sprint_laps = next((h["sprint_laps"] for h in hist if h.get("sprint_laps")), None)
    base["laps"], base["sprint_laps"] = laps, sprint_laps
    try:
        info = fetch_circuit_info(meeting.get("circuit_info_url"))
        base["corners"] = len(info.get("corners", [])) or None
    except Exception:  # noqa: BLE001
        base["corners"] = None
    base["history"] = [[h["year"], h["winner"], h["pole"]] for h in hist]
    base["safetycars"] = [[h["year"], h["sc"], h["vsc"]] for h in hist]
    mob = [h["mobility"] for h in hist if h.get("mobility") is not None]
    base["mobility"] = round(sum(mob) / len(mob), 1) if mob else None
    base["weatherHistory"] = {"year": hist[0]["year"], "rows": hist[0]["weather"]} if hist and hist[0]["weather"] else None
    base["metrics"] = None
    for h in hist:  # misure dal giro della pole dell'edizione più recente che ha i dati
        pl = h.get("pole_lap")
        if not pl or not pl.get("time"):
            continue
        try:
            laps_q = [l for l in get("laps", f"session_key={pl['session_key']}", f"driver_number={pl['driver']}")
                      if l.get("lap_duration") and abs(l["lap_duration"] - pl["time"]) <= 0.0015 and l.get("date_start")]
            if not laps_q:
                continue
            raw = load_lap(pl["session_key"], pl["driver"], laps_q[0])
            base["metrics"] = {"year": h["year"], "rows": lap_metrics(raw)}
            break
        except Pending:
            raise
        except Exception as e:  # noqa: BLE001
            warn(f"Metriche circuito {h['year']}: {type(e).__name__}: {e}")
    return base


def circuit_notes(base, weekend_sessions, off):
    notes = []
    practice = [s for s in weekend_sessions if s["id"].startswith("FP")]
    if any(s["id"] == "SPR" for s in weekend_sessions):
        notes.append(f"Weekend sprint: {'una sola sessione' if len(practice) == 1 else str(len(practice)) + ' sessioni'} "
                     f"di libere prima della Sprint Qualifying.")
    hours = [(s["short"], local_dt(s["start"], off).hour + local_dt(s["start"], off).minute / 60) for s in weekend_sessions]
    day = [n for n, h in hours if h < 17.5]
    eve = [n for n, h in hours if h >= 18.5]
    if day and eve:
        notes.append(f"Luce diversa tra le sessioni: di giorno {', '.join(day)}; di sera {', '.join(eve)}. "
                     f"Confronta le temperature dell'asfalto prima di usare i long run sulla gara.")
    sc = base.get("safetycars") or []
    if sc:
        notes.append("Safety Car nelle ultime edizioni: " + ", ".join(
            f"{y} {a}" + (f" (+{b} VSC)" if b else "") for y, a, b in sc) + ".")
    if base.get("mobility") is not None:
        notes.append(f"Rimonta media in gara dalla griglia: {it_num(base['mobility'])} posizioni a pilota "
                     f"(ultime {len(base['history'])} edizioni).")
    return notes


def build_circuit(base, weekend_sessions, off):
    stats = []
    L, laps = base.get("length"), base.get("laps")
    if L:
        stats.append([it_num(L / 1000, 3), "km", "lunghezza" + ("" if base.get("lengthKnown") else " (stimata)")])
    if laps:
        stats.append([str(laps), "", "giri"])
        if L:
            stats.append([it_num(laps * L / 1000, 1), "km", "distanza gara"])
    if base.get("corners"):
        stats.append([str(base["corners"]), "", "curve"])
    if base.get("sprint_laps") and any(s["id"] == "SPR" for s in weekend_sessions):
        stats.append([str(base["sprint_laps"]), "", "giri sprint"])
    race = next((s for s in weekend_sessions if s["id"] == "R"), None)
    night = race and local_dt(race["start"], off).hour >= 18
    place = ", ".join(x for x in [base.get("location"), base.get("country")] if x)
    if base.get("type"):
        place += f", circuito {base['type']}"
    if night:
        place += ", gara in notturna"
    return {
        "key": base["key"], "builtYear": base["builtYear"], "name": base["name"], "place": place, "stats": stats,
        "metrics": base.get("metrics"), "weatherHistory": base.get("weatherHistory"),
        "notes": circuit_notes(base, weekend_sessions, off), "history": base.get("history") or [],
        "base": base,
    }


def build_weekend(meeting, rnd, rounds, sessions, race_laps, sprint_laps):
    off = parse_offset(meeting.get("gmt_offset"))
    ss = [s for s in sorted(sessions, key=lambda x: x["date_start"])
          if not s.get("is_cancelled") and s.get("session_name") in SESSION_META]
    out = []
    for s in ss:
        sid, short, nm, market = SESSION_META[s["session_name"]]
        if sid == "R" and race_laps:
            nm = f"Gara, {race_laps} giri"
        if sid == "SPR" and sprint_laps:
            nm = f"Sprint, {sprint_laps} giri"
        out.append({"id": sid, "short": short, "name": nm, "market": market, "start": s["date_start"], "end": s["date_end"]})
    first, last = local_dt(ss[0]["date_start"], off), local_dt(ss[-1]["date_start"], off)
    if first.month == last.month:
        dates = f"{first.day}–{last.day} {IT_MONTHS[first.month - 1]}"
    else:
        dates = f"{first.day} {IT_MONTHS[first.month - 1][:3]} – {last.day} {IT_MONTHS[last.month - 1][:3]}"
    short = meeting.get("circuit_short_name") or meeting.get("location") or meeting.get("country_name")
    return {"id": f"{meeting['year']}-{rnd:02d}-{slug(short)}", "meeting_key": meeting["meeting_key"], "round": rnd, "rounds": rounds,
            "name": meeting.get("meeting_name") or f"GP {short}", "short": short,
            "circuit": meeting.get("location") or short, "dates": dates,
            "format": "sprint" if any(x["id"] in ("SPR", "SQ") for x in out) else "standard", "sessions": out}, off


def build_standings(race, rnd_done, dm):
    k = race["session_key"]
    drv = sorted(get("championship_drivers", f"session_key={k}"), key=lambda r: r.get("position_current") or 999)
    tms = sorted(get("championship_teams", f"session_key={k}"), key=lambda r: r.get("position_current") or 999)
    if not drv:
        raise Pending("classifica piloti non ancora disponibile")
    return {"after": f"Dopo il round {rnd_done}.",
            "drivers": [[dm[int(r["driver_number"])]["code"], r["points_current"]] for r in drv if int(r["driver_number"]) in dm],
            "teams": [[slug(r["team_name"]), r["points_current"]] for r in tms]}


def build_last_race(race, meeting_sessions, rnd, rounds, short, meeting_name, dm, standings):
    k = race["session_key"]
    res = sorted([r for r in get("session_result", f"session_key={k}") if r.get("position")], key=lambda r: r["position"])
    code = lambda n: (dm.get(int(n)) or {}).get("code") or f"#{n}"
    rows = []
    for r in res[:10]:
        gap = r.get("gap_to_leader")
        rows.append([code(r["driver_number"]), None if r["position"] == 1 else (gap if isinstance(gap, (int, float, str)) else None)])
    pole = None
    q = next((s for s in meeting_sessions if s["session_name"] == "Qualifying"), None)
    if q:
        qr = get("session_result", f"session_key={q['session_key']}", "position=1")
        if qr:
            dur = qr[0].get("duration")
            vals = [x for x in (dur if isinstance(dur, list) else [dur]) if isinstance(x, (int, float))]
            if vals:
                pole = [code(qr[0]["driver_number"]), fmt_lap(vals[-1])]
    fastest = None
    valid = [l for l in get("laps", f"session_key={k}") if l.get("lap_duration") and not l.get("is_pit_out_lap")]
    if valid:
        f = min(valid, key=lambda l: l["lap_duration"])
        fastest = [code(f["driver_number"]), fmt_lap(f["lap_duration"])]
    notes = []
    laps = max((r.get("number_of_laps") or 0 for r in res), default=0)
    if laps:
        notes.append(f"Gara su {laps} giri.")
    dnfs = [f"{code(r['driver_number'])} (giro {(r.get('number_of_laps') or 0) + 1})" for r in
            get("session_result", f"session_key={k}") if r.get("dnf")]
    if dnfs:
        notes.append("Ritirati: " + ", ".join(dnfs[:5]) + ".")
    try:
        sc, vsc = count_safety_cars(k)
        reds = [r for r in get("race_control", f"session_key={k}", "flag=RED") if (r.get("scope") or "Track") == "Track"]
        bits = ([f"Safety Car {sc}"] if sc else []) + ([f"VSC {vsc}"] if vsc else []) + ([f"bandiera rossa {len(reds)}"] if reds else [])
        notes.append((", ".join(bits) + ".") if bits else "Nessuna neutralizzazione.")
    except Pending:
        pass
    w = get("weather", f"session_key={k}")
    if any((x.get("rainfall") or 0) > 0 for x in w):
        notes.append("Pioggia durante la gara.")
    sd = standings.get("drivers") or []
    if len(sd) >= 2 and rounds:
        name = lambda c: next((v["short"] for v in dm.values() if v["code"] == c), c)
        notes.append(f"{name(sd[0][0])} guida con {sd[0][1]} punti, +{sd[0][1] - sd[1][1]} su {name(sd[1][0])}. "
                     f"Restano {max(rounds - rnd, 0)} GP.")
    off = parse_offset(race.get("gmt_offset"))
    d = local_dt(race["date_start"], off)
    return {"name": short, "gp": meeting_name, "round": rnd, "date": f"{d.day} {IT_MONTHS[d.month - 1]}", "results": rows,
            "pole": pole, "fastest": fastest, "notes": " ".join(notes)}


def update_info(data, now, tele_meeting=None, tele_length=None):
    """Aggiorna data['info'] (weekend, classifiche, ultimo GP, circuito). Restituisce True se è cambiato."""
    prev = data.get("info") or {}
    year = datetime.fromtimestamp(now, UTC).year
    ms = season_meetings(year)
    if not ms:
        raise Pending("calendario non disponibile")
    idx, cur = pick_current(ms, now)
    rnd, rounds = idx + 1, len(ms)
    length_hint = tele_length if tele_meeting == cur["meeting_key"] else None
    race = last_finished_race(year, now)
    rk = race["session_key"] if race else None
    age = now - (prev.get("updatedTs") or 0)
    if prev and prev.get("meeting_key") == cur["meeting_key"] and prev.get("lastRaceKey") == rk and age < 86400:
        log("Informazioni del weekend già aggiornate.")
        return False
    log(f"Aggiorno le informazioni del weekend: {cur.get('meeting_name')} (round {rnd} di {rounds})")
    problems = []

    def bad(msg):
        problems.append(msg)
        warn(msg)

    info = dict(prev)
    info.update({"meeting_key": cur["meeting_key"], "updatedTs": int(now),
                 "updated": datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")})

    # piloti e squadre dall'ultima gara conclusa (o dalla sessione più recente)
    dm = {}
    try:
        ref_key = rk
        if not ref_key:
            ref_key = (get("sessions", "session_key=latest") or [{}])[0].get("session_key")
        dm = driver_map(ref_key) if ref_key else {}
        if dm:
            info["teams"] = {slug(v["team"]): {"name": v["team"], "color": "#" + (v["colour"] or "888888")} for v in dm.values()}
            info["drivers"] = {v["code"]: {"name": v["name"], "short": v["short"], "team": slug(v["team"])} for v in dm.values()}
    except Pending as e:
        bad(f"Piloti: {e}")
    except Exception as e:  # noqa: BLE001
        bad(f"Piloti: {type(e).__name__}: {e}")

    sessions = get("sessions", f"meeting_key={cur['meeting_key']}")

    # circuito (parte annuale in cache)
    base = (prev.get("circuit") or {}).get("base")
    try:
        if not base or base.get("key") != cur.get("circuit_key") or base.get("builtYear") != cur["year"] or not base.get("history"):
            base = build_circuit_base(cur, now, length_hint)
        elif not base.get("lengthKnown") and length_hint:
            base["length"] = length_hint
    except Pending as e:
        bad(f"Circuito: {e}")
    except Exception as e:  # noqa: BLE001
        bad(f"Circuito: {type(e).__name__}: {e}")
        base = base or None

    try:
        weekend, off = build_weekend(cur, rnd, rounds, sessions, (base or {}).get("laps"), (base or {}).get("sprint_laps"))
        info["weekend"] = weekend
        if base:
            info["circuit"] = build_circuit(base, weekend["sessions"], off)
    except Exception as e:  # noqa: BLE001
        bad(f"Weekend: {type(e).__name__}: {e}")

    # classifiche e ultima gara
    if race and dm:
        try:
            done_idx = next((i for i, m in enumerate(ms) if m["meeting_key"] == race["meeting_key"]), None)
            rnd_done = (done_idx + 1) if done_idx is not None else rnd - 1
            st = build_standings(race, rnd_done, dm)
            info["standings"] = st
            m_done = ms[done_idx] if done_idx is not None else cur
            short = m_done.get("circuit_short_name") or m_done.get("location") or ""
            st["after"] = f"Dopo il round {rnd_done} ({short})."
            msess = get("sessions", f"meeting_key={race['meeting_key']}")
            info["lastRace"] = build_last_race(race, msess, rnd_done, rounds, short, m_done.get("meeting_name") or short, dm, st)
            info["lastRaceKey"] = rk
        except Pending as e:
            bad(f"Classifiche: {e}")
        except Exception as e:  # noqa: BLE001
            bad(f"Classifiche/ultimo GP: {type(e).__name__}: {e}")
    if not race:
        info["lastRaceKey"] = None
    if problems:  # riprova alla prossima esecuzione (fra ~30 minuti) invece di aspettare 24 ore
        info["updatedTs"] = int(now) - 86400 + 1500
        log(f"Informazioni incomplete ({len(problems)} problemi): nuovo tentativo alla prossima esecuzione.")
    data["info"] = info
    return True



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="telemetry.json")
    ap.add_argument("--session", help="session_key da (ri)elaborare")
    ap.add_argument("--no-info", action="store_true", help="salta le informazioni del weekend")
    ap.add_argument("--now", help=argparse.SUPPRESS)
    a = ap.parse_args()
    now = ts(a.now) if a.now else time.time()

    data = {"sessions": []}
    if os.path.exists(a.out):
        try:
            data = json.load(open(a.out, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    changed = False

    # --- telemetria delle sessioni concluse ---
    try:
        if a.session:
            if not a.session.isdigit():
                sys.exit("--session deve essere un numero")
            sessions = get("sessions", f"session_key={a.session}")
        else:
            sessions = get("sessions", "meeting_key=latest")
    except Pending as e:
        warn(str(e))
        sessions = []
    if sessions:
        mk = sessions[0]["meeting_key"]
        meeting = (get("meetings", f"meeting_key={mk}") or [None])[0]
        if data.get("meeting_key") != mk:
            data = {"meeting_key": mk, "sessions": [], "info": data.get("info")}
        done = {s["id"] for s in data["sessions"]}
        todo = []
        for s in sorted(sessions, key=lambda x: x["date_start"]):
            if s.get("is_cancelled") or s.get("session_name") not in LABELS:
                continue
            sid = f"{s['meeting_key']}-{s['session_key']}"
            forced = a.session and str(s["session_key"]) == a.session
            if sid in done and not forced:
                continue
            if not forced and ts(s["date_end"]) + MARGIN_MIN * 60 > now:
                continue
            todo.append(s)
        if not todo:
            log("Telemetria: nessuna sessione finita da elaborare.")
        for s in todo:
            try:
                out = process_session(s, meeting)
            except Pending as e:
                warn(f"{s['session_name']}: {e}. Riprovo alla prossima esecuzione.")
                continue
            except Exception as e:  # noqa: BLE001
                warn(f"{s['session_name']}: errore {type(e).__name__}: {e}")
                continue
            data["sessions"] = [x for x in data["sessions"] if x["id"] != out["id"]] + [out]
            changed = True
        if changed:
            data["sessions"].sort(key=lambda x: x["date"])
            data["meeting_key"] = mk
            data["gp"] = data["sessions"][-1]["gp"]
    else:
        log("Nessuna sessione trovata.")

    # --- calendario, classifiche, ultimo GP, circuito ---
    if not a.no_info and not a.session:
        try:
            tele_len = data["sessions"][-1].get("length") if data.get("sessions") else None
            if update_info(data, now, data.get("meeting_key"), tele_len):
                changed = True
        except Pending as e:
            warn(f"Informazioni del weekend: {e}")
        except Exception as e:  # noqa: BLE001
            warn(f"Informazioni del weekend: {type(e).__name__}: {e}")

    if data.get("sessions") and (changed or data.get("summaryV") != SUMMARY_V):
        data["summary"] = build_summary(data["sessions"])
        data["summaryV"] = SUMMARY_V
        changed = True
    if not changed:
        return
    data["updated"] = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ordered(data), f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, a.out)
    log(f"Scritto {a.out}: {len(data.get('sessions', []))} sessioni di telemetria"
        + (", informazioni del weekend aggiornate." if data.get("info") else "."))


if __name__ == "__main__":
    main()
