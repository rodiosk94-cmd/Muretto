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
KNOWN_LENGTHS = {
    "singapore": 4927, "austin": 5513, "mexico city": 4304, "são paulo": 4309, "sao paulo": 4309,
    "interlagos": 4309, "las vegas": 6201, "lusail": 5419, "yas marina": 5281, "yas marina circuit": 5281,
    "sepang": 5543,
}
STEP_M = 10.0          # passo della griglia in metri
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
    return {"t0": t0, "T": T, "tg": tg, "v": v, "th": th, "br": br, "g": g, "d": d, "D": float(d[-1])}


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

    res, sec_d1, sec_d2 = {}, [], []
    for num, raw in raws.items():
        lap = chosen[num]
        rs = rs_all[num]
        code = drv[num]["code"]
        d = {
            "lap": round(raw["T"], 3), "lapNo": lap.get("lap_number"),
            "v": [round(float(x), 1) for x in rs["v"]],
            "th": [int(round(float(x))) for x in rs["th"]],
            "br": [int(x) for x in rs["br"]],
            "g": [int(x) for x in rs["g"]],
        }
        s1, s2, s3 = (lap.get("duration_sector_1"), lap.get("duration_sector_2"), lap.get("duration_sector_3"))
        if all(isinstance(x, (int, float)) for x in (s1, s2, s3)):
            d["sec"] = [round(s1, 3), round(s2, 3), round(s3, 3)]
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

    corners = corners_from_circuit(meeting, key, ref, chosen[ref], raws[ref], L)
    auto = False
    if not corners:
        vm = np.median([r["v"] for r in rs_all.values()], axis=0)
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


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="telemetry.json")
    ap.add_argument("--session", help="session_key da (ri)elaborare")
    ap.add_argument("--now", help=argparse.SUPPRESS)
    a = ap.parse_args()
    now = ts(a.now) if a.now else time.time()

    try:
        if a.session:
            if not a.session.isdigit():
                sys.exit("--session deve essere un numero")
            sessions = get("sessions", f"session_key={a.session}")
        else:
            sessions = get("sessions", "meeting_key=latest")
    except Pending as e:
        warn(str(e))
        return
    if not sessions:
        log("Nessuna sessione trovata.")
        return

    mk = sessions[0]["meeting_key"]
    meeting = (get("meetings", f"meeting_key={mk}") or [None])[0]

    data = {"sessions": []}
    if os.path.exists(a.out):
        try:
            data = json.load(open(a.out, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    if data.get("meeting_key") != mk:
        data = {"meeting_key": mk, "sessions": []}
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
        log("Niente da fare: nessuna sessione finita da elaborare.")
        return

    changed = False
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
    if not changed:
        return
    data["sessions"].sort(key=lambda x: x["date"])
    data["meeting_key"] = mk
    data["gp"] = data["sessions"][-1]["gp"]
    data["updated"] = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, a.out)
    log(f"Scritto {a.out}: {len(data['sessions'])} sessioni.")


if __name__ == "__main__":
    main()
