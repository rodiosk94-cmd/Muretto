#!/usr/bin/env python3
"""
export_fetch.py — Muretto: il pacchetto di analisi di ogni sessione, al posto del notebook Colab
(F1_export_v3). Stessi file e stesse regole del notebook, salvati nel repo invece che in Drive.

Fonte: OpenF1 (dati gratuiti da 30 minuti dopo la fine della sessione).
Piloti con telemetria completa: top team (Mercedes, McLaren, Red Bull, Ferrari; Racing Bulls escluso).

Uscita: export/<anno>-<luogo>/<SESSIONE>/ con
    laps.csv.gz                tutti i giri di TUTTI i piloti (gomma, vita gomma, settori, trappole)
    sectors_summary.csv        top team: migliori S1/S2/S3, giro ideale, velocità alle trappole
    telemetry_top4.csv.gz      top team: telemetria di ogni giro (TelValid = dati affidabili)
    track_fastest_top4.csv.gz  top team: giro più veloce con coordinate X/Y/Z
    clipping_events.csv        top team: tratti a gas pieno con velocità in calo o piatta
    weather.csv, race_control.csv
    info.json                  evento, sessione, piloti, numero di giri, data dello scaricamento
e l'indice export/indice.json.

Uso:
  python export_fetch.py                         automatico: le sessioni del weekend in corso finite
                                                 da almeno 35 minuti e non ancora scaricate
  python export_fetch.py --anno 2026 --gara Austin --sessione FP1    una sessione precisa
  (--forza riscarica anche se c'è già)
Sessioni: FP1 FP2 FP3 SQ S Q R
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

BASE = os.environ.get("OPENF1_BASE", "https://api.openf1.org/v1")
TOP_TEAMS = ["mercedes", "mclaren", "red bull", "ferrari"]  # "Racing Bulls" escluso
SESSION_NAMES = {"FP1": ["practice 1"], "FP2": ["practice 2"], "FP3": ["practice 3"],
                 "Q": ["qualifying"], "R": ["race"], "S": ["sprint"],
                 "SQ": ["sprint qualifying", "sprint shootout"]}
CODICI = {n: c for c, ns in SESSION_NAMES.items() for n in ns}

THR_FULL = 98       # % acceleratore = gas pieno
V_MIN = 200         # km/h minimi per valutare il clipping
ACC_DECEL = -1.0    # km/h/s: sotto = velocità in calo a gas pieno (super clipping)
ACC_PLATEAU = 1.0   # km/h/s: |acc| entro = velocità piatta
PAUSE = float(os.environ.get("OPENF1_PAUSA", "2.2"))  # secondi tra richieste (limite gratuito)
ATTESA_MIN = 35     # minuti dopo la fine della sessione prima di scaricare

_last = [0.0]


class NonPronto(Exception):
    """sessione ancora 'live' per OpenF1 o senza dati: si riprova al giro successivo"""


def get(endpoint, query):
    url = f"{BASE}/{endpoint}?{query}"
    for attempt in range(5):
        wait = PAUSE - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(url, timeout=120)
        except requests.RequestException as e:
            print(f"  rete: {e}, riprovo")
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code in (401, 403):
            raise NonPronto("Accesso negato: la sessione è ancora 'live' (fino a 30 min dopo la fine).")
        if r.status_code == 404:
            return pd.DataFrame()
        if r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
    raise RuntimeError(f"Troppe richieste rifiutate: {url}")


def to_dt(s):
    return pd.to_datetime(s, utc=True, format="ISO8601")


def is_top(team):
    t = str(team).lower()
    return any(k in t for k in TOP_TEAMS)


def slug(s):
    s = re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")
    return s or "gp"


def find_meeting(year, gara):
    meetings = get("meetings", f"year={year}")
    if meetings.empty:
        sys.exit(f"Nessun evento trovato per il {year}.")
    fields = [c for c in ["meeting_name", "meeting_official_name", "country_name",
                          "location", "circuit_short_name"] if c in meetings]
    g = gara.lower()
    mask = meetings[fields].astype(str).apply(
        lambda col: col.str.lower().str.contains(g, regex=False)).any(axis=1)
    m = meetings[mask].copy()
    if m.empty:
        sys.exit(f"Gara '{gara}' non trovata. Eventi {year}: "
                 + ", ".join(meetings["meeting_name"].astype(str)))
    m["date_start"] = to_dt(m["date_start"])
    past = m[m["date_start"] <= pd.Timestamp.now(tz="UTC")]
    return (past if not past.empty else m).sort_values("date_start").iloc[-1]


def current_meeting(now):
    """evento del weekend in corso o appena finito (fino a 2 giorni dopo)"""
    meetings = get("meetings", f"year={now.year}")
    if meetings.empty:
        return None
    meetings = meetings.copy()
    meetings["date_start"] = to_dt(meetings["date_start"])
    meetings["date_end"] = to_dt(meetings["date_end"])
    ts = pd.Timestamp(now)
    hit = meetings[(meetings["date_start"] <= ts) & (meetings["date_end"] + pd.Timedelta(days=2) >= ts)]
    if "meeting_name" in hit:
        hit = hit[~hit["meeting_name"].astype(str).str.lower().str.contains("test")]
    if hit.empty:
        return None
    return hit.sort_values("date_start").iloc[-1]


def cartella(out_dir, meeting, codice):
    luogo = meeting.get("location") or meeting.get("circuit_short_name") or meeting.get("meeting_name")
    anno = pd.Timestamp(meeting["date_start"]).year
    return os.path.join(out_dir, f"{anno}-{slug(luogo)}", codice)


def clipping_events(t, drv, lap_n):
    t = t.sort_values("Time").reset_index(drop=True)
    dt = t["Time"].diff().replace(0, np.nan)
    acc = t["Speed"].diff() / dt
    full = (t["Throttle"] >= THR_FULL) & (~t["Brake"]) & (t["Speed"] >= V_MIN)
    kind = pd.Series("", index=t.index)
    kind[full & (acc < ACC_DECEL)] = "decel"
    kind[full & (acc.abs() <= ACC_PLATEAU)] = "plateau"
    out = []
    for _, g in t.groupby((kind != kind.shift()).cumsum()):
        k = kind[g.index[0]]
        if not k or len(g) < 2:
            continue
        i0 = max(g.index[0] - 1, 0)
        v0 = t.loc[i0, "Speed"]
        out.append({
            "Driver": drv, "LapNumber": lap_n, "Type": k,
            "DistStart_m": round(t.loc[i0, "Distance"], 1),
            "DistEnd_m": round(g["Distance"].iloc[-1], 1),
            "Duration_s": round(g["Time"].iloc[-1] - t.loc[i0, "Time"], 2),
            "SpeedStart": v0, "SpeedEnd": g["Speed"].iloc[-1],
            "SpeedLost": v0 - g["Speed"].iloc[-1],
            "RPM_avg": int(g["RPM"].mean()) if g["RPM"].notna().any() else None,
            "Gear": int(g["nGear"].mode().iloc[0]) if g["nGear"].notna().any() else None,
        })
    return out


def esporta(meeting, sess, out):
    """stessa elaborazione del notebook F1_export_v3; scrive i file nella cartella out"""
    sk = int(sess["session_key"])
    print(f"{meeting['meeting_name']} - {sess['session_name']} (session_key {sk})")
    tmp = out + ".tmp"
    os.makedirs(tmp, exist_ok=True)

    # Piloti
    drivers = get("drivers", f"session_key={sk}")
    if drivers.empty:
        raise NonPronto("Nessun dato piloti: sessione non ancora disponibile.")
    drivers = drivers.drop_duplicates("driver_number")
    acr = dict(zip(drivers["driver_number"], drivers["name_acronym"]))
    team = dict(zip(drivers["driver_number"], drivers["team_name"]))
    top_nums = [n for n in drivers["driver_number"] if is_top(team.get(n))]
    print("Piloti top team:", ", ".join(acr[n] for n in top_nums))

    # Giri + gomme
    laps = get("laps", f"session_key={sk}")
    if laps.empty:
        raise NonPronto("Nessun giro disponibile: riprova più tardi.")
    laps["date_start"] = to_dt(laps["date_start"])
    stints = get("stints", f"session_key={sk}")
    if not stints.empty:
        mm = laps[["driver_number", "lap_number"]].merge(
            stints[["driver_number", "stint_number", "lap_start", "lap_end",
                    "compound", "tyre_age_at_start"]], on="driver_number")
        mm = mm[(mm["lap_number"] >= mm["lap_start"]) & (mm["lap_number"] <= mm["lap_end"])]
        mm["tyre_life"] = mm["tyre_age_at_start"] + mm["lap_number"] - mm["lap_start"]
        mm = mm.drop_duplicates(["driver_number", "lap_number"])
        laps = laps.merge(mm[["driver_number", "lap_number", "stint_number", "compound",
                              "tyre_life"]], on=["driver_number", "lap_number"], how="left")
    for c in ["stint_number", "compound", "tyre_life", "is_pit_out_lap",
              "duration_sector_1", "duration_sector_2", "duration_sector_3",
              "i1_speed", "i2_speed", "st_speed", "lap_duration"]:
        if c not in laps:
            laps[c] = np.nan
    # Settore mancante ricavabile dal tempo giro e dagli altri due
    laps["SectorImputed"] = False
    for a, b, c in [("duration_sector_1", "duration_sector_2", "duration_sector_3"),
                    ("duration_sector_2", "duration_sector_1", "duration_sector_3"),
                    ("duration_sector_3", "duration_sector_1", "duration_sector_2")]:
        miss = laps[a].isna() & laps[b].notna() & laps[c].notna() & laps["lap_duration"].notna()
        laps.loc[miss, a] = (laps["lap_duration"] - laps[b] - laps[c]).round(3)
        laps.loc[miss, "SectorImputed"] = True
    laps["Driver"] = laps["driver_number"].map(acr)
    laps["Team"] = laps["driver_number"].map(team)
    laps["TopTeam"] = laps["Team"].apply(is_top)
    laps.rename(columns={
        "lap_number": "LapNumber", "lap_duration": "LapTime",
        "duration_sector_1": "S1", "duration_sector_2": "S2", "duration_sector_3": "S3",
        "i1_speed": "SpeedI1", "i2_speed": "SpeedI2", "st_speed": "SpeedST",
        "is_pit_out_lap": "PitOutLap", "stint_number": "Stint",
        "compound": "Compound", "tyre_life": "TyreLife"}, inplace=True)
    laps = laps.sort_values(["Driver", "LapNumber"])

    # Riepilogo settori
    summ = []
    for n in top_nums:
        d = laps[(laps["driver_number"] == n) & laps["LapTime"].notna()]
        v = d[d["PitOutLap"] != True]  # noqa: E712
        if v.empty:
            continue
        s1, s2, s3 = v["S1"].min(), v["S2"].min(), v["S3"].min()
        best = v["LapTime"].min()
        summ.append({"Driver": acr[n], "Team": team[n], "Laps": len(d), "BestLap": best,
                     "BestS1": s1, "BestS2": s2, "BestS3": s3, "IdealLap": s1 + s2 + s3,
                     "GapToIdeal": best - (s1 + s2 + s3),
                     "MaxSpeedI1": d["SpeedI1"].max(), "MaxSpeedI2": d["SpeedI2"].max(),
                     "MaxSpeedST": d["SpeedST"].max()})
    if summ:
        pd.DataFrame(summ).sort_values("BestLap").round(3).to_csv(
            f"{tmp}/sectors_summary.csv", index=False)

    # Telemetria
    tel_all, locs = [], {}
    for n in top_nums:
        drv = acr[n]
        print(f"  telemetria {drv}...")
        car = get("car_data", f"session_key={sk}&driver_number={n}")
        dl = laps[(laps["driver_number"] == n) & laps["LapTime"].notna()
                  & laps["date_start"].notna()].sort_values("date_start").copy()
        if car.empty or dl.empty:
            print(f"  {drv}: nessun dato")
            continue
        car["date"] = to_dt(car["date"])
        car = car.sort_values("date").drop_duplicates("date")
        dl["date_end"] = dl["date_start"] + pd.to_timedelta(dl["LapTime"], unit="s")
        m = pd.merge_asof(car, dl[["date_start", "date_end", "LapNumber", "Compound"]],
                          left_on="date", right_on="date_start", direction="backward")
        m = m[m["date"] < m["date_end"]].copy()
        if m.empty:
            continue
        m["LapNumber"] = m["LapNumber"].astype(int)
        m.rename(columns={"speed": "Speed", "rpm": "RPM", "n_gear": "nGear",
                          "throttle": "Throttle", "brake": "Brake", "drs": "DRS"},
                 inplace=True)
        for c in ("RPM", "nGear", "Throttle", "Brake", "DRS"):
            if c not in m:
                m[c] = np.nan
        # Valori sentinella del feed = dato non valido
        m.loc[m["Throttle"] > 100, "Throttle"] = np.nan
        m.loc[(m["nGear"] < 0) | (m["nGear"] > 8), "nGear"] = np.nan
        m["Brake"] = m["Brake"] > 0
        m["Time"] = (m["date"] - m["date_start"]).dt.total_seconds()
        m = m.sort_values(["LapNumber", "date"])
        dt = m.groupby("LapNumber")["Time"].diff().fillna(m["Time"])
        m["Distance"] = (m["Speed"] / 3.6 * dt).groupby(m["LapNumber"]).cumsum()
        m["Driver"] = drv
        m["driver_number"] = n
        tel_all.append(m)
        locs[n] = get("location", f"session_key={sk}&driver_number={n}")

    track_all, clip_all, valide = [], [], (0, 0)
    if tel_all:
        tel = pd.concat(tel_all)
        # Qualità per giro: feed congelato, sentinelle, distanza incoerente
        q = tel.groupby(["driver_number", "LapNumber"]).agg(
            dist=("Distance", "max"), vmax=("Speed", "max"), sstd=("Speed", "std"),
            bad=("Throttle", lambda x: x.isna().mean())).reset_index()
        ok = (q["bad"] < 0.02) & (q["vmax"] > 200) & (q["sstd"] > 20)
        ref = q.loc[ok, "dist"].median()
        q["TelValid"] = ok & ((q["dist"] / ref - 1).abs() < 0.05)
        valide = (int(q["TelValid"].sum()), len(q))
        print(f"Lunghezza giro stimata: {ref:.0f} m - giri con telemetria valida: {valide[0]}/{valide[1]}")
        tel = tel.merge(q[["driver_number", "LapNumber", "TelValid"]],
                        on=["driver_number", "LapNumber"], how="left")
        laps = laps.merge(q[["driver_number", "LapNumber", "TelValid"]],
                          on=["driver_number", "LapNumber"], how="left")

        for (dn, ln), t in tel[tel["TelValid"] == True].groupby(["driver_number", "LapNumber"]):  # noqa: E712
            clip_all += clipping_events(t, acr[dn], int(ln))

        # Giro più veloce con telemetria valida + coordinate
        for n in top_nums:
            loc = locs.get(n)
            v = laps[(laps["driver_number"] == n) & (laps["TelValid"] == True)  # noqa: E712
                     & (laps["PitOutLap"] != True)]  # noqa: E712
            if loc is None or loc.empty or v.empty:
                continue
            fl = v.loc[v["LapTime"].idxmin()]
            start = fl["date_start"]
            end = start + pd.to_timedelta(fl["LapTime"], unit="s")
            loc = loc.copy()
            loc["date"] = to_dt(loc["date"])
            loc = loc[(loc["date"] >= start) & (loc["date"] < end)]
            lap_tel = tel[(tel["driver_number"] == n) & (tel["LapNumber"] == fl["LapNumber"])]
            if loc.empty or lap_tel.empty:
                continue
            p = pd.merge_asof(loc.sort_values("date"),
                              lap_tel[["date", "Distance", "Speed", "Throttle", "Brake",
                                       "nGear"]].sort_values("date"),
                              on="date", direction="nearest")
            p = p.rename(columns={"x": "X", "y": "Y", "z": "Z"})
            p.insert(0, "Driver", acr[n])
            p.insert(1, "LapNumber", int(fl["LapNumber"]))
            track_all.append(p[["Driver", "LapNumber", "Distance", "X", "Y", "Z", "Speed",
                                "Throttle", "Brake", "nGear"]])

        cols = ["Driver", "LapNumber", "TelValid", "Compound", "Time", "Distance", "Speed",
                "RPM", "nGear", "Throttle", "Brake"]
        if tel["DRS"].notna().any():
            cols.append("DRS")
        tel[cols].round({"Time": 3, "Distance": 1}).to_csv(
            f"{tmp}/telemetry_top4.csv.gz", index=False)
    if track_all:
        pd.concat(track_all).round({"Distance": 1}).to_csv(
            f"{tmp}/track_fastest_top4.csv.gz", index=False)
    pd.DataFrame(clip_all).to_csv(f"{tmp}/clipping_events.csv", index=False)

    if "TelValid" not in laps:
        laps["TelValid"] = np.nan
    laps[["Driver", "Team", "TopTeam", "LapNumber", "LapTime", "S1", "S2", "S3",
          "SectorImputed", "SpeedI1", "SpeedI2", "SpeedST", "PitOutLap", "Stint",
          "Compound", "TyreLife", "TelValid", "date_start"]].to_csv(
        f"{tmp}/laps.csv.gz", index=False)

    # Meteo e direzione gara
    w = get("weather", f"session_key={sk}")
    if not w.empty:
        w.to_csv(f"{tmp}/weather.csv", index=False)
    rc = get("race_control", f"session_key={sk}")
    if not rc.empty:
        rc.to_csv(f"{tmp}/race_control.csv", index=False)

    info = {"evento": str(meeting["meeting_name"]), "luogo": str(meeting.get("location")),
            "sessione": str(sess["session_name"]), "session_key": sk,
            "inizio": pd.Timestamp(sess.get("date_start")).isoformat(), "fine": pd.Timestamp(sess.get("date_end")).isoformat(),
            "piloti_top": [acr[n] for n in top_nums], "giri": int(laps["LapTime"].notna().sum()),
            "giri_telemetria_valida": list(valide),
            "scaricato": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(f"{tmp}/info.json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)

    # sostituisce la cartella solo a lavoro finito
    if os.path.isdir(out):
        for f in os.listdir(out):
            os.remove(os.path.join(out, f))
        os.rmdir(out)
    os.rename(tmp, out)
    dim = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"Pronto: {out} ({dim / 1e6:.1f} MB)")
    return info


def aggiorna_indice(out_dir):
    righe = []
    for root, _, files in os.walk(out_dir):
        if "info.json" in files:
            with open(os.path.join(root, "info.json")) as f:
                i = json.load(f)
            i["cartella"] = os.path.relpath(root, out_dir)
            righe.append(i)
    righe.sort(key=lambda i: i.get("inizio") or "")
    with open(os.path.join(out_dir, "indice.json"), "w") as f:
        json.dump({"aggiornato": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "sessioni": righe}, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="export")
    ap.add_argument("--anno", type=int)
    ap.add_argument("--gara")
    ap.add_argument("--sessione", help="FP1 FP2 FP3 SQ S Q R (vuoto = tutte le sessioni finite)")
    ap.add_argument("--forza", action="store_true", help="riscarica anche le sessioni già presenti")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    now = datetime.now(timezone.utc)

    if a.gara:
        meeting = find_meeting(a.anno or now.year, a.gara)
    else:
        meeting = current_meeting(now)
        if meeting is None:
            print("Nessun weekend in corso.")
            return
    sessions = get("sessions", f"meeting_key={meeting['meeting_key']}")
    if sessions.empty:
        print("Nessuna sessione trovata.")
        return
    sessions = sessions.copy()
    sessions["codice"] = sessions["session_name"].astype(str).str.lower().map(CODICI)
    sessions["date_end"] = to_dt(sessions["date_end"])
    if a.sessione:
        cod = a.sessione.upper()
        if cod not in SESSION_NAMES:
            sys.exit(f"Sessione '{cod}' non valida. Usa: {' '.join(SESSION_NAMES)}")
        sessions = sessions[sessions["codice"] == cod]
        if sessions.empty:
            sys.exit(f"Sessione {cod} non trovata per {meeting['meeting_name']}.")
    pronte = sessions[sessions["codice"].notna()
                      & (sessions["date_end"] + pd.Timedelta(minutes=ATTESA_MIN) <= pd.Timestamp(now))]
    fatte, attese = 0, 0
    for _, s in pronte.sort_values("date_end").iterrows():
        out = cartella(a.out_dir, meeting, s["codice"])
        if os.path.exists(os.path.join(out, "info.json")) and not a.forza:
            continue
        try:
            esporta(meeting, s, out)
            fatte += 1
        except NonPronto as e:
            print(f"{s['codice']}: {e}")
            attese += 1
            tmp = out + ".tmp"
            if os.path.isdir(tmp):
                for f in os.listdir(tmp):
                    os.remove(os.path.join(tmp, f))
                os.rmdir(tmp)
    aggiorna_indice(a.out_dir)
    print(f"Sessioni scaricate: {fatte}; da riprovare: {attese}.")


if __name__ == "__main__":
    main()
