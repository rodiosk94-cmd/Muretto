#!/usr/bin/env python3
"""
verifica_integrita.py — Muretto: controllo di completezza dei dati scaricati da OpenF1.

Francesco (5/11/2026): i modelli si basano sulla telemetria; ogni sessione scaricata deve essere completa,
e lo storico dell'ultima stagione completo. Questo script dice, sessione per sessione, che cosa manca.

Storico (storico/<anno>/*.json.gz), per ogni sessione:
  piloti      chi compare nei giri o nella classifica ma non nell'elenco piloti; squadra o sigla mancante
  giri        giri in archivio contro i giri della classifica ufficiale, pilota per pilota; numeri di giro saltati
  inizio      primi dati (meteo, direzione gara, giri) troppo tardi rispetto all'inizio della sessione
              (Messico FP3 2026: OpenF1 senza i primi 14 minuti)
  meteo       letture rispetto alla durata (almeno l'80% dei minuti), temperature della pista impossibili (0 °C)
  gomme       stint senza mescola; piloti con giri ma senza stint
  gara        distacchi (intervals) per tutti i classificati, soste, telemetria della partenza per ogni pilota
  energia     blocchi di telemetria vuoti o con errori
Export (export/<anno>-<luogo>/<SESSIONE>/):
  file        file attesi presenti
  giri        righe di laps.csv.gz contro lo Storico
  telemetria  piloti con giri ma senza telemetria; giri con tempo ma senza telemetria; quota di giri con
              telemetria valida; campioni al secondo (atteso circa 3,7)

Uscita: elenco dei problemi (o «completo») per sessione; con --out un JSON per il repo; con --github le
annotazioni ::warning:: per GitHub Actions. Codice di uscita 1 se --esito e la sessione richiesta non è completa.

Uso, dalla radice di Muretto:
  python verifica_integrita.py --anni 2026                       tutto lo Storico 2026 (+ Export se c'è)
  python verifica_integrita.py --anni 2026 --round 19 --export   un weekend, anche l'Export
  python verifica_integrita.py --sessione "Practice 3" --fine 2026-10-31T18:30:00Z --esito
        (per il workflow Storico: 0 solo se la sessione è in archivio e completa)
"""
import argparse
import datetime as dt
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

CODICI = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Qualifying": "Q", "Race": "R",
          "Sprint": "S", "Sprint Qualifying": "SQ", "Sprint Shootout": "SQ"}
GARE = ("Race", "Sprint")
MAX_RITARDO_MIN = 5.0     # primi dati entro 5 minuti dall'inizio della sessione
COPERTURA_METEO = 0.8     # letture meteo: almeno l'80% dei minuti della sessione
HZ_ATTESI = 3.7           # campioni al secondo della telemetria OpenF1
FILE_EXPORT = ("info.json", "laps.csv.gz", "telemetry_top4.csv.gz", "weather.csv", "race_control.csv")


def t(s):
    if not s:
        return None
    d = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "gp"


def controlla_sessione(s):
    """Problemi di una sessione dello Storico: lista di (gravità, testo). Gravità: 'grave' o 'lieve'."""
    P = []
    nome = s["session"]["session_name"]
    inizio, fine = t(s["session"].get("date_start")), t(s["session"].get("date_end"))
    laps = s.get("laps") or []
    res = s.get("session_result") or []
    drv = {d["driver_number"]: d for d in s.get("drivers") or []}
    if not laps:
        P.append(("grave", "nessun giro"))
    if len(res) < 18:
        P.append(("grave", "classifica con %d piloti" % len(res)))
    # piloti
    numeri = {l["driver_number"] for l in laps} | {r["driver_number"] for r in res}
    fuori = sorted(n for n in numeri if n not in drv)
    if fuori:
        P.append(("lieve", "piloti senza scheda: %s" % ", ".join(map(str, fuori))))
    senza_team = [str(d.get("name_acronym") or n) for n, d in drv.items() if not d.get("team_name")]
    if senza_team:
        P.append(("lieve", "piloti senza squadra: %s" % ", ".join(senza_team)))
    sigla = {n: (d.get("name_acronym") or str(n)) for n, d in drv.items()}
    # giri contro classifica
    per = defaultdict(set)
    for l in laps:
        per[l["driver_number"]].add(l["lap_number"])
    mancano = []
    for r in res:
        n, nl = r["driver_number"], r.get("number_of_laps") or 0
        if nl and len(per[n]) < nl:
            mancano.append("%s %d/%d" % (sigla.get(n, n), len(per[n]), nl))
    if mancano:
        tot = sum(int(x.split()[1].split("/")[1]) - int(x.split()[1].split("/")[0]) for x in mancano)
        P.append(("grave" if tot > 3 else "lieve", "giri mancanti (%d): %s" % (tot, ", ".join(mancano))))
    buchi = []
    for n, g in per.items():
        if g:
            salt = sorted(set(range(1, max(g) + 1)) - g)
            if salt:
                buchi.append("%s %s" % (sigla.get(n, n), ",".join(map(str, salt[:6])) + ("…" if len(salt) > 6 else "")))
    if buchi:
        P.append(("lieve", "numeri di giro saltati: %s" % "; ".join(buchi[:8])))
    # inizio dei dati
    w = s.get("weather") or []
    if inizio and fine:
        prime = [t(x.get("date")) for x in w[:1]] + [t(min((l["date_start"] for l in laps if l.get("date_start")), default=None))]
        prime = [p for p in prime if p]
        if laps and nome not in GARE:
            p0 = min(t(l["date_start"]) for l in laps if l.get("date_start"))
            # nelle libere e in qualifica i primi giri arrivano entro pochi minuti
            rit = (p0 - inizio).total_seconds() / 60
            wv0 = sorted(t(x["date"]) for x in w if x.get("date"))
            meteo_tardi = not wv0 or (wv0[0] - inizio).total_seconds() / 60 > MAX_RITARDO_MIN
            if rit > 12 and meteo_tardi:
                # niente giri e niente meteo all'inizio: OpenF1 ha perso l'inizio della sessione
                P.append(("grave", "dati dal minuto %.0f della sessione: inizio perso da OpenF1" % rit))
            elif rit > 12:
                P.append(("lieve", "primo giro al minuto %.0f (meteo regolare: probabilmente nessuno in pista)" % rit))
        if w:
            w0 = t(w[0]["date"]) if w[0].get("date") else None
            wv = sorted(t(x["date"]) for x in w if x.get("date"))
            w0 = wv[0] if wv else None
            if w0 and (w0 - inizio).total_seconds() / 60 > MAX_RITARDO_MIN:
                P.append(("lieve", "meteo dal minuto %.0f della sessione" % ((w0 - inizio).total_seconds() / 60)))
            a0, a1 = inizio, fine
            if nome in GARE and laps:   # gara e Sprint: la finestra in calendario è più lunga della corsa
                ds = [t(l["date_start"]) for l in laps if l.get("date_start")]
                a0, a1 = min(ds), max(ds)
            durata = (a1 - a0).total_seconds() / 60
            dentro = [x for x in wv if a0 <= x <= a1]
            if durata > 0 and len(dentro) < COPERTURA_METEO * durata:
                P.append(("lieve", "meteo: %d letture su %.0f minuti" % (len(dentro), durata)))
        else:
            P.append(("grave", "nessun dato meteo"))
    zero = sum(1 for x in w if (x.get("track_temperature") or 0) <= 1)
    if zero:
        P.append(("lieve", "meteo: %d letture con la pista a 0 °C (da scartare)" % zero))
    if not s.get("race_control"):
        P.append(("lieve", "nessun messaggio della direzione gara"))
    # gomme
    st = s.get("stints") or []
    senza = sum(1 for x in st if not x.get("compound"))
    if senza:
        P.append(("grave" if senza > 3 else "lieve", "%d stint senza mescola" % senza))
    con_stint = {x["driver_number"] for x in st}
    no_st = [sigla.get(n, str(n)) for n in per if n not in con_stint]
    if no_st:
        P.append(("lieve", "piloti con giri ma senza stint: %s" % ", ".join(no_st)))
    # gara
    if nome in GARE:
        iv = s.get("intervals")
        if iv is None:
            P.append(("grave", "distacchi (intervals) assenti"))
        else:
            senza_iv = [sigla.get(r["driver_number"], r["driver_number"]) for r in res
                        if (r.get("number_of_laps") or 0) > 1 and str(r["driver_number"]) not in iv.get("piloti", {})]
            if senza_iv:
                P.append(("lieve", "distacchi mancanti per: %s" % ", ".join(map(str, senza_iv))))
        if not s.get("pit") and any(l.get("is_pit_out_lap") and l["lap_number"] > 1 for l in laps):
            P.append(("lieve", "soste (pit) assenti"))
        part = (s.get("partenza") or {}).get("car_data") or {}
        if not part:
            P.append(("grave", "telemetria della partenza assente"))
        else:
            vuoti = [sigla.get(int(k), k) for k, v in part.items() if not (v.get("speed") or [])]
            if vuoti:
                P.append(("lieve", "partenza senza campioni per: %s" % ", ".join(map(str, vuoti))))
    # energia
    en = s.get("energia")
    if isinstance(en, dict):
        if en.get("errori"):
            P.append(("lieve", "energia: %d errori" % (en["errori"] if isinstance(en["errori"], int) else len(en["errori"]))))
        if en.get("vuoti"):
            P.append(("lieve", "energia: %d blocchi vuoti" % en["vuoti"]))
    return P


def controlla_export(cart, s):
    """Problemi di una cartella dell'Export confrontata con la sessione dello Storico."""
    import numpy as np
    import pandas as pd
    P = []
    for f in FILE_EXPORT:
        if not os.path.exists(os.path.join(cart, f)):
            P.append(("grave", "file mancante: %s" % f))
    if any(g == "grave" for g, _ in P):
        return P
    info = json.load(open(os.path.join(cart, "info.json")))
    L = pd.read_csv(os.path.join(cart, "laps.csv.gz"))
    T = pd.read_csv(os.path.join(cart, "telemetry_top4.csv.gz"), usecols=["Driver", "LapNumber", "TelValid", "Time"])
    n_st = len(s.get("laps") or [])
    if len(L) < n_st:
        P.append(("grave" if n_st - len(L) > 3 else "lieve", "giri nell'Export %d contro %d nello Storico" % (len(L), n_st)))
    attesi = set(info.get("piloti_telemetria") or info.get("piloti_top") or [])
    con_giri = set(L[L.LapTime.notna()].Driver.dropna())
    if info.get("piloti_telemetria"):
        manca_p = sorted(p for p in con_giri if p not in set(T.Driver))
    else:
        manca_p = sorted(p for p in attesi if p in con_giri and p not in set(T.Driver))
        P.append(("lieve", "telemetria solo dei top team (scaricato prima del 5/11/2026)"))
    if manca_p:
        P.append(("grave", "piloti senza telemetria: %s" % ", ".join(manca_p)))
    tel_giri = set(zip(T.Driver, T.LapNumber))
    lt = L[L.LapTime.notna() & L.Driver.isin(set(T.Driver))]
    senza = [(r.Driver, int(r.LapNumber)) for r in lt.itertuples() if (r.Driver, r.LapNumber) not in tel_giri]
    if senza:
        P.append(("grave" if len(senza) > 0.02 * max(len(lt), 1) else "lieve",
                  "%d giri con tempo ma senza telemetria su %d" % (len(senza), len(lt))))
    q = T.groupby(["Driver", "LapNumber"]).agg(n=("Time", "size"), dur=("Time", "max"), ok=("TelValid", "first"))
    if len(q):
        val = float((q.ok == True).mean())  # noqa: E712
        hz = float(np.median(q.n / q.dur.clip(lower=1)))
        if val < 0.85:
            P.append(("lieve", "telemetria valida nel %.0f%% dei giri" % (100 * val)))
        if hz < 0.8 * HZ_ATTESI:
            P.append(("grave", "campioni al secondo %.1f (attesi circa %.1f)" % (hz, HZ_ATTESI)))
    return P


def cartella_export(export_dir, d, nome):
    luogo = d["meeting"].get("location") or d["meeting"].get("circuit_short_name") or d["meeting"].get("meeting_name")
    return os.path.join(export_dir, "%d-%s" % (d["anno"], slug(luogo)), CODICI.get(nome, "?"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storico", default="storico")
    ap.add_argument("--export-dir", default="export")
    ap.add_argument("--anni", default=str(dt.datetime.now(dt.timezone.utc).year))
    ap.add_argument("--round", type=int)
    ap.add_argument("--export", action="store_true", help="controlla anche le cartelle dell'Export")
    ap.add_argument("--sessione", help="solo questa sessione (nome OpenF1, es. 'Practice 3')")
    ap.add_argument("--fine", help="con --sessione: fine prevista in UTC, per trovarla fra i file della stagione")
    ap.add_argument("--esito", action="store_true", help="codice 1 se la sessione richiesta manca o ha problemi gravi")
    ap.add_argument("--out", help="JSON con i problemi per sessione")
    ap.add_argument("--github", action="store_true", help="annotazioni ::warning:: per GitHub Actions")
    ap.add_argument("--solo-problemi", action="store_true")
    a = ap.parse_args()
    fine = t(a.fine) if a.fine else None
    risultati, trovata, gravi_trovata = [], False, False
    for anno in [int(x) for x in a.anni.split(",")]:
        for f in sorted(glob.glob(os.path.join(a.storico, str(anno), "*.json.gz"))):
            d = json.load(gzip.open(f))
            if a.round and d.get("round") != a.round:
                continue
            for s in d.get("sessioni") or []:
                nome = s["session"]["session_name"]
                if a.sessione and nome != a.sessione:
                    continue
                if fine:
                    de = t(s["session"].get("date_end"))
                    if not de or abs((de - fine).total_seconds()) > 6 * 3600:
                        continue
                P = controlla_sessione(s)
                fonte = "storico"
                if a.export:
                    c = cartella_export(a.export_dir, d, nome)
                    if os.path.isdir(c):
                        P += [(g, "Export: " + x) for g, x in controlla_export(c, s)]
                    else:
                        P.append(("grave", "Export assente (%s)" % c))
                if a.sessione:
                    trovata = True
                    gravi_trovata = gravi_trovata or any(g == "grave" for g, _ in P)
                risultati.append(dict(anno=anno, round=d.get("round"), gp=d["meeting"].get("location"), sessione=nome,
                                      session_key=s["session"].get("session_key"), parziale=bool(d.get("parziale")),
                                      stato="completo" if not P else ("lacune gravi" if any(g == "grave" for g, _ in P) else "lacune lievi"),
                                      problemi=[{"gravita": g, "testo": x} for g, x in P]))
    stati = Counter(r["stato"] for r in risultati)
    for r in risultati:
        if a.solo_problemi and r["stato"] == "completo":
            continue
        print("%d %02d %-14s %-18s %s" % (r["anno"], r["round"] or 0, (r["gp"] or "")[:14], r["sessione"], r["stato"]))
        for p in r["problemi"]:
            print("      [%s] %s" % (p["gravita"], p["testo"]))
            if a.github and p["gravita"] == "grave":
                print("::warning::%d %s %s: %s" % (r["anno"], r["gp"], r["sessione"], p["testo"]))
    print("Sessioni: %d — %s" % (len(risultati), ", ".join("%s %d" % kv for kv in sorted(stati.items()))))
    if a.out:
        with open(a.out, "w") as fo:
            json.dump({"controllato": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "riepilogo": dict(stati), "sessioni": risultati}, fo, ensure_ascii=False, indent=1)
    if a.esito:
        sys.exit(0 if trovata and not gravi_trovata else 1)


if __name__ == "__main__":
    main()
