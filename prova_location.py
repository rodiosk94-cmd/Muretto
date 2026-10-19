"""Prova dei dati di posizione in pista (endpoint `location` di OpenF1).

Scarica, per un weekend già presente nello Storico:
- la partenza della gara, per tutte le auto, nella stessa finestra della chiave `partenza`;
- il giro migliore in qualifica dei primi N, negli stessi istanti dei blocchi `energia`.

Serve a confrontare le distanze ricavate dalle coordinate con quelle ricavate
integrando la velocità (metri percorsi dopo il via, allineamento dei giri).

Uso: python prova_location.py --file storico/2026/17-singapore-grand-prix.json.gz \
         --out storico/prove/location-singapore.json.gz --primi 4
"""
import argparse
import gzip
import json
import sys
from datetime import datetime, timezone

from backfill_fetch import get, iso, ts, log, warn, salva, _count

MARGINE_GIRO = 2.0  # secondi in più prima e dopo il giro di qualifica


def compatta(rows):
    rows = sorted(rows or [], key=lambda r: r.get("date") or "")
    return {f: [r.get(f) for r in rows] for f in ("date", "x", "y", "z")}


def posizioni(key, num, t0, t1):
    rows = get("location", f"session_key={key}", f"driver_number={num}",
               f"date>={iso(t0)}", f"date<={iso(t1)}")
    return compatta(rows)


def sessione(data, nome):
    for s in data.get("sessioni", []):
        if s.get("session", {}).get("session_name") == nome:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="file dello Storico del weekend")
    ap.add_argument("--out", required=True, help="file di uscita (.json.gz)")
    ap.add_argument("--primi", type=int, default=4, help="piloti della qualifica da scaricare")
    a = ap.parse_args()

    with gzip.open(a.file, "rt", encoding="utf-8") as f:
        data = json.load(f)

    out = {"fonte": a.file, "creato": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "errori": []}

    # Partenza della gara
    gara = sessione(data, "Race")
    if gara and gara.get("partenza"):
        key = gara["session"]["session_key"]
        p = gara["partenza"]
        t0, t1 = ts(p["da"]), ts(p["a"])
        piloti = sorted(p["car_data"].keys(), key=int)
        log(f"Gara {key}: partenza {p['da']} - {p['a']}, {len(piloti)} auto")
        loc = {}
        for n in piloti:
            try:
                loc[n] = posizioni(key, int(n), t0, t1)
                log(f"  auto {n}: {len(loc[n]['date'])} punti")
            except Exception as e:  # noqa: BLE001
                warn(f"gara, auto {n}: {e}")
                out["errori"].append({"sessione": "Race", "pilota": n, "errore": str(e)})
        out["gara"] = {"session_key": key, "da": p["da"], "a": p["a"], "location": loc}
    else:
        warn("nessuna partenza di gara nel file")

    # Giro migliore in qualifica dei primi N
    q = sessione(data, "Qualifying")
    if q and q.get("energia"):
        key = q["session"]["session_key"]
        def pos(r):
            p = r.get("position")
            return p if isinstance(p, int) else 99  # "RT", "NC" e simili in fondo

        ordine = [r["driver_number"] for r in sorted(q.get("session_result") or [], key=pos)]
        primi = ordine[:a.primi]
        blocchi = {b["pilota"]: b for b in q["energia"].get("blocchi", []) if b.get("tipo") == "migliore"}
        giri = {}
        for num in primi:
            b = blocchi.get(num)
            if not b:
                warn(f"qualifica, auto {num}: nessun giro migliore nei blocchi")
                continue
            inizio = ts(b["da"]) + b["inizi"][0] / 100.0
            fine = inizio + b["durate"][0]
            try:
                pos = posizioni(key, num, inizio - MARGINE_GIRO, fine + MARGINE_GIRO)
                giri[str(num)] = {"giro": b["giri"][0], "inizio": iso(inizio), "durata": b["durate"][0], **pos}
                log(f"Qualifica, auto {num}: giro {b['giri'][0]}, {len(pos['date'])} punti")
            except Exception as e:  # noqa: BLE001
                warn(f"qualifica, auto {num}: {e}")
                out["errori"].append({"sessione": "Qualifying", "pilota": num, "errore": str(e)})
        out["qualifica"] = {"session_key": key, "giri": giri}
    else:
        warn("nessun giro di qualifica nei blocchi energia")

    out["richieste"] = _count[0]
    salva(a.out, out)
    log(f"Salvato {a.out}: {_count[0]} richieste, {len(out['errori'])} errori")
    return 1 if out["errori"] else 0


if __name__ == "__main__":
    sys.exit(main())
