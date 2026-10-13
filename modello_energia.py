#!/usr/bin/env python3
"""
modello_energia.py — Muretto: energia e motore delle monoposto 2026.

Un solo file, solo numpy (Python 3.9+). Legge l'archivio OpenF1 di Muretto (storico/):
  - la telemetria dei giri scelti che backfill_fetch.py salva in ogni sessione 2026 (chiave 'energia':
    giro migliore di libere e qualifiche, long run delle libere, blocchi di 4 giri in gara e Sprint);
  - il giro migliore in qualifica dei primi 6 nelle edizioni 2024 e 2025 dei circuiti del
    calendario (storico/circuiti/), per misurare tutte le piste con le stesse auto;
  - le velocità alle trappole di tutti i giri (sempre presenti, anche senza telemetria).

Misure di un giro (velocità, gas, freno a circa 3,7 campioni al secondo):
  vmax         velocità massima
  spinta       accelerazione media a gas pieno fra 240 e 290 km/h (km/h al secondo): potenza
               meno resistenza dell'aria. spinta_alta: lo stesso fra 290 e 330 km/h
  clip         metri a gas pieno, sopra 220 km/h, senza più guadagnare velocità a fine rettilineo
               (batteria che non spinge più o che ricarica); clip_quota: frazione dei metri a gas pieno
  calo         km/h persi a gas pieno a fine rettilineo, sommati sui rettilinei
  coast        secondi senza gas né freno sopra 150 km/h nei 3 s prima delle frenate (lift and coast)
  frenate      numero ed energia delle frenate (kJ/kg): quanto si può recuperare
  pieno        quota del giro a gas pieno, rettilinei lunghi, esposizione (metri oltre 400 m per tratto)

Profili:
  team      scarti dalla mediana dei team a ogni evento (qualifica e gara), media sui più recenti
  circuito  fame di energia della pista misurata con le auto 2024-2025 (uguale per tutte le piste,
            anche quelle che restano) e, sulle piste già corse, quella misurata nel 2026

Tempo per classi di pista: ogni giro, allineato sui punti di corda, è diviso in curve, primi 400 m dei
rettilinei e resto dei rettilinei (dove la batteria si svuota). Lo scarto di ogni team in ogni classe,
riportato con il peso delle classi su un circuito nuovo, dà lo spostamento previsto del team su quella pista.

Uso:
  python modello_energia.py scheda --storico storico --round 18 [--circuito Austin]
  python modello_energia.py prova  --storico storico [--out risultati]
  python modello_energia.py telemetria --file telemetry.json     (giri migliori dei primi 4 team nel weekend)
"""
import argparse
import glob
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

try:  # distacchi in qualifica e passo gara come nel modello di gara (stesso repo)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import modello_gara as MG
except Exception:  # noqa: BLE001
    MG = None

GAS = 98                    # gas «pieno» (OpenF1 arriva fino a 104)
V_CLIP = 220                # km/h: sotto questa velocità non si parla di clipping
SPINTA_BANDA = (240, 290)   # km/h
SPINTA_ALTA = (290, 330)    # km/h
EMIVITA = 4.0               # eventi: peso 1/2 dopo 4 eventi
DENSITA_MIN = 2.5           # campioni al secondo sotto cui il giro si scarta

# Misure usate per i profili (nome, descrizione, unità)
MISURE_TEAM = (
    ('vmax', 'velocità massima', 'km/h'),
    ('spinta', 'spinta 240-290 km/h', 'km/h/s'),
    ('spinta_alta', 'spinta 290-330 km/h', 'km/h/s'),
    ('clip_quota', 'clipping (quota del gas pieno)', '%'),
    ('calo', 'velocità persa a gas pieno', 'km/h'),
    ('coast', 'lift and coast', 's'),
)
MISURE_CIRCUITO = (
    ('quota_pieno', 'giro a gas pieno', '%'),
    ('esposizione_km', 'rettilinei lunghi (m oltre 400 m per km)', 'm/km'),
    ('run_max', 'tratto a gas pieno più lungo', 'm'),
    ('lunghi', 'tratti a gas pieno oltre 500 m', ''),
    ('frenate_km', 'energia delle frenate per km', 'kJ/kg/km'),
    ('v_media', 'velocità media', 'km/h'),
    ('vmax', 'velocità massima', 'km/h'),
)


def _t(s):
    if s is None:
        return None
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def carica_weekend(path):
    with gzip.open(path) as f:
        return json.load(f)


def sessioni(d):
    return {s['session']['session_name']: s for s in d['sessioni']}


def _arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


# ---------------------------------------------------------------------------
# 1. Misure di un giro
# ---------------------------------------------------------------------------

def misure_giro(t, v, gas, freno, rpm=None):
    """Misure energetiche di un giro. t in secondi, v in km/h, gas 0-100+, freno 0/1 (o 0/100).
    Ritorna un dict oppure None se i campioni sono troppo pochi."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    gas = np.nan_to_num(np.asarray(gas, float))
    freno = np.nan_to_num(np.asarray(freno, float)) > 0
    ok = ~np.isnan(t) & ~np.isnan(v)
    t, v, gas, freno = t[ok], v[ok], gas[ok], freno[ok]
    n = len(t)
    if n < 50:
        return None
    o = np.argsort(t, kind='stable')
    t, v, gas, freno = t[o], v[o], gas[o], freno[o]
    dt = np.diff(t)
    keep = np.r_[True, dt > 0]
    t, v, gas, freno = t[keep], v[keep], gas[keep], freno[keep]
    n = len(t)
    dt = np.diff(t)
    ds = (v[1:] + v[:-1]) / 2 / 3.6 * dt
    d = np.concatenate([[0.0], np.cumsum(ds)])
    pieno = gas >= GAS
    # un solo campione sotto il pieno (ma sopra l'80%) fra due pieni non spezza il rettilineo
    buco = np.zeros(n, bool)
    buco[1:-1] = ~pieno[1:-1] & pieno[:-2] & pieno[2:] & (gas[1:-1] >= 80) & ~freno[1:-1]
    pieno = pieno | buco
    tratti = []
    i = 0
    while i < n:
        if pieno[i]:
            j = i
            while j + 1 < n and pieno[j + 1]:
                j += 1
            tratti.append((i, j))
            i = j + 1
        else:
            i += 1
    run = []
    clip_tot = calo_tot = 0.0
    for i, j in tratti:
        lung = d[j] - d[i]
        if lung < 150:
            continue
        seg = v[i:j + 1]
        vmax = float(seg.max())
        # clipping: tratto finale a gas pieno, sopra V_CLIP, dopo il quale la velocità non sale più di
        # 2 km/h (velocità piatta o in calo con il gas a fondo)
        suff = np.maximum.accumulate(seg[::-1])[::-1]
        okc = (suff - seg < 2.0) & (seg >= V_CLIP)
        clip_len = calo = 0.0
        if okc.any():
            k0 = i + int(np.argmax(okc))
            clip_len = float(d[j] - d[k0])
            if clip_len < 60:
                clip_len = 0.0
            else:
                calo = float(max(v[k0:j + 1]) - v[j])
        clip_tot += clip_len
        calo_tot += calo
        run.append(dict(inizio=float(d[i]), lung=float(lung), v_in=float(v[i]), vmax=vmax,
                        v_fine=float(v[j]), clip=clip_len, calo=calo))

    def _spinta(banda):
        gv = gt = 0.0
        for i, j in tratti:
            for k in range(i, j):
                vm = (v[k] + v[k + 1]) / 2
                if banda[0] <= vm <= banda[1]:
                    gv += v[k + 1] - v[k]
                    gt += dt[k]
        return float(gv / gt) if gt >= 1.0 else None

    # lift and coast: gas sotto il 10% senza freno a più di 150 km/h, nei 3 s prima di una frenata
    coast = 0.0
    for k in np.where(freno[1:] & ~freno[:-1])[0] + 1:
        a = k - 1
        while a >= 0 and t[a] >= t[k] - 3.0 and gas[a] < 10 and not freno[a] and v[a] > 150:
            coast += t[a + 1] - t[a]
            a -= 1
    # frenate: numero ed energia cinetica dissipata per unità di massa (kJ/kg)
    zone = 0
    energia = 0.0
    i = 0
    while i < n:
        if freno[i]:
            j = i
            while j + 1 < n and freno[j + 1]:
                j += 1
            if t[j] - t[i] >= 0.5:
                zone += 1
                v0 = v[max(i - 1, 0)] / 3.6
                v1 = v[min(j + 1, n - 1)] / 3.6
                energia += max(0.0, 0.5 * (v0 ** 2 - v1 ** 2)) / 1000.0
            i = j + 1
        else:
            i += 1
    dist = float(d[-1])
    durata = float(t[-1] - t[0])
    pieno_m = float(sum(r['lung'] for r in run))
    km = max(dist / 1000.0, 0.1)
    out = dict(durata=durata, distanza=dist, v_media=dist / max(durata, 1e-6) * 3.6, vmax=float(v.max()),
               quota_pieno=100.0 * pieno_m / max(dist, 1.0),
               tempo_pieno=100.0 * float(np.sum(dt[pieno[:-1] & pieno[1:]])) / max(durata, 1e-6),
               pieno_m=pieno_m, lunghi=int(sum(1 for r in run if r['lung'] >= 500)),
               run_max=float(max([r['lung'] for r in run] or [0.0])),
               esposizione=float(sum(max(0.0, r['lung'] - 400) for r in run)),
               esposizione_km=float(sum(max(0.0, r['lung'] - 400) for r in run)) / km,
               clip=float(clip_tot), clip_quota=100.0 * clip_tot / max(pieno_m, 1.0), calo=float(calo_tot),
               spinta=_spinta(SPINTA_BANDA), spinta_alta=_spinta(SPINTA_ALTA),
               coast=float(coast), frenate=zone, energia_frenate=float(energia), frenate_km=float(energia) / km,
               tratti=run)
    if rpm is not None:
        r_ = np.asarray(rpm, float)
        r_ = r_[~np.isnan(r_)]
        out['rpm_max'] = float(np.percentile(r_, 99)) if len(r_) else None
    return out


# ---------------------------------------------------------------------------
# 2. Lettura della telemetria salvata da backfill_fetch.py
# ---------------------------------------------------------------------------

def giri_blocco(b):
    """Divide un blocco nei suoi giri: lista di (giro, t dall'inizio del giro, colonne, durata)."""
    t = np.asarray(b['t'], float) / 100.0
    col = {k: _arr(b[k]) for k in ('v', 'gas', 'freno', 'rpm', 'marcia') if k in b}
    out = []
    inizi = b.get('inizi') or [int(round(100 * 1.0))]
    durate = b.get('durate') or [float(t[-1] - t[0] - 2.0)]
    for g, ini, dur in zip(b['giri'], inizi, durate):
        a = ini / 100.0
        m = (t >= a - 0.15) & (t <= a + dur + 0.15)
        if m.sum() < 50:
            continue
        out.append((g, t[m] - a, {k: c[m] for k, c in col.items()}, float(dur)))
    return out


def misure_blocco(b):
    """Misure di ogni giro di un blocco (giri con troppi pochi campioni scartati)."""
    res = []
    for g, t, c, dur in giri_blocco(b):
        if len(t) / max(dur, 1.0) < DENSITA_MIN:
            continue
        m = misure_giro(t, c['v'], c['gas'], c['freno'], c.get('rpm'))
        if m is None:
            continue
        m.update(giro=g, durata_ufficiale=dur, pilota=b['pilota'], tipo=b['tipo'])
        res.append(m)
    return res


def bagnata(s):
    """Sessione bagnata: gomme intermedie o da bagnato in più del 10% dei giri, o pioggia segnalata."""
    tot = wet = 0
    for st in s.get('stints') or []:
        nn = max(0, (st.get('lap_end') or 0) - (st.get('lap_start') or 0) + 1)
        tot += nn
        if st.get('compound') in ('INTERMEDIATE', 'WET'):
            wet += nn
    if tot and wet / tot > 0.10:
        return True
    w = s.get('weather') or []
    return bool(w) and sum(1 for x in w if x.get('rainfall')) / len(w) > 0.2


def trappole(s, qualifica):
    """Velocità alla trappola per pilota: massimo sui giri veloci (qualifica) o mediana dei giri di
    passo (gara e libere)."""
    by = defaultdict(list)
    for l in s.get('laps') or []:
        if l.get('st_speed') and l.get('lap_duration') and not l.get('is_pit_out_lap') and l['lap_number'] > 1:
            by[l['driver_number']].append(l)
    out = {}
    for dn, ls in by.items():
        tt = np.array([l['lap_duration'] for l in ls])
        st = np.array([l['st_speed'] for l in ls], float)
        if qualifica:
            m = tt <= 1.02 * tt.min()
            out[dn] = float(np.max(st[m]))
        else:
            m = tt <= 1.07 * np.median(tt)
            if m.sum() >= 3:
                out[dn] = float(np.median(st[m]))
    return out


def evento_energia(d):
    """Riassunto energetico di un weekend dell'archivio."""
    S = sessioni(d)
    m = d['meeting']
    ev = dict(anno=d['anno'], round=d['round'], circuit_key=m.get('circuit_key'),
              circuito=m.get('circuit_short_name'), luogo=m.get('location'), gp=m.get('meeting_name'),
              parziale=bool(d.get('parziale')), data=m.get('date_start'))
    team = {}
    for s in d['sessioni']:
        for x in s.get('drivers') or []:
            if x.get('team_name'):
                team[x['driver_number']] = x['team_name']
    ev['team'] = team
    ev['misure'] = {}
    ev['bagnato'] = {}
    ev['trappola'] = {}
    for nome, s in S.items():
        ev['bagnato'][nome] = bagnata(s)
        ev['trappola'][nome] = trappole(s, nome in ('Qualifying', 'Sprint Qualifying', 'Sprint Shootout'))
        e = s.get('energia')
        if not e:
            continue
        rows = []
        for b in e.get('blocchi') or []:
            rows += misure_blocco(b)
        ev['misure'][nome] = rows
    # tempo per classi di pista (curve, rettilinei brevi e lunghi)
    ev['classi'] = {}
    for nome, tipo in (('Qualifying', 'migliore'), ('Race', 'gara'), ('Sprint', 'gara'),
                       ('Practice 1', 'migliore'), ('Practice 2', 'migliore'), ('Practice 3', 'migliore'),
                       ('Practice 1', 'long_run'), ('Practice 2', 'long_run')):
        s = S.get(nome)
        e = (s or {}).get('energia')
        if not e or ev['bagnato'].get(nome):
            continue
        a = analisi_classi(profili_sessione(e.get('blocchi') or [], tipo), team)
        if a:
            ev['classi'][f'{nome}|{tipo}'] = a
    # distacchi in qualifica e passo gara (come nel modello di gara)
    ev['distacchi'] = {}
    ev['passo'] = {}
    if MG is not None:
        q = S.get('Qualifying')
        if q and q.get('session_result'):
            try:
                ev['distacchi'], _ = MG.distacchi_qualifica(q)
            except Exception:  # noqa: BLE001
                pass
        r = S.get('Race')
        if r and r.get('laps') and r.get('stints'):
            try:
                ev['passo'], _ = MG.passo_gara(r)
            except Exception:  # noqa: BLE001
                pass
    return ev


def carica_stagione(storico, anno=2026):
    files = sorted(glob.glob(os.path.join(storico, str(anno), '*.json.gz')))
    out = []
    for f in files:
        try:
            out.append(evento_energia(carica_weekend(f)))
        except Exception as e:  # noqa: BLE001
            print(f'ATTENZIONE: {os.path.basename(f)}: {type(e).__name__}: {e}', file=sys.stderr)
    out.sort(key=lambda e: e['round'])
    return out


# ---------------------------------------------------------------------------
# 3. Profili di team
# ---------------------------------------------------------------------------

def per_pilota(rows, tipo=None, aggrega='media'):
    """Misure per pilota: media dei giri (o il solo giro, per il giro migliore)."""
    by = defaultdict(list)
    for r in rows:
        if tipo is None or r['tipo'] == tipo:
            by[r['pilota']].append(r)
    out = {}
    for dn, rs in by.items():
        x = {}
        for k, _, _ in MISURE_TEAM + MISURE_CIRCUITO:
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals:
                x[k] = float(np.median(vals)) if aggrega == 'mediana' else float(np.mean(vals))
        x['n'] = len(rs)
        out[dn] = x
    return out


def scarti_team(valori, team):
    """Da valori per pilota a scarti per team dalla mediana dei team (media dei due piloti)."""
    bt = defaultdict(list)
    for dn, v in valori.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        t = team.get(dn)
        if t:
            bt[t].append(v)
    tv = {t: float(np.mean(v)) for t, v in bt.items()}
    if len(tv) < 4:
        return {}
    med = float(np.median(list(tv.values())))
    return {t: v - med for t, v in tv.items()}


def profilo_evento(ev, sessione='Qualifying', tipo='migliore'):
    """Scarti dei team dalla mediana per ogni misura in una sessione dell'evento."""
    rows = ev['misure'].get(sessione) or []
    pp = per_pilota(rows, tipo)
    out = {}
    for k, _, _ in MISURE_TEAM:
        out[k] = scarti_team({dn: x.get(k) for dn, x in pp.items()}, ev['team'])
    return out


def media_recente(eventi, funzione, emivita=EMIVITA):
    """Media pesata (peso 1/2 dopo 'emivita' eventi) di dict team->valore prodotti da funzione(ev)."""
    num, den = defaultdict(float), defaultdict(float)
    for i, ev in enumerate(eventi[::-1]):
        w = 0.5 ** (i / emivita)
        for t, v in (funzione(ev) or {}).items():
            if v is not None and not math.isnan(v):
                num[t] += w * v
                den[t] += w
    return {t: num[t] / den[t] for t in den if den[t] > 0}, dict(den)


def team_gap_qualifica(ev):
    return scarti_team(ev.get('distacchi') or {}, ev['team'])


def team_passo(ev):
    return scarti_team(ev.get('passo') or {}, ev['team'])


def team_gara_meno_qualifica(ev):
    """Quanto il team va peggio in gara che in qualifica (scarti dalla mediana, % sul giro)."""
    p, q = team_passo(ev), team_gap_qualifica(ev)
    return {t: p[t] - q[t] for t in p if t in q}


def team_trappola(ev, sessione='Qualifying'):
    if ev['bagnato'].get(sessione):
        return {}
    return scarti_team(ev['trappola'].get(sessione) or {}, ev['team'])


def team_misura(ev, misura, sessione='Qualifying', tipo='migliore'):
    if ev['bagnato'].get(sessione):
        return {}
    rows = [r for r in ev['misure'].get(sessione) or [] if r['tipo'] == tipo]
    pp = per_pilota(rows)
    return scarti_team({dn: x.get(misura) for dn, x in pp.items()}, ev['team'])


# ---------------------------------------------------------------------------
# 4. Profili dei circuiti
# ---------------------------------------------------------------------------

def circuiti_riferimento(storico):
    """Misure dei circuiti dalle qualifiche 2024-2025 (giro migliore dei primi 6), sessioni asciutte.
    Ritorna circuit_key -> dict(nome, anni, misure medie, n giri)."""
    out = {}
    per = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(storico, 'circuiti', '*.json.gz'))):
        try:
            d = carica_weekend(f)
        except Exception:  # noqa: BLE001
            continue
        w = d.get('weather') or []
        bag = bool(w) and sum(1 for x in w if x.get('rainfall')) / len(w) > 0.2
        rows = []
        for b in (d.get('energia') or {}).get('blocchi') or []:
            rows += misure_blocco(b)
        if not rows:
            continue
        per[d['meeting'].get('circuit_key')].append((d['anno'], bag, rows, d['meeting']))
    for ck, lst in per.items():
        asciutte = [x for x in lst if not x[1]] or lst
        rows = [r for x in asciutte for r in x[2]]
        mis = {}
        for k, _, _ in MISURE_CIRCUITO + (('clip_quota', '', ''), ('calo', '', ''), ('spinta', '', '')):
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                mis[k] = float(np.median(vals))
        out[ck] = dict(nome=asciutte[-1][3].get('circuit_short_name'), luogo=asciutte[-1][3].get('location'),
                       anni=sorted({x[0] for x in asciutte}), bagnato=all(x[1] for x in lst),
                       misure=mis, n=len(rows))
    return out


def circuito_2026(ev):
    """Misure del circuito dai giri migliori in qualifica 2026 di tutti i piloti (se asciutta)."""
    if ev['bagnato'].get('Qualifying'):
        return None
    rows = [r for r in ev['misure'].get('Qualifying') or [] if r['tipo'] == 'migliore']
    if len(rows) < 8:
        return None
    mis = {}
    for k, _, _ in MISURE_CIRCUITO + (('clip_quota', '', ''), ('calo', '', ''), ('spinta', '', ''),
                                     ('coast', '', '')):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            mis[k] = float(np.median(vals))
    return mis


def indice_circuito(rif, chiave='quota_pieno'):
    """Indice di fame di energia (scarto standardizzato sul calendario) per ogni circuito."""
    vals = {ck: c['misure'].get(chiave) for ck, c in rif.items() if c['misure'].get(chiave) is not None}
    if len(vals) < 3:
        return {}
    a = np.array(list(vals.values()))
    mu, sd = float(a.mean()), float(a.std()) or 1.0
    return {ck: (v - mu) / sd for ck, v in vals.items()}


# ---------------------------------------------------------------------------
# 5. Dove si perde il tempo: curve, rettilinei brevi e lunghi
# ---------------------------------------------------------------------------
# Ogni giro è riportato sulla distanza (0-1 del giro). Sulla pista si separano, con la mediana del
# gruppo, i tratti a gas pieno (rettilinei, divisi nei primi 400 m e in quello che segue) dal resto
# (curve e frenate). Per ogni pilota si misura il tempo speso in ciascuna classe: lo scarto di un team
# in % del tempo della classe, stimato sui weekend passati, si riporta su un circuito nuovo con il peso
# di ogni classe in quel circuito. I rettilinei oltre i 400 m sono quelli in cui la batteria si svuota.

GRIGLIA = 1000
SOGLIA_RETTILINEO = 0.7     # quota del gruppo a gas pieno perché un punto sia rettilineo
BREVE = 400.0               # m: primi metri di ogni rettilineo
CLASSI = ('curve', 'rett_breve', 'rett_lungo')


def profilo_distanza(t, v, gas, durata, n=GRIGLIA):
    """Giro riportato sulla distanza: tempo, velocità e gas pieno su n+1 punti fra 0 e 1 del giro.
    t in secondi dall'inizio del giro ufficiale (che dura 'durata')."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    gas = np.nan_to_num(np.asarray(gas, float))
    ok = ~np.isnan(t) & ~np.isnan(v)
    t, v, gas = t[ok], v[ok], gas[ok]
    o = np.argsort(t, kind='stable')
    t, v, gas = t[o], v[o], gas[o]
    keep = np.r_[True, np.diff(t) > 0]
    t, v, gas = t[keep], v[keep], gas[keep]
    if len(t) < 50 or t[0] > 1.0 or t[-1] < durata - 1.0:
        return None
    d = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 / 3.6 * np.diff(t))])
    d0, d1 = np.interp(0.0, t, d), np.interp(durata, t, d)
    if d1 - d0 < 1000:
        return None
    s = (d - d0) / (d1 - d0)
    g = np.linspace(0.0, 1.0, n + 1)
    tg = np.interp(g, s, t)
    tg[0], tg[-1] = 0.0, durata
    return dict(t=tg, v=np.interp(g, s, v), pieno=np.interp(g, s, (gas >= GAS).astype(float)),
                lunghezza=float(d1 - d0), durata=float(durata))


def profili_sessione(rows_blocchi, tipo='migliore'):
    """Profili sulla distanza dei giri di una sessione: lista di (pilota, giro, profilo)."""
    out = []
    for b in rows_blocchi:
        if b['tipo'] != tipo:
            continue
        for g, t, c, dur in giri_blocco(b):
            if len(t) / max(dur, 1.0) < DENSITA_MIN:
                continue
            p = profilo_distanza(t, c['v'], c['gas'], dur)
            if p is not None:
                out.append((b['pilota'], g, p))
    return out


def trova_apici(v, finestra=15, profondita=30.0):
    """Punti di corda: minimi locali della velocità (indici della griglia) almeno 'profondita' km/h sotto
    il massimo dei 300 m intorno."""
    n = len(v)
    out = []
    for i in range(finestra, n - finestra):
        if v[i] > v[i - finestra:i + finestra + 1].min():
            continue
        a, b = max(0, i - 60), min(n, i + 61)
        if v[a:b].max() - v[i] >= profondita and (not out or i - out[-1] > finestra):
            out.append(i)
    return out


def registra(profili):
    """Allinea sulla distanza i giri di una sessione: i punti di corda di ogni giro vanno su quelli della
    mediana del gruppo (la distanza integrata dalla velocità sbaglia di 10-20 m lungo il giro).
    Aggiunge a ogni profilo le chiavi t_r, v_r, pieno_r sulla griglia comune."""
    if not profili:
        return profili
    V = np.array([p['v'] for _, _, p in profili])
    apici = trova_apici(np.median(V, axis=0))
    g = np.arange(GRIGLIA + 1, dtype=float)
    for _, _, p in profili:
        src, dst = [0.0], [0.0]
        for a in apici:
            lo, hi = max(1, a - 12), min(GRIGLIA - 1, a + 12)
            k = lo + int(np.argmin(p['v'][lo:hi + 1]))
            if k > src[-1] and a > dst[-1]:
                src.append(float(k))
                dst.append(float(a))
        src.append(float(GRIGLIA))
        dst.append(float(GRIGLIA))
        idx = np.interp(g, dst, src)
        p['t_r'] = np.interp(idx, g, p['t'])
        p['v_r'] = np.interp(idx, g, p['v'])
        p['pieno_r'] = np.interp(idx, g, p['pieno'])
    return profili


def classi_pista(profili, lunghezza=None):
    """Classe di ogni intervallo della griglia (0 curve, 1 rettilineo breve, 2 rettilineo lungo) dalla
    mediana del gruppo, e lunghezza della pista (m)."""
    P = np.array([p.get('pieno_r', p['pieno']) for _, _, p in profili])
    L = lunghezza or float(np.median([p['lunghezza'] for _, _, p in profili]))
    pieno = np.median(P, axis=0) if len(P) >= 3 else P.mean(axis=0)
    mid = (pieno[1:] + pieno[:-1]) / 2 >= SOGLIA_RETTILINEO
    cl = np.zeros(GRIGLIA, int)
    passo = L / GRIGLIA
    i = 0
    while i < GRIGLIA:
        if mid[i]:
            j = i
            while j + 1 < GRIGLIA and mid[j + 1]:
                j += 1
            for k in range(i, j + 1):
                cl[k] = 1 if (k - i) * passo < BREVE else 2
            i = j + 1
        else:
            i += 1
    return cl, L


def tempi_classi(profilo, cl):
    dt = np.diff(profilo.get('t_r', profilo['t']))
    return np.array([dt[cl == c].sum() for c in range(len(CLASSI))])


def analisi_classi(profili, team):
    """Per una sessione: classi della pista, peso di ogni classe (quota del tempo sul giro) e scarto %
    di ogni team in ogni classe dalla mediana dei team."""
    if len(profili) < 6:
        return None
    # giri rotti (interrotti, traffico, errori): oltre il 5% della mediana dei giri della sessione
    dmed = float(np.median([p['durata'] for _, _, p in profili]))
    profili = [x for x in profili if x[2]['durata'] <= 1.05 * dmed]
    if len(profili) < 6:
        return None
    registra(profili)
    cl, L = classi_pista(profili)
    tp = defaultdict(list)
    for dn, g, p in profili:
        tp[dn].append(tempi_classi(p, cl))
    pil = {dn: np.mean(v, axis=0) for dn, v in tp.items()}
    tot = np.median(np.array(list(pil.values())), axis=0)
    bt = defaultdict(list)
    for dn, v in pil.items():
        if team.get(dn):
            bt[team[dn]].append(v)
    tv = {t: np.mean(v, axis=0) for t, v in bt.items()}
    if len(tv) < 4:
        return None
    med = np.median(np.array(list(tv.values())), axis=0)
    scarti = {t: 100.0 * (v - med) / np.maximum(med, 1e-6) for t, v in tv.items()}
    pesi = tot / tot.sum()
    metri = np.array([(cl == c).sum() * L / GRIGLIA for c in range(len(CLASSI))])
    return dict(classi=cl, lunghezza=L, pesi=pesi, metri=metri, tempo=float(tot.sum()), scarti=scarti,
                giro={t: float(np.sum(v) - np.sum(med)) / float(np.sum(med)) * 100.0 for t, v in tv.items()})


def composizione_riferimento(storico):
    """Peso delle classi (quota del tempo sul giro) per ogni circuito dalle qualifiche 2024-2025."""
    per = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(storico, 'circuiti', '*.json.gz'))):
        try:
            d = carica_weekend(f)
        except Exception:  # noqa: BLE001
            continue
        w = d.get('weather') or []
        bag = bool(w) and sum(1 for x in w if x.get('rainfall')) / len(w) > 0.2
        prof = profili_sessione((d.get('energia') or {}).get('blocchi') or [])
        if len(prof) >= 3:
            per[d['meeting'].get('circuit_key')].append((bag, prof, d['meeting']))
    out = {}
    for ck, lst in per.items():
        asciutte = [x for x in lst if not x[0]] or lst
        pesi, metri = [], []
        for _, prof, _ in asciutte:
            registra(prof)
            cl, L = classi_pista(prof)
            tt = np.median(np.array([tempi_classi(p, cl) for _, _, p in prof]), axis=0)
            pesi.append(tt / tt.sum())
            metri.append(np.array([(cl == c).sum() * L / GRIGLIA for c in range(len(CLASSI))]))
        out[ck] = dict(nome=asciutte[-1][2].get('circuit_short_name'), pesi=np.mean(pesi, axis=0),
                       metri=np.mean(metri, axis=0))
    return out


# ---------------------------------------------------------------------------
# 6. Prova in sequenza: motore x circuito
# ---------------------------------------------------------------------------

def prova_interazione(eventi, obiettivo, profilo, indice, dal=5, emivita=EMIVITA, lam=1.0, verbose=False):
    """Prova in sequenza. Per ogni evento k (dal k-esimo): previsione dello scarto di ogni team
    (obiettivo(ev) -> team->valore) con
      A  la forma (media pesata degli scarti precedenti)
      B  la forma + beta * profilo motore del team * indice del circuito,
         con beta stimato sugli eventi precedenti (una sola pendenza per tutti i team)
    profilo(eventi_precedenti) -> team->valore; indice(ev) -> numero o None.
    Ritorna dict con errori e dettagli."""
    righe = []
    for k in range(len(eventi)):
        ev = eventi[k]
        y = obiettivo(ev)
        if not y or k < dal - 1:
            continue
        prec = eventi[:k]
        c = indice(ev)
        if c is None:
            continue
        forma, _ = media_recente([p for p in prec if obiettivo(p)], obiettivo, emivita)
        prof = profilo(prec)
        # beta dagli eventi precedenti: residuo della forma (calcolata con i soli eventi prima di ognuno)
        xs, rs = [], []
        for j in range(2, k):
            yj = obiettivo(eventi[j])
            cj = indice(eventi[j])
            if not yj or cj is None:
                continue
            fj, _ = media_recente([p for p in eventi[:j] if obiettivo(p)], obiettivo, emivita)
            pj = profilo(eventi[:j])
            for t, v in yj.items():
                if t in fj and t in pj:
                    xs.append(pj[t] * cj)
                    rs.append(v - fj[t])
        xs, rs = np.array(xs), np.array(rs)
        beta = float(np.sum(xs * rs) / (np.sum(xs * xs) + lam)) if len(xs) else 0.0
        for t, v in y.items():
            if t not in forma:
                continue
            pa = forma[t]
            pb = forma[t] + beta * prof.get(t, 0.0) * c
            righe.append(dict(round=ev['round'], circuito=ev['circuito'], team=t, vero=v, A=pa, B=pb,
                              beta=beta, prof=prof.get(t), c=c))
    return _riassunto(righe)


def _riassunto(righe, chiavi=('A', 'B')):
    if not righe:
        return None
    out = dict(n=len(righe), eventi=len({r['round'] for r in righe}), righe=righe,
               beta_finale=righe[-1].get('beta'))
    for k in chiavi:
        out['rmse_' + k] = float(np.sqrt(np.mean([(r['vero'] - r[k]) ** 2 for r in righe])))
    ev_ids = sorted({r['round'] for r in righe})
    da = {e: [] for e in ev_ids}
    for r in righe:
        da[r['round']].append((r['vero'] - r[chiavi[1]]) ** 2 - (r['vero'] - r[chiavi[0]]) ** 2)
    diffs = np.array([np.mean(v) for v in da.values()])
    rng = np.random.default_rng(5)
    bs = [np.mean(rng.choice(diffs, len(diffs))) for _ in range(4000)]
    out['diff_mse'] = float(np.mean(diffs))
    out['diff_int90'] = (float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))
    return out


def prova_classi(eventi, analisi, obiettivo, pesi_circuito, dal=5, emivita=EMIVITA, k0=2.0):
    """Prova in sequenza del modello per classi. analisi(ev) -> risultato di analisi_classi o None;
    obiettivo(ev) -> team->scarto % da prevedere; pesi_circuito(ev) -> pesi delle classi del circuito.
      A  forma (media pesata degli scarti precedenti)
      C  forma + somma sulle classi di (peso del circuito - peso medio dei circuiti passati) x scarto
         del team nella classe (media pesata, ristretta verso lo scarto sul giro con k0 eventi)"""
    righe = []
    for k in range(len(eventi)):
        ev = eventi[k]
        y = obiettivo(ev)
        w = pesi_circuito(ev)
        if not y or w is None or k < dal - 1:
            continue
        prec = [p for p in eventi[:k] if obiettivo(p)]
        forma, _ = media_recente(prec, obiettivo, emivita)
        num, den = defaultdict(lambda: np.zeros(len(CLASSI))), defaultdict(float)
        wnum, wden = np.zeros(len(CLASSI)), 0.0
        for i, p in enumerate(eventi[:k][::-1]):
            a = analisi(p)
            if not a:
                continue
            q = 0.5 ** (i / emivita)
            for t, s in a['scarti'].items():
                num[t] += q * s
                den[t] += q
            wp = pesi_circuito(p)
            if wp is not None:
                wnum += q * wp
                wden += q
        if wden == 0:
            continue
        wbar = wnum / wden
        for t, v in y.items():
            if t not in forma:
                continue
            c = forma[t]
            if den.get(t):
                r = num[t] / den[t]
                # restringimento delle classi verso la media del team (scarto uniforme sul giro)
                rg = float(np.dot(wbar, r))
                r = rg + (r - rg) * den[t] / (den[t] + k0)
                c = forma[t] + float(np.dot(w - wbar, r))
            righe.append(dict(round=ev['round'], circuito=ev['circuito'], team=t, vero=v, A=forma[t], B=c))
    return _riassunto(righe)


# ---------------------------------------------------------------------------
# 7. Scheda del circuito
# ---------------------------------------------------------------------------

def _fmt(x, nd=1, segno=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '-'
    return f'{x:+.{nd}f}' if segno else f'{x:.{nd}f}'


def prova_dopo_qualifica(eventi, C, sessione='Race|gara', dal=5, emivita=EMIVITA, lam=3.0):
    """Passo gara dei team previsto dopo la qualifica, come nel modello di gara: regressione sugli eventi
    precedenti di passo ~ 1 + distacco in qualifica + forma (A) e, in più, lo spostamento per classi
    di pista del circuito (B). Tutto a livello di team, scarti dalla mediana in %."""
    def spostamenti(j):
        pr, _, _ = proiezione(eventi[:j], C, eventi[j]['circuit_key'], sessione, emivita)
        return {t: v[0] for t, v in pr.items()}

    cache = {}

    def riga_evento(j):
        if j in cache:
            return cache[j]
        e = eventi[j]
        y, q = team_passo(e), team_gap_qualifica(e)
        f, _ = media_recente([p for p in eventi[:j] if team_passo(p)], team_passo, emivita)
        s = spostamenti(j)
        rows = [(t, y[t], q[t], f[t], s.get(t, 0.0)) for t in y if t in q and t in f]
        cache[j] = rows
        return rows

    def fit(X, y):
        A = np.vstack([X, np.sqrt(lam) * np.eye(X.shape[1])])
        b = np.concatenate([y, np.zeros(X.shape[1])])
        return np.linalg.lstsq(A, b, rcond=None)[0]

    righe = []
    for k in range(len(eventi)):
        if k < dal - 1 or not team_passo(eventi[k]):
            continue
        tr = [r for j in range(1, k) for r in riga_evento(j)]
        if len(tr) < 20:
            continue
        T = np.array([r[1:] for r in tr], float)
        y, q, f, s = T.T
        one = np.ones_like(y)
        ba = fit(np.c_[one, q, f], y)
        bb = fit(np.c_[one, q, f, s], y)
        for t, yy, qq, ff, ss in riga_evento(k):
            righe.append(dict(round=eventi[k]['round'], circuito=eventi[k]['circuito'], team=t, vero=yy,
                              A=float(ba @ [1, qq, ff]), B=float(bb @ [1, qq, ff, ss]), beta=float(bb[3])))
    return _riassunto(righe)


def _calibra(coppie):
    """Retta y = a + b x dalle coppie (x, y); con meno di 3 coppie solo lo spostamento medio."""
    if not coppie:
        return lambda x: x
    x = np.array([c[0] for c in coppie], float)
    y = np.array([c[1] for c in coppie], float)
    if len(coppie) < 3 or np.std(x) < 1e-9:
        dm = float(np.mean(y - x))
        return lambda v: v + dm
    b, a0 = np.polyfit(x, y, 1)
    return lambda v: a0 + b * v


class Circuiti:
    """Misure e composizione dei circuiti sulla stessa scala (auto 2024-2025). Le piste senza edizioni
    2024-2025 (Madrid, Sepang) le prendono dal 2026, riportate sulla scala di riferimento con una retta
    stimata sulle piste che hanno tutte e due."""

    def __init__(self, storico, eventi):
        self.rif = circuiti_riferimento(storico)
        self.comp = composizione_riferimento(storico)
        self.eventi = {e['circuit_key']: e for e in eventi if not e['parziale']}
        self._cal = {}

    def misura(self, ck, chiave):
        r = self.rif.get(ck)
        if r and r['misure'].get(chiave) is not None:
            return r['misure'][chiave]
        ev = self.eventi.get(ck)
        m = circuito_2026(ev) if ev else None
        if not m or m.get(chiave) is None:
            return None
        if chiave not in self._cal:
            cop = []
            for k, e in self.eventi.items():
                m2 = circuito_2026(e)
                r2 = self.rif.get(k)
                if m2 and r2 and m2.get(chiave) is not None and r2['misure'].get(chiave) is not None:
                    cop.append((m2[chiave], r2['misure'][chiave]))
            self._cal[chiave] = _calibra(cop)
        return float(self._cal[chiave](m[chiave]))

    def indice(self, chiave):
        """Funzione ev -> scarto standardizzato della misura (media e deviazione sulle piste di riferimento)."""
        vals = [r['misure'].get(chiave) for r in self.rif.values() if r['misure'].get(chiave) is not None]
        if len(vals) < 3:
            return lambda ev: None
        mu, sd = float(np.mean(vals)), float(np.std(vals)) or 1.0

        def f(ev):
            v = self.misura(ev['circuit_key'], chiave)
            return None if v is None else (v - mu) / sd
        return f

    def pesi(self, ck):
        c = self.comp.get(ck)
        if c is not None:
            return np.asarray(c['pesi'])
        ev = self.eventi.get(ck)
        a = (ev or {}).get('classi', {}).get('Qualifying|migliore')
        if not a:
            return None
        # stesso spostamento medio fra 2026 e riferimento delle piste che hanno tutte e due
        diffs = []
        for k, e in self.eventi.items():
            a2 = e['classi'].get('Qualifying|migliore')
            if a2 and k in self.comp:
                diffs.append(np.asarray(self.comp[k]['pesi']) - a2['pesi'])
        w = a['pesi'] + (np.mean(diffs, axis=0) if diffs else 0.0)
        w = np.clip(w, 0.0, None)
        return w / w.sum()

    def nome(self, ck):
        r = self.rif.get(ck) or self.comp.get(ck)
        if r:
            return r['nome']
        ev = self.eventi.get(ck)
        return ev['circuito'] if ev else str(ck)


def _diff(a, b):
    return {t: a[t] - b[t] for t in a if t in b}


def analisi_differenza(ev):
    """Classi: scarti in gara meno scarti in qualifica (stesso evento)."""
    q, r = ev['classi'].get('Qualifying|migliore'), ev['classi'].get('Race|gara')
    if not q or not r:
        return None
    return dict(scarti={t: r['scarti'][t] - q['scarti'][t] for t in r['scarti'] if t in q['scarti']})


def _senza_ev(e, escl):
    """Copia leggera dell'evento senza i piloti dei team esclusi."""
    e2 = dict(e)
    e2['team'] = {d: t for d, t in e['team'].items() if t not in escl}
    return e2


def senza_team(obiettivo, escl):
    return lambda e: {t: v for t, v in (obiettivo(e) or {}).items() if t not in escl}


TOP = ('Mercedes', 'Ferrari', 'McLaren', 'Red Bull Racing')


def esegui_prova(storico, out=None, dal=5, verbose=True):
    """Tutte le prove in sequenza sul 2026. Ritorna un dict con i risultati (e lo scrive in out)."""
    eventi = [e for e in carica_stagione(storico) if not e['parziale']]
    C = Circuiti(storico, eventi)
    pesi = lambda ev: C.pesi(ev['circuit_key'])  # noqa: E731
    ris = {'eventi': [e['circuito'] for e in eventi], 'prove': []}

    def riga(nome, obiettivo_nome, r, extra=''):
        if not r:
            return
        top = [x for x in r['righe'] if x['team'] in TOP]
        rt = {k: float(np.sqrt(np.mean([(x['vero'] - x[k]) ** 2 for x in top]))) for k in ('A', 'B')} if top else {}
        ris['prove'].append(dict(prova=nome, obiettivo=obiettivo_nome, n=r['n'], eventi=r['eventi'],
                                 rmse_forma=r['rmse_A'], rmse_modello=r['rmse_B'], diff_mse=r['diff_mse'],
                                 int90=r['diff_int90'], beta=r.get('beta_finale'),
                                 rmse_top_forma=rt.get('A'), rmse_top_modello=rt.get('B')))
        if verbose:
            print(f"{nome:<40}{obiettivo_nome:<11}{r['n']:>4}{r['rmse_A']:>8.3f}{r['rmse_B']:>8.3f}"
                  f"{r['diff_mse']:>+9.4f} ({r['diff_int90'][0]:+.4f};{r['diff_int90'][1]:+.4f})"
                  + (f"   top4 {rt['A']:.3f}->{rt['B']:.3f}" if rt else '') + extra)

    if verbose:
        print(f'Prova in sequenza dal {dal}° weekend: {len(eventi)} weekend 2026, '
              f'{len(C.rif)} circuiti di riferimento')
        print(f"{'modello':<40}{'obiettivo':<11}{'n':>4}{'forma':>8}{'model.':>8}{'diff MSE':>9} (int. 90%)")
    obiettivi = (('qualifica', team_gap_qualifica), ('passo gara', team_passo), ('gara-quali', team_gara_meno_qualifica))
    # 1. classi di pista
    for escl in ((), ('Aston Martin',)):
        tag = '' if not escl else ' senza AM'
        for on, ob in obiettivi:
            ob2 = senza_team(ob, escl)
            r = prova_classi(eventi, lambda e: e['classi'].get('Qualifying|migliore'), ob2, pesi, dal=dal)
            riga('classi (qualifica)' + tag, on, r)
        r = prova_classi(eventi, lambda e: e['classi'].get('Race|gara'), senza_team(team_passo, escl), pesi, dal=dal)
        riga('classi (gara)' + tag, 'passo gara', r)
        r = prova_classi(eventi, analisi_differenza, senza_team(team_gara_meno_qualifica, escl), pesi, dal=dal)
        riga('classi (gara - qualifica)' + tag, 'gara-quali', r)
    # 1b. passo gara dopo la qualifica (come nel modello di gara): con e senza lo spostamento per classi
    for sess in ('Race|gara', 'Qualifying|migliore'):
        for escl in ((), ('Aston Martin',)):
            ev2 = eventi
            r = prova_dopo_qualifica(ev2, C, sess, dal=dal) if not escl else None
            if escl:
                # stessa prova senza il team: si tolgono le sue righe dagli eventi
                r = prova_dopo_qualifica([_senza_ev(e, escl) for e in eventi], C, sess, dal=dal)
            riga(f"dopo qualifica + classi {sess.split('|')[0][:4]}" + (' senza AM' if escl else ''), 'passo gara', r,
                 f"   peso {r['beta_finale']:+.2f}" if r else '')
    # 2. profilo del team x indice del circuito
    profili = [('trappola Q', lambda prec: media_recente(prec, team_trappola)[0]),
               ('clip_quota R-Q', lambda prec: media_recente(prec, lambda e: _diff(
                   team_misura(e, 'clip_quota', 'Race', 'gara'), team_misura(e, 'clip_quota')))[0]),
               ('coast R', lambda prec: media_recente(prec, lambda e: team_misura(e, 'coast', 'Race', 'gara'))[0])]
    for k in ('vmax', 'spinta', 'clip_quota', 'calo'):
        profili.append((f'{k} Q', lambda prec, k=k: media_recente(prec, lambda e: team_misura(e, k))[0]))
    for chiave in ('quota_pieno', 'esposizione_km'):
        idx = C.indice(chiave)
        for pn, pf in profili:
            for escl in ((), ('Aston Martin',)):
                for on, ob in obiettivi:
                    r = prova_interazione(eventi, senza_team(ob, escl), pf, idx, dal=dal)
                    riga(f"{pn} x {chiave}" + (' senza AM' if escl else ''), on, r,
                         f"   beta {r['beta_finale']:+.4f}" if r else '')
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'energia_prova.json'), 'w', encoding='utf-8') as f:
            json.dump(ris, f, ensure_ascii=False, indent=1)
    return ris


def proiezione(eventi, C, ck, sessione='Qualifying|migliore', emivita=EMIVITA, k0=2.0):
    """Scarto per classi dei team (media pesata, ristretta) e spostamento previsto sul circuito ck
    rispetto ai circuiti recenti: team -> (spostamento %, scarti per classe)."""
    w = C.pesi(ck)
    if w is None:
        return {}, None, None
    num, den = defaultdict(lambda: np.zeros(len(CLASSI))), defaultdict(float)
    wnum, wden = np.zeros(len(CLASSI)), 0.0
    for i, p in enumerate(eventi[::-1]):
        a = p['classi'].get(sessione)
        if not a:
            continue
        q = 0.5 ** (i / emivita)
        for t, s in a['scarti'].items():
            num[t] += q * s
            den[t] += q
        wp = C.pesi(p['circuit_key'])
        if wp is not None:
            wnum += q * wp
            wden += q
    if not wden:
        return {}, w, None
    wbar = wnum / wden
    out = {}
    for t in num:
        r = num[t] / den[t]
        rg = float(np.dot(wbar, r))
        r = rg + (r - rg) * den[t] / (den[t] + k0)
        out[t] = (float(np.dot(w - wbar, r)), r)
    return out, w, wbar


def scheda(storico, rnd=None, circuito=None, anno=2026):
    eventi = carica_stagione(storico, anno)
    fatti = [e for e in eventi if not e['parziale'] and (rnd is None or e['round'] < rnd)]
    C = Circuiti(storico, fatti)
    print(f'Modello energia — {len(fatti)} weekend {anno} prima del round {rnd or "-"}, '
          f'{len(C.rif)} circuiti di riferimento')
    ck = None
    corrente = next((e for e in eventi if e['round'] == rnd), None) if rnd else None
    if corrente:
        ck = corrente['circuit_key']
    elif circuito:
        c = circuito.lower()
        for k, r in C.rif.items():
            if c in (r['nome'] or '').lower() or c in (r['luogo'] or '').lower():
                ck = k
                break
    nome = C.nome(ck) if ck is not None else (circuito or '?')
    # 1. circuito
    keys = (('quota_pieno', '% gas pieno', 1), ('esposizione_km', 'rett. lunghi m/km', 0),
            ('run_max', 'rett. max m', 0), ('frenate_km', 'frenate kJ/kg/km', 2), ('v_media', 'v media', 0))
    print('\n== Circuiti del calendario (auto 2024-2025: giro migliore dei primi 6 in qualifica)')
    print('circuito'.ljust(20) + ''.join(f'{d:>18}' for _, d, _ in keys) + '   curve/rett.brevi/rett.lunghi (% tempo)')
    cks = set(C.rif) | set(C.eventi)
    ordine = sorted(cks, key=lambda k: -(C.misura(k, 'esposizione_km') or 0))
    for k in ordine:
        w = C.pesi(k)
        mark = '  <==' if k == ck else ''
        print(f"{C.nome(k)[:19]:<20}" + ''.join(f"{_fmt(C.misura(k, x), nd, False):>18}" for x, _, nd in keys)
              + ('   ' + '/'.join(f'{100 * x:.0f}' for x in w) if w is not None else '') + mark)
    # 2. profili dei team
    print('\n== Team in qualifica: scarti dalla mediana dei team (media pesata sui weekend recenti)')
    cols = [(k, media_recente(fatti, lambda e, k=k: team_misura(e, k))[0]) for k, _, _ in MISURE_TEAM]
    trap, _ = media_recente(fatti, team_trappola)
    gap, _ = media_recente(fatti, team_gap_qualifica)
    teams = sorted(gap, key=lambda t: gap[t])
    print('team'.ljust(16) + f"{'distacco %':>11}{'trappola':>9}" + ''.join(f'{k[:11]:>12}' for k, _ in cols))
    for t in teams:
        print(f'{t[:15]:<16}{_fmt(gap.get(t), 2):>11}{_fmt(trap.get(t), 1):>9}'
              + ''.join(f"{_fmt(p.get(t), 2):>12}" for _, p in cols))
    print('\n== Team in gara (blocchi di 4 giri): scarti dalla mediana e differenza con la qualifica')
    colsr = [(k, media_recente(fatti, lambda e, k=k: team_misura(e, k, 'Race', 'gara'))[0]) for k, _, _ in MISURE_TEAM]
    passo, _ = media_recente(fatti, team_passo)
    print('team'.ljust(16) + f"{'passo %':>9}" + ''.join(f'{k[:11]:>12}' for k, _ in colsr))
    for t in teams:
        print(f'{t[:15]:<16}{_fmt(passo.get(t), 2):>9}'
              + ''.join(f"{_fmt(p.get(t), 2):>12}" for _, p in colsr))
    # 3. il weekend in corso (libere e qualifica già nell'archivio)
    if corrente and corrente['classi']:
        print(f"\n== {nome}, questo weekend: scarti dei team per classe di pista (% del tempo della classe)")
        print('sessione'.ljust(22) + 'team'.ljust(16) + ''.join(f'{c:>12}' for c in CLASSI) + f"{'giro %':>9}")
        for chiave, a in corrente['classi'].items():
            for t in sorted(a['giro'], key=lambda t: a['giro'][t]):
                print(f"{chiave.replace('|', ' ')[:21]:<22}{t[:15]:<16}"
                      + ''.join(f'{x:>+12.2f}' for x in a['scarti'][t]) + f"{a['giro'][t]:>+9.2f}")
        for sess, tipo in (('Practice 1', 'migliore'), ('Practice 2', 'migliore'), ('Practice 3', 'migliore'),
                           ('Qualifying', 'migliore'), ('Practice 2', 'long_run'), ('Practice 1', 'long_run')):
            if not (corrente['misure'].get(sess)):
                continue
            vals = [(k, team_misura(corrente, k, sess, tipo)) for k, _, _ in MISURE_TEAM]
            if not any(v for _, v in vals):
                continue
            print(f"\n   {sess} ({tipo}): scarti dalla mediana dei team")
            print('   ' + 'team'.ljust(16) + ''.join(f'{k[:11]:>12}' for k, _ in vals))
            for t in sorted(vals[0][1], key=lambda t: -vals[0][1][t]):
                print('   ' + f'{t[:15]:<16}' + ''.join(f"{_fmt(v.get(t), 2):>12}" for _, v in vals))
    # 4. proiezione sul circuito
    if ck is not None:
        for sess, titolo in (('Qualifying|migliore', 'qualifica'), ('Race|gara', 'gara')):
            pr, w, wbar = proiezione(fatti, C, ck, sess)
            if not pr:
                continue
            print(f'\n== {nome}: dove si perde il tempo ({titolo}) e spostamento previsto rispetto ai circuiti recenti')
            print(f"pesi del circuito (curve/rett.brevi/rett.lunghi): {'/'.join(f'{100 * x:.0f}' for x in w)} %"
                  f" — circuiti recenti {'/'.join(f'{100 * x:.0f}' for x in wbar)} %")
            print('team'.ljust(16) + ''.join(f'{c:>12}' for c in CLASSI) + f"{'spostamento':>13}")
            for t in sorted(pr, key=lambda t: pr[t][0]):
                s, r = pr[t]
                print(f'{t[:15]:<16}' + ''.join(f'{x:>+12.2f}' for x in r) + f'{s:>+12.3f}%')
    return eventi, C


# ---------------------------------------------------------------------------
# 8. Telemetria di Muretto (telemetry.json): i giri migliori dei primi 4 team nel weekend
# ---------------------------------------------------------------------------

def da_traccia(v, th, br, lunghezza, tempo):
    """Giro di telemetry.json (velocità, gas, freno su punti equidistanti lungo la pista) riportato sul
    tempo: t, v, gas, freno, con la durata riportata al tempo ufficiale del giro."""
    v = np.maximum(np.asarray(v, float), 5.0)
    n = len(v)
    ds = lunghezza / n
    vm = (v + np.roll(v, -1)) / 2
    dt = ds / (vm / 3.6)
    t = np.concatenate([[0.0], np.cumsum(dt[:-1])])
    tot = float(np.sum(dt))
    if tempo:
        t = t * tempo / tot
        tot = tempo
    return t, v, np.asarray(th, float), np.asarray(br, float), tot


def telemetria_muretto(path):
    """Misure energetiche e tempo per classi di pista dei giri di telemetry.json, sessione per sessione."""
    d = json.load(open(path, encoding='utf-8'))
    teams = {k: v['name'] for k, v in (d.get('info') or {}).get('teams', {}).items()}
    dteam = {c: teams.get(x.get('team'), x.get('team')) for c, x in (d.get('info') or {}).get('drivers', {}).items()}
    out = []
    for S in d.get('sessions') or []:
        L = float(S.get('length') or 5000)
        righe, profili = {}, []
        for code, x in (S.get('drivers') or {}).items():
            if not x.get('v'):
                continue
            t, v, th, br, dur = da_traccia(x['v'], x['th'], x['br'], L, x.get('lap'))
            m = misure_giro(t, v, th, br)
            if m is None:
                continue
            m['tempo'] = x.get('lap')
            m['gomma'] = x.get('tyre')
            righe[code] = m
            p = profilo_distanza(np.r_[t, dur], np.r_[v, v[0]], np.r_[th, th[0]], dur)
            if p is not None:
                profili.append((code, 1, p))
        a = None
        if len(profili) >= 4:
            registra(profili)
            cl, _ = classi_pista(profili, L)
            tc = {code: tempi_classi(p, cl) for code, _, p in profili}
            med = np.median(np.array(list(tc.values())), axis=0)
            a = dict(classi=cl, tempi={c: v - med for c, v in tc.items()},
                     metri=np.array([(cl == c).sum() * L / GRIGLIA for c in range(len(CLASSI))]))
        out.append(dict(sessione=S.get('label') or S.get('name'), misure=righe, classi=a, team=dteam))
    return out


def stampa_telemetria(path):
    for s in telemetria_muretto(path):
        print(f"\n== {s['sessione']}: giri migliori (scarti dalla mediana dei piloti presenti)")
        m = s['misure']
        if not m:
            continue
        keys = [k for k, _, _ in MISURE_TEAM]
        med = {k: np.median([x[k] for x in m.values() if x.get(k) is not None] or [np.nan]) for k in keys}
        intest = 'pilota'.ljust(18) + f"{'giro':>9}{'gomma':>6}" + ''.join(f'{k[:11]:>12}' for k in keys)
        if s['classi']:
            intest += ''.join(f'{c + " s":>13}' for c in CLASSI)
        print(intest)
        for code in sorted(m, key=lambda c: m[c].get('tempo') or 1e9):
            x = m[code]
            riga = (f"{code} {(s['team'].get(code) or '')[:12]:<14}{x.get('tempo') or 0:>9.3f}{x.get('gomma') or '-':>6}"
                    + ''.join(f"{_fmt(None if x.get(k) is None else x[k] - med[k], 2):>12}" for k in keys))
            if s['classi'] and code in s['classi']['tempi']:
                riga += ''.join(f'{v:>+13.3f}' for v in s['classi']['tempi'][code])
            print(riga)
        if s['classi']:
            print('metri per classe: ' + ', '.join(f'{c} {mm:.0f}' for c, mm in zip(CLASSI, s['classi']['metri'])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('scheda')
    a1.add_argument('--storico', default='storico')
    a1.add_argument('--round', type=int)
    a1.add_argument('--circuito')
    a2 = sub.add_parser('prova')
    a2.add_argument('--storico', default='storico')
    a2.add_argument('--out', default='risultati')
    a2.add_argument('--dal', type=int, default=5)
    a3 = sub.add_parser('telemetria')
    a3.add_argument('--file', default='telemetry.json')
    a = ap.parse_args()
    if a.cmd == 'scheda':
        scheda(a.storico, a.round, a.circuito)
    elif a.cmd == 'telemetria':
        stampa_telemetria(a.file)
    else:
        esegui_prova(a.storico, a.out, a.dal)
    return 0


if __name__ == '__main__':
    sys.exit(main())
