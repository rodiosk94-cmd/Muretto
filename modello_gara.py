#!/usr/bin/env python3
"""
modello_gara.py — Muretto: modello di gara F1 (vittoria e podio) e prova a ritroso.

Un solo file, solo numpy (Python 3.9+). Legge l'archivio OpenF1 di Muretto (storico/).

Uso:
  python modello_gara.py prova   --storico storico --out risultati
      prova a ritroso sul 2026: ogni gara e Sprint prevista con i soli dati precedenti;
      scrive risultati/prova_righe.csv e risultati/prova_riepilogo.json
  python modello_gara.py prevedi --storico storico --round 18 --tipo R \
         [--ricalibra risultati/prova_righe.csv] [--griglia "VER,NOR,..."] [--pioggia 0.2]
      previsione dopo la qualifica (o dopo la Sprint Shootout con --tipo S) di un weekend
      dell'archivio; il weekend in corso ci arriva con backfill_fetch.py (file "parziale")
  --shift-sorpassi -1   (prova e prevedi) variante con sorpassi più facili di un livello
  --gomme               (prova e prevedi) simula mescole e strategie con modello_gomme.py
                        (sperimentale: nella prova 2026 non migliora vittoria e podio)
  Con modello_gomme.py nella stessa cartella, prevedi stampa anche degrado e strategie previste.

Componenti (parametri stimati solo con eventi precedenti al via):
  griglia        griglia ufficiale (penalità comprese), DNS tolti; Sprint: ultime posizioni
                 prima del via
  passo          regressione sui giri puliti di ogni corsa asciutta (pilota, giro, mescola,
                 età gomma; aria libera, niente SC/VSC) -> passo in % per pilota.
                 Previsione: forma (team + scarto dal compagno, peso ai più recenti) e
                 distacco in qualifica, con una regressione rifatta prima di ogni corsa.
                 Ordine tra i primi 8 in qualifica: probit sulle coppie (distacco, log della
                 posizione in qualifica, forma). Errore con parte comune al team, code larghe.
  partenza       primo giro: rumore normale + incidente (stimati sulle coppie vicine dei
                 primi 10), rumore proprio della pole, scatto (metri a 5 s dal via) ristretto
  ritiri e guai  tasso per team ristretto verso la media; giri a potenza ridotta
  SC e VSC       tassi separati aggiornati gara per gara, cittadini a parte, storia del
                 circuito se c'è; SC ricompatta il gruppo, VSC no; sosta scontata in tutte e due
  soste          piano comune (una o due soste) con scarti individuali, copertura
                 dell'undercut, degrado effettivo, giro di uscita, perdita ai box del circuito
  sorpassi       difficoltà per circuito (tabella a priori, spostabile con --shift-sorpassi),
                 vantaggio di passo, aria sporca
  pioggia        rischio pioggia dal messaggio FIA prima del via (o da --pioggia)
  gomme          (solo con --gomme) mescole, degrado con crollo e piani di sosta da modello_gomme.py
  ricalibrazione temperatura sulle probabilità di vittoria e logistica sul podio, stimate
                 sulle previsioni passate (opzionale)
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

try:  # modello gomme (stesso repo): mescole, degrado, crollo, strategie
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import modello_gomme as G
except ImportError:  # senza il file il modello usa il degrado unico della versione 3
    G = None

# ---------------------------------------------------------------------------
# 0. Costanti e tabelle a priori
# ---------------------------------------------------------------------------

# Difficoltà di sorpasso per circuito (0 facile ... 4 quasi impossibile).
# Conoscenza a priori, disponibile prima del weekend.
SORPASSO = {
    'Melbourne': 2, 'Shanghai': 1, 'Suzuka': 2, 'Miami Gardens': 2, 'Montréal': 1,
    'Monte Carlo': 4, 'Barcelona': 2, 'Spielberg': 1, 'Silverstone': 1,
    'Spa-Francorchamps': 0, 'Budapest': 3, 'Zandvoort': 3, 'Monza': 0, 'Madrid': 2,
    'Baku': 0, 'Kuala Lumpur': 1, 'Marina Bay': 3, 'Austin': 1, 'Mexico City': 2,
    'São Paulo': 1, 'Las Vegas': 1, 'Lusail': 1, 'Yas Island': 2, 'Jeddah': 1,
    'Sakhir': 0, 'Imola': 3,
}
CITTADINI = {'Melbourne', 'Miami Gardens', 'Montréal', 'Monte Carlo', 'Madrid',
             'Baku', 'Marina Bay', 'Las Vegas', 'Jeddah'}

# Parametri strutturali (a priori; vedi documento dei risultati)
P = dict(
    n_sim=8000,               # simulazioni per corsa (quelle della prova nel documento dei risultati)
    # passo
    emivita_forma=4.0,        # gare: peso 1/2 dopo 4 eventi
    shrink_pilota=2.0,        # peso (eventi recenti) che serve per metà fiducia nello scarto dal compagno
    sigma_pace_min=0.12,      # % minimo di errore della stima di passo
    rumore_giro=0.30,         # s, rumore giro per giro
    errore_giro_p=0.010,      # probabilità di errore (perdita 1.5-5 s) per giro
    # partenza
    metri_per_posizione=8.0,  # distanza tra due caselle
    peso_scatto=0.9,          # posizioni guadagnate ogni 8 m di scatto in più (dati 2026: 1 ogni 9 m)
    gap_lap1=0.75,            # s tra due auto dopo il primo giro
    # gomme e soste
    quota_degrado=0.45,       # frazione del degrado lordo che separa gomma nuova e usata
    giro_uscita=1.0,          # s persi nel giro di uscita (gomme fredde)
    aria_sporca=0.15,         # s/giro persi a meno di 1 s da chi precede (Singapore 2026)
    usura_scia=1.6,           # la gomma invecchia 1,6 volte più in fretta in scia
    rumore_box=0.4,           # s, variabilità della sosta
    p_due_soste=0.25,
    p_altra_strategia=0.10,   # quota di piloti che sceglie l'altro piano (una/due soste)
    finestra_sosta=(0.35, 0.62),
    perdita_box_default=22.0,
    sc_sconto_box=0.50,       # costo della sosta sotto SC (frazione)
    vsc_sconto_box=0.55,
    # neutralizzazioni
    sc_prior=(0.8, 3.0),      # (tasso a priori per gara, peso in gare)
    vsc_prior=(0.7, 3.0),
    mult_cittadino=1.35,
    sc_giri=(3, 6),
    vsc_giri=(1, 3),
    gap_sc=0.8,
    # ritiri e guai
    dnf_prior_peso=20.0,      # partenze equivalenti per restringere il tasso di team
    dnf_quota_giro1=0.08,     # quota dei ritiri al primo giro (2026: 7%)
    p_guaio=0.06, guaio_giri=(3, 10), guaio_perdita=(0.3, 1.0),
    # pioggia
    sigma_bagnato=0.45,       # % di rumore in più sul passo in caso di gara bagnata
    mult_sc_bagnato=1.6, mult_dnf_bagnato=1.4,
    # sorpassi: per livello di difficoltà (soglia s/giro, probabilità massima per giro)
    sorpasso=[(0.20, 0.70), (0.35, 0.55), (0.55, 0.40), (0.85, 0.25), (2.00, 0.05)],
    sorpasso_shift=0.0,       # spostamento di tutti i circuiti nella tabella (negativo = più facile)
    gomme=False,              # simulazione con mescole e strategie del modello gomme (sperimentale: nella
                              # prova 2026 non migliora vittoria e podio, vedi documento)
    gap_min=0.35,
)


def _t(s):
    if s is None:
        return None
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


# ---------------------------------------------------------------------------
# 1. Caricamento ed estrazione
# ---------------------------------------------------------------------------

def carica_weekend(path):
    with gzip.open(path) as f:
        return json.load(f)


def _driver_info(sess):
    return {d['driver_number']: (d['name_acronym'], d['team_name']) for d in sess['drivers']}


def distacchi_qualifica(sess):
    """Distacco % dal più veloce, con le fasi concatenate (Q1->Q2->Q3)."""
    res = sess['session_result']
    t = {}
    for r in res:
        dur = r.get('duration')
        if isinstance(dur, list):
            t[r['driver_number']] = [x if isinstance(x, (int, float)) and x > 0 else None for x in dur]
        elif isinstance(dur, (int, float)):
            t[r['driver_number']] = [dur]
    if not t:
        return {}, None
    nseg = max(len(v) for v in t.values())
    impr = []
    for s in range(nseg - 1):
        dd = [v[s] - v[s + 1] for v in t.values() if len(v) > s + 1 and v[s] and v[s + 1]]
        impr.append(float(np.median(dd)) if dd else 0.0)
    adj = {}
    for dn, v in t.items():
        # tempo dell'ultima fase disputata, riportato alla fase finale (ordine ufficiale)
        last = max([i for i, x in enumerate(v) if x] or [-1])
        if last >= 0:
            adj[dn] = v[last] - sum(impr[last:])
    if not adj:
        return {}, None
    m = min(adj.values())
    return {dn: 100.0 * (a - m) / m for dn, a in adj.items()}, m


def intervalli_neutralizzati(sess, giro_rif):
    """Intervalli di tempo (inizio, fine) con SC, VSC o bandiera rossa."""
    iv = []
    aperto = {}
    for m in sorted(sess['race_control'], key=lambda x: x['date']):
        msg = (m['message'] or '').upper()
        t = _t(m['date'])
        if msg == 'SAFETY CAR DEPLOYED':
            aperto['SC'] = t
        elif msg.startswith('SAFETY CAR IN THIS LAP') and 'SC' in aperto:
            iv.append((aperto.pop('SC'), t + 1.5 * giro_rif, 'SC'))
        elif msg in ('VSC DEPLOYED', 'VIRTUAL SAFETY CAR DEPLOYED'):
            aperto['VSC'] = t
        elif msg in ('VSC ENDING', 'VIRTUAL SAFETY CAR ENDING') and 'VSC' in aperto:
            iv.append((aperto.pop('VSC'), t + 30.0, 'VSC'))
        elif m.get('flag') == 'RED' or msg.startswith('RED FLAG'):
            for k in ('SC', 'VSC'):
                if k in aperto:
                    iv.append((aperto.pop(k), t, k))
            aperto.setdefault('RED', t)
        elif m.get('category') == 'SessionStatus' and msg == 'SESSION STARTED' and 'RED' in aperto:
            iv.append((aperto.pop('RED'), t + 2.5 * giro_rif, 'RED'))
    for k, t in aperto.items():
        iv.append((t, t + 1e5, k))
    return iv


def conta_neutralizzazioni(sess):
    sc = vsc = red = 0
    redt = []
    for m in sess['race_control']:
        msg = (m['message'] or '').upper()
        if msg == 'SAFETY CAR DEPLOYED':
            sc += 1
        elif msg in ('VSC DEPLOYED', 'VIRTUAL SAFETY CAR DEPLOYED'):
            vsc += 1
        elif m.get('flag') == 'RED' or msg.startswith('RED FLAG'):
            redt.append(_t(m['date']))
    # più messaggi di bandiera rossa a pochi minuti = una sola sospensione
    redt.sort()
    for i, t in enumerate(redt):
        if i == 0 or t - redt[i - 1] > 600:
            red += 1
    return sc, vsc, red


def rischio_pioggia(sess, t_via):
    """Ultimo messaggio 'RISK OF RAIN' prima del via, in [0,1]."""
    best = None
    for m in sess['race_control']:
        msg = (m['message'] or '').upper()
        if 'RISK OF RAIN' in msg and '%' in msg:
            t = _t(m['date'])
            if t is not None and t_via is not None and t > t_via:
                continue
            try:
                v = float(msg.split(' IS ')[-1].replace('%', '').strip()) / 100.0
            except ValueError:
                continue
            best = v
    return best


def stint_giri(sess, giri_pilota, dn):
    """Mescola ed età gomma per giro, con gli stint ricostruiti dai giri di uscita dai box.
    Un tratto senza una voce di stint che inizia esattamente lì resta senza mescola
    (l'export a volte non azzera lo stint alla sosta o perde il primo stint)."""
    out_laps = sorted(l['lap_number'] for l in giri_pilota if l.get('is_pit_out_lap') and l['lap_number'] > 1)
    starts = [1] + out_laps
    st = [s for s in sess['stints'] if s['driver_number'] == dn]
    mapping = {}
    maxlap = max([l['lap_number'] for l in giri_pilota] or [0])
    for i, s0 in enumerate(starts):
        s1 = starts[i + 1] - 1 if i + 1 < len(starts) else maxlap
        cand = [s for s in st if s['lap_start'] == s0]
        comp = cand[0]['compound'] if cand else None
        age0 = (cand[0].get('tyre_age_at_start') or 0) if cand else 0
        for lap in range(s0, s1 + 1):
            mapping[lap] = (comp, age0 + lap - s0)
    return mapping, True


def passo_gara(sess, min_giri=5):
    """Passo per pilota in % (negativo = più veloce della mediana) da giri puliti.
    Ritorna (dict pilota->%, info) oppure ({}, info) se sessione bagnata o dati scarsi."""
    laps = [l for l in sess['laps'] if l.get('lap_duration') and l.get('date_start')]
    info = {'bagnato': False, 'n_giri': 0}
    if not laps:
        return {}, info
    wet_laps = 0
    tot_laps = 0
    for s in sess['stints']:
        n = (s.get('lap_end') or 0) - (s.get('lap_start') or 0) + 1
        tot_laps += max(n, 0)
        if s.get('compound') in ('INTERMEDIATE', 'WET'):
            wet_laps += max(n, 0)
    if tot_laps and wet_laps / tot_laps > 0.10:
        info['bagnato'] = True
        return {}, info
    med = float(np.median([l['lap_duration'] for l in laps]))
    iv = intervalli_neutralizzati(sess, med)
    # tempi di passaggio sul traguardo di tutte le auto
    cross = np.array(sorted(_t(l['date_start']) for l in laps))
    pit_laps = defaultdict(set)
    for p in sess.get('pit', []) or []:
        pit_laps[p['driver_number']].add(p['lap_number'])
    by_d = defaultdict(list)
    for l in sess['laps']:
        by_d[l['driver_number']].append(l)
    rows = []
    for dn, gl in by_d.items():
        mp, ok = stint_giri(sess, gl, dn)
        if not ok:
            continue
        for l in gl:
            ln = l['lap_number']
            if ln <= 1 or not l.get('lap_duration') or not l.get('date_start'):
                continue
            if l.get('is_pit_out_lap') or ln in pit_laps[dn]:
                continue
            t0 = _t(l['date_start'])
            t1 = t0 + l['lap_duration']
            if any(a <= t1 and t0 <= b for a, b, _ in iv):
                continue
            k = np.searchsorted(cross, t0) - 1
            # passaggio precedente (escluso il proprio)
            while k >= 0 and abs(cross[k] - t0) < 1e-6:
                k -= 1
            if k >= 0 and t0 - cross[k] < 1.5:
                continue
            comp, age = mp.get(ln, (None, None))
            if comp not in ('SOFT', 'MEDIUM', 'HARD'):
                continue
            if l['lap_duration'] > 1.07 * med:
                continue
            rows.append((dn, ln, comp, age, l['lap_duration']))
    if len(rows) < 50:
        return {}, info
    R = rows
    for _ in range(4):
        cnt = defaultdict(int)
        for r in R:
            cnt[r[0]] += 1
        R = [r for r in R if cnt[r[0]] >= min_giri]
        ds = sorted(set(r[0] for r in R))
        di = {d: i for i, d in enumerate(ds)}
        nd = len(ds)
        X = np.zeros((len(R), nd + 6))
        y = np.zeros(len(R))
        for i, (dn, ln, comp, age, dur) in enumerate(R):
            X[i, di[dn]] = 1
            X[i, nd] = ln
            X[i, nd + 1] = comp == 'SOFT'
            X[i, nd + 2] = comp == 'HARD'
            X[i, nd + 3] = age * (comp == 'SOFT')
            X[i, nd + 4] = age * (comp == 'MEDIUM')
            X[i, nd + 5] = age * (comp == 'HARD')
            y[i] = dur
        # priori deboli su benzina, mescole e degrado: rendono stimabile anche la Sprint
        # (giro ed età gomma quasi collineari quando nessuno si ferma)
        mu = np.array([-0.05, -0.35, 0.35, 0.08, 0.06, 0.05])
        w = np.array([10.0, 1.0, 1.0, 7.5, 7.5, 7.5])
        Xa = np.zeros((6, nd + 6))
        Xa[np.arange(6), nd + np.arange(6)] = w
        ya = w * mu
        XX = np.vstack([X, Xa])
        yy = np.concatenate([y, ya])
        b, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        cols = list(range(nd + 6))
        res = y - X @ b
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        keepR = np.abs(res) <= max(3.0 * mad, 0.4)
        if keepR.all():
            break
        R = [r for r, k in zip(R, keepR) if k]
    coef = {d: b[di[d]] for d in ds}
    m = float(np.median(list(coef.values())))
    info['n_giri'] = len(R)
    info['residuo'] = float(mad)
    bcols = dict(zip(cols, b))
    info['degrado'] = {k: float(bcols.get(nd + j, np.nan)) for j, k in zip((3, 4, 5), ('SOFT', 'MEDIUM', 'HARD'))}
    info['benzina'] = float(bcols.get(nd, np.nan))
    info['giro_med'] = med
    return {d: 100.0 * (c - m) / med for d, c in coef.items()}, info


def ordine_dopo_giro1(sess):
    by = {}
    l1 = {}
    for l in sess['laps']:
        if l['lap_number'] == 2 and l.get('date_start'):
            by[l['driver_number']] = _t(l['date_start'])
        if l['lap_number'] == 1 and l.get('date_start') and l.get('lap_duration'):
            l1[l['driver_number']] = _t(l['date_start']) + l['lap_duration']
    for d, t in l1.items():
        by.setdefault(d, t)
    order = sorted(by, key=lambda d: by[d])
    return {d: i + 1 for i, d in enumerate(order)}


def metri_scatto(sess, secondi=5.0):
    """Metri percorsi nei primi `secondi` dopo lo spegnimento dei semafori, per pilota.
    L'istante del via si ricava dalla telemetria (ultima velocità nulla, mediana tra le auto):
    l'inizio del giro 1 nell'export può essere spostato di oltre un secondo.
    Ritorna (dict, partenza_lanciata)."""
    p = sess.get('partenza') or {}
    cd = p.get('car_data') or {}
    t_ref = None
    for l in sess['laps']:
        if l['lap_number'] == 1 and l.get('date_start'):
            t_ref = _t(l['date_start'])
            break
    if t_ref is None or not cd:
        return {}, False
    serie = {}
    fermi = {}
    for dn, c in cd.items():
        ts = np.array([_t(x) for x in c['date']])
        sp = np.array([x if x is not None else np.nan for x in c['speed']], float)
        if len(ts) < 5:
            continue
        serie[int(dn)] = (ts, sp)
        w = (ts >= t_ref - 15) & (ts <= t_ref + 5) & (sp == 0)
        if w.any():
            fermi[int(dn)] = ts[w][-1]
    if len(fermi) < 0.5 * max(len(serie), 1):
        return {}, True
    t_via = float(np.median(list(fermi.values())))
    out = {}
    for dn, (ts, sp) in serie.items():
        if dn not in fermi or fermi[dn] > t_via + 1.0:
            continue  # partito dalla corsia box o fermo a lungo
        mask = (ts >= t_via - 0.5) & (ts <= t_via + secondi + 0.5)
        tt = ts[mask]
        vv = sp[mask] / 3.6
        if len(tt) < 5 or np.isnan(vv).any():
            continue
        gt = np.linspace(t_via, t_via + secondi, 201)
        vi = np.interp(gt, tt, vv)
        out[dn] = float(np.sum((vi[1:] + vi[:-1]) * 0.5 * np.diff(gt)))
    return out, False


def griglia_sprint(sess, t_via):
    """Ordine sulla griglia: ultima posizione nota prima del via (i primi record della
    sessione possono essere vecchi)."""
    pos = {}
    for p in sorted(sess.get('position') or [], key=lambda x: x['date']):
        t = _t(p['date'])
        if t_via is not None and t > t_via + 1:
            break
        pos[p['driver_number']] = p['position']
    return pos


def costruisci_evento(d, tipo):
    """Evento (gara 'R' o sprint 'S') da un weekend dell'archivio. Se la corsa non è ancora
    stata disputata l'evento ha futuro=True e solo i dati disponibili prima del via."""
    S = {s['session']['session_name']: s for s in d['sessioni']}
    nome, qnome = ('Sprint', 'Sprint Qualifying') if tipo == 'S' else ('Race', 'Qualifying')
    s = S.get(nome)
    q = S.get(qnome)
    if not q or not q.get('session_result'):
        return None
    futuro = not s or not s.get('laps') or not s.get('session_result')
    e = dict(anno=d['anno'], round=d['round'], tipo=tipo, luogo=d['meeting'].get('location'),
             gp=d['meeting'].get('meeting_name'), futuro=futuro)
    e['circuito'] = chiave_circuito(e['luogo'], e['gp'])
    info = _driver_info(s) if s else {}
    info.update({k: v for k, v in _driver_info(q).items() if k not in info})
    e['piloti'] = info
    lap1 = [l for l in s['laps'] if l['lap_number'] == 1 and l.get('date_start')] if not futuro else []
    if lap1:
        t_via = _t(lap1[0]['date_start'])
    elif s:
        t_via = _t(s['session']['date_start'])
    else:
        t_via = _t(q['session']['date_start']) + 86400
    e['t_via'] = t_via
    e['data'] = s['session']['date_start'] if s else None
    # griglia
    g = {}
    if tipo == 'R':
        g = {x['driver_number']: x['position'] for x in (q.get('starting_grid') or [])}
    elif s and not futuro:
        g = griglia_sprint(s, t_via)
        if len(g) < 0.8 * len(s['drivers']):
            g = {}
    if not g:
        qr = [r for r in q['session_result'] if isinstance(r.get('position'), int)]
        g = {r['driver_number']: r['position'] for r in qr}
        # chi non ha posizione numerica (es. 'RT') parte in fondo
        rest = [r['driver_number'] for r in q['session_result'] if not isinstance(r.get('position'), int)]
        for i, dn in enumerate(rest):
            g[dn] = len(qr) + 1 + i
    res = {r['driver_number']: r for r in (s['session_result'] if s else [])}
    dns = {dn for dn, r in res.items() if r.get('dns')}
    e['dns'] = dns
    gg = {dn: p for dn, p in g.items() if dn not in dns}
    order = sorted(gg, key=lambda x: gg[x])
    e['griglia'] = {dn: i + 1 for i, dn in enumerate(order)}
    e['griglia_uff'] = g
    e['quali'], e['pole_t'] = distacchi_qualifica(q)
    # gomme: long run delle libere del weekend (prima della qualifica) e, a gara disputata, l'analisi
    # gomme della gara per il modello di stagione
    e['long_run'] = []
    e['gomme'] = None
    if G is not None and tipo == 'R':
        for nm in ('Practice 1', 'Practice 2', 'Practice 3'):
            if nm in S:
                try:
                    e['long_run'] += G.long_run(S[nm])
                except Exception:  # noqa: BLE001
                    pass
        if not futuro:
            try:
                e['gomme'] = G.analisi_weekend(d)
            except Exception:  # noqa: BLE001
                e['gomme'] = None
    if futuro:
        e.update(arrivo={}, ritirati=set(), dsq=set(), giri=0, passo={}, passo_info={}, giro1={},
                 scatto={}, lanciata=False, sc=0, vsc=0, red=0, bagnato=False, perdita_box=None,
                 soste={}, giri_rit={}, pioggia_rischio=rischio_pioggia(s, None) if s else None)
        return e
    e['arrivo'] = {dn: r['position'] for dn, r in res.items() if isinstance(r.get('position'), int)}
    e['ritirati'] = {dn for dn, r in res.items() if (r.get('dnf') or (r.get('position') is None and not r.get('dns')))}
    e['dsq'] = {dn for dn, r in res.items() if r.get('dsq')}
    e['giri'] = max([r.get('number_of_laps') or 0 for r in res.values()] or [0])
    e['passo'], e['passo_info'] = passo_gara(s)
    e['giro1'] = ordine_dopo_giro1(s)
    e['scatto'], e['lanciata'] = metri_scatto(s)
    e['sc'], e['vsc'], e['red'] = conta_neutralizzazioni(s)
    e['pioggia_rischio'] = rischio_pioggia(s, e['t_via'])
    comps = {st['compound'] for st in s['stints']}
    e['bagnato'] = bool(comps & {'INTERMEDIATE', 'WET'})
    lane = [p['lane_duration'] for p in (s.get('pit') or []) if p.get('lane_duration') and 15 < p['lane_duration'] < 45]
    e['perdita_box'] = float(np.median(lane)) if lane else None
    nst = defaultdict(int)
    for p in (s.get('pit') or []):
        nst[p['driver_number']] += 1
    e['soste'] = dict(nst)
    e['giri_rit'] = {dn: r.get('number_of_laps') or 0 for dn, r in res.items() if dn in e['ritirati']}
    return e


def estrai_eventi(storico_dir, anni=(2026,), futuri=False):
    """Lista di eventi (gare e sprint) in ordine cronologico, con tutto ciò che serve."""
    files = sorted(glob.glob(os.path.join(storico_dir, '*', '*.json.gz')))
    ev = []
    for f in files:
        d = carica_weekend(f)
        if 'sessioni' not in d:  # storico/circuiti: solo il giro di riferimento, non un weekend
            continue
        if d['anno'] not in anni:
            continue
        for tipo in ('S', 'R'):
            e = costruisci_evento(d, tipo)
            if e is None or (e['futuro'] and not futuri):
                continue
            ev.append(e)
    ev.sort(key=lambda e: e['t_via'])
    return ev


def _phi(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x):
    # funzione di ripartizione normale (approssimazione di Abramowitz-Stegun 7.1.26, errore < 1.5e-7)
    z = np.abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + 0.3275911 * z)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * y)


def probit_ridge(X, y, prior_mu, prior_sd, iters=60):
    """Probit con priore normale sui coefficienti (massimo a posteriori, Newton)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu0 = np.asarray(prior_mu, float)
    s0 = np.asarray(prior_sd, float)
    b = mu0.copy()
    for _ in range(iters):
        z = X @ b if len(X) else np.zeros(0)
        Pz = np.clip(_Phi(z), 1e-9, 1 - 1e-9)
        ph = _phi(z)
        lam = np.where(y > 0, ph / Pz, -ph / (1 - Pz))
        g = X.T @ lam - (b - mu0) / s0 ** 2 if len(X) else -(b - mu0) / s0 ** 2
        w = lam * (lam + z)  # -d2 loglik / dz2 per il probit
        H = (X * w[:, None]).T @ X + np.diag(1 / s0 ** 2) if len(X) else np.diag(1 / s0 ** 2)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-6:
            break
    return b


# ---------------------------------------------------------------------------
# 2. Stima dei parametri dallo storico (solo eventi precedenti)
# ---------------------------------------------------------------------------

def _qrel(e):
    q = e['quali']
    if not q:
        return {}
    m = float(np.median(list(q.values())))
    return {d: v - m for d, v in q.items()}


def _team(e, d):
    return e['piloti'].get(d, ('?', '?'))[1]


def forma(prec, chiave='passo', emivita=None, shrink=None):
    """Forma per team e scarto del pilota dal compagno, con peso ai più recenti.
    Ritorna (team->valore, team->peso, pilota->scarto ristretto, pilota->peso)."""
    emivita = emivita or P['emivita_forma']
    shrink = shrink or P['shrink_pilota']
    ev = [p for p in prec if p.get(chiave)]
    tn, td = defaultdict(float), defaultdict(float)
    on, od = defaultdict(float), defaultdict(float)
    for i, p in enumerate(ev[::-1]):
        val = p[chiave] if chiave != 'quali' else _qrel(p)
        rel = min(1.0, p['passo_info'].get('n_giri', 400) / 400.0) if chiave == 'passo' else 1.0
        w = 0.5 ** (i / emivita) * rel
        bt = defaultdict(list)
        for d, v in val.items():
            bt[_team(p, d)].append((d, v))
        for t, lst in bt.items():
            m = float(np.mean([v for _, v in lst]))
            tn[t] += w * m
            td[t] += w
            if len(lst) == 2:
                for d, v in lst:
                    on[d] += w * (v - m)
                    od[d] += w
    team = {t: tn[t] / td[t] for t in td if td[t] > 0}
    # restringimento verso il compagno: peso effettivo / (peso + k)
    off = {d: (on[d] / od[d]) * od[d] / (od[d] + shrink) for d in od if od[d] > 0}
    return team, td, off, od


class Storia:
    """Tutto ciò che si stima dagli eventi precedenti a quello da prevedere."""

    def __init__(self, prec):
        self.prec = prec
        self.gomme = G.StagioneGomme([p.get('gomme') for p in prec if p['tipo'] == 'R']) if G is not None else None
        self._regressione_passo()
        self._scatto()
        self._giro1()
        self._affidabilita()
        self._neutralizzazioni()
        self._strategia()
        self._griglia_pl()

    # --- passo: regressione distacco in qualifica + forma -> passo gara
    def _righe_passo(self):
        rows = []
        for k, e in enumerate(self.prec):
            if not e['passo']:
                continue
            prima = self.prec[:k]
            tf, tw, off, ow = forma(prima, 'passo')
            qr = _qrel(e)
            qrank = {d: i + 1 for i, d in enumerate(sorted(qr, key=qr.get))}
            for d, y in e['passo'].items():
                t = _team(e, d)
                f = tf[t] + off.get(d, 0.0) if t in tf and tw[t] >= 0.3 else np.nan
                rows.append((k, d, t, y, qr.get(d, np.nan), f, qrank.get(d, 99)))
        return rows

    BANDE = ((1, 6), (7, 12), (13, 99))
    SD_PRIORI = (0.30, 0.38, 0.60)

    @staticmethod
    def _peso_rank(r):
        # conta di più chi parte davanti: è lì che si decidono vittoria e podio
        return 1.0 if r <= 10 else 0.35

    def _fit(self, y, X, w, prior, lam=30.0):
        sw = np.sqrt(w)
        Xa = np.sqrt(lam) * np.eye(len(prior))
        ya = np.sqrt(lam) * np.asarray(prior)
        b, *_ = np.linalg.lstsq(np.vstack([X * sw[:, None], Xa]), np.concatenate([y * sw, ya]), rcond=None)
        return b

    def _sd_bande(self, res, rk, scala=1.0):
        out = []
        for (lo, hi), pr in zip(self.BANDE, self.SD_PRIORI):
            r = res[(rk >= lo) & (rk <= hi)] if len(res) else np.zeros(0)
            n0 = 15.0
            out.append(max(math.sqrt((np.sum(r ** 2) + n0 * (pr * scala) ** 2) / (len(r) + n0)), P['sigma_pace_min']))
        return out

    def _centra(self, res, mask, rk):
        """Toglie a ogni evento lo scarto medio dei primi 12 in qualifica: uno spostamento
        comune a tutto il gruppo di testa non cambia chi vince."""
        ks = np.array([r[0] for r, m in zip(self.righe_passo, mask) if m])
        rr = rk[mask]
        out = res.copy()
        for k in set(ks.tolist()):
            sel = (ks == k) & (rr <= 12)
            if sel.sum() >= 3:
                out[sel] = res[sel] - res[sel].mean() * 1.0
        return out

    def _regressione_passo(self):
        rows = self._righe_passo()
        self.righe_passo = rows
        A = np.array([(r[3], r[4], r[5], r[6]) for r in rows], float) if rows else np.zeros((0, 4))
        y, q, f, rk = (A.T if len(A) else (np.zeros(0),) * 4)
        w = np.array([self._peso_rank(r) for r in rk]) if len(A) else np.zeros(0)
        one = np.ones_like(y)
        # completo: passo = a + bq*qualifica + bf*forma
        ok = ~np.isnan(q) & ~np.isnan(f)
        self.b_full = self._fit(y[ok], np.c_[one[ok], q[ok], f[ok]], w[ok], [0.0, 0.5, 0.5])
        res = y[ok] - np.c_[one[ok], q[ok], f[ok]] @ self.b_full
        self.sd_full = self._sd_bande(self._centra(res, ok, rk), rk[ok])
        # solo qualifica (pilota o team senza storia)
        okq = ~np.isnan(q)
        self.b_quali = self._fit(y[okq], np.c_[one[okq], q[okq]], w[okq], [0.0, 0.9])
        resq = y[okq] - np.c_[one[okq], q[okq]] @ self.b_quali
        self.sd_quali = self._sd_bande(self._centra(resq, okq, rk), rk[okq], 1.15)
        # solo forma (nessun tempo in qualifica)
        okf = ~np.isnan(f)
        self.b_forma = self._fit(y[okf], np.c_[one[okf], f[okf]], w[okf], [0.0, 0.9])
        resf = y[okf] - np.c_[one[okf], f[okf]] @ self.b_forma
        self.sd_forma = self._sd_bande(resf, rk[okf], 1.4)
        # quota dell'errore comune ai due piloti dello stesso team
        rho = 0.45
        if ok.sum() > 20:
            R = [r for r, k in zip(rows, ok) if k]
            byev = defaultdict(lambda: defaultdict(list))
            for r, rr in zip(R, res):
                byev[r[0]][r[2]].append(rr)
            pr = [v for d in byev.values() for v in d.values() if len(v) == 2]
            if len(pr) >= 10:
                pr = np.array(pr)
                c = float(np.mean(pr[:, 0] * pr[:, 1]))
                v = float(np.mean(pr ** 2))
                n = len(pr)
                rho = float(np.clip((c / v * n + 0.45 * 15) / (n + 15), 0.0, 0.8))
        self.rho_team = rho
        self._modello_testa()

    def _modello_testa(self, top=12):
        """Modello per i primi `top` in qualifica, centrato evento per evento: conta solo
        l'ordine tra chi può vincere, non lo scarto dal resto del gruppo.
        passo - media(primi) = bq*(q - media) + bf*(f - media)."""
        by = defaultdict(list)
        for r in self.righe_passo:
            if r[6] <= top and not np.isnan(r[4]):
                by[r[0]].append(r)
        Yf, Xf, Yq, Xq, Kf, team_f = [], [], [], [], [], []
        for k, lst in by.items():
            if len(lst) < 6:
                continue
            y = np.array([r[3] for r in lst])
            q = np.array([r[4] for r in lst])
            Yq += list(y - y.mean())
            Xq += list(q - q.mean())
            lf = [r for r in lst if not np.isnan(r[5])]
            if len(lf) >= 6:
                y2 = np.array([r[3] for r in lf])
                q2 = np.array([r[4] for r in lf])
                f2 = np.array([r[5] for r in lf])
                Yf += list(y2 - y2.mean())
                Xf += list(zip(q2 - q2.mean(), f2 - f2.mean()))
                Kf += [k] * len(lf)
                team_f += [r[2] for r in lf]
        lam = 20.0
        Xf = np.array(Xf).reshape(-1, 2)
        Yf = np.array(Yf)
        self.bt_full = self._fit(Yf, Xf, np.ones(len(Yf)), [0.7, 0.5], lam) if len(Yf) else np.array([0.7, 0.5])
        Xq = np.array(Xq).reshape(-1, 1)
        Yq = np.array(Yq)
        self.bt_quali = self._fit(Yq, Xq, np.ones(len(Yq)), [1.2], lam) if len(Yq) else np.array([1.2])
        rf = Yf - Xf @ self.bt_full if len(Yf) else np.zeros(0)
        rq = Yq - Xq @ self.bt_quali if len(Yq) else np.zeros(0)
        n0 = 20.0
        self.sdt_full = max(math.sqrt((np.sum(rf ** 2) + n0 * 0.35 ** 2) / (len(rf) + n0)), P['sigma_pace_min'])
        self.sdt_quali = max(math.sqrt((np.sum(rq ** 2) + n0 * 0.42 ** 2) / (len(rq) + n0)), P['sigma_pace_min'])
        # correlazione tra compagni nel gruppo di testa
        rho = 0.45
        if len(rf) > 20:
            byk = defaultdict(lambda: defaultdict(list))
            for k, t, r in zip(Kf, team_f, rf):
                byk[k][t].append(r)
            pr = np.array([v for d in byk.values() for v in d.values() if len(v) == 2])
            if len(pr) >= 8:
                c = float(np.mean(pr[:, 0] * pr[:, 1]))
                v = float(np.mean(pr ** 2))
                n = len(pr)
                rho = float(np.clip((c / v * n + 0.45 * 15) / (n + 15), 0.0, 0.8))
        self.rho_testa = min(rho, 0.45)
        self._ordine_testa()

    TOP_ORDINE = 8

    def _ordine_testa(self):
        """Ordine di passo tra i primi 8 in qualifica: probit sulle coppie (chi dei due ha avuto
        il passo gara migliore) con distacco in qualifica, log della posizione in qualifica e forma.
        Sulle gare 2026 l'ordine in qualifica conta più della forma per i primi."""
        by = defaultdict(list)
        for r in self.righe_passo:
            if r[6] <= self.TOP_ORDINE and not np.isnan(r[4]):
                by[r[0]].append(r)
        Xf, yf, Xq, yq = [], [], [], []
        for k, lst in by.items():
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    a, b = lst[i], lst[j]
                    dq = b[4] - a[4]
                    dr = math.log(b[6]) - math.log(a[6])
                    out = 1.0 if b[3] > a[3] else 0.0  # a più veloce di b
                    Xq.append((dq, dr))
                    yq.append(out)
                    if not np.isnan(a[5]) and not np.isnan(b[5]):
                        Xf.append((dq, dr, b[5] - a[5]))
                        yf.append(out)
        self.beta_ordine = probit_ridge(np.array(Xf).reshape(-1, 3), yf, [1.0, 0.5, 0.4], [0.6, 0.6, 0.6])
        self.beta_ordine_q = probit_ridge(np.array(Xq).reshape(-1, 2), yq, [1.2, 0.5], [0.6, 0.6])
        self.n_coppie_ordine = len(yf)

    def _banda(self, rank):
        for i, (lo, hi) in enumerate(self.BANDE):
            if lo <= rank <= hi:
                return i
        return len(self.BANDE) - 1

    def passo_previsto(self, e):
        """Passo previsto (%) e deviazione standard per ogni pilota al via."""
        tf, tw, off, ow = forma(self.prec, 'passo')
        qr = _qrel(e)
        qrank = {d: i + 1 for i, d in enumerate(sorted(qr, key=qr.get))}
        mu, sd = {}, {}
        for d in e['griglia']:
            t = _team(e, d)
            f = tf[t] + off.get(d, 0.0) if t in tf and tw[t] >= 0.3 else None
            q = qr.get(d)
            bi = self._banda(qrank.get(d, 99))
            if q is not None and f is not None:
                b = self.b_full
                dev = q - f
                if abs(dev) > 1.0:
                    # qualifica molto lontana dalla forma (guasto, errore): si tronca lo scarto
                    q = f + math.copysign(1.0, dev)
                mu[d] = b[0] + b[1] * q + b[2] * f
                sd[d] = self.sd_full[bi] * (1.2 if abs(dev) > 1.0 else 1.0)
            elif q is not None:
                b = self.b_quali
                mu[d] = b[0] + b[1] * q
                sd[d] = self.sd_quali[bi]
            elif f is not None:
                b = self.b_forma
                mu[d] = b[0] + b[1] * f
                sd[d] = self.sd_forma[2]
            else:
                mu[d] = 1.5
                sd[d] = 1.0
        # gruppo di testa (primi 8 in qualifica): ordine dal probit sulle coppie, tradotto in
        # passo con la stessa deviazione standard usata nel simulatore; livello medio dal
        # modello generale
        testa = [d for d in e['griglia'] if qrank.get(d, 99) <= self.TOP_ORDINE]
        if len(testa) >= 4:
            s = self.sdt_full * math.sqrt(2)
            idx = {}
            for d in testa:
                t = _team(e, d)
                qd = qr[d]
                lr = math.log(qrank[d])
                if t in tf and tw[t] >= 0.3:
                    fd = tf[t] + off.get(d, 0.0)
                    if abs(qd - fd) > 1.0:
                        qd = fd + math.copysign(1.0, qd - fd)
                    idx[d] = float(np.dot(self.beta_ordine, (qd, lr, fd)))
                    sd[d] = self.sdt_full
                else:
                    idx[d] = float(np.dot(self.beta_ordine_q, (qd, lr)))
                    sd[d] = self.sdt_quali
            livello = float(np.mean([mu[d] for d in testa]))
            m = float(np.mean(list(idx.values())))
            for d in testa:
                mu[d] = livello + s * (idx[d] - m)
        return mu, sd

    # --- partenza
    @staticmethod
    def _scatto_evento(e):
        sc = e['scatto']
        if not sc:
            return {}
        med = float(np.median(list(sc.values())))
        return {d: float(np.clip(v - med, -25, 25)) for d, v in sc.items()
                if d in e['griglia'] and e['griglia'][d] <= 20}

    @staticmethod
    def _scatto_da_storia(hist, histT, e, d):
        """Metri in più a 5 s rispetto alla mediana, ristretti verso il team e verso 0.
        Restringimento forte (12 partenze per metà fiducia): nel 2026 lo scarto di una
        partenza si ripete solo per metà circa (pendenza 0,54 sulla media delle precedenti)."""
        t = _team(e, d)
        vt = histT.get(t, [])[-16:]
        team = (np.mean(vt) * len(vt) / (len(vt) + 12.0)) if vt else 0.0
        vd = hist.get(d, [])
        if vd:
            return team + (np.mean(vd) - team) * len(vd) / (len(vd) + 12.0)
        return team

    def _scatto(self):
        self.hist_scatto = defaultdict(list)
        self.hist_scatto_t = defaultdict(list)
        for e in self.prec:
            for d, r in self._scatto_evento(e).items():
                self.hist_scatto[d].append(r)
                self.hist_scatto_t[_team(e, d)].append(r)

    def scatto_previsto(self, e, d):
        return self._scatto_da_storia(self.hist_scatto, self.hist_scatto_t, e, d)

    def _giro1(self):
        """Rumore del primo giro sigma(g) = a + b*(g-1). `a` per massima verosimiglianza sulle
        coppie di auto vicine tra i primi 10 in griglia (chi è davanti dopo il primo giro),
        tenendo conto dello scatto previsto con la storia di allora; priore N(1.0, 0.3).
        `b` è fisso: conta solo per il centro gruppo."""
        from math import erf, sqrt, log
        b = 0.06
        k = P['peso_scatto'] / P['metri_per_posizione']
        coppie = []
        lead = []
        lead2 = []
        self._g1_casi = []
        hist, histT = defaultdict(list), defaultdict(list)
        for e in self.prec:
            g = e['griglia']
            l1 = e['giro1']
            if not e['lanciata'] and l1:
                ds = sorted([d for d in g if d in l1], key=lambda d: g[d])
                if len(ds) >= 10:
                    gr = {d: i + 1 for i, d in enumerate(ds)}
                    off = {d: k * self._scatto_da_storia(hist, histT, e, d) for d in ds[:13]}
                    for i, di in enumerate(ds[:10]):
                        for dj in ds[i + 1:i + 4]:
                            if gr[dj] > 10:
                                continue
                            m = (gr[dj] - off[dj]) - (gr[di] - off[di])
                            coppie.append((gr[di], gr[dj], m, l1[di] < l1[dj]))
                    lead.append(l1[ds[0]] == 1)
                    lead2.append(l1[ds[1]] == 1)
                    self._g1_casi.append((e, np.array([k * self._scatto_da_storia(hist, histT, e, d) for d in ds])))
            for d, r in self._scatto_evento(e).items():
                hist[d].append(r)
                histT[_team(e, d)].append(r)
        self.giro1_obs = len(lead)
        self.giro1_lead_oss = float(np.mean(lead)) if lead else None
        if not coppie:
            self.sig1 = (1.0, b, 1.0)
            self.p_incidente1 = 0.05
            self.delta2 = 0.0
            self.giro1_target = (0.68, 0.12)
            return
        # mescolanza: rumore normale + incidente (auto che perde 4-12 posizioni) con prob. p
        best = None
        for a in np.arange(0.4, 2.01, 0.05):
            for p in np.arange(0.0, 0.151, 0.01):
                ll = -0.5 * ((a - 1.0) / 0.3) ** 2 - 0.5 * ((p - 0.05) / 0.03) ** 2
                for gi, gj, m, ok in coppie:
                    s = sqrt((a + b * (gi - 1)) ** 2 + (a + b * (gj - 1)) ** 2)
                    ph = 0.5 * (1 + erf(m / s / sqrt(2)))
                    pr = ((1 - p) ** 2 + p ** 2) * ph + p * (1 - p)
                    pr = min(max(pr, 1e-4), 1 - 1e-4)
                    ll += log(pr if ok else 1 - pr)
                if best is None or ll > best[0]:
                    best = (ll, a, p)
        self.p_incidente1 = float(best[2])
        a = float(best[1])
        # Prima fila: la pole ha un rumore suo e la seconda casella una penalità (nel 2026 dopo il
        # primo giro era in testa la pole 16 volte su 23, la seconda casella una sola). Tutti e due
        # scelti perché le quote medie di pole e seconda casella in testa dopo il giro 1, simulate
        # sulle partenze precedenti, tornino con quelle osservate (priori 0,68 e 0,12, peso 8).
        t1 = (sum(lead) + 0.68 * 8) / (len(lead) + 8)
        t2 = (sum(lead2) + 0.12 * 8) / (len(lead2) + 8)
        self.giro1_target = (t1, t2)
        pi = self.p_incidente1
        casi = [off for _, off in self._g1_casi[-25:]]
        best_p = (9, a, 0.0)
        for d2 in (np.arange(0.0, 2.01, 0.25) if casi else []):
            for ap in np.arange(0.3, a + 0.001, 0.1):
                rng = np.random.default_rng(11)
                m = np.zeros(2)
                for off in casi:
                    n = len(off)
                    g = np.arange(1, n + 1, dtype=float)
                    sig = a + b * (g - 1)
                    sig[0] = ap
                    sc = g[None, :] - off[None, :] + sig[None, :] * rng.standard_normal((600, n))
                    sc[:, 1] += d2
                    sc = sc + (rng.random((600, n)) < pi) * rng.uniform(4, 12, (600, n))
                    m += np.bincount(np.argmin(sc, 1), minlength=n)[:2] / 600
                m /= len(casi)
                loss = (m[0] - t1) ** 2 + (m[1] - t2) ** 2
                if loss < best_p[0]:
                    best_p = (loss, float(ap), float(d2))
        self.sig1 = (a, b, best_p[1])
        self.delta2 = best_p[2]

    # --- ritiri
    def _affidabilita(self):
        st, dn = defaultdict(int), defaultdict(int)
        for e in self.prec:
            for d in e['griglia']:
                t = _team(e, d)
                st[t] += 1
                if d in e['ritirati']:
                    dn[t] += 1
        tot_s = sum(st.values())
        tot_d = sum(dn.values())
        self.dnf_medio = (tot_d + 0.15 * 40) / (tot_s + 40)
        k = P['dnf_prior_peso']
        self.dnf_team = {t: (dn[t] + k * self.dnf_medio) / (st[t] + k) for t in st}

    def p_ritiro(self, e, d):
        return self.dnf_team.get(_team(e, d), self.dnf_medio)

    # --- safety car e VSC
    def _neutralizzazioni(self):
        sc = defaultdict(float)
        n = defaultdict(float)
        for e in self.prec:
            tipo = 'citt' if e['circuito'] in CITTADINI else 'perm'
            # la Sprint è circa un terzo di gara
            w = 1.0 if e['tipo'] == 'R' else 0.35
            sc[('SC', tipo)] += e['sc'] + e['red']
            sc[('VSC', tipo)] += e['vsc']
            n[tipo] += w
        self.lam = {}
        for k, (m0, n0) in (('SC', P['sc_prior']), ('VSC', P['vsc_prior'])):
            tot = (sc[(k, 'citt')] + sc[(k, 'perm')] + m0 * n0) / (n['citt'] + n['perm'] + n0)
            for tipo, mult in (('citt', P['mult_cittadino']), ('perm', 1.0 / P['mult_cittadino'] ** 0.5)):
                prior = tot * mult
                self.lam[(k, tipo)] = (sc[(k, tipo)] + prior * 4.0) / (n[tipo] + 4.0)

    # --- strategia
    def _strategia(self):
        due = uno = 0
        degs = []
        for e in self.prec:
            if e['tipo'] != 'R' or e['bagnato']:
                continue
            for d, pos in e['arrivo'].items():
                if pos <= 10:
                    s = e['soste'].get(d, 0)
                    if s == 1:
                        uno += 1
                    elif s >= 2:
                        due += 1
            dg = e['passo_info'].get('degrado') or {}
            v = [x for x in dg.values() if x == x]
            if v:
                degs.append(float(np.mean(v)))
        # le soste in più dovute a SC/VSC le genera il simulatore: si prende metà dell'eccesso
        p2 = (due + P['p_due_soste'] * 30) / (uno + due + 30)
        self.p_due = float(np.clip(0.5 * p2 + 0.5 * P['p_due_soste'], 0.1, 0.6))
        # il degrado della regressione è lordo (per giro di età gomma); il guadagno vero di chi
        # anticipa la sosta è circa il 40-45% (Singapore 2026: 0,67 s/giro con 25 giri di
        # differenza di età contro 0,069 di degrado lordo)
        lordo = float(np.median(degs)) if len(degs) >= 3 else 0.065
        self.degrado = P['quota_degrado'] * lordo

    # --- modello solo griglia (Plackett-Luce con forza exp(-b*(g-1)))
    def _griglia_pl(self):
        """Due parametri stimati sugli eventi precedenti: b_vitt sulla sola vittoria,
        b_podio sull'ordine dei primi tre. Priore N(0.6, 0.3) su entrambi."""
        data = []
        for e in self.prec:
            g = e['griglia']
            top = sorted([d for d in e['arrivo'] if d in g], key=lambda d: e['arrivo'][d])[:3]
            if len(top) < 3:
                continue
            data.append((np.array([g[d] for d in g], float), [g[d] for d in top]))
        best_v = best_p = None
        for b in np.arange(0.05, 2.5, 0.01):
            llv = llp = -0.5 * ((b - 0.6) / 0.3) ** 2
            for gs, top in data:
                s = np.exp(-b * (gs - 1))
                tot = s.sum()
                for i, gp in enumerate(top):
                    sp = math.exp(-b * (gp - 1))
                    l = math.log(sp / tot)
                    llp += l
                    if i == 0:
                        llv += l
                    tot -= sp
            if best_v is None or llv > best_v[0]:
                best_v = (llv, b)
            if best_p is None or llp > best_p[0]:
                best_p = (llp, b)
        self.b_griglia = (float(best_v[1]), float(best_p[1]))


def prob_griglia(e, bb, n=40000, seed=1):
    """Vittoria e podio dal solo modello griglia (Plackett-Luce, Gumbel-max)."""
    b_v, b_p = bb
    ds = sorted(e['griglia'], key=lambda d: e['griglia'][d])
    g = np.array([e['griglia'][d] for d in ds], float)
    sv = np.exp(-b_v * (g - 1))
    win = sv / sv.sum()
    rng = np.random.default_rng(seed)
    keys = -b_p * (g - 1)[None, :] + rng.gumbel(size=(n, len(ds)))
    order = np.argsort(-keys, 1)[:, :3]
    pod = np.zeros(len(ds))
    for k in range(3):
        pod += np.bincount(order[:, k], minlength=len(ds)) / n
    return {d: float(w) for d, w in zip(ds, win)}, {d: float(p) for d, p in zip(ds, pod)}


# ---------------------------------------------------------------------------
# 3. Simulatore Monte Carlo
# ---------------------------------------------------------------------------

# Giri in programma (gara, Sprint; 0 = niente Sprint), 2026. Servono per prevedere una corsa non
# ancora disputata e, nella prova a ritroso, al posto dei giri corsi (una corsa accorciata non si
# conosce prima del via).
GIRI = {'Melbourne': (58, 0), 'Shanghai': (56, 19), 'Suzuka': (53, 0), 'Miami Gardens': (57, 19),
        'Montréal': (70, 23), 'Monte Carlo': (78, 0), 'Barcelona': (66, 0), 'Spielberg': (71, 0),
        'Silverstone': (52, 17), 'Spa-Francorchamps': (44, 0), 'Budapest': (70, 0), 'Zandvoort': (72, 24),
        'Monza': (53, 0), 'Madrid': (57, 0), 'Baku': (51, 0), 'Marina Bay': (62, 21),
        'Austin': (56, 19), 'Mexico City': (71, 0), 'São Paulo': (71, 24), 'Las Vegas': (50, 0),
        'Lusail': (57, 19), 'Yas Island': (58, 0)}

# Nomi alternativi: se OpenF1 cambia la località, si riconosce il circuito dal nome del GP
ALIAS_CIRCUITO = (('united states', 'Austin'), ('mexico', 'Mexico City'), ('são paulo', 'São Paulo'),
                  ('sao paulo', 'São Paulo'), ('brazil', 'São Paulo'), ('las vegas', 'Las Vegas'),
                  ('qatar', 'Lusail'), ('abu dhabi', 'Yas Island'), ('yas marina', 'Yas Island'),
                  ('singapore', 'Marina Bay'), ('monaco', 'Monte Carlo'), ('belgian', 'Spa-Francorchamps'),
                  ('hungarian', 'Budapest'), ('dutch', 'Zandvoort'), ('italian', 'Monza'),
                  ('azerbaijan', 'Baku'), ('austrian', 'Spielberg'), ('british', 'Silverstone'),
                  ('japanese', 'Suzuka'), ('chinese', 'Shanghai'), ('australian', 'Melbourne'),
                  ('canadian', 'Montréal'), ('miami', 'Miami Gardens'))


def chiave_circuito(luogo, gp):
    """Nome del circuito usato nelle tabelle (sorpassi, cittadini, giri, storia del circuito)."""
    if luogo in SORPASSO or luogo in GIRI:
        return luogo
    t = f"{luogo or ''} {gp or ''}".lower()
    for k, v in ALIAS_CIRCUITO:
        if k in t:
            return v
    return luogo


def prepara(e, st, sorpasso_liv=None, pioggia=None, circuito=None, giri=None, shift=None):
    """Ingressi del simulatore per l'evento `e` con la storia `st` (solo dati prima del via).
    `circuito`: eventi passati sullo stesso circuito (altri anni) per SC, VSC, box e giri."""
    ds = sorted(e['griglia'], key=lambda d: e['griglia'][d])
    mu, sd = st.passo_previsto(e)
    teams = sorted({_team(e, d) for d in ds})
    ti = {t: i for i, t in enumerate(teams)}
    tipo = 'citt' if e['circuito'] in CITTADINI else 'perm'
    rr = e.get('pioggia_rischio')
    if pioggia is not None:
        rr = pioggia
    if rr is None:
        rr = 0.1
    liv = SORPASSO.get(e['circuito'], 2) if sorpasso_liv is None else sorpasso_liv
    giro_rif = (e['pole_t'] or 90.0) * 1.06
    sprint = e['tipo'] == 'S'
    lam_sc = st.lam[('SC', tipo)]
    lam_vsc = st.lam[('VSC', tipo)]
    box = e.get('perdita_box')
    in_tab = GIRI.get(e['circuito'], (0, 0))[1 if sprint else 0]
    n_giri = giri or in_tab or e.get('giri') or 0
    if circuito:
        stessi = [c for c in circuito if c['tipo'] == e['tipo']]
        gare = [c for c in circuito if c['tipo'] == 'R']
        if gare:
            # tasso del circuito ristretto verso quello della stagione (peso 2 gare)
            n_c = len(gare)
            lam_sc = (sum(c['sc'] + c['red'] for c in gare) + 2.0 * lam_sc) / (n_c + 2.0)
            lam_vsc = (sum(c['vsc'] for c in gare) + 2.0 * lam_vsc) / (n_c + 2.0)
            if box is None:
                bx = [c['perdita_box'] for c in gare if c['perdita_box']]
                box = float(np.median(bx)) if bx else None
        if not n_giri and stessi:
            n_giri = max(c['giri'] for c in stessi)
    if not n_giri:
        n_giri = 19 if sprint else 57
    gomme = None
    piani = None
    if st.gomme is not None and not sprint and P.get('gomme'):
        gomme = st.gomme.previsione(runs=e.get('long_run') or None)
        piani = G.piani_strategia(gomme, n_giri, box or P['perdita_box_default'])
    return dict(
        gomme=gomme,
        piani=piani,
        piloti=ds,
        sigle=[e['piloti'][d][0] for d in ds],
        griglia=np.array([e['griglia'][d] for d in ds], float),
        team=np.array([ti[_team(e, d)] for d in ds]),
        n_team=len(teams),
        mu=np.array([mu[d] for d in ds]),
        sd=np.array([sd[d] for d in ds]),
        rho=st.rho_testa,
        scatto=np.array([st.scatto_previsto(e, d) for d in ds]),
        sig1=st.sig1,
        p_incidente1=st.p_incidente1,
        delta2=st.delta2,
        p_rit=np.array([st.p_ritiro(e, d) for d in ds]) * (0.45 if sprint else 1.0),
        lam_sc=lam_sc * (0.35 if sprint else 1.0),
        lam_vsc=lam_vsc * (0.35 if sprint else 1.0),
        p_bagnato=float(np.clip(0.7 * rr, 0, 1)),
        giri=int(n_giri),
        giro_rif=giro_rif,
        perdita_box=box or P['perdita_box_default'],
        degrado=st.degrado,
        p_due=st.p_due,
        soste=not sprint,
        sorpasso=livello_sorpasso(liv, P['sorpasso_shift'] if shift is None else shift),
    )


def livello_sorpasso(liv, shift=0.0):
    """(soglia, probabilità massima) per un livello di difficoltà anche non intero."""
    x = min(max(liv + shift, 0.0), len(P['sorpasso']) - 1.0)
    i = int(math.floor(x))
    j = min(i + 1, len(P['sorpasso']) - 1)
    f = x - i
    (t0, p0), (t1, p1) = P['sorpasso'][i], P['sorpasso'][j]
    return (t0 + f * (t1 - t0), p0 + f * (p1 - p0))


def _t5(rng, size, df=5):
    """Student-t con varianza 1."""
    return rng.standard_t(df, size) * math.sqrt((df - 2) / df)


def _rimescolamento(rank1, fin, giro_rit, L):
    """Media di |posizione finale - posizione dopo il primo giro| tra chi arriva al traguardo,
    con le posizioni ricontate tra i soli arrivati."""
    n, D = fin.shape
    ar = np.arange(n)
    rf = np.empty_like(fin)
    rf[ar[:, None], fin] = np.arange(D)[None, :]
    ok = giro_rit > L
    tot = 0.0
    cnt = 0
    for i in range(min(n, 400)):
        m = ok[i]
        if m.sum() < 5:
            continue
        a = np.argsort(np.argsort(rank1[i][m]))
        b = np.argsort(np.argsort(rf[i][m]))
        tot += np.abs(a - b).mean()
        cnt += 1
    return float(tot / max(cnt, 1))


def simula(inp, n=None, seed=0, dettagli=False):
    """Simula la corsa n volte. Ritorna probabilità di vittoria, podio e altri conteggi."""
    n = n or P['n_sim']
    rng = np.random.default_rng(seed)
    D = len(inp['piloti'])
    L = inp['giri']
    T = inp['giro_rif']
    ar = np.arange(n)

    bagnato = rng.random(n) < inp['p_bagnato']
    # passo vero di questa corsa (s/giro, relativo)
    tn = _t5(rng, (n, inp['n_team']))
    ind = _t5(rng, (n, D))
    rho = inp['rho']
    err = inp['sd'][None, :] * (math.sqrt(rho) * tn[:, inp['team']] + math.sqrt(1 - rho) * ind)
    err = err + bagnato[:, None] * P['sigma_bagnato'] * rng.standard_normal((n, D))
    passo = T / 100.0 * (inp['mu'][None, :] + err)
    rumore = P['rumore_giro'] * (1 + 0.8 * bagnato)[:, None]

    # ritiri
    pr = inp['p_rit'][None, :] * (1 + (P['mult_dnf_bagnato'] - 1) * bagnato[:, None])
    rit = rng.random((n, D)) < pr
    giro_rit = np.where(rng.random((n, D)) < P['dnf_quota_giro1'], 1, rng.integers(2, L + 1, (n, D)))
    giro_rit = np.where(rit, giro_rit, L + 10)
    # guai: qualche giro a potenza ridotta
    gu = rng.random((n, D)) < P['p_guaio']
    gu_in = rng.integers(1, max(L - 3, 2), (n, D))
    gu_len = rng.integers(P['guaio_giri'][0], P['guaio_giri'][1] + 1, (n, D))
    gu_loss = rng.uniform(*P['guaio_perdita'], (n, D)) * gu

    # neutralizzazioni: stato per giro (0 verde, 1 SC, 2 VSC)
    stato = np.zeros((n, L + 2), np.int8)
    mult = 1 + (P['mult_sc_bagnato'] - 1) * bagnato
    nsc = rng.poisson(inp['lam_sc'] * mult)
    nvsc = rng.poisson(inp['lam_vsc'] * mult)
    inizio_sc = np.zeros((n, L + 2), bool)
    for kind, cnt, (l0, l1) in ((1, nsc, P['sc_giri']), (2, nvsc, P['vsc_giri'])):
        for j in range(int(cnt.max()) if len(cnt) else 0):
            idx = np.where(cnt > j)[0]
            if kind == 1:
                # 13% delle SC ai giri 1-2, il resto distribuito sulla gara
                st0 = np.where(rng.random(len(idx)) < 0.13, rng.integers(2, 4, len(idx)),
                               rng.integers(2, max(L - 1, 3), len(idx)))
            else:
                st0 = rng.integers(max(2, int(0.1 * L)), max(L - 1, 3), len(idx))
            ln = rng.integers(l0, l1 + 1, len(idx))
            for i, s0, k in zip(idx, st0, ln):
                seg = stato[i, s0:s0 + k]
                seg[seg == 0] = kind
                if kind == 1:
                    inizio_sc[i, s0] = True

    # strategia: un piano comune per simulazione, seguito da ognuno con uno scarto di qualche giro;
    # una piccola quota sceglie un altro piano. Con il modello gomme i piani (mescole e giri di sosta)
    # vengono dalle strategie migliori della gara; senza, una o due soste in finestre fisse.
    piani = inp.get('piani') if inp['soste'] else None
    MAXS = 3
    if piani:
        gm = inp['gomme']
        pk = np.array([x['p'] for x in piani], float)
        pk /= pk.sum()
        K = len(piani)
        comune = rng.choice(K, size=n, p=pk)
        altro = rng.random((n, D)) < P['p_altra_strategia']
        propria = np.where(altro, rng.choice(K, size=(n, D), p=pk), comune[:, None])
        stop_lap = np.full((n, D, MAXS), L + 10, int)
        mesc = np.full((n, D, MAXS + 1), 1, int)
        for k, pl in enumerate(piani):
            m = propria == k
            if not m.any():
                continue
            for j, c in enumerate(pl['sequenza'][:MAXS + 1]):
                mesc[..., j][m] = G.COMP.index(c)
            for j, sl in enumerate(pl['soste'][:MAXS]):
                jit = np.round(rng.normal(0, 2.0, int(m.sum()))).astype(int)
                stop_lap[..., j][m] = np.clip(sl + jit, 3, L - 2)
        for j in range(1, MAXS):
            prev = stop_lap[..., j - 1]
            cur = stop_lap[..., j]
            spinto = np.maximum(cur, prev + 4)
            stop_lap[..., j] = np.where(cur <= L, np.where(spinto <= L - 2, spinto, L + 10), cur)
        nsoste = (stop_lap <= L).sum(2)
        off_c = np.array([gm['off'][c] for c in G.COMP])
        deg_c = np.array([max(gm['deg'][c], 0.005) for c in G.COMP])
        curv_c = np.array([gm['curv'][c] for c in G.COMP])
        ef_g = gm['efficace']
        gin = gm['ginocchio']
        ETA_SOFT = 3
        s1 = s2 = None
    elif inp['soste']:
        w0, w1 = P['finestra_sosta']
        due_c = rng.random(n) < inp['p_due']
        altro = rng.random((n, D)) < P['p_altra_strategia']
        due = due_c[:, None] ^ altro
        c1u = rng.uniform(w0, w1, n)[:, None] * L
        c1d = rng.uniform(0.25, 0.38, n)[:, None] * L
        c2d = rng.uniform(0.58, 0.70, n)[:, None] * L
        s1 = np.where(due, c1d, c1u) + rng.normal(0, 2.0, (n, D))
        s2 = np.where(due, c2d + rng.normal(0, 2.0, (n, D)), L + 10)
        s1 = np.clip(np.round(s1), 3, L - 3).astype(int)
        s2 = np.where(due, np.clip(np.round(s2), s1 + 5, L - 2), L + 10).astype(int)
    else:
        s1 = np.full((n, D), L + 10)
        s2 = np.full((n, D), L + 10)
    fatte = np.zeros((n, D), int)
    uscita = np.zeros((n, D), bool)
    scia = np.zeros((n, D), bool)
    if piani:
        eta = np.where(mesc[..., 0] == 0, ETA_SOFT, 0).astype(float)
    else:
        nsoste = np.where(s2 <= L, 2, np.where(s1 <= L, 1, 0))
        eta = np.zeros((n, D))

    # primo giro: scatto e rimescolamento
    a, b, a_pole = inp['sig1']
    g = inp['griglia']
    sig = a + b * (g - 1)
    sig[g == 1] = a_pole
    score = g[None, :] - P['peso_scatto'] * inp['scatto'][None, :] / P['metri_per_posizione'] \
        + sig[None, :] * rng.standard_normal((n, D))
    score[:, g == 2] += inp.get('delta2', 0.0)
    inc = rng.random((n, D)) < inp['p_incidente1']
    score = score + inc * rng.uniform(4, 12, (n, D))
    score = np.where(giro_rit == 1, 1e6 + score, score)
    ordine = np.argsort(score, 1)
    rank = np.empty_like(ordine)
    rank[ar[:, None], ordine] = np.arange(D)[None, :]
    gaps = P['gap_lap1'] * rng.uniform(0.5, 1.5, (n, D))
    gaps[:, 0] = 0
    cum_pos = np.cumsum(gaps, 1)
    cum = np.empty((n, D))
    cum[ar[:, None], ordine] = cum_pos
    leader1 = ordine[:, 0].copy()
    rank1 = rank.copy()
    eta += 1
    attivo = giro_rit > 1
    cum[~attivo] = 1e9 + rng.random((~attivo).sum())

    thr, pmax = inp['sorpasso']
    gmin = P['gap_min']
    n_sorpassi = 0
    big = 1e8
    for lap in range(2, L + 1):
        stt = stato[:, lap]
        verde = stt == 0
        sc = stt == 1
        vsc = stt == 2
        # giri a potenza ridotta
        g_on = (lap >= gu_in) & (lap < gu_in + gu_len)
        if piani:
            c_now = np.take_along_axis(mesc, np.minimum(fatte, MAXS)[..., None], axis=2)[..., 0]
            gomma = off_c[c_now] + ef_g * (deg_c[c_now] * eta + curv_c[c_now] * np.maximum(0.0, eta - gin) ** 2)
        else:
            gomma = inp['degrado'] * eta
        t_sis = passo + gomma + gu_loss * g_on + P['giro_uscita'] * uscita \
            + P['aria_sporca'] * scia
        uscita = np.zeros((n, D), bool)
        err_g = (rng.random((n, D)) < P['errore_giro_p']) * rng.uniform(1.5, 5.0, (n, D))
        t_giro = t_sis + rumore * rng.standard_normal((n, D)) + err_g
        t_adv = t_sis + err_g  # vantaggio "vero" per il sorpasso: passo, gomme, guai, errori
        # in neutralizzazione nessuno guadagna sugli altri
        t_giro = np.where(verde[:, None], t_giro, 0.05 * rng.standard_normal((n, D)))
        new = cum + t_giro
        # SC: gruppo ricompattato al primo giro di SC
        start_sc = inizio_sc[:, lap]
        if start_sc.any():
            idx = np.where(start_sc)[0]
            sub = new[idx]
            o = np.argsort(sub, 1)
            pos = np.empty_like(o)
            pos[np.arange(len(idx))[:, None], o] = np.arange(D)[None, :]
            lead = sub[np.arange(len(idx)), o[:, 0]]
            comp = lead[:, None] + P['gap_sc'] * pos
            alive = sub < big
            new[idx] = np.where(alive, comp, sub)
        # sorpassi (solo giri verdi, auto in corsa)
        if verde.any():
            for p in range(1, D):
                ia = ordine[:, p - 1]
                ib = ordine[:, p]
                ca = new[ar, ia]
                cb = new[ar, ib]
                vivi = (ca < big) & (cb < big) & verde
                vicino = vivi & (cb < ca + gmin)
                if not vicino.any():
                    continue
                adv = t_adv[ar, ia] - t_adv[ar, ib]
                pp = pmax / (1 + np.exp(-(adv - thr) / 0.10))
                ok = vicino & (rng.random(n) < pp)
                ko = vicino & ~ok
                new[ar[ko], ib[ko]] = ca[ko] + gmin
                if ok.any():
                    n_sorpassi += ok.sum()
                    r_ = ar[ok]
                    nb = np.minimum(cb[ok], ca[ok] - 0.1)
                    new[r_, ib[ok]] = nb
                    new[r_, ia[ok]] = np.maximum(ca[ok], nb + 0.3)
                    ordine[r_, p - 1] = ib[ok]
                    ordine[r_, p] = ia[ok]
        # soste
        if inp['soste']:
            if piani:
                prossima = np.where(fatte < nsoste,
                                    np.take_along_axis(stop_lap, np.minimum(fatte, MAXS - 1)[..., None], axis=2)[..., 0],
                                    L + 10)
            else:
                prossima = np.where(fatte == 0, s1, s2)
            # anticipo della sosta sotto neutralizzazione
            neutr = sc | vsc
            if neutr.any():
                fin = np.where(sc, 0.35 * L, 0.15 * L)[:, None]
                anticipa = neutr[:, None] & (fatte < nsoste) & (prossima > lap) & (prossima - lap <= fin) \
                    & (lap >= 0.2 * L)
                if anticipa.any():
                    prossima = np.where(anticipa, lap, prossima)
                    if piani:
                        r_, d_ = np.where(anticipa)
                        stop_lap[r_, d_, fatte[r_, d_]] = lap
                    else:
                        s1 = np.where(anticipa & (fatte == 0), lap, s1)
                        s2 = np.where(anticipa & (fatte == 1), lap, s2)
            box = (fatte < nsoste) & (prossima == lap)
            box &= new < big
            if box.any():
                # chi è davanti copre il tentativo di undercut di chi lo segue entro 3 s
                for p in range(D - 1):
                    ia = ordine[:, p]
                    ib = ordine[:, p + 1]
                    prox_a = prossima[ar, ia]
                    cop = box[ar, ib] & ~box[ar, ia] & (fatte[ar, ia] == fatte[ar, ib]) \
                        & (cum[ar, ib] - cum[ar, ia] < 3.0) & (prox_a > lap) & (prox_a - lap <= 8) & verde
                    if cop.any():
                        r_ = ar[cop]
                        a_ = ia[cop]
                        if piani:
                            stop_lap[r_, a_, fatte[r_, a_]] = lap + 1
                        else:
                            f0 = fatte[r_, a_] == 0
                            s1[r_[f0], a_[f0]] = lap + 1
                            s2[r_[~f0], a_[~f0]] = lap + 1
                perdita = inp['perdita_box'] * np.where(sc, P['sc_sconto_box'], np.where(vsc, P['vsc_sconto_box'], 1.0))
                lento = rng.random((n, D)) < 0.03
                costo = perdita[:, None] + P['rumore_box'] * np.abs(rng.standard_normal((n, D))) + lento * rng.uniform(3, 8, (n, D))
                new = new + box * costo
                uscita = box.copy()
                fatte = fatte + box
                if piani:
                    c_new = np.take_along_axis(mesc, np.minimum(fatte, MAXS)[..., None], axis=2)[..., 0]
                    eta = np.where(box, np.where(c_new == 0, ETA_SOFT, 0), eta)
                else:
                    eta = np.where(box, 0, eta)
        # ritiri di questo giro
        r_now = giro_rit == lap
        if r_now.any():
            new = np.where(r_now, 1e9 + (L - lap) * 1000 + rng.random((n, D)), new)
        cum = new
        ordine = np.argsort(cum, 1, kind='stable')
        # chi è a meno di 1 s da chi lo precede: passo e gomme peggiori il giro dopo
        cs = cum[ar[:, None], ordine]
        dgap = np.diff(cs, axis=1)
        scia_o = np.zeros((n, D), bool)
        scia_o[:, 1:] = (dgap < 1.0) & (cs[:, 1:] < big)
        scia = np.zeros((n, D), bool)
        scia[ar[:, None], ordine] = scia_o
        scia &= verde[:, None]
        eta += np.where(scia, P['usura_scia'], 1.0)

    fin = ordine
    win = np.bincount(fin[:, 0], minlength=D) / n
    pod = np.zeros(D)
    for k in range(3):
        pod += np.bincount(fin[:, k], minlength=D) / n
    out = dict(
        vittoria={d: float(x) for d, x in zip(inp['piloti'], win)},
        podio={d: float(x) for d, x in zip(inp['piloti'], pod)},
        testa_giro1={d: float(x) for d, x in zip(inp['piloti'], np.bincount(leader1, minlength=D) / n)},
        testa1_vince=float(np.mean(fin[:, 0] == leader1)),
        pole_vince=float(np.mean(fin[:, 0] == int(np.argmin(inp['griglia'])))),
        veloce_vince=float(np.mean(fin[:, 0] == np.argmin(passo + 1e6 * (giro_rit < 0.25 * L), 1))),
        pole_testa1_vince=float(np.mean((fin[:, 0] == int(np.argmin(inp['griglia'])))[leader1 == int(np.argmin(inp['griglia']))]))
        if np.any(leader1 == int(np.argmin(inp['griglia']))) else float('nan'),
        pole_ritiro=float(np.mean(giro_rit[:, int(np.argmin(inp['griglia']))] <= L)),
        sorpassi=float(n_sorpassi / n),
        rimescolamento=_rimescolamento(rank1, fin, giro_rit, L),
    )
    if dettagli:
        out['ordine'] = fin
    return out


# ---------------------------------------------------------------------------
# 4. Prova a ritroso e punteggi
# ---------------------------------------------------------------------------

def esito(e):
    """Vincitore e podio reali (posizioni di arrivo, squalifiche comprese come da classifica)."""
    cl = sorted([d for d in e['arrivo'] if d in e['griglia']], key=lambda d: e['arrivo'][d])
    return cl[0], set(cl[:3])


def punteggi(prob_v, prob_p, e):
    vinc, pod = esito(e)
    ds = list(e['griglia'])
    bv = sum((prob_v.get(d, 0.0) - (d == vinc)) ** 2 for d in ds)
    bp = sum((prob_p.get(d, 0.0) - (d in pod)) ** 2 for d in ds)
    lv = -math.log(max(prob_v.get(vinc, 0.0), 1e-3))
    lp = -sum(math.log(max(prob_p.get(d, 0.0), 1e-3)) if d in pod else
              math.log(max(1 - prob_p.get(d, 0.0), 1e-3)) for d in ds) / len(ds)
    return dict(brier_v=bv, brier_p=bp, log_v=lv, log_p=lp)


def ricalibrazione(passate):
    """Correzione di taratura stimata sulle sole previsioni passate.
    Vittoria: p' proporzionale a p^tau. Podio: p' = logistica(a + b*logit(p)).
    Priori: tau ~ N(1, 0.35), b ~ N(1, 0.35), a ~ N(0, 0.5)."""
    tau, a, b = 1.0, 0.0, 1.0
    if len(passate) >= 3:
        best = None
        for t in np.arange(0.5, 3.01, 0.05):
            ll = -0.5 * ((t - 1) / 0.35) ** 2
            for pv, pp, vinc, pod in passate:
                p = np.clip(np.array(list(pv.values())), 1e-4, 1)
                pw = max(pv.get(vinc, 0.0), 1e-4)
                ll += t * math.log(pw) - math.log(np.sum(p ** t))
            if best is None or ll > best[0]:
                best = (ll, t)
        tau = float(best[1])
        x = np.concatenate([np.array(list(pp.values())) for pv, pp, vinc, pod in passate])
        y = np.concatenate([np.array([d in pod for d in pp]) for pv, pp, vinc, pod in passate]).astype(float)
        lx = np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
        best = None
        for aa in np.arange(-1.5, 1.51, 0.1):
            for bb in np.arange(0.5, 2.51, 0.05):
                z = aa + bb * lx
                ll = np.sum(y * z - np.logaddexp(0, z)) - 0.5 * ((bb - 1) / 0.35) ** 2 - 0.5 * (aa / 0.5) ** 2
                if best is None or ll > best[0]:
                    best = (ll, aa, bb)
        a, b = float(best[1]), float(best[2])
    return tau, a, b


def applica_ricalibrazione(pv, pp, cal):
    tau, a, b = cal
    ds = list(pv)
    p = np.clip(np.array([pv[d] for d in ds]), 1e-6, 1) ** tau
    p = p / p.sum()
    q = np.clip(np.array([pp[d] for d in ds]), 1e-4, 1 - 1e-4)
    z = a + b * np.log(q / (1 - q))
    q2 = 1 / (1 + np.exp(-z))
    return dict(zip(ds, p.tolist())), dict(zip(ds, q2.tolist()))


def prova(eventi, n_sim=None, dal_round=4, seed=0, verbose=True, modifica=None):
    """Previsione di ogni evento con i soli eventi precedenti. Ritorna righe per pilota ed evento."""
    righe = []
    passate = []
    for k, e in enumerate(eventi):
        if k == 0:
            continue
        prec = [x for x in eventi[:k] if x['t_via'] < e['t_via']]
        st = Storia(prec)
        inp = prepara(e, st)
        if modifica:
            modifica(inp, e, st)
        out = simula(inp, n=n_sim, seed=seed + k)
        gv, gp = prob_griglia(e, st.b_griglia)
        vinc, pod = esito(e)
        cal = ricalibrazione(passate)
        cv, cp = applica_ricalibrazione(out['vittoria'], out['podio'], cal)
        passate.append((out['vittoria'], out['podio'], vinc, pod))
        for d in e['griglia']:
            righe.append(dict(
                anno=e['anno'], round=e['round'], tipo=e['tipo'], luogo=e['luogo'], pilota=e['piloti'][d][0],
                team=_team(e, d), griglia=e['griglia'][d],
                p_vitt=out['vittoria'][d], p_podio=out['podio'][d], p_testa1=out['testa_giro1'][d],
                c_vitt=cv[d], c_podio=cp[d], tau=cal[0],
                g_vitt=gv[d], g_podio=gp[d],
                vince=int(d == vinc), podio=int(d in pod), valutato=int(e['round'] >= dal_round),
                testa1=int(bool(e['giro1']) and e['giro1'].get(d) == 1)))
        if verbose:
            s1 = punteggi(out['vittoria'], out['podio'], e)
            s2 = punteggi(gv, gp, e)
            top = sorted(out['vittoria'].items(), key=lambda x: -x[1])[:3]
            print(f"{e['round']:>2} {e['tipo']} {e['luogo'][:12]:<12} vince {e['piloti'][vinc][0]} "
                  f"(mod {out['vittoria'][vinc]:.2f}, gri {gv[vinc]:.2f})  "
                  f"Brier V {s1['brier_v']:.3f}/{s2['brier_v']:.3f}  P {s1['brier_p']:.3f}/{s2['brier_p']:.3f}  "
                  f"fav: " + ', '.join(f"{e['piloti'][d][0]} {p:.2f}" for d, p in top), flush=True)
    return righe


def riepilogo(righe, filtro=lambda r: r['valutato']):
    R = [r for r in righe if filtro(r)]
    ev = defaultdict(list)
    for r in R:
        ev[(r['anno'], r['round'], r['tipo'])].append(r)
    tab = {}
    for nome, pv, pp in (('modello', 'p_vitt', 'p_podio'), ('ricalibrato', 'c_vitt', 'c_podio'),
                         ('griglia', 'g_vitt', 'g_podio')):
        bv = [sum((r[pv] - r['vince']) ** 2 for r in rs) for rs in ev.values()]
        bp = [sum((r[pp] - r['podio']) ** 2 for r in rs) for rs in ev.values()]
        lv = [-math.log(max(sum(r[pv] for r in rs if r['vince']), 1e-3)) for rs in ev.values()]
        tab[nome] = dict(brier_v=float(np.mean(bv)), brier_p=float(np.mean(bp)), log_v=float(np.mean(lv)),
                         bv=bv, bp=bp, lv=lv)
    tab['n_eventi'] = len(ev)
    return tab


def bootstrap_diff(a, b, n=20000, seed=3):
    """Differenza media (a - b) con intervallo al 90% per ricampionamento degli eventi."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 5)), float(np.percentile(m, 95)), float((d < 0).mean())


FASCE = [(0, .05), (.05, .10), (.10, .20), (.20, .35), (.35, .50), (.50, .70), (.70, 1.0001)]


def taratura(righe, col, esito_col, filtro=lambda r: r['valutato']):
    R = [r for r in righe if filtro(r)]
    out = []
    for lo, hi in FASCE:
        s = [r for r in R if lo <= r[col] < hi]
        if not s:
            out.append((lo, hi, 0, None, None))
            continue
        out.append((lo, hi, len(s), float(np.mean([r[col] for r in s])), float(np.mean([r[esito_col] for r in s]))))
    return out


# ---------------------------------------------------------------------------
# 5. Uscite e riga di comando
# ---------------------------------------------------------------------------

def scrivi_csv(path, righe):
    if not righe:
        return
    cols = list(righe[0].keys())
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in righe:
            f.write(','.join(f"{r[c]:.5f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + '\n')


def leggi_csv(path):
    out = []
    with open(path, encoding='utf-8') as f:
        cols = f.readline().strip().split(',')
        for line in f:
            v = line.strip().split(',')
            r = {}
            for c, x in zip(cols, v):
                try:
                    r[c] = float(x) if '.' in x else int(x)
                except ValueError:
                    r[c] = x
            out.append(r)
    return out


def passate_da_csv(righe):
    """Ricostruisce le previsioni passate (per la ricalibrazione) da prova_righe.csv."""
    ev = defaultdict(list)
    for r in righe:
        ev[(r['anno'], r['round'], r['tipo'])].append(r)
    out = []
    for k in sorted(ev):
        rs = ev[k]
        pv = {r['pilota']: r['p_vitt'] for r in rs}
        pp = {r['pilota']: r['p_podio'] for r in rs}
        vinc = [r['pilota'] for r in rs if r['vince']]
        pod = {r['pilota'] for r in rs if r['podio']}
        if vinc:
            out.append((pv, pp, vinc[0], pod))
    return out


def cmd_prova(a):
    ev = estrai_eventi(a.storico, anni=(a.anno,))
    os.makedirs(a.out, exist_ok=True)
    righe = prova(ev, n_sim=a.n, dal_round=a.dal_round)
    scrivi_csv(os.path.join(a.out, 'prova_righe.csv'), righe)
    rie = {}
    for nome, f in (('tutti', lambda r: r['valutato']), ('gare', lambda r: r['valutato'] and r['tipo'] == 'R'),
                    ('sprint', lambda r: r['valutato'] and r['tipo'] == 'S')):
        T = riepilogo(righe, f)
        if not T['n_eventi']:
            continue
        rie[nome] = {
            'n_eventi': T['n_eventi'],
            'punteggi': {m: {k: v for k, v in T[m].items() if not isinstance(v, list)}
                         for m in ('modello', 'ricalibrato', 'griglia')},
            'diff_modello_griglia': {'brier_v': bootstrap_diff(T['modello']['bv'], T['griglia']['bv']),
                                     'brier_p': bootstrap_diff(T['modello']['bp'], T['griglia']['bp']),
                                     'log_v': bootstrap_diff(T['modello']['lv'], T['griglia']['lv'])},
        }
    rie['taratura'] = {c: taratura(righe, c, es) for c, es in
                       (('p_vitt', 'vince'), ('c_vitt', 'vince'), ('g_vitt', 'vince'),
                        ('p_podio', 'podio'), ('c_podio', 'podio'), ('g_podio', 'podio'),
                        ('p_testa1', 'testa1'))}
    with open(os.path.join(a.out, 'prova_riepilogo.json'), 'w', encoding='utf-8') as f:
        json.dump(rie, f, indent=1, ensure_ascii=False)
    T = rie['tutti']['punteggi']
    print('\nBrier vittoria  modello %.3f  ricalibrato %.3f  griglia %.3f' % (
        T['modello']['brier_v'], T['ricalibrato']['brier_v'], T['griglia']['brier_v']))
    print('Brier podio     modello %.3f  ricalibrato %.3f  griglia %.3f' % (
        T['modello']['brier_p'], T['ricalibrato']['brier_p'], T['griglia']['brier_p']))


def cmd_prevedi(a):
    tutti = estrai_eventi(a.storico, anni=(a.anno,), futuri=True)
    target = [e for e in tutti if e['round'] == a.round and e['tipo'] == a.tipo]
    if not target:
        sys.exit('Evento non trovato: serve almeno la qualifica nel file del weekend.')
    e = target[0]
    if a.griglia:
        # griglia data a mano (sigle in ordine), utile prima che OpenF1 pubblichi quella ufficiale
        per_sigla = {v[0]: d for d, v in e['piloti'].items()}
        ordine = [per_sigla[s.strip().upper()] for s in a.griglia.split(',') if s.strip().upper() in per_sigla]
        resto = [d for d in sorted(e['griglia'], key=lambda d: e['griglia'][d]) if d not in ordine]
        e['griglia'] = {d: i + 1 for i, d in enumerate(ordine + resto)}
    prec = [x for x in tutti if not x['futuro'] and x['t_via'] < e['t_via']]
    altri = estrai_eventi(a.storico, anni=tuple(y for y in range(2018, 2030) if y != a.anno))
    circ = [x for x in altri if x['circuito'] == e['circuito']]
    st = Storia(prec)
    inp = prepara(e, st, pioggia=a.pioggia, circuito=circ, giri=a.giri)
    out = simula(inp, n=a.n, seed=1)
    gv, _ = prob_griglia(e, st.b_griglia)
    cv, cp = out['vittoria'], out['podio']
    if a.ricalibra and os.path.exists(a.ricalibra):
        cal = ricalibrazione(passate_da_csv(leggi_csv(a.ricalibra)))
        cv, cp = applica_ricalibrazione(out['vittoria'], out['podio'], cal)
        print(f"Ricalibrazione da {a.ricalibra}: tau {cal[0]:.2f}, podio a {cal[1]:.2f} b {cal[2]:.2f}")
    print(f"{e['gp']} {e['anno']} — {'Gara' if e['tipo'] == 'R' else 'Sprint'}: {inp['giri']} giri, "
          f"{len(prec)} eventi di storia, {len(circ)} precedenti sul circuito, "
          f"pioggia {inp['p_bagnato']:.0%}, SC {inp['lam_sc']:.2f}, VSC {inp['lam_vsc']:.2f}, box {inp['perdita_box']:.1f} s"
          + (f", sorpassi spostati di {P['sorpasso_shift']:+.1f} livelli" if P['sorpasso_shift'] else ''))
    if st.gomme is not None and e['tipo'] == 'R':
        # informazione sulle gomme (le probabilità qui sotto usano le gomme solo con --gomme)
        gp = st.gomme.previsione(runs=e.get('long_run') or None)
        piani = G.piani_strategia(gp, inp['giri'], inp['perdita_box'])
        pst = defaultdict(float)
        for pl in piani:
            pst[len(pl['soste'])] += pl['p']
        print(f"Gomme ({gp['fonte']}): degrado soft {gp['deg']['SOFT']:.3f}, medium {gp['deg']['MEDIUM']:.3f}, "
              f"hard {gp['deg']['HARD']:.3f} s/giro; soste " + ', '.join(f"{k}: {v:.0%}" for k, v in sorted(pst.items()))
              + '; piano più probabile ' + '-'.join(c[0] for c in piani[0]['sequenza'])
              + ' al giro ' + ', '.join(str(x) for x in piani[0]['soste'])
              + (' (simulato)' if P.get('gomme') else ''))
    print(f"{'pilota':<5}{'gr':>4}{'vitt':>8}{'q.min':>7}{'podio':>8}{'q.min':>7}{'testa g1':>10}{'griglia':>9}")
    ds = sorted(e['griglia'], key=lambda d: -cv[d])
    for d in ds[:12]:
        v, p = cv[d], cp[d]
        qv = f"{1 / v:7.2f}" if v >= 0.002 else '      -'
        qp = f"{1 / p:7.2f}" if p >= 0.002 else '      -'
        print(f"{e['piloti'][d][0]:<5}{e['griglia'][d]:>4}{v:>8.1%}{qv}{p:>8.1%}{qp}"
              f"{out['testa_giro1'][d]:>10.1%}{gv[d]:>9.1%}")
    fav = ds[0]
    print(f"\nFavorito: {e['piloti'][fav][0]} — probabilità {cv[fav]:.1%} — quota minima {1 / cv[fav]:.2f}")


def main():
    ap = argparse.ArgumentParser(description='Modello di gara Muretto')
    sp = ap.add_subparsers(dest='cmd', required=True)
    p1 = sp.add_parser('prova', help='prova a ritroso sulla stagione')
    p1.add_argument('--storico', default='storico')
    p1.add_argument('--anno', type=int, default=2026)
    p1.add_argument('--out', default='risultati')
    p1.add_argument('--n', type=int, default=P['n_sim'])
    p1.add_argument('--dal-round', dest='dal_round', type=int, default=4)
    p1.add_argument('--shift-sorpassi', dest='shift', type=float, default=0.0,
                    help='sposta tutti i circuiti nella tabella dei sorpassi (-1 = un livello più facile)')
    p1.add_argument('--gomme', action='store_true', help='simula mescole e strategie con modello_gomme.py (sperimentale)')
    p2 = sp.add_parser('prevedi', help='previsione di una gara o sprint dopo la qualifica')
    p2.add_argument('--storico', default='storico')
    p2.add_argument('--anno', type=int, default=2026)
    p2.add_argument('--round', type=int, required=True)
    p2.add_argument('--tipo', choices=['R', 'S'], default='R')
    p2.add_argument('--giri', type=int, default=None)
    p2.add_argument('--pioggia', type=float, default=None, help='rischio pioggia 0-1 (se manca il messaggio FIA)')
    p2.add_argument('--n', type=int, default=20000)
    p2.add_argument('--ricalibra', default=None, help='prova_righe.csv della prova a ritroso')
    p2.add_argument('--griglia', default=None, help='sigle in ordine di griglia, es. "VER,LEC,HAM" (penalità)')
    p2.add_argument('--shift-sorpassi', dest='shift', type=float, default=0.0,
                    help='sposta tutti i circuiti nella tabella dei sorpassi (-1 = un livello più facile)')
    p2.add_argument('--gomme', action='store_true', help='simula mescole e strategie con modello_gomme.py (sperimentale)')
    a = ap.parse_args()
    P['sorpasso_shift'] = a.shift
    P['gomme'] = bool(getattr(a, 'gomme', False))
    if P['gomme'] and G is None:
        sys.exit('Per --gomme serve modello_gomme.py nella stessa cartella.')
    {'prova': cmd_prova, 'prevedi': cmd_prevedi}[a.cmd](a)


if __name__ == '__main__':
    main()
