#!/usr/bin/env python3
"""
modello_gomme.py — Muretto: modello gomme F1 (usura, crollo, strategie) e lettura dei long run.

Un solo file, solo numpy (Python 3.9+). Legge l'archivio OpenF1 di Muretto (storico/).
Lo usa anche modello_gara.py, se sta nella stessa cartella (opzione --gomme e strategie in uscita).

Uso:
  python modello_gomme.py scheda --storico storico --round 18
      scheda gomme del weekend: stagione, storia del circuito, long run delle libere (se già
      nell'archivio), degrado previsto, vita utile, strategie e finestre di sosta, undercut
  python modello_gomme.py scheda --storico storico --round 18 --circuito Austin --giri 56
      la stessa prima del weekend, quando il file del round non c'è ancora
  python modello_gomme.py prova --storico storico
      verifica sul 2026, solo dati precedenti: degrado di gara previsto, numero di soste,
      undercut, passo dei long run contro passo di gara

Modello (per mescola, tempo sul giro rispetto a una medium nuova):
  perdita(età) = scarto mescola + degrado * età + crollo * max(0, età - 5)^2
  - scarto e degrado: mediane delle gare precedenti (giri giovani: soft fino a 10 giri, medium 15,
    hard 22), il degrado corretto con i long run del weekend (riscalati sulle coppie libere-gara)
  - crollo: stimato su tutti i giri delle gare precedenti oltre i giri giovani
  - efficace: quota della perdita che si vede nei duelli (chi si ferma contro chi resta fuori),
    stimata sulle soste delle gare precedenti
  - strategie: tutte le sequenze ammesse (almeno due mescole, soft solo prima o ultima con un treno
    usato in qualifica, partenza con la hard penalizzata), suddivisione ottima degli stint,
    soste anticipate di 3 giri (pressione dell'undercut)
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

COMP = ('SOFT', 'MEDIUM', 'HARD')
ASCIUTTE = set(COMP)


# ---------------------------------------------------------------------------
# 0. Utilità
# ---------------------------------------------------------------------------

def _t(s):
    if s is None:
        return None
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def carica_weekend(path):
    with gzip.open(path) as f:
        return json.load(f)


def sessioni(d):
    return {s['session']['session_name']: s for s in d['sessioni']}


def intervalli_bandiere(sess, giro_rif, gialle=False):
    """Intervalli (inizio, fine, tipo) di SC, VSC, bandiera rossa e, se richiesto, doppie gialle."""
    iv = []
    aperto = {}
    settori = {}
    for m in sorted(sess.get('race_control') or [], key=lambda x: x['date']):
        msg = (m.get('message') or '').upper()
        t = _t(m['date'])
        flag = m.get('flag')
        if msg == 'SAFETY CAR DEPLOYED':
            aperto['SC'] = t
        elif msg.startswith('SAFETY CAR IN THIS LAP') and 'SC' in aperto:
            iv.append((aperto.pop('SC'), t + 1.5 * giro_rif, 'SC'))
        elif msg in ('VSC DEPLOYED', 'VIRTUAL SAFETY CAR DEPLOYED'):
            aperto['VSC'] = t
        elif msg in ('VSC ENDING', 'VIRTUAL SAFETY CAR ENDING') and 'VSC' in aperto:
            iv.append((aperto.pop('VSC'), t + 30.0, 'VSC'))
        elif flag == 'RED' or msg.startswith('RED FLAG'):
            for k in ('SC', 'VSC'):
                if k in aperto:
                    iv.append((aperto.pop(k), t, k))
            aperto.setdefault('RED', t)
        elif m.get('category') == 'SessionStatus' and msg == 'SESSION STARTED' and 'RED' in aperto:
            iv.append((aperto.pop('RED'), t + 2.5 * giro_rif, 'RED'))
        if gialle:
            sett = m.get('sector')
            if flag == 'DOUBLE YELLOW' and sett is not None:
                settori.setdefault(sett, t)
            elif flag == 'CLEAR' and sett is not None and sett in settori:
                iv.append((settori.pop(sett), t + 5.0, 'GIALLA'))
            elif flag in ('CLEAR', 'GREEN') and m.get('scope') == 'Track':
                for k, t0 in list(settori.items()):
                    iv.append((t0, t + 5.0, 'GIALLA'))
                    settori.pop(k)
    for k, t in aperto.items():
        iv.append((t, t + 1e5, k))
    for k, t in settori.items():
        iv.append((t, t + 120.0, 'GIALLA'))
    return iv


def stint_per_giro(sess, giri_pilota, dn):
    """Mescola, età gomma e numero di stint per giro, con gli stint ricostruiti dai giri di uscita.
    Un tratto senza una voce di stint che inizia lì resta senza mescola."""
    out_laps = sorted(l['lap_number'] for l in giri_pilota if l.get('is_pit_out_lap') and l['lap_number'] > 1)
    starts = [1] + out_laps
    st = [s for s in sess.get('stints') or [] if s['driver_number'] == dn]
    mapping = {}
    maxlap = max([l['lap_number'] for l in giri_pilota] or [0])
    for i, s0 in enumerate(starts):
        s1 = starts[i + 1] - 1 if i + 1 < len(starts) else maxlap
        cand = [s for s in st if s['lap_start'] == s0]
        if not cand and i == 0:
            cand = [s for s in st if (s.get('lap_start') or 0) <= 1]
        comp = cand[0]['compound'] if cand else None
        age0 = (cand[0].get('tyre_age_at_start') or 0) if cand else 0
        for lap in range(s0, s1 + 1):
            mapping[lap] = (comp, age0 + lap - s0, i)
    return mapping


def temperatura(sess):
    """Funzione tempo -> temperatura asfalto (interpolata), e media della sessione."""
    w = sorted([(_t(x['date']), x['track_temperature']) for x in sess.get('weather') or []
                if x.get('track_temperature') is not None])
    if not w:
        return (lambda t: None), None
    ts = np.array([a for a, _ in w])
    vs = np.array([b for _, b in w], float)
    t0 = _t(sess['session']['date_start'])
    t1 = _t(sess['session'].get('date_end')) or (t0 + 7200)
    m = (ts >= t0) & (ts <= t1)
    media = float(vs[m].mean()) if m.any() else float(vs.mean())
    return (lambda t: float(np.interp(t, ts, vs))), media


def bagnata(sess, soglia=0.10):
    tot = wet = 0
    for st in sess.get('stints') or []:
        n = max((st.get('lap_end') or 0) - (st.get('lap_start') or 0) + 1, 0)
        tot += n
        if st.get('compound') in ('INTERMEDIATE', 'WET'):
            wet += n
    return bool(tot and wet / tot > soglia)


# ---------------------------------------------------------------------------
# 1. Estrazione: giri di gara e long run delle libere
# ---------------------------------------------------------------------------

def giri_gara(sess, max_lento=1.07):
    """Giri di gara utilizzabili per il modello gomme, uno per riga (dict).
    Tolti: giro 1, giri di uscita e di rientro, giri con SC/VSC/rossa, giri oltre il 107% della
    mediana, mescole da bagnato. Tenuti i giri nel traffico, con la distanza da chi precede."""
    laps = [l for l in sess.get('laps') or [] if l.get('lap_duration') and l.get('date_start')]
    if not laps:
        return []
    med = float(np.median([l['lap_duration'] for l in laps]))
    iv = intervalli_bandiere(sess, med)
    cross = np.array(sorted(_t(l['date_start']) for l in laps))
    pit = defaultdict(set)
    for p in sess.get('pit') or []:
        pit[p['driver_number']].add(p['lap_number'])
    team = {x['driver_number']: x.get('team_name') for x in sess.get('drivers') or []}
    sigla = {x['driver_number']: x.get('name_acronym') for x in sess.get('drivers') or []}
    temp_f, _ = temperatura(sess)
    by = defaultdict(list)
    for l in sess['laps']:
        by[l['driver_number']].append(l)
    n_tot = max(l['lap_number'] for l in laps)
    rows = []
    for dn, gl in by.items():
        mp = stint_per_giro(sess, gl, dn)
        for l in gl:
            ln = l['lap_number']
            if ln <= 1 or not l.get('lap_duration') or not l.get('date_start'):
                continue
            if l.get('is_pit_out_lap') or ln in pit[dn]:
                continue
            t0 = _t(l['date_start'])
            t1 = t0 + l['lap_duration']
            if any(a <= t1 and t0 <= b for a, b, _ in iv):
                continue
            comp, age, k = mp.get(ln, (None, None, None))
            if comp not in ASCIUTTE or l['lap_duration'] > max_lento * med:
                continue
            j = np.searchsorted(cross, t0) - 1
            while j >= 0 and abs(cross[j] - t0) < 1e-6:
                j -= 1
            gap = float(t0 - cross[j]) if j >= 0 else 99.0
            # primo giro lanciato dello stint (il giro dopo quello di uscita)
            primo = (ln - 1) in mp and mp[ln - 1][2] == k and any(
                x['lap_number'] == ln - 1 and x.get('is_pit_out_lap') for x in gl)
            rows.append(dict(pilota=dn, sigla=sigla.get(dn), team=team.get(dn), giro=ln, n_giri=n_tot,
                             mescola=comp, eta=age, stint=k, tempo=l['lap_duration'], gap=gap,
                             primo=bool(primo), t=t0, temp=temp_f(t0)))
    return rows


def long_run(sess, min_giri=5, banda=0.03, gap_traffico=1.0):
    """Long run di una sessione di libere: per ogni pilota e ogni treno di gomme, i giri a ritmo gara.
    Un long run è il tratto più lungo di giri consecutivi (con al più due giri scartati in mezzo)
    entro ±banda dalla mediana dello stint, senza rossa né doppia gialla.
    Ritorna una lista di run: dict con pilota, team, mescola, giri [(indice nel run, età, tempo, t, traffico)]."""
    laps = [l for l in sess.get('laps') or [] if l.get('lap_duration') and l.get('date_start')]
    if not laps or bagnata(sess, 0.3):
        return []
    med = float(np.median([l['lap_duration'] for l in laps]))
    iv = intervalli_bandiere(sess, med, gialle=True)
    cross = np.array(sorted(_t(l['date_start']) for l in laps))
    team = {x['driver_number']: x.get('team_name') for x in sess.get('drivers') or []}
    sigla = {x['driver_number']: x.get('name_acronym') for x in sess.get('drivers') or []}
    temp_f, _ = temperatura(sess)
    t_sess = _t(sess['session']['date_start'])
    by = defaultdict(list)
    for l in sess['laps']:
        by[l['driver_number']].append(l)
    runs = []
    for dn, gl in by.items():
        gl = sorted(gl, key=lambda l: l['lap_number'])
        mp = stint_per_giro(sess, gl, dn)
        stints = defaultdict(list)
        for l in gl:
            if l['lap_number'] in mp:
                stints[mp[l['lap_number']][2]].append(l)
        for sl in stints.values():
            comp = mp[sl[0]['lap_number']][0]
            if comp not in ASCIUTTE:
                continue
            cand = []
            for l in sl:
                if l.get('is_pit_out_lap') or not l.get('lap_duration') or not l.get('date_start'):
                    continue
                t0 = _t(l['date_start'])
                t1 = t0 + l['lap_duration']
                bandiera = any(a <= t1 and t0 <= b for a, b, _ in iv)
                j = np.searchsorted(cross, t0) - 1
                while j >= 0 and abs(cross[j] - t0) < 1e-6:
                    j -= 1
                gap = float(t0 - cross[j]) if j >= 0 else 99.0
                cand.append((l['lap_number'], mp[l['lap_number']][1], l['lap_duration'], t0, gap, bandiera))
            if len(cand) < min_giri:
                continue
            tempi = np.array([c[2] for c in cand])
            # mediana dei giri «di ritmo»: tolgo i giri lenti (raffreddamento, rientro)
            base = np.median(np.sort(tempi)[:max(min_giri, len(tempi) // 2 + 1)])
            ok = (np.abs(tempi / base - 1) <= banda)
            # tratto consecutivo più lungo con al più 2 buchi
            best, cur, buchi = [], [], 0
            for c, o in zip(cand, ok):
                if o and not c[5]:
                    cur.append(c)
                    buchi = 0
                else:
                    buchi += 1
                    if buchi > 2:
                        if len(cur) > len(best):
                            best = cur
                        cur, buchi = [], 0
            if len(cur) > len(best):
                best = cur
            if len(best) < min_giri:
                continue
            l0 = best[0][0]
            runs.append(dict(pilota=dn, sigla=sigla.get(dn), team=team.get(dn), mescola=comp,
                             sessione=sess['session']['session_name'],
                             giri=[dict(i=c[0] - l0, eta=c[1], tempo=c[2], t=c[3] - t_sess, traffico=c[4] < gap_traffico,
                                        temp=temp_f(c[3])) for c in best]))
    return runs


# ---------------------------------------------------------------------------
# 2. Stime con priori (minimi quadrati con righe di priore e scarto robusto)
# ---------------------------------------------------------------------------

def _mq_priori(X, y, mu, sd, w=None, sigma=0.35, giri=4, soglia=3.0, minimo=0.4):
    """Minimi quadrati con priori normali indipendenti (mu, sd in unità dei coefficienti) e
    scarto iterativo dei giri anomali. sd=inf = nessun priore. Ritorna (b, cov, resid, tenuti)."""
    n, p = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    keep = np.ones(n, bool)
    sd = np.asarray(sd, float)
    mu = np.asarray(mu, float)
    has = np.isfinite(sd)
    for _ in range(giri):
        Xk = X[keep] * np.sqrt(w[keep])[:, None]
        yk = y[keep] * np.sqrt(w[keep])
        A = np.zeros((has.sum(), p))
        A[np.arange(has.sum()), np.where(has)[0]] = sigma / sd[has]
        XX = np.vstack([Xk, A])
        yy = np.concatenate([yk, sigma / sd[has] * mu[has]])
        b, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        res = y - X @ b
        r = res[keep]
        mad = 1.4826 * np.median(np.abs(r - np.median(r))) if len(r) else sigma
        sigma = max(mad, 0.05)
        nk = np.abs(res) <= max(soglia * mad, minimo)
        if (nk == keep).all():
            break
        keep = nk
    XtX = XX.T @ XX
    cov = sigma ** 2 * np.linalg.pinv(XtX)
    return b, cov, y - X @ b, keep, sigma


def stima_libere(runs, prior=None, solo=None, sd_run=0.3, piloti_ok=None):
    """Passo dalle libere: tempo = pilota + mescola + degrado(mescola)*età + benzina*indice nel run
    + sessione + traffico. Ritorna dict con passo per pilota (s, riferito a medium nuova a parità
    di benzina), errore, degrado per mescola e scarti tra mescole della settimana."""
    pr = dict(benzina=(-0.05, 0.02), off_S=(-0.3, 0.4), off_H=(0.3, 0.4),
              deg={'SOFT': (0.15, 0.10), 'MEDIUM': (0.12, 0.10), 'HARD': (0.10, 0.10)},
              traffico=(0.2, 0.3), sessione=(0.0, 1.0))
    if prior:
        pr.update(prior)
    righe = []
    run_id = []
    nrun = 0
    for r in runs:
        if solo and r['sessione'] not in solo:
            continue
        if piloti_ok is not None and r['pilota'] not in piloti_ok:
            continue
        for g in r['giri']:
            righe.append((r['pilota'], r['mescola'], g['eta'], g['i'], r['sessione'], g['traffico'], g['tempo'],
                          r['team'], r['sigla']))
            run_id.append(nrun)
        nrun += 1
    if len(righe) < 20:
        return None
    piloti = sorted({x[0] for x in righe})
    sess = sorted({x[4] for x in righe})
    pi = {d: i for i, d in enumerate(piloti)}
    si = {s: i for i, s in enumerate(sess)}
    nd, ns = len(piloti), len(sess)
    # colonne: piloti | off_S off_H | deg S M H | benzina | traffico | sessioni (tranne la prima)
    c0 = nd
    p0 = nd + 2 + 3 + 1 + 1 + (ns - 1)
    p = p0 + (nrun if sd_run else 0)
    X = np.zeros((len(righe), p))
    y = np.zeros(len(righe))
    for k, (dn, comp, eta, i, s, tr, t, *_r) in enumerate(righe):
        X[k, pi[dn]] = 1
        if comp == 'SOFT':
            X[k, c0] = 1
        elif comp == 'HARD':
            X[k, c0 + 1] = 1
        X[k, c0 + 2 + COMP.index(comp)] = eta
        X[k, c0 + 5] = i
        X[k, c0 + 6] = tr
        if si[s] > 0:
            X[k, c0 + 6 + si[s]] = 1
        if sd_run:
            X[k, p0 + run_id[k]] = 1
        y[k] = t
    mu = np.zeros(p)
    sd = np.full(p, np.inf)
    mu[c0], sd[c0] = pr['off_S']
    mu[c0 + 1], sd[c0 + 1] = pr['off_H']
    for j, c in enumerate(COMP):
        mu[c0 + 2 + j], sd[c0 + 2 + j] = pr['deg'][c]
    mu[c0 + 5], sd[c0 + 5] = pr['benzina']
    mu[c0 + 6], sd[c0 + 6] = pr['traffico']
    for j in range(ns - 1):
        mu[c0 + 7 + j], sd[c0 + 7 + j] = pr['sessione']
    if sd_run:
        # effetto del singolo run (benzina, motore): casuale, centrato su zero
        mu[p0:], sd[p0:] = 0.0, sd_run
    # piloti: priore largo intorno alla mediana dei tempi (rende stabile chi ha pochi giri)
    med = float(np.median(y))
    mu[:nd], sd[:nd] = med, 3.0
    b, cov, _, keep, sig = _mq_priori(X, y, mu, sd)
    # passo di riferimento: medium, età 0, inizio run, prima sessione
    passo = {d: b[pi[d]] for d in piloti}
    se = {d: math.sqrt(max(cov[pi[d], pi[d]], 0)) for d in piloti}
    n_giri = defaultdict(int)
    for x, k in zip(righe, keep):
        if k:
            n_giri[x[0]] += 1
    info = {x[0]: (x[7], x[8]) for x in righe}
    return dict(passo=passo, se=se, n=dict(n_giri), info=info,
                off={'SOFT': b[c0], 'MEDIUM': 0.0, 'HARD': b[c0 + 1]},
                deg={c: b[c0 + 2 + j] for j, c in enumerate(COMP)},
                se_deg={c: math.sqrt(cov[c0 + 2 + j, c0 + 2 + j]) for j, c in enumerate(COMP)},
                benzina=b[c0 + 5], traffico=b[c0 + 6], sigma=sig,
                giri_mescola={c: sum(1 for x, k in zip(righe, keep) if k and x[1] == c) for c in COMP},
                sessioni={s: (b[c0 + 6 + si[s]] if si[s] > 0 else 0.0) for s in sess})


def stima_gara(righe, prior=None, quadratico=True, per_team=False, sd_team=0.03):
    """Modello gomme di una gara: tempo = pilota + tendenza*giro (benzina e pista) + scarto mescola
    + degrado(mescola)*età [+ curvatura*età^2] + primo giro lanciato(mescola) + aria sporca
    [+ scarto di degrado del team*età]. Ritorna dict con i parametri, gli errori e i residui."""
    pr = dict(tendenza=(-0.05, 0.03), off_S=(-0.3, 0.5), off_H=(0.3, 0.5),
              deg={'SOFT': (0.07, 0.10), 'MEDIUM': (0.06, 0.10), 'HARD': (0.05, 0.10)},
              curv=(0.0, 0.002), primo=(0.3, 0.5), scia=(0.2, 0.3))
    if prior:
        pr.update(prior)
    if len(righe) < 100:
        return None
    piloti = sorted({r['pilota'] for r in righe})
    teams = sorted({r['team'] for r in righe})
    pi = {d: i for i, d in enumerate(piloti)}
    ti = {t: i for i, t in enumerate(teams)}
    nd = len(piloti)
    c0 = nd
    # colonne: piloti | tendenza | off_S off_H | deg S M H | curv | primo S M H | scia | team*età
    p = nd + 1 + 2 + 3 + 1 + 3 + 1 + (len(teams) if per_team else 0)
    X = np.zeros((len(righe), p))
    y = np.zeros(len(righe))
    for k, r in enumerate(righe):
        c = r['mescola']
        j = COMP.index(c)
        X[k, pi[r['pilota']]] = 1
        X[k, c0] = r['giro']
        if c == 'SOFT':
            X[k, c0 + 1] = 1
        elif c == 'HARD':
            X[k, c0 + 2] = 1
        X[k, c0 + 3 + j] = r['eta']
        if quadratico:
            X[k, c0 + 6] = r['eta'] ** 2
        if r['primo']:
            X[k, c0 + 7 + j] = 1
        X[k, c0 + 10] = 1.0 if r['gap'] < 1.0 else 0.0
        if per_team:
            X[k, c0 + 11 + ti[r['team']]] = r['eta']
        y[k] = r['tempo']
    mu = np.zeros(p)
    sd = np.full(p, np.inf)
    mu[:nd], sd[:nd] = float(np.median(y)), 5.0
    mu[c0], sd[c0] = pr['tendenza']
    mu[c0 + 1], sd[c0 + 1] = pr['off_S']
    mu[c0 + 2], sd[c0 + 2] = pr['off_H']
    for j, c in enumerate(COMP):
        mu[c0 + 3 + j], sd[c0 + 3 + j] = pr['deg'][c]
        mu[c0 + 7 + j], sd[c0 + 7 + j] = pr['primo']
    mu[c0 + 6], sd[c0 + 6] = pr['curv'] if quadratico else (0.0, 1e-6)
    mu[c0 + 10], sd[c0 + 10] = pr['scia']
    if per_team:
        mu[c0 + 11:], sd[c0 + 11:] = 0.0, sd_team
    b, cov, res, keep, sig = _mq_priori(X, y, mu, sd)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    n_c = {c: sum(1 for r, k in zip(righe, keep) if k and r['mescola'] == c) for c in COMP}
    out = dict(tendenza=b[c0], off={'SOFT': b[c0 + 1], 'MEDIUM': 0.0, 'HARD': b[c0 + 2]},
               se_off={'SOFT': se[c0 + 1], 'MEDIUM': 0.0, 'HARD': se[c0 + 2]},
               deg={c: b[c0 + 3 + j] for j, c in enumerate(COMP)},
               se_deg={c: se[c0 + 3 + j] for j, c in enumerate(COMP)},
               curv=b[c0 + 6], se_curv=se[c0 + 6],
               primo={c: b[c0 + 7 + j] for j, c in enumerate(COMP)},
               scia=b[c0 + 10], se_scia=se[c0 + 10], sigma=sig, n=int(keep.sum()), n_c=n_c,
               passo={d: b[pi[d]] for d in piloti}, residui=res, tenuti=keep)
    if per_team:
        tv = {t: b[c0 + 11 + ti[t]] for t in teams}
        m = float(np.mean(list(tv.values())))
        out['deg_team'] = {t: v - m for t, v in tv.items()}
        out['se_deg_team'] = {t: se[c0 + 11 + ti[t]] for t in teams}
    return out


# ---------------------------------------------------------------------------
# 3. Modello di stagione: quello che si sa delle gomme prima di una gara
# ---------------------------------------------------------------------------

# giri «giovani» su cui si stima il degrado lineare di ogni gara (prima del crollo)
LIM_GIOVANI = {'SOFT': 10, 'MEDIUM': 15, 'HARD': 22}
PRIORI = dict(
    off={'SOFT': -0.25, 'MEDIUM': 0.0, 'HARD': 0.25},      # s al giro rispetto alla medium nuova
    deg={'SOFT': 0.07, 'MEDIUM': 0.06, 'HARD': 0.05},       # s per giro di età, giri giovani
    curv={'SOFT': 0.0025, 'MEDIUM': 0.0012, 'HARD': 0.0003},  # crollo: s per (giro oltre il ginocchio)^2
    ginocchio=5,
    scia=0.25,          # s al giro persi a meno di 1 s da chi precede
    tendenza=-0.055,    # s al giro guadagnati per giro (benzina e pista)
    efficace=0.7,       # quota dell'effetto gomme che si vede nei duelli (priore; stimata sulle soste)
    libere=(0.6, 0.01),  # degrado gara = a * degrado libere + b
)


def casi_undercut(sess, fit, finestra=4.0):
    """Coppie di piloti vicini (entro `finestra` s sul traguardo) quando uno dei due si ferma e l'altro
    resta fuori almeno 3 giri: variazione della differenza sul giro (dopo la sosta meno prima).
    Ritorna dict con, per caso: osservato, parte dovuta allo scarto tra mescole, parte dovuta al
    degrado lineare (dal modello della gara) e i termini per calcolare il crollo (mescola, età, peso)."""
    vuoto = dict(oss=np.zeros(0), off=np.zeros(0), lin=np.zeros(0), crollo=[])
    laps = [l for l in sess.get('laps') or [] if l.get('lap_duration') and l.get('date_start')]
    if not laps:
        return vuoto
    med = float(np.median([l['lap_duration'] for l in laps]))
    iv = intervalli_bandiere(sess, med)
    by = defaultdict(dict)
    for l in sess['laps']:
        by[l['driver_number']][l['lap_number']] = l
    mp = {dn: stint_per_giro(sess, list(v.values()), dn) for dn, v in by.items()}
    pits = defaultdict(list)
    for p in sess.get('pit') or []:
        pits[p['driver_number']].append(p['lap_number'])

    def neutro(ln, dn):
        l = by[dn].get(ln)
        if not l or not l.get('date_start') or not l.get('lap_duration'):
            return True
        t0 = _t(l['date_start'])
        t1 = t0 + l['lap_duration']
        return any(a <= t1 and t0 <= b for a, b, _ in iv)

    def tc(dn, ln):
        l = by[dn].get(ln)
        return _t(l['date_start']) if l and l.get('date_start') else None

    oss_, off_, lin_, cr_ = [], [], [], []
    for A_, plist in list(pits.items()):
        for L_ in plist:
            for B_ in list(by):
                if B_ == A_:
                    continue
                ta, tb = tc(A_, L_), tc(B_, L_)
                if ta is None or tb is None or abs(ta - tb) > finestra:
                    continue
                nb = [x for x in pits.get(B_, []) if x >= L_]
                k = (min(nb) - L_) if nb else 99
                if k < 4:
                    continue
                prima = [ln for ln in range(L_ - 3, L_) if not neutro(ln, A_) and not neutro(ln, B_)
                         and ln - 1 not in pits.get(A_, []) and ln - 1 not in pits.get(B_, [])]
                dopo = [ln for ln in range(L_ + 2, L_ + min(k, 7)) if not neutro(ln, A_) and not neutro(ln, B_)]
                if len(prima) < 2 or len(dopo) < 2:
                    continue
                if any(mp[A_].get(ln, (None,))[0] not in ASCIUTTE or mp[B_].get(ln, (None,))[0] not in ASCIUTTE
                       for ln in prima + dopo):
                    continue
                dt = lambda ln: by[A_][ln]['lap_duration'] - by[B_][ln]['lap_duration']  # noqa: E731
                oss = np.mean([dt(ln) for ln in dopo]) - np.mean([dt(ln) for ln in prima])
                if abs(oss) > 6:
                    continue
                o_p = l_p = 0.0
                termini = []
                for finestra_, segno in ((dopo, 1.0), (prima, -1.0)):
                    w = segno / len(finestra_)
                    for ln in finestra_:
                        for dn, s_ in ((A_, 1.0), (B_, -1.0)):
                            c, age, _ = mp[dn][ln]
                            o_p += w * s_ * fit['off'][c]
                            l_p += w * s_ * fit['deg'][c] * age
                            termini.append((COMP.index(c), age, w * s_))
                oss_.append(oss)
                off_.append(o_p)
                lin_.append(l_p)
                cr_.append(np.array(termini, float))
    return dict(oss=np.array(oss_), off=np.array(off_), lin=np.array(lin_), crollo=cr_)


def analisi_weekend(d):
    """Riassunto gomme di un weekend disputato (gara asciutta), per il modello di stagione."""
    S = sessioni(d)
    s = S.get('Race')
    if not s or bagnata(s):
        return None
    rows = giri_gara(s)
    young = [r for r in rows if r['eta'] <= LIM_GIOVANI[r['mescola']]]
    o = stima_gara(young, quadratico=False)
    if o is None:
        return None
    res = []
    for r in rows:
        if r['pilota'] not in o['passo']:
            continue
        c = r['mescola']
        pred = (o['passo'][r['pilota']] + o['tendenza'] * r['giro'] + o['off'][c] + o['deg'][c] * r['eta']
                + (o['scia'] if r['gap'] < 1 else 0.0))
        e = r['tempo'] - pred
        if abs(e) <= 3:
            res.append((COMP.index(c), r['eta'], e))
    runs = []
    for nm in ('Practice 1', 'Practice 2', 'Practice 3'):
        if nm in S:
            runs += long_run(S[nm])
    fp = stima_libere(runs, prior=dict(deg={c: (0.1, 0.3) for c in COMP}, off_S=(0, 1.0), off_H=(0, 1.0))) if runs else None
    lane = [p['lane_duration'] for p in (s.get('pit') or []) if p.get('lane_duration') and 15 < p['lane_duration'] < 45]
    _, tm = temperatura(s)
    try:
        uc = casi_undercut(s, o)
    except Exception:  # noqa: BLE001
        uc = dict(oss=np.zeros(0), off=np.zeros(0), lin=np.zeros(0), crollo=[])
    # strategie dei primi dieci: numero di soste e giro della prima
    res_ = {x['driver_number']: x for x in s.get('session_result') or []}
    top = sorted([dn for dn, x in res_.items() if isinstance(x.get('position'), int)], key=lambda dn: res_[dn]['position'])[:10]
    soste_top = [sorted(p['lap_number'] for p in (s.get('pit') or []) if p['driver_number'] == dn) for dn in top]
    return dict(anno=d['anno'], round=d['round'], luogo=d['meeting'].get('location'), gp=d['meeting'].get('meeting_name'),
                undercut=uc, soste_top=soste_top,
                n_giri=max([x.get('number_of_laps') or 0 for x in res_.values()] or [0]),
                off=o['off'], deg=o['deg'], se_deg=o['se_deg'], n_c=o['n_c'], tendenza=o['tendenza'], scia=o['scia'],
                residui=np.array(res, float).reshape(-1, 3),
                fp_deg=(fp['deg'] if fp else None), fp_n=(fp['giri_mescola'] if fp else None),
                box=float(np.median(lane)) if lane else None, temp=tm)


class StagioneGomme:
    """Parametri gomme della stagione stimati sulle gare precedenti (analisi_weekend)."""

    def __init__(self, analisi, priori=None):
        self.pr = dict(PRIORI)
        if priori:
            self.pr.update(priori)
        A = [a for a in analisi if a]
        self.n_gare = len(A)
        pr = self.pr
        # scarti tra mescole: mediana delle gare in cui la mescola e la medium hanno abbastanza giri,
        # ristretta verso il priore (peso 3 gare)
        self.off = {'MEDIUM': 0.0}
        self.sd_off = {'MEDIUM': 0.0}
        for c in ('SOFT', 'HARD'):
            v = [a['off'][c] for a in A if a['n_c'][c] > 50 and a['n_c']['MEDIUM'] > 50]
            m = float(np.median(v)) if v else pr['off'][c]
            self.off[c] = (m * len(v) + pr['off'][c] * 3) / (len(v) + 3)
            self.sd_off[c] = float(np.std(v)) if len(v) > 2 else 0.3
        # degrado dei giri giovani: mediana per mescola (peso del priore 3 gare)
        self.deg = {}
        self.sd_deg = {}
        for c in COMP:
            v = [a['deg'][c] for a in A if a['n_c'][c] > 50]
            m = float(np.median(v)) if v else pr['deg'][c]
            self.deg[c] = (m * len(v) + pr['deg'][c] * 3) / (len(v) + 3)
            self.sd_deg[c] = float(np.std(v)) if len(v) > 2 else 0.05
        # crollo oltre il ginocchio, stimato su tutti i residui insieme
        g = pr['ginocchio']
        self.curv = {}
        R = np.vstack([a['residui'] for a in A]) if A else np.zeros((0, 3))
        for j, c in enumerate(COMP):
            Rc = R[R[:, 0] == j] if len(R) else R
            h = np.maximum(0, Rc[:, 1] - g) ** 2 if len(Rc) else np.zeros(0)
            lam = 200 * (15.0 ** 2) ** 2  # priore con il peso di 200 giri a 15 giri oltre il ginocchio
            num = float(np.sum(h * Rc[:, 2])) if len(Rc) else 0.0
            den = float(np.sum(h * h)) if len(Rc) else 0.0
            self.curv[c] = max(0.0, (num + lam * pr['curv'][c]) / (den + lam))
        self.ginocchio = g
        # aria sporca e tendenza
        v = [a['scia'] for a in A]
        self.scia = (float(np.median(v)) * len(v) + pr['scia'] * 3) / (len(v) + 3) if v else pr['scia']
        v = [a['tendenza'] for a in A]
        self.tendenza = float(np.median(v)) if v else pr['tendenza']
        # quota dell'effetto età (degrado + crollo) che si vede nei duelli: regressione senza intercetta
        # di (osservato - parte mescole) sulla parte età, nei casi di sosta delle gare precedenti
        # (priore 0,7 con il peso di 40 casi)
        num = den = 0.0
        n_u = 0
        curv_arr = np.array([self.curv[c] for c in COMP])
        for a in A:
            uc = a.get('undercut')
            if not uc or not len(uc['oss']):
                continue
            for o_, of_, li_, cr in zip(uc['oss'], uc['off'], uc['lin'], uc['crollo']):
                cl = float(np.sum(cr[:, 2] * curv_arr[cr[:, 0].astype(int)] * np.maximum(0, cr[:, 1] - g) ** 2)) if len(cr) else 0.0
                x = li_ + cl
                num += (o_ - of_) * x
                den += x * x
                n_u += 1
        lam = 40.0
        self.efficace = min(max((num + lam * 0.7) / (den + lam), 0.3), 1.2) if n_u else 0.7
        self.n_undercut = n_u
        # libere -> gara: regressione del degrado di gara su quello delle libere (coppie mescola-weekend)
        coppie = [(a['fp_deg'][c], a['deg'][c]) for a in A if a['fp_deg'] for c in COMP
                  if a['fp_n'][c] > 25 and a['n_c'][c] > 50]
        a0, b0 = pr['libere']
        if len(coppie) >= 4:
            X = np.array(coppie)
            lam = 4.0
            # minimi quadrati con priore (a0, b0) del peso di 4 coppie
            M_ = np.c_[X[:, 0], np.ones(len(X))]
            A_ = np.vstack([M_, math.sqrt(lam) * np.eye(2) * np.array([0.1, 1.0])])
            y_ = np.concatenate([X[:, 1], math.sqrt(lam) * np.array([0.1 * a0, b0])])
            (a1, b1), *_ = np.linalg.lstsq(A_, y_, rcond=None)
            res_fp = X[:, 1] - (a1 * X[:, 0] + b1)
            self.libere = (float(a1), float(b1))
            self.sd_libere = float(np.sqrt(np.mean(res_fp ** 2) + 1e-6))
            # errore della mediana di stagione sulle stesse coppie
            self.sd_stagione = float(np.sqrt(np.mean([(gd - self.deg[COMP[0]]) ** 2 for _, gd in coppie]))) if False else \
                float(np.sqrt(np.mean([(a['deg'][c] - self.deg[c]) ** 2 for a in A for c in COMP if a['n_c'][c] > 50])))
        else:
            self.libere = (a0, b0)
            self.sd_libere = 0.06
            self.sd_stagione = 0.06
        self.n_coppie = len(coppie)

    def previsione(self, runs=None, storia=None):
        """Parametri gomme per la prossima gara: degrado dalla stagione e, se ci sono, dai long run del
        weekend (riscalati); gravità del circuito estesa alle mescole senza long run."""
        deg = dict(self.deg)
        fonte = 'stagione'
        dettaglio = {}
        if runs:
            fp = stima_libere(runs, prior=dict(deg={c: (0.1, 0.3) for c in COMP}, off_S=(0, 1.0), off_H=(0, 1.0)))
            if fp:
                a, b = self.libere
                w_fp = self.sd_stagione ** 2 / (self.sd_stagione ** 2 + self.sd_libere ** 2)
                fatt = []
                for c in COMP:
                    if fp['giri_mescola'][c] > 25:
                        stima_fp = min(max(a * fp['deg'][c] + b, 0.0), 0.35)
                        nuovo = (1 - w_fp) * self.deg[c] + w_fp * stima_fp
                        dettaglio[c] = dict(libere=fp['deg'][c], riscalato=stima_fp, giri=fp['giri_mescola'][c])
                        if self.deg[c] > 0.01:
                            fatt.append(nuovo / self.deg[c])
                        deg[c] = nuovo
                if fatt:
                    f = float(np.median(fatt))
                    for c in COMP:
                        if c not in dettaglio:
                            deg[c] = self.deg[c] * f
                    fonte = 'stagione + libere'
        return dict(off=dict(self.off), deg=deg, curv=dict(self.curv), ginocchio=self.ginocchio,
                    scia=self.scia, tendenza=self.tendenza, efficace=self.efficace, fonte=fonte,
                    dettaglio=dettaglio, sd_off=dict(self.sd_off))


# ---------------------------------------------------------------------------
# 4. Strategia: tempo di gara delle sequenze di mescole e soste migliori
# ---------------------------------------------------------------------------

def perdita_stint(par, c, lunghezza, eta0=0):
    """Secondi persi in uno stint rispetto a una medium sempre nuova (scarto mescola + età)."""
    a = np.arange(eta0 + 1, eta0 + lunghezza + 1, dtype=float)
    return float(np.sum(par['off'][c] + par['deg'][c] * a + par['curv'][c] * np.maximum(0, a - par['ginocchio']) ** 2))


# Sequenze ammesse (2026: si parte quasi sempre con medium, la soft si usa solo come primo o ultimo
# stint e con un treno già usato in qualifica, la hard è la gomma lunga)
def sequenze_ammesse(max_soste=3):
    import itertools
    out = []
    for n in range(2, max_soste + 2):
        for seq in itertools.product(COMP, repeat=n):
            if len(set(seq)) < 2:
                continue
            soft = [i for i, c in enumerate(seq) if c == 'SOFT']
            if len(soft) > 1 or (soft and soft[0] not in (0, n - 1)):
                continue
            out.append(seq)
    return out


def _curva_cum(par, c, L, eta0=0, efficace=1.0):
    a = np.arange(eta0 + 1, eta0 + L + 1, dtype=float)
    d = max(par['deg'][c], 0.005)
    perg = par['off'][c] + efficace * (d * a + par['curv'][c] * np.maximum(0, a - par['ginocchio']) ** 2)
    return np.concatenate([[0.0], np.cumsum(perg)])


PARTENZA_HARD = 3.0  # s persi partendo con la hard (gomma fredda e meno aderenza al via); nel 2026 quasi nessuno


def strategie(par, giri, perdita_box, max_soste=3, giro_uscita=1.0, min_stint=4, eta_soft=3, efficace=None):
    """Migliore suddivisione in stint per ogni sequenza di mescole ammessa (almeno due mescole).
    La soft parte con eta_soft giri (treno usato in qualifica). L'effetto età è moltiplicato per
    `efficace` (default: quello del modello). Ritorna una lista ordinata di dict."""
    L = int(giri)
    ef = par.get('efficace', 1.0) if efficace is None else efficace
    cum = {c: _curva_cum(par, c, L, eta_soft if c == 'SOFT' else 0, ef) for c in COMP}
    P = perdita_box + giro_uscita
    out = []
    for seq in sequenze_ammesse(max_soste):
        n = len(seq)
        best = None
        if n == 2:
            for l1 in range(min_stint, L - min_stint + 1):
                t = cum[seq[0]][l1] + cum[seq[1]][L - l1]
                if best is None or t < best[0]:
                    best = (t, (l1, L - l1))
        elif n == 3:
            for l1 in range(min_stint, L - 2 * min_stint + 1):
                for l2 in range(min_stint, L - l1 - min_stint + 1):
                    t = cum[seq[0]][l1] + cum[seq[1]][l2] + cum[seq[2]][L - l1 - l2]
                    if best is None or t < best[0]:
                        best = (t, (l1, l2, L - l1 - l2))
        else:
            for l1 in range(min_stint, L - 3 * min_stint + 1, 2):
                for l2 in range(min_stint, L - l1 - 2 * min_stint + 1, 2):
                    for l3 in range(min_stint, L - l1 - l2 - min_stint + 1, 2):
                        t = cum[seq[0]][l1] + cum[seq[1]][l2] + cum[seq[2]][l3] + cum[seq[3]][L - l1 - l2 - l3]
                        if best is None or t < best[0]:
                            best = (t, (l1, l2, l3, L - l1 - l2 - l3))
        if best is None:
            continue
        t = best[0] + (n - 1) * P + (PARTENZA_HARD if seq[0] == 'HARD' else 0.0)
        soste = list(np.cumsum(best[1])[:-1])
        out.append(dict(tempo=t, sequenza=seq, stint=best[1], soste=[int(x) for x in soste], n_soste=n - 1))
    out.sort(key=lambda x: x['tempo'])
    t0 = out[0]['tempo']
    for x in out:
        x['delta'] = x['tempo'] - t0
    return out


def prob_soste(par, giri, perdita_box, sd_deg=None, n=200, tau=4.0, seed=0):
    """Probabilità del numero di soste della strategia migliore, con l'incertezza sul degrado
    (moltiplicatore lognormale) e una scelta «morbida» tra strategie vicine (tau secondi)."""
    rng = np.random.default_rng(seed)
    cont = defaultdict(float)
    sd = 0.35 if sd_deg is None else sd_deg
    for _ in range(n):
        m = math.exp(rng.normal(0, sd))
        p2 = dict(par)
        p2['deg'] = {c: v * m for c, v in par['deg'].items()}
        p2['curv'] = {c: v * m for c, v in par['curv'].items()}
        st = strategie(p2, giri, perdita_box, max_soste=3)
        best = {}
        for x in st:
            best.setdefault(x['n_soste'], x['delta'])
        w = {k: math.exp(-v / tau) for k, v in best.items()}
        z = sum(w.values())
        for k, v in w.items():
            cont[k] += v / z / n
    return dict(sorted(cont.items()))


ANTICIPO_SOSTA = 3  # giri: nel 2026 le prime soste arrivano prima dell'ottimo delle sole gomme (undercut)


def piani_strategia(par, giri, perdita_box, tau=4.0, molt=(0.7, 1.0, 1.4), pesi=(0.25, 0.5, 0.25),
                    soglia=15.0, max_piani=6, anticipo=ANTICIPO_SOSTA):
    """Piani di gara probabili (sequenza di mescole e giri di sosta) con la loro probabilità:
    scelta «morbida» tra le strategie entro `soglia` secondi dalla migliore (temperatura tau),
    mediata su tre livelli di degrado (incertezza della previsione)."""
    acc = {}
    for m, w in zip(molt, pesi):
        p2 = dict(par)
        p2['deg'] = {c: v * m for c, v in par['deg'].items()}
        p2['curv'] = {c: v * m for c, v in par['curv'].items()}
        st = strategie(p2, giri, perdita_box)
        cand = [x for x in st if x['delta'] <= soglia]
        ws = np.array([math.exp(-x['delta'] / tau) for x in cand])
        ws /= ws.sum()
        for x, wx in zip(cand, ws):
            soste = []
            for sl in x['soste']:
                v = max(sl - anticipo, (soste[-1] + 4) if soste else 4)
                soste.append(int(v))
            k = (x['sequenza'], tuple(soste))
            acc[k] = acc.get(k, 0.0) + w * wx
    piani = sorted(acc.items(), key=lambda kv: -kv[1])[:max_piani]
    z = sum(v for _, v in piani)
    return [dict(sequenza=k[0], soste=list(k[1]), p=v / z) for k, v in piani]


# ---------------------------------------------------------------------------
# 5. Scheda del weekend e verifica
# ---------------------------------------------------------------------------

GIRI = {'Melbourne': 58, 'Shanghai': 56, 'Suzuka': 53, 'Miami Gardens': 57, 'Montréal': 70, 'Monte Carlo': 78,
        'Barcelona': 66, 'Spielberg': 71, 'Silverstone': 52, 'Spa-Francorchamps': 44, 'Budapest': 70,
        'Zandvoort': 72, 'Monza': 53, 'Madrid': 57, 'Baku': 51, 'Kuala Lumpur': 56, 'Marina Bay': 62,
        'Austin': 56, 'Mexico City': 71, 'São Paulo': 71, 'Las Vegas': 50, 'Lusail': 57, 'Yas Island': 58}
ALIAS = (('united states', 'Austin'), ('austin', 'Austin'), ('mexico', 'Mexico City'), ('são paulo', 'São Paulo'),
         ('sao paulo', 'São Paulo'), ('brazil', 'São Paulo'), ('las vegas', 'Las Vegas'), ('qatar', 'Lusail'),
         ('lusail', 'Lusail'), ('abu dhabi', 'Yas Island'), ('yas', 'Yas Island'), ('singapore', 'Marina Bay'))


def chiave_circuito(luogo, gp=''):
    if luogo in GIRI:
        return luogo
    t = f"{luogo or ''} {gp or ''}".lower()
    for k, v in ALIAS:
        if k in t:
            return v
    return luogo


def vita_utile(par, c, soglia=0.5):
    """Giri di età oltre i quali il crollo aggiunge più di `soglia` s al giro."""
    k = par['curv'][c]
    if k <= 0:
        return None
    return par['ginocchio'] + math.sqrt(soglia / k)


def _f(x, d=2):
    return f"{x:.{d}f}".replace('.', ',')


def _neutre(sess, laps_pit):
    laps = [l for l in sess.get('laps') or [] if l.get('lap_duration') and l.get('date_start')]
    med = float(np.median([l['lap_duration'] for l in laps])) if laps else 90.0
    iv = intervalli_bandiere(sess, med)
    by = {(l['driver_number'], l['lap_number']): l for l in sess.get('laps') or []}
    out = 0
    for dn, ln in laps_pit:
        l = by.get((dn, ln))
        if not l or not l.get('date_start'):
            continue
        t0 = _t(l['date_start'])
        t1 = t0 + (l.get('lap_duration') or med)
        if any(a <= t1 + 30 and t0 <= b for a, b, _ in iv):
            out += 1
    return out


def carica_archivio(storico):
    W = []
    for f in sorted(glob.glob(os.path.join(storico, '*', '*.json.gz'))):
        try:
            d = carica_weekend(f)
        except Exception:  # noqa: BLE001
            continue
        if 'sessioni' in d:  # storico/circuiti: solo il giro di riferimento, non un weekend
            W.append(d)
    return W


def scheda(storico, anno, rnd, circuito=None, giri=None, box=None, analisi=None, archivio=None):
    """Testo della scheda gomme del weekend `rnd` dell'anno `anno`."""
    W = archivio if archivio is not None else carica_archivio(storico)
    weekend = next((d for d in W if d['anno'] == anno and d['round'] == rnd), None)
    if analisi is None:
        analisi = {d['round']: analisi_weekend(d) for d in W if d['anno'] == anno and d['round'] < rnd}
    st = StagioneGomme([analisi[k] for k in sorted(analisi) if k < rnd])
    if weekend:
        circ = chiave_circuito(weekend['meeting'].get('location'), weekend['meeting'].get('meeting_name'))
        nome = f"{weekend['meeting'].get('meeting_name')} {anno}"
    else:
        circ = chiave_circuito(circuito or '')
        nome = f"{circ} {anno}"
    L = giri or GIRI.get(circ)
    if not L:
        raise SystemExit('Giri della gara sconosciuti: usa --giri')
    righe = [f"SCHEDA GOMME — {nome} (round {rnd}), {circ}, {L} giri", '']
    # stagione
    righe.append(f"Stagione {anno} fino al round {rnd - 1} ({st.n_gare} gare; giri sul bagnato esclusi):")
    righe.append(f"  scarto dalla medium nuova: soft {_f(st.off['SOFT'])} s/giro, hard {'+' if st.off['HARD'] >= 0 else ''}{_f(st.off['HARD'])} s/giro")
    righe.append(f"  degrado nei primi giri (s per giro di età): soft {_f(st.deg['SOFT'], 3)}, medium {_f(st.deg['MEDIUM'], 3)}, hard {_f(st.deg['HARD'], 3)}")
    par0 = dict(off=st.off, deg=st.deg, curv=st.curv, ginocchio=st.ginocchio)
    vu = {c: vita_utile(par0, c) for c in COMP}
    righe.append("  vita utile (crollo oltre 0,5 s al giro): " + ', '.join(
        f"{c.lower()} {int(round(v))} giri" if v else f"{c.lower()} oltre 50 giri" for c, v in vu.items()))
    righe.append(f"  aria sporca entro 1 s: {_f(st.scia)} s/giro; benzina e pista: {_f(st.tendenza, 3)} s/giro")
    righe.append(f"  quota del degrado che si vede nei duelli (undercut): {int(round(100 * st.efficace))}% ({st.n_undercut} soste misurate)")
    righe.append('')
    # storia del circuito
    storia = [d for d in W if d['anno'] != anno and chiave_circuito(d['meeting'].get('location'), d['meeting'].get('meeting_name')) == circ]
    boxes = []
    if storia:
        righe.append('Storia del circuito (auto e gomme diverse, solo indicativa):')
        for d in sorted(storia, key=lambda d: d['anno']):
            a = analisi_weekend(d)
            if not a:
                righe.append(f"  {d['anno']}: gara bagnata o dati insufficienti")
                continue
            if a['box']:
                boxes.append(a['box'])
            dg = [a['deg'][c] for c in COMP if a['n_c'][c] > 50]
            ns = defaultdict(int)
            for x in a['soste_top']:
                ns[len(x)] += 1
            prime = [x[0] for x in a['soste_top'] if x]
            righe.append(f"  {d['anno']}: degrado medio {_f(float(np.mean(dg)), 3) if dg else '-'} s/giro, soste dei primi dieci "
                         + ', '.join(f"{k}: {v}" for k, v in sorted(ns.items()))
                         + (f", prima sosta mediana al giro {int(np.median(prime))}" if prime else '')
                         + (f", box {_f(a['box'], 1)} s" if a['box'] else ''))
        righe.append('')
    perdita = box or (float(np.median(boxes)) if boxes else 22.0)
    # libere del weekend
    runs = []
    if weekend:
        S = sessioni(weekend)
        for nm in ('Practice 1', 'Practice 2', 'Practice 3'):
            if nm in S:
                runs += long_run(S[nm])
    par = st.previsione(runs or None)
    if runs:
        righe.append(f"Libere del weekend: {len(runs)} long run, {sum(len(r['giri']) for r in runs)} giri")
        fp = stima_libere(runs, prior=dict(deg={c: (0.1, 0.3) for c in COMP}, off_S=(0, 1.0), off_H=(0, 1.0)))
        for c, v in par['dettaglio'].items():
            righe.append(f"  {c.lower()}: degrado nei long run {_f(v['libere'], 3)} -> previsto in gara {_f(v['riscalato'], 3)} ({v['giri']} giri)")
        if fp:
            tit = {}
            for r in runs:
                tit.setdefault(r['pilota'], (r['sigla'], r['team']))
            best = min(fp['passo'].values())
            righe.append('  passo dei long run (medium nuova, stessa benzina; s dal migliore, ± errore):')
            for dn in sorted(fp['passo'], key=fp['passo'].get):
                sg, tm = tit.get(dn, ('?', '?'))
                mesc = sorted({r['mescola'][0] for r in runs if r['pilota'] == dn})
                righe.append(f"    {sg:<4} {str(tm)[:16]:<16} +{_f(fp['passo'][dn] - best)} ± {_f(fp['se'][dn])}  ({fp['n'].get(dn, 0)} giri, {'/'.join(mesc)})")
            righe.append('  nota: nel 2026 il passo dei long run non aggiunge niente dopo la qualifica; prima della qualifica')
            righe.append('  aggiunge poco alla forma (circa 3% di errore in meno sul passo gara).')
        righe.append('')
    else:
        righe.append('Libere del weekend: non ancora nell\'archivio (la scheda usa solo la stagione).')
        righe.append('')
    # previsione
    righe.append(f"Previsione per la gara ({par['fonte']}), box {_f(perdita, 1)} s:")
    righe.append(f"  degrado: soft {_f(par['deg']['SOFT'], 3)}, medium {_f(par['deg']['MEDIUM'], 3)}, hard {_f(par['deg']['HARD'], 3)} s per giro di età")
    stt = strategie(par, L, perdita)
    best = {}
    for x in stt:
        best.setdefault(x['n_soste'], x)
    for k in sorted(best):
        x = best[k]
        righe.append(f"  migliore con {k} {'sosta' if k == 1 else 'soste'}: {'-'.join(c[0] for c in x['sequenza'])}, "
                     f"soste al giro {', '.join(str(v) for v in x['soste'])} (ottimo gomme), +{_f(x['delta'], 1)} s")
    piani = piani_strategia(par, L, perdita)
    pst = defaultdict(float)
    for pl in piani:
        pst[len(pl['soste'])] += pl['p']
    righe.append('  probabilità del numero di soste: ' + ', '.join(f"{k}: {int(round(100 * v))}%" for k, v in sorted(pst.items())))
    prime = [pl['soste'][0] for pl in piani for _ in range(max(1, int(round(pl['p'] * 20))))]
    if prime:
        righe.append(f"  finestra della prima sosta (con l'anticipo da undercut): giri {int(np.percentile(prime, 10))}-{int(np.percentile(prime, 90))}")
    righe.append('  piani più probabili: ' + '; '.join(f"{'-'.join(c[0] for c in pl['sequenza'])} giri {', '.join(str(v) for v in pl['soste'])} ({int(round(100 * pl['p']))}%)" for pl in piani[:4]))
    for eta in (15, 25):
        g = par['efficace'] * (max(par['deg']['MEDIUM'], 0.005) * eta + par['curv']['MEDIUM'] * max(0, eta - par['ginocchio']) ** 2)
        righe.append(f"  undercut: medium nuova contro medium di {eta} giri, circa {_f(g, 1)} s al giro nei primi giri")
    return '\n'.join(righe)


def prova(storico, anno=2026):
    """Verifica sul campionato: ogni gara prevista solo con le precedenti."""
    W = [d for d in carica_archivio(storico) if d['anno'] == anno]
    W.sort(key=lambda d: d['round'])
    AN = {d['round']: analisi_weekend(d) for d in W}
    out = dict(degrado=[], soste=[], undercut=[], passo=[])
    for d in W:
        r = d['round']
        a = AN.get(r)
        prev = [AN[k] for k in AN if k < r]
        if not a or len([x for x in prev if x]) < 1:
            continue
        st = StagioneGomme(prev)
        S = sessioni(d)
        runs = [x for nm in ('Practice 1', 'Practice 2', 'Practice 3') if nm in S for x in long_run(S[nm])]
        par = st.previsione(runs or None)
        for c in COMP:
            if a['n_c'][c] > 50:
                out['degrado'].append(dict(round=r, mescola=c, gara=a['deg'][c], stagione=st.deg[c], previsto=par['deg'][c]))
        circ = chiave_circuito(d['meeting'].get('location'), d['meeting'].get('meeting_name'))
        L = GIRI.get(circ) or a['n_giri']
        piani = piani_strategia(par, L, a['box'] or 22.0)
        pst = defaultdict(float)
        for pl in piani:
            pst[len(pl['soste'])] += pl['p']
        if a['soste_top']:
            ns = defaultdict(int)
            for x in a['soste_top']:
                ns[len(x)] += 1
            moda = max(ns, key=ns.get)
            s_ = S['Race']
            # quota di prime soste fatte con SC, VSC o bandiera rossa
            res_ = {x['driver_number']: x for x in s_.get('session_result') or []}
            top = sorted([dn for dn, x in res_.items() if isinstance(x.get('position'), int)], key=lambda dn: res_[dn]['position'])[:10]
            pit_first = []
            pit_all = []
            for dn in top:
                v = sorted(p['lap_number'] for p in (s_.get('pit') or []) if p['driver_number'] == dn)
                if v:
                    pit_first.append((dn, v[0]))
                    pit_all += [(dn, x) for x in v]
            # quota delle soste dei primi dieci fatte con SC, VSC o bandiera rossa
            forzate = _neutre(s_, pit_all) / max(len(pit_all), 1)
            out['soste'].append(dict(round=r, previsto=max(pst, key=pst.get), p_reale=pst.get(moda, 0.0), reale=moda,
                                     prima_prevista=float(sum(pl['p'] * pl['soste'][0] for pl in piani)),
                                     prima_reale=float(np.median([x[1] for x in pit_first])) if pit_first else None,
                                     forzate=forzate))
        uc = a.get('undercut')
        if uc and len(uc['oss']):
            out['undercut'].append(dict(round=r, casi=int(len(uc['oss'])), efficace_previsto=st.efficace))
        # passo dei long run contro passo di gara (piloti in comune)
        if runs:
            fp = stima_libere(runs)
            rg = stima_gara(giri_gara(S['Race']))
            if fp and rg:
                com = [dn for dn in fp['passo'] if dn in rg['passo']]
                if len(com) >= 8:
                    x = np.array([fp['passo'][dn] for dn in com])
                    y = np.array([rg['passo'][dn] for dn in com])
                    out['passo'].append(dict(round=r, n=len(com), r=float(np.corrcoef(x, y)[0, 1])))
    return out


def cmd_scheda(a):
    print(scheda(a.storico, a.anno, a.round, circuito=a.circuito, giri=a.giri, box=a.box))


def cmd_prova(a):
    o = prova(a.storico, a.anno)
    D = o['degrado']
    e_s = np.array([x['gara'] - x['stagione'] for x in D])
    e_p = np.array([x['gara'] - x['previsto'] for x in D])
    print(f"Degrado di gara (s per giro di età), {len(D)} casi mescola-gara:")
    print(f"  errore con la sola stagione {_f(float(np.sqrt(np.mean(e_s ** 2))), 3)}, con stagione + libere {_f(float(np.sqrt(np.mean(e_p ** 2))), 3)}")
    Sx = [x for x in o['soste'] if x['forzate'] <= 0.2]
    print(f"Numero di soste dei primi dieci, gare con al più il 20% di soste fatte con SC/VSC/rossa ({len(Sx)}):")
    print(f"  giusto {sum(x['previsto'] == x['reale'] for x in Sx)} su {len(Sx)}, probabilità media data al numero reale {_f(float(np.mean([x['p_reale'] for x in Sx])))}")
    dif = [x['prima_prevista'] - x['prima_reale'] for x in Sx if x['prima_reale']]
    if dif:
        print(f"  prima sosta: prevista in media {_f(float(np.mean(dif)), 1)} giri dopo quella reale (errore medio assoluto {_f(float(np.mean(np.abs(dif))), 1)})")
    for x in o['soste']:
        print(f"    round {x['round']:>2}: previsto {x['previsto']} soste (p reale {_f(x['p_reale'])}), reale {x['reale']}, "
              f"prima sosta {_f(x['prima_prevista'], 0)} contro {_f(x['prima_reale'], 0) if x['prima_reale'] else '-'}"
              + (f", forzate {int(round(100 * x['forzate']))}%" if x['forzate'] else ''))
    U = o['undercut']
    if U:
        print(f"Undercut: quota efficace prevista da {_f(U[0]['efficace_previsto'])} a {_f(U[-1]['efficace_previsto'])} ({sum(x['casi'] for x in U)} casi)")
    Pp = o['passo']
    if Pp:
        print(f"Passo dei long run contro passo di gara: correlazione mediana {_f(float(np.median([x['r'] for x in Pp])))} ({len(Pp)} gare)")


def main():
    ap = argparse.ArgumentParser(description='Modello gomme Muretto')
    sp = ap.add_subparsers(dest='cmd', required=True)
    p1 = sp.add_parser('scheda', help='scheda gomme del weekend')
    p1.add_argument('--storico', default='storico')
    p1.add_argument('--anno', type=int, default=2026)
    p1.add_argument('--round', type=int, required=True)
    p1.add_argument('--circuito', default=None, help='se il file del weekend non c\'è ancora (es. Austin)')
    p1.add_argument('--giri', type=int, default=None)
    p1.add_argument('--box', type=float, default=None, help='secondi persi ai box (se manca la storia del circuito)')
    p2 = sp.add_parser('prova', help='verifica sulla stagione')
    p2.add_argument('--storico', default='storico')
    p2.add_argument('--anno', type=int, default=2026)
    a = ap.parse_args()
    {'scheda': cmd_scheda, 'prova': cmd_prova}[a.cmd](a)


if __name__ == '__main__':
    main()
