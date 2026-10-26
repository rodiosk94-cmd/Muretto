#!/usr/bin/env python3
"""
backtest_sottrazione.py — Muretto: prova di sottrazione dei modelli (26/10/2026).

Domanda: ogni pezzo della pipeline migliora la probabilità del favorito? Si toglie un pezzo alla volta e
si rifà la prova a ritroso sugli stessi eventi. Se l'errore non peggiora, il pezzo è peso morto.

Un solo metro: Brier score delle probabilità date a tutti i piloti, (probabilità - esito)^2 sommato sui
piloti, con esito 1 per chi vince e 0 per gli altri; nel testa a testa la stessa cosa sulla coppia.
Più basso è meglio. Ogni evento è previsto con i soli dati precedenti (nessun dato futuro).

Uscite provate (stagioni 2024-2026, dal round 4; la qualifica usa anche il 2023 per tarare le probabilità):
  pole              prima della qualifica: giro secco di taratura_matrice.py e, a parte, pole_telemetria.py
  vittoria_preQ     vittoria in gara letta prima della qualifica (base della regola A)
  vittoria          vittoria in gara dopo la qualifica (modello_gara.py)
  libere            primo nelle libere (modello_libere.py)
  h2h               testa a testa in qualifica fra compagni (modello_h2h_quali.py)
Il modello energia non entra in nessuna probabilità. Il modello gomme entra come «+gomme» nella gara e come
«+long run» (passo dei long run) nella pole e nella vittoria prima della qualifica.

Regola di decisione. Per ogni variante:
  Δ = Brier della variante - Brier del completo, media per evento, intervallo al 90% ricampionando gli eventi.
  «-pezzo» (tolto):   resta se Δ > 0 con intervallo tutto sopra zero; esce se Δ <= 0; altrimenti non dimostrato
  «+pezzo» (in prova): entra se Δ < 0 con intervallo tutto sotto zero; altrimenti resta fuori
  «banale»:           il modello vale se Δ > 0 con intervallo tutto sopra zero (batte la regola banale);
                      se Δ <= 0 basta la regola banale; altrimenti non dimostrato
  «altro strumento»:  pole_telemetria.py contro il giro secco sugli stessi weekend: meglio, peggio o equivalente
  «snello»:           versione ridotta del modello: perde precisione solo se peggiora in modo dimostrato
  Controllo aggiunto dopo i primi risultati (va nel senso prudente, tiene più pezzi): la stessa prova si fa
  sulla log-loss, e un pezzo non esce se toglierlo peggiora in modo dimostrato anche solo la log-loss. Il Brier
  è poco sensibile a come la probabilità si divide fra piloti con probabilità piccole (per esempio fra due
  compagni), che conta quando si gioca un pilota preciso.
  Secondo controllo, stesso senso prudente: la quota minima si calcola sul favorito, quindi un pezzo non esce
  se senza di lui il favorito riceve più probabilità di quante volte vince, in modo dimostrato rispetto al
  completo (scarto appaiato evento per evento con intervallo sopra zero). Nella gara il Brier migliora
  togliendo partenza, ritiri e safety car, ma il favorito sale di 2-6 punti mentre nel completo è già giusto.

Uso, dalla radice di Muretto (servono i modelli nella stessa cartella):
  python backtest_sottrazione.py --storico storico --out risultati
  python backtest_sottrazione.py --solo pole,h2h --anni 2025,2026
Opzioni: --n simulazioni per corsa nella gara (4000; stessi numeri casuali per tutte le varianti),
         --processi per la gara (una stagione per processo), --dal primo round valutato (4).
Scrive risultati/sottrazione_righe.csv (una riga per evento e variante) e risultati/sottrazione_riepilogo.json.
"""
import argparse
import collections
import csv
import glob
import gzip
import json
import math
import os
import re
import sys
import time

import numpy as np

QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, QUI)

N_BOOT = 20000


# ---------------------------------------------------------------------------------------------
# 0. Punteggi e decisione
# ---------------------------------------------------------------------------------------------

def brier_multi(prob, vincitore, candidati):
    """somma su tutti i candidati di (p - esito)^2; chi vince fuori dai candidati conta con p = 0"""
    b = sum((prob.get(c, 0.0) - (c == vincitore)) ** 2 for c in candidati)
    if vincitore not in candidati:
        b += 1.0
    return float(b)


def logloss(p):
    return -math.log(max(p, 1e-3))


def bootstrap(d, seed=3):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()) if len(d) else float('nan'), float('nan'), float('nan')
    idx = np.random.default_rng(seed).integers(0, len(d), (N_BOOT, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 5)), float(np.percentile(m, 95))


def verdetto(nome, b, l, gonfia=False):
    """b, l: (Δ, inizio, fine dell'intervallo) per Brier e log-loss. Il Brier decide; la log-loss fa da
    controllo: un pezzo non esce se toglierlo peggiora in modo dimostrato anche solo la log-loss (il Brier
    è poco sensibile a come la probabilità si divide fra piloti con probabilità piccole, per esempio fra
    due compagni). gonfia: senza il pezzo il favorito riceve più probabilità di quante volte vince, in modo
    dimostrato rispetto al completo; il pezzo resta, perché la quota minima si calcola sul favorito."""
    peggio = (b[0] > 0 and b[1] > 0) or (l[0] > 0 and l[1] > 0)    # la variante è peggio, dimostrato
    meglio = (b[0] < 0 and b[2] < 0) or (l[0] < 0 and l[2] < 0)    # la variante è meglio, dimostrato
    mai_peggio = b[0] <= 0 and l[0] <= 0
    if nome.startswith('snello'):
        if peggio:
            return 'perde precisione: non basta'
        if gonfia:
            return 'gonfia il favorito: non basta'
        return 'non perde precisione' if mai_peggio else 'perdita non dimostrata'
    if nome.startswith('altro strumento'):
        if meglio and not peggio:
            return 'meglio del giro secco'
        return 'peggio del giro secco' if peggio and not meglio else 'equivalente al giro secco'
    if nome.startswith('+'):
        return 'entra' if (meglio and not peggio) else 'resta fuori'
    if nome.startswith('banale'):
        if peggio:
            return 'il modello batte la regola banale'
        return 'basta la regola banale' if mai_peggio else 'non dimostrato'
    if peggio:
        return 'resta'
    if gonfia:
        return 'resta: senza, il favorito è gonfiato'
    return 'esce' if mai_peggio else 'non dimostrato'


def riga(uscita, anno, rnd, evento, variante, brier, ll, preso, extra=None):
    r = dict(uscita=uscita, anno=anno, round=rnd, evento=evento, variante=variante,
             brier=round(float(brier), 6), logloss=round(float(ll), 6), preso=round(float(preso), 4))
    if extra:
        r.update(extra)
    return r


# ---------------------------------------------------------------------------------------------
# 1. Vittoria dopo la qualifica: modello_gara.py, un componente alla volta
# ---------------------------------------------------------------------------------------------

def _senza_passo(inp, M):
    inp['mu'] = np.full_like(inp['mu'], float(np.mean(inp['mu'])))


def _senza_partenza(inp, M):
    inp['scatto'] = np.zeros_like(inp['scatto'])
    inp['sig1'] = (0.0, 0.0, 0.0)
    inp['delta2'] = 0.0
    inp['p_incidente1'] = 0.0


def _senza_ritiri(inp, M):
    inp['p_rit'] = np.zeros_like(inp['p_rit'])
    M.P['p_guaio'] = 0.0


def _senza_sc(inp, M):
    inp['lam_sc'] = 0.0
    inp['lam_vsc'] = 0.0


def _senza_soste(inp, M):
    inp['soste'] = False


def _senza_sorpassi(inp, M):
    inp['sorpasso'] = M.livello_sorpasso(2, M.P['sorpasso_shift'])


def _senza_scia(inp, M):
    M.P['aria_sporca'] = 0.0
    M.P['usura_scia'] = 1.0


def _senza_pioggia(inp, M):
    inp['p_bagnato'] = 0.0


def _insieme(*mods):
    def f(inp, M):
        for m in mods:
            m(inp, M)
    return f


VAR_GARA = [
    ('-passo (forma e distacco in qualifica)', _senza_passo),
    ('-partenza (scatto e primo giro)', _senza_partenza),
    ('-ritiri e guai', _senza_ritiri),
    ('-safety car e VSC', _senza_sc),
    ('-soste', _senza_soste),
    ('-sorpassi per circuito', _senza_sorpassi),
    ('-aria sporca e scia', _senza_scia),
    ('-pioggia', _senza_pioggia),
    # versioni snelle (scelte sugli stessi dati: la verifica vera è dal Messico in poi)
    ('snello: via aria sporca, partenza, ritiri e safety car', _insieme(_senza_scia, _senza_partenza, _senza_ritiri,
                                                                      _senza_sc)),
    ('snello: solo passo e griglia', _insieme(_senza_scia, _senza_partenza, _senza_ritiri, _senza_sc, _senza_soste,
                                              _senza_sorpassi, _senza_pioggia)),
]


def gara_stagione(args):
    storico, anno, n_sim, dal = args[:4]
    filtro = args[4] if len(args) > 4 else None    # solo alcune varianti (le altre si riusano con --righe)
    import modello_gara as M
    t0 = time.time()
    ev = M.estrai_eventi(storico, anni=(anno,))
    righe, passate = [], []
    for k, e in enumerate(ev):
        if k == 0:
            continue
        prec = [x for x in ev[:k] if x['t_via'] < e['t_via']]
        st = M.Storia(prec)
        M.P['gomme'] = False
        inp0 = M.prepara(e, st)
        out0 = M.simula(inp0, n=n_sim, seed=k)
        vinc, pod = M.esito(e)
        cal = M.ricalibrazione(passate)
        cv, _ = M.applica_ricalibrazione(out0['vittoria'], out0['podio'], cal)
        passate.append((out0['vittoria'], out0['podio'], vinc, pod))
        if e['round'] < dal:
            continue
        gv, _ = M.prob_griglia(e, st.b_griglia)
        probs = {'completo': out0['vittoria'], 'banale: griglia': gv, '+ricalibrazione': cv}
        for nome, modifica in VAR_GARA:
            if filtro and nome not in filtro:
                continue
            inp = dict(inp0)
            salva = dict(M.P)
            modifica(inp, M)
            try:
                probs[nome] = M.simula(inp, n=n_sim, seed=k)['vittoria']
            finally:
                M.P.clear()
                M.P.update(salva)
        if filtro and '+gomme (mescole e strategie)' not in filtro:
            pass
        elif e['tipo'] == 'R' and st.gomme is not None:
            M.P['gomme'] = True
            try:
                probs['+gomme (mescole e strategie)'] = M.simula(M.prepara(e, st), n=n_sim, seed=k)['vittoria']
            finally:
                M.P['gomme'] = False
        else:
            probs['+gomme (mescole e strategie)'] = out0['vittoria']
        ds = list(e['griglia'])
        nome_ev = '%s %s' % ('Gara' if e['tipo'] == 'R' else 'Sprint', e['luogo'])
        for nome, pv in probs.items():
            righe.append(riga('vittoria', anno, e['round'], nome_ev, nome, brier_multi(pv, vinc, ds),
                              logloss(pv.get(vinc, 0.0)), max(pv, key=pv.get) == vinc,
                              {'tipo': e['tipo'], 'p_fav': round(max(pv.values()), 4)}))
        print('  gara %d r%02d %-22s fatto (%.0fs)' % (anno, e['round'], nome_ev[:22], time.time() - t0), flush=True)
    return righe


def sezione_gara(storico, anni, n_sim, dal, processi, filtro=None):
    jobs = [(storico, a, n_sim, dal, filtro) for a in anni]
    if processi > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(min(processi, len(jobs))) as pool:
            parti = pool.map(gara_stagione, jobs)
    else:
        parti = [gara_stagione(j) for j in jobs]
    return [r for p in parti for r in p]


# ---------------------------------------------------------------------------------------------
# 2. Pole e vittoria prima della qualifica: giro secco di taratura_matrice.py, un pezzo alla volta
# ---------------------------------------------------------------------------------------------

def _secco_varianti(tm, st, w, sg, tit):
    """distacco previsto (% del giro) per pilota in ogni variante del giro secco"""
    def comb(f):
        return {dn: v for dn, v in tm.combina({dn: [s for s in lst if f(s)] for dn, lst in sg.items()}).items()
                if dn in tit}
    out = {'completo': comb(lambda s: True),
           '-forma stagionale': comb(lambda s: s[0] != 'forma_q'),
           '-libere del weekend (solo forma)': comb(lambda s: s[0] == 'forma_q')}
    # senza abitudini: i distacchi delle libere presi così come sono
    sg2 = collections.defaultdict(list)
    for dn, lst in sg.items():
        for nome, v, p in lst:
            if nome == 'forma_q':
                sg2[dn].append((nome, v, p))
    for k in ('FP1', 'FP2', 'FP3', 'SQ'):
        x = w['libere'].get(k)
        if not x or x['bagnata']:
            continue
        for dn, g in x['gap'].items():
            sg2[dn].append((k, g, 1.0 / tm._var(st.err, k)))
    out['-abitudini delle squadre'] = {dn: v for dn, v in tm.combina(sg2).items() if dn in tit}
    # senza pesi per errore: media semplice dei segnali
    out['-pesi per errore (media semplice)'] = {dn: float(np.mean([v for _, v, _ in lst]))
                                                for dn, lst in sg.items() if lst and dn in tit}
    # regola banale: distacchi dell'ultima sessione asciutta prima della qualifica (il foglio tempi)
    kf, _, _, _ = tm.ultimo_foglio(w, tit)
    if kf:
        out['banale: foglio tempi'] = {dn: g for dn, g in w['libere'][kf]['gap'].items() if dn in tit}
    return out


def _nll(theta, casi, l2):
    ll, g = 0.0, np.zeros(len(theta))
    H = np.zeros((len(theta), len(theta)))
    for X, i in casi:
        u = X @ theta
        u = u - u.max()
        p = np.exp(u)
        p /= p.sum()
        ll += math.log(max(p[i], 1e-300))
        m = p @ X
        g += X[i] - m
        H -= (X * p[:, None]).T @ X - np.outer(m, m)
    ll -= 0.5 * l2 * float(theta[1:] @ theta[1:])
    g[1:] -= l2 * theta[1:]
    H[1:, 1:] -= l2 * np.eye(len(theta) - 1)
    return -ll, -g, -H


def fit_clogit(casi, dim, l2=1.0):
    """logit condizionale P(d) ∝ exp(theta · x_d); il primo coefficiente è sul distacco in secondi"""
    th = np.zeros(dim)
    th[0] = -5.0
    for _ in range(40):
        _, g, H = _nll(th, casi, l2)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -5, 5)
        th = th - step
        if np.abs(step).max() < 1e-7:
            break
    return th


def _prob_clogit(theta, X):
    u = X @ theta
    u = u - u.max()
    p = np.exp(u)
    return p / p.sum()


def sezione_qualifica(storico, anni, dal, anno0=2023, min_casi=15):
    import taratura_matrice as tm
    t0 = time.time()
    ws = tm.carica(storico, list(range(min(anno0, min(anni)), max(anni) + 1)))
    print('  qualifica: %d weekend letti (%.0fs)' % (len(ws), time.time() - t0), flush=True)
    st = tm.Stato()
    casi = collections.defaultdict(list)     # (uscita, variante) -> [(X, indice del vincitore)]
    righe, per_pole = [], {}
    for w in ws:
        st.nuova_stagione(w['anno'])
        tit = st.titolari(w)
        sg_s = st.segnali_secco(w)
        sg_l = st.segnali_lr(w)
        giro = tm.giro_rif(w)
        if not w['futuro'] and giro:
            var = _secco_varianti(tm, st, w, sg_s, tit)
            lr = tm.combina(sg_l)
            vinc_gara = None
            if w['arrivo']:
                primi = [dn for dn, pos in w['arrivo'].items() if pos == 1]
                vinc_gara = primi[0] if primi else None
            uscite = (('pole', w['pole'], w['pole'] is not None and not w['q_bagnata']),
                      ('vittoria_preQ', vinc_gara, vinc_gara is not None))
            if 'banale: foglio tempi' in var:
                for uscita, vinc, ok in uscite:
                    if not ok:
                        continue
                    base = dict(var)
                    base['+correzione Ferrari'] = var['completo']
                    base['+long run (modello gomme)'] = var['completo']
                    probs = {}
                    for nome, xs in base.items():
                        cand = sorted(xs)
                        if len(cand) < 5:
                            continue
                        b0 = min(xs.values())
                        cols = [[(xs[d] - b0) * giro / 100.0 for d in cand]]
                        if nome == '+correzione Ferrari':
                            cols.append([1.0 if 'Ferrari' in (w['team'].get(d) or '') else 0.0 for d in cand])
                        if nome == '+long run (modello gomme)':
                            vals = [lr[d] for d in cand if d in lr]
                            if len(vals) >= 5:
                                l0, med = min(vals), float(np.median(vals))
                                cols.append([((lr[d] if d in lr else med) - l0) * giro / 100.0 for d in cand])
                            else:
                                cols.append([0.0] * len(cand))
                        X = np.array(cols).T
                        chiave = (uscita, nome)
                        if w['anno'] in anni and w['round'] >= dal and len(casi[chiave]) >= min_casi:
                            th = fit_clogit(casi[chiave], X.shape[1])
                            probs[nome] = dict(zip(cand, _prob_clogit(th, X)))
                        if vinc in cand:
                            casi[chiave].append((X, cand.index(vinc)))
                    if len(probs) == len(base):
                        ev = '%s' % (w['luogo'] or w['gp'])
                        candidati = set(tit)
                        for nome, pv in probs.items():
                            righe.append(riga(uscita, w['anno'], w['round'], ev, nome,
                                              brier_multi(pv, vinc, candidati), logloss(pv.get(vinc, 0.0)),
                                              max(pv, key=pv.get) == vinc, {'p_fav': round(float(max(pv.values())), 4)}))
                        if uscita == 'pole':
                            per_pole[(w['anno'], w['round'])] = dict(
                                completo={w['piloti'][d][0]: p for d, p in probs['completo'].items()},
                                foglio={w['piloti'][d][0]: p for d, p in probs['banale: foglio tempi'].items()},
                                pole=w['piloti'][vinc][0], luogo=w['luogo'],
                                candidati={w['piloti'][d][0] for d in candidati if d in w['piloti']})
        if not w['futuro']:
            st.aggiorna(w, sg_s, sg_l)
    print('  qualifica: fatto (%.0fs)' % (time.time() - t0), flush=True)
    return righe, per_pole


# ---------------------------------------------------------------------------------------------
# 3. Strumenti per la pole: pole_telemetria.py (copia della sua logica, ma solo con i weekend precedenti)
# ---------------------------------------------------------------------------------------------

def _soft(s):
    """pilota -> (giro migliore su Soft, squadra), anche cancellato per i limiti (come pole_telemetria.py)"""
    ac = {x['driver_number']: x.get('name_acronym') for x in s.get('drivers') or []}
    sq = {x['driver_number']: x.get('team_name') for x in s.get('drivers') or []}
    comp = {}
    for x in s.get('stints') or []:
        for g in range((x.get('lap_start') or 0), (x.get('lap_end') or -1) + 1):
            comp[(x['driver_number'], g)] = x.get('compound')
    out = {}
    for lp in s.get('laps') or []:
        t, dn = lp.get('lap_duration'), lp['driver_number']
        if not t or lp.get('is_pit_out_lap') or comp.get((dn, lp['lap_number'])) != 'SOFT':
            continue
        if dn not in out or t < out[dn]:
            out[dn] = t
    return {ac[dn]: (t, sq.get(dn)) for dn, t in out.items() if dn in ac}


def _distacchi_soft(s, lim=1.5):
    g = _soft(s)
    if not g:
        return {}
    b0 = min(v[0] for v in g.values())
    return {p: (v[0] - b0, v[1]) for p, v in g.items() if v[0] - b0 < lim}


def _weekend_telemetria(path, tm):
    d = json.load(gzip.open(path))
    S = {s['session']['session_name']: s for s in d['sessioni']}
    if 'Sprint' in S or 'Qualifying' not in S or tm._bagnata(S['Qualifying']):
        return None
    if any(k not in S or tm._bagnata(S[k]) for k in ('Practice 2', 'Practice 3')):
        return None
    ac = {x['driver_number']: x.get('name_acronym') for x in S['Qualifying'].get('drivers') or []}
    q = {}
    for r in S['Qualifying'].get('session_result') or []:
        t = tm._migliore(r.get('duration'))
        if t and r['driver_number'] in ac:
            q[ac[r['driver_number']]] = t
    if len(q) < 10:
        return None
    q0 = min(q.values())
    return dict(anno=d['anno'], rnd=d['round'], q={p: t - q0 for p, t in q.items()}, pole=min(q, key=q.get),
                fp={'FP2': _distacchi_soft(S['Practice 2']), 'FP3': _distacchi_soft(S['Practice 3'])})


def _previsione_tel(fp, med):
    out = {}
    for p in sorted(set(fp['FP2']) | set(fp['FP3'])):    # ordine fisso (vedi _prob_tel)
        v = [fp[k][p][0] + med.get((k, fp[k][p][1]), 0.0) for k in ('FP2', 'FP3') if p in fp[k]]
        if v:
            out[p] = float(np.mean(v))
    return out


def _stima_tel(W, usa_spostamento=True):
    sh = collections.defaultdict(list)
    if usa_spostamento:
        for w in W:
            for k in ('FP2', 'FP3'):
                for p, (g, sq) in w['fp'][k].items():
                    if p in w['q'] and w['q'][p] < 2.0:
                        sh[(k, sq)].append(w['q'][p] - g)
    med = {key: float(np.median(v)) for key, v in sh.items() if len(v) >= 4}
    res = []
    for w in W:
        for p, mu in _previsione_tel(w['fp'], med).items():
            if p in w['q'] and w['q'][p] < 2.0:
                res.append(w['q'][p] - mu)
    return med, np.array(res)


def _prob_tel(mu, res, n=20000, seed=0):
    piloti = sorted(mu)    # ordine fisso: con l'ordine di un set il rumore cambierebbe da un lancio all'altro
    m = np.array([mu[p] for p in piloti])
    sims = m[None, :] + np.random.default_rng(seed).choice(res, size=(n, len(piloti)))
    return dict(zip(piloti, np.bincount(sims.argmin(axis=1), minlength=len(piloti)) / n))


def sezione_strumenti_pole(storico, anni, dal, per_pole):
    import taratura_matrice as tm
    t0 = time.time()
    W = []
    for anno in range(min(anni) - 1, max(anni) + 1):
        for f in sorted(glob.glob(os.path.join(storico, str(anno), '*.json.gz'))):
            x = _weekend_telemetria(f, tm)
            if x:
                W.append(x)
    righe = []
    for i, w in enumerate(W):
        if w['anno'] not in anni or w['rnd'] < dal or (w['anno'], w['rnd']) not in per_pole:
            continue
        prec = [x for x in W[:i] if x['anno'] >= w['anno'] - 1]
        if len(prec) < 6:
            continue
        rif = per_pole[(w['anno'], w['rnd'])]
        cand = rif['candidati']
        probs = {'completo': rif['completo'], 'banale: foglio tempi': rif['foglio']}
        for nome, usa in (('altro strumento: pole_telemetria con spostamento squadra', True),
                          ('altro strumento: pole_telemetria senza spostamento', False)):
            med, res = _stima_tel(prec, usa)
            mu = _previsione_tel(w['fp'], med)
            if not mu or len(res) < 20:
                break
            probs[nome] = _prob_tel(mu, res)
        if len(probs) < 4:
            continue
        for nome, pv in probs.items():
            righe.append(riga('pole_strumenti', w['anno'], w['rnd'], rif['luogo'], nome,
                              brier_multi(pv, rif['pole'], cand | set(pv)), logloss(pv.get(rif['pole'], 0.0)),
                              max(pv, key=pv.get) == rif['pole'], {'p_fav': round(float(max(pv.values())), 4)}))
    print('  strumenti pole: %d weekend confrontabili (%.0fs)' % (len({(r['anno'], r['round']) for r in righe}),
                                                                time.time() - t0), flush=True)
    return righe


# ---------------------------------------------------------------------------------------------
# 4. Primo nelle libere: modello_libere.py, un pezzo alla volta
# ---------------------------------------------------------------------------------------------

def sezione_libere(storico, anni, dal):
    import modello_libere as L
    t0 = time.time()
    P0 = json.loads(json.dumps(L.P))
    orig = {k: getattr(L, k) for k in ('segnali', 'correzione_fp1', 'caratteristiche_compagni')}

    def solo_segnali(filtro):
        def f(w, k, team, S):
            return [s for s in orig['segnali'](w, k, team, S) if filtro(s[0])]
        return f

    varianti = [
        ('completo', {}, {}),
        ('-abitudini delle squadre', {'abitudine': {'FP1': 'no', 'FP2': 'no', 'FP3': 'no'}}, {}),
        ('-tipo di pista (FP2 lente/altre)', {'abitudine': {'FP1': 'no', 'FP2': 'team', 'FP3': 'team'}}, {}),
        ('-una macchina (rookie in FP1)', {'una_macchina': 0.0}, {}),
        ('-correzione FP1 per squadra', {}, {'correzione_fp1': lambda prev, squadre: {t: 1.0 for t in squadre}}),
        ('-scelta fra compagni (50/50)', {}, {'caratteristiche_compagni': lambda *a, **k: np.zeros(3)}),
        ('-forma stagionale', {}, {'segnali': solo_segnali(lambda n: n != 'forma')}),
        ('-libere già corse (solo forma)', {}, {'segnali': solo_segnali(lambda n: n == 'forma')}),
        # versione snella: restano forma, libere già corse, una macchina e scelta fra compagni; via abitudini,
        # tipo di pista e correzione FP1 (scelta sugli stessi dati: la verifica vera è dal Messico in poi)
        ('snello: forma + libere + una macchina + compagni', {'abitudine': {'FP1': 'no', 'FP2': 'no', 'FP3': 'no'}},
         {'correzione_fp1': lambda prev, squadre: {t: 1.0 for t in squadre}}),
    ]
    righe = []
    circuiti = L.leggi_circuiti(storico)
    for anno in anni:
        storia = L.prepara(L.carica(storico, anno))
        piste = L.Piste(storia, circuiti)
        for nome, p_mod, f_mod in varianti:
            L.P.update(json.loads(json.dumps(P0)))
            L.P.update(p_mod)
            for k, f in f_mod.items():
                setattr(L, k, f)
            try:
                prev, _, _ = L.motore(storia, piste)
            finally:
                for k, f in orig.items():
                    setattr(L, k, f)
                L.P.update(json.loads(json.dumps(P0)))
            for r in prev:
                if r['round'] < dal:
                    continue
                ev = '%s %s' % (r['sessione'], r['luogo'])
                cand = set(r['pp'])
                righe.append(riga('libere', anno, r['round'], ev, nome, brier_multi(r['pp'], r['primo'], cand),
                                  logloss(r['pp'].get(r['primo'], 0.0)), max(r['pp'], key=r['pp'].get) == r['primo'],
                                  {'sessione': r['sessione'], 'p_fav': round(float(max(r['pp'].values())), 4)}))
                if nome == 'completo':
                    pi = r['pp_ing']
                    righe.append(riga('libere', anno, r['round'], ev, 'banale: forma e libere senza correzioni',
                                      brier_multi(pi, r['primo'], set(pi)), logloss(pi.get(r['primo'], 0.0)),
                                      max(pi, key=pi.get) == r['primo'],
                                      {'sessione': r['sessione'], 'p_fav': round(float(max(pi.values())), 4)}))
        print('  libere %d fatto (%.0fs)' % (anno, time.time() - t0), flush=True)
    return righe


# ---------------------------------------------------------------------------------------------
# 5. Testa a testa in qualifica: modello_h2h_quali.py, una caratteristica alla volta
# ---------------------------------------------------------------------------------------------

def sezione_h2h(storico, anni, dal):
    import modello_h2h_quali as H
    t0 = time.time()
    C = H.CARATT   # ['stagione', 'fp1', 'fp2', 'fp3', 'sq']
    varianti = [('completo', C),
                ('-media stagionale della coppia', [c for c in C if c != 'stagione']),
                ('-FP1', [c for c in C if c != 'fp1']),
                ('-FP2', [c for c in C if c != 'fp2']),
                ('-FP3', [c for c in C if c != 'fp3']),
                ('-Sprint Shootout', [c for c in C if c != 'sq']),
                ('-libere (solo stagione e Shootout)', ['stagione', 'sq']),
                # versione snella (scelta sugli stessi dati: verifica vera dal Messico in poi)
                ('snello: media stagionale + FP3', ['stagione', 'fp3'])]
    righe = []
    for anno in anni:
        storia = [w for w in H.carica(storico, anno) if 'Qualifying' in w['sessioni']]
        tutte = H.tabella(storia)
        for rnd in sorted({r[2]['round'] for r in tutte}):
            if rnd < dal:
                continue
            train = [r for r in tutte if r[2]['round'] < rnd]
            test = [r for r in tutte if r[2]['round'] == rnd]
            if len(train) < 20 or not test:
                continue
            Xtr = np.array([r[0] for r in train])
            Ytr = [r[1] for r in train]
            y = np.array([r[1] for r in test], float)
            Xte = np.array([r[0] for r in test])
            ev = 'Qualifica %s' % test[0][2]['luogo']
            for nome, usa in varianti:
                m = np.array([c in usa for c in C], float)
                beta = H.stima_pesi([(x * m, yy, None) for x, yy in zip(Xtr, Ytr)])
                p = 1 / (1 + np.exp(-(Xte * m) @ beta))
                b = float(np.mean((p - y) ** 2))
                ll = float(-np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1))))
                righe.append(riga('h2h', anno, rnd, ev, nome, b, ll, np.mean((p > 0.5) == (y == 1)), {'coppie': len(test)}))
            pb = []
            for r in test:
                v, s = r[2]['det']['bilancio']      # (davanti, dietro) dal punto di vista di A
                pb.append((v + 1) / (v + s + 2) if (v + s) else 0.5)
            pb = np.array(pb)
            righe.append(riga('h2h', anno, rnd, ev, 'banale: solo bilancio stagionale', float(np.mean((pb - y) ** 2)),
                              float(-np.mean(y * np.log(np.clip(pb, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - pb, 1e-4, 1)))),
                              np.mean((pb > 0.5) == (y == 1)), {'coppie': len(test)}))
        print('  h2h %d fatto (%.0fs)' % (anno, time.time() - t0), flush=True)
    return righe


# ---------------------------------------------------------------------------------------------
# 6. Riepilogo
# ---------------------------------------------------------------------------------------------

ORDINE_USCITE = ['pole', 'pole_strumenti', 'vittoria_preQ', 'vittoria', 'libere', 'h2h']
TITOLI = {'pole': 'Pole, prima della qualifica (giro secco)',
          'pole_strumenti': 'Pole: strumenti a confronto, stessi weekend (riferimento giro secco completo)',
          'vittoria_preQ': 'Vittoria in gara, prima della qualifica (giro secco)',
          'vittoria': 'Vittoria in gara, dopo la qualifica (modello_gara.py)',
          'libere': 'Primo nelle libere (modello_libere.py)',
          'h2h': 'Testa a testa in qualifica (modello_h2h_quali.py)'}


def riepilogo(righe, anni):
    out = {}
    for uscita in ORDINE_USCITE:
        R = [r for r in righe if r['uscita'] == uscita and r['anno'] in anni]
        if not R:
            continue
        per_var = collections.defaultdict(dict)
        for r in R:
            per_var[r['variante']][(r['anno'], r['round'], r['evento'])] = r
        if 'completo' not in per_var:
            continue
        ref = per_var['completo']
        voci = []
        for nome, ev in per_var.items():
            chiavi = sorted(set(ev) & set(ref))
            if not chiavi:
                continue
            d = [ev[c]['brier'] - ref[c]['brier'] for c in chiavi]
            dl = [ev[c]['logloss'] - ref[c]['logloss'] for c in chiavi]
            delta, lo, hi = bootstrap(d)
            bl = bootstrap(dl)
            stagioni = {}
            for a in anni:
                da = [ev[c]['brier'] - ref[c]['brier'] for c in chiavi if c[0] == a]
                if da:
                    stagioni[a] = (round(float(np.mean(da)), 4), len(da))
            pf = [float(ev[c]['p_fav']) for c in chiavi if ev[c].get('p_fav') not in (None, '')]
            # taratura del favorito: probabilità media data al favorito meno quante volte vince; scarto
            # appaiato con il completo, evento per evento
            fav = None
            if len(pf) == len(chiavi) and all(ref[c].get('p_fav') not in (None, '') for c in chiavi):
                sv = np.array([float(ev[c]['p_fav']) - float(ev[c]['preso']) for c in chiavi])
                sr = np.array([float(ref[c]['p_fav']) - float(ref[c]['preso']) for c in chiavi])
                dd = bootstrap(sv - sr)
                fav = dict(scarto=float(sv.mean()), scarto_completo=float(sr.mean()), delta=dd[0], lo=dd[1], hi=dd[2])
            gonfia = bool(fav and fav['scarto'] > 0 and fav['lo'] > 0)
            voci.append(dict(variante=nome, n=len(chiavi),
                             p_fav=float(np.mean(pf)) if len(pf) == len(chiavi) else None, favorito=fav,
                             brier=float(np.mean([ev[c]['brier'] for c in chiavi])),
                             brier_completo=float(np.mean([ref[c]['brier'] for c in chiavi])),
                             delta=delta, lo=lo, hi=hi, delta_logloss=bl[0], lo_logloss=bl[1], hi_logloss=bl[2],
                             presi=float(np.mean([ev[c]['preso'] for c in chiavi])),
                             stagioni=stagioni,
                             verdetto='riferimento' if nome == 'completo' else verdetto(nome, (delta, lo, hi), bl,
                                                                                        gonfia)))
        voci.sort(key=lambda v: (v['variante'] != 'completo', not v['variante'].startswith('banale'),
                                 v['variante'].startswith('+'), v['variante']))
        out[uscita] = voci
    return out


def stampa(rie, anni):
    print('\nProva di sottrazione, stagioni %s. Δ = Brier della variante - Brier del completo (per evento),'
          ' intervallo al 90%%.' % ','.join(map(str, anni)))
    for uscita, voci in rie.items():
        print('\n%s' % TITOLI[uscita])
        for v in voci:
            if v['variante'] == 'completo':
                print('  %-50s n=%3d  Brier %.4f  favorito preso %3.0f%%%s' % (
                    'completo', v['n'], v['brier'], 100 * v['presi'],
                    '' if v['p_fav'] is None else ' (probabilità media data al favorito %.0f%%)' % (100 * v['p_fav'])))
                continue
            st = ' '.join('%d:%+.3f' % (a, x[0]) for a, x in sorted(v['stagioni'].items()))
            pf = '' if v['p_fav'] is None else ' (p media %2.0f%%)' % (100 * v['p_fav'])
            if v.get('favorito'):
                pf += ' favorito %+.1f punti sul completo (%+.1f; %+.1f)' % (
                    100 * v['favorito']['delta'], 100 * v['favorito']['lo'], 100 * v['favorito']['hi'])
            print('  %-50s n=%3d  Brier %.4f  Δ %+.4f (%+.4f; %+.4f)  Δ log-loss %+.3f (%+.3f; %+.3f)  presi %3.0f%%%s'
                  '  [%s]  -> %s' % (v['variante'][:50], v['n'], v['brier'], v['delta'], v['lo'], v['hi'],
                                     v['delta_logloss'], v['lo_logloss'], v['hi_logloss'], 100 * v['presi'], pf, st,
                                     v['verdetto']))


def scrivi(out_dir, righe, rie, meta):
    os.makedirs(out_dir, exist_ok=True)
    cols = ['uscita', 'anno', 'round', 'evento', 'variante', 'brier', 'logloss', 'preso', 'p_fav', 'tipo', 'sessione',
            'coppie']
    with open(os.path.join(out_dir, 'sottrazione_righe.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        wr.writeheader()
        for r in righe:
            wr.writerow(r)
    with open(os.path.join(out_dir, 'sottrazione_riepilogo.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(meta=meta, uscite=rie), f, indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description='Prova di sottrazione dei modelli Muretto')
    ap.add_argument('--storico', default='storico')
    ap.add_argument('--anni', default='2024,2025,2026')
    ap.add_argument('--out', default='risultati')
    ap.add_argument('--solo', default='gara,pole,libere,h2h', help='sezioni: gara, pole, libere, h2h')
    ap.add_argument('--n', type=int, default=4000, help='simulazioni per corsa nella gara')
    ap.add_argument('--processi', type=int, default=2)
    ap.add_argument('--dal', type=int, default=4)
    ap.add_argument('--righe', default=None, help='riusa le righe di sottrazione_righe.csv già scritti (più file con la virgola)')
    ap.add_argument('--varianti-gara', dest='varianti_gara', default=None,
                    help='nella gara simula solo queste varianti, separate da «;» (completo, griglia e ricalibrazione sempre)')
    a = ap.parse_args()
    anni = [int(x) for x in a.anni.split(',')]
    solo = set(a.solo.split(','))
    t0 = time.time()
    righe = []
    if a.righe:
        visti = set()
        for path in a.righe.split(','):
            with open(path, encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    for k in ('anno', 'round'):
                        r[k] = int(r[k])
                    for k in ('brier', 'logloss', 'preso'):
                        r[k] = float(r[k])
                    chiave = (r['uscita'], r['anno'], r['round'], r['evento'], r['variante'])
                    if chiave not in visti:      # la stessa riga in due file (es. il completo) conta una volta
                        visti.add(chiave)
                        righe.append(r)
    if 'pole' in solo:
        rq, per_pole = sezione_qualifica(a.storico, anni, a.dal)
        righe += rq
        righe += sezione_strumenti_pole(a.storico, anni, a.dal, per_pole)
    if 'libere' in solo:
        righe += sezione_libere(a.storico, anni, a.dal)
    if 'h2h' in solo:
        righe += sezione_h2h(a.storico, anni, a.dal)
    if 'gara' in solo:
        filtro = set(a.varianti_gara.split(';')) if a.varianti_gara else None
        righe += sezione_gara(a.storico, anni, a.n, a.dal, a.processi, filtro)
    rie = riepilogo(righe, anni)
    stampa(rie, anni)
    scrivi(a.out, righe, rie, dict(anni=anni, n_sim=a.n, dal=a.dal, sezioni=sorted(solo),
                                  durata_s=round(time.time() - t0)))
    print('\nScritti %s e %s (%.0f s).' % (os.path.join(a.out, 'sottrazione_righe.csv'),
                                           os.path.join(a.out, 'sottrazione_riepilogo.json'), time.time() - t0))


if __name__ == '__main__':
    main()
