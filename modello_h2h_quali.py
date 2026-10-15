#!/usr/bin/env python3
"""
modello_h2h_quali.py — Muretto: testa a testa in qualifica tra compagni di squadra.

Un solo file, solo numpy (Python 3.9+). Legge l'archivio OpenF1 di Muretto (storico/), lo stesso
di modello_gara.py: per ogni weekend la classifica di qualifica e i giri delle sessioni che la
precedono (FP1, FP2, FP3 nei weekend normali; FP1 e Sprint Shootout in quelli con la Sprint).

Che cosa stima: la probabilità che il pilota A si classifichi davanti al compagno B nella qualifica
ufficiale (classifica della sessione, senza penalità in griglia). Chi non fa il tempo o viene
escluso dalla classifica conta come «dietro» (vedi il regolamento del mercato prima di giocare).

Ingredienti (distacchi in % sul giro, positivi = A più veloce):
  stagione  distacchi in qualifica e Sprint Shootout della coppia nei weekend precedenti, nel segmento
            (Q1/Q2/Q3) più profondo fatto da entrambi; pesati verso i più recenti (emivita 6 eventi)
            e ristretti verso zero quando i casi sono pochi
  fp1..fp3  distacco nelle libere: media fra giro migliore (su gomma morbida se l'hanno usata
            entrambi) e giro ideale (somma dei settori migliori). Si scarta la sessione se è bagnata,
            se uno dei due ha meno di 3 giri cronometrati o se il distacco supera l'1,5% (giro non
            rappresentativo: guasto, programma diverso, bandiera rossa)
  sq        distacco nello Sprint Shootout, nel segmento più profondo fatto da entrambi
Tutti i distacchi sono tagliati a ±1%. La probabilità viene da una regressione logistica senza
intercetta (scambiare A e B cambia solo il segno), stimata su tutte le coppie di tutti i team
nei weekend già corsi: i pesi sono gli stessi per tutti, cambiano solo i dati della coppia.

Uso:
  python modello_h2h_quali.py prova  --storico storico [--dal 4] [--out risultati]
  python modello_h2h_quali.py scheda --storico storico --round 18 [--squadre "Alpine,Cadillac"]
                                     [--quote "GAS-COL=1.40/2.75,PER-BOT=1.85/1.85"] [--tutte]
Per la scheda del weekend in corso serve il file del weekend nello storico con le libere già finite:
Actions → Storico → Run workflow dopo la FP3 (il file «parziale» contiene le sole sessioni finite).
"""
import argparse
import glob
import gzip
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

P = {
    'taglio': 1.0,        # % : distacchi tagliati a ±1%
    'max_libere': 1.5,    # % : distacco nelle libere oltre questo = sessione non rappresentativa per la coppia
    'min_giri': 3,        # giri cronometrati minimi per pilota in una sessione di libere
    'emivita': 6.0,       # eventi: peso 1/2 dopo 6 eventi nella media stagionale
    'k0': 3.0,            # restringimento della media stagionale: n/(n+k0)
    'l2': 0.5,            # regolarizzazione della logistica
    'margine': 0.05,      # si gioca solo con quota >= quota minima x 1,05
    'dal_round': 4,       # prova a ritroso: primo round previsto
}

# Scuderie scelte per i testa a testa (15 ottobre 2026), vedi claude/modello-h2h-qualifica-2026.md
SQUADRE_SCELTE = ['Alpine', 'Cadillac']
TOP4 = {'Mercedes', 'Ferrari', 'McLaren', 'Red Bull Racing'}

CARATT = ['stagione', 'fp1', 'fp2', 'fp3', 'sq']
LIBERE = {'fp1': 'Practice 1', 'fp2': 'Practice 2', 'fp3': 'Practice 3'}
ORDINE_SESSIONI = ['Practice 1', 'Practice 2', 'Practice 3', 'Sprint Qualifying', 'Sprint', 'Qualifying']


# ---------------------------------------------------------------------------------------------
# 1. Lettura dello storico
# ---------------------------------------------------------------------------------------------

def _giri(sess):
    """per pilota: giro migliore, migliore su SOFT, giro ideale, numero di giri cronometrati"""
    mescola = {}
    for st in sess.get('stints') or []:
        dn = st.get('driver_number')
        a, b = st.get('lap_start') or 0, st.get('lap_end') or -1
        for g in range(a, b + 1):
            mescola[(dn, g)] = st.get('compound')
    out = {}
    for lp in sess.get('laps') or []:
        dn = lp.get('driver_number')
        o = out.setdefault(dn, {'best': None, 'soft': None, 'sett': [None, None, None], 'n': 0})
        if lp.get('is_pit_out_lap'):
            continue
        t = lp.get('lap_duration')
        if not t or not (50 < t < 200):
            continue
        o['n'] += 1
        if o['best'] is None or t < o['best']:
            o['best'] = t
        if mescola.get((dn, lp.get('lap_number'))) == 'SOFT' and (o['soft'] is None or t < o['soft']):
            o['soft'] = t
        for i in range(3):
            x = lp.get('duration_sector_%d' % (i + 1))
            if x and x > 5 and (o['sett'][i] is None or x < o['sett'][i]):
                o['sett'][i] = x
    for o in out.values():
        o['ideale'] = sum(o['sett']) if all(o['sett']) else None
        del o['sett']
    return out


def _classifica(sess):
    """classifica di qualifica o Sprint Shootout: posizione (None se senza tempo/esclusa) e tempi Q1-Q3"""
    res = {}
    for r in sess.get('session_result') or []:
        pos = r.get('position')
        pos = pos if isinstance(pos, int) else None
        dur = r.get('duration')
        if not isinstance(dur, list):
            dur = [dur, None, None]
        dur = (list(dur) + [None, None, None])[:3]
        res[r.get('driver_number')] = {'pos': pos, 'tempi': dur, 'dnf': bool(r.get('dnf')),
                                       'dns': bool(r.get('dns')), 'dsq': bool(r.get('dsq'))}
    return res


def _pioggia(sess):
    return any((x.get('rainfall') or 0) > 0 for x in sess.get('weather') or [])


def leggi_weekend(path):
    d = json.load(gzip.open(path))
    sess = {s['session']['session_name']: s for s in d.get('sessioni') or []}
    w = {'round': d.get('round'), 'file': os.path.basename(path), 'luogo': d.get('meeting', {}).get('location'),
         'gp': d.get('meeting', {}).get('meeting_name'), 'sprint': 'Sprint Qualifying' in sess,
         'parziale': bool(d.get('parziale')), 'sessioni': {}, 'squadre': {}, 'sigle': {}}
    for n in ORDINE_SESSIONI:
        if n not in sess:
            continue
        s = sess[n]
        x = {'pioggia': _pioggia(s)}
        if n in ('Practice 1', 'Practice 2', 'Practice 3'):
            x['giri'] = _giri(s)
        if n in ('Sprint Qualifying', 'Qualifying'):
            x['class'] = _classifica(s)
        x['piloti'] = {dr['driver_number']: (dr.get('team_name'), dr.get('name_acronym')) for dr in s.get('drivers') or []}
        w['sessioni'][n] = x
    # formazione: dalla sessione più avanzata (la FP1 può avere i terzi piloti)
    for n in reversed(ORDINE_SESSIONI):
        if n in w['sessioni'] and w['sessioni'][n]['piloti']:
            for dn, (team, sigla) in w['sessioni'][n]['piloti'].items():
                w['squadre'][dn] = team
                w['sigle'][dn] = sigla
            break
    for x in w['sessioni'].values():
        for dn, (team, sigla) in x['piloti'].items():
            w['sigle'].setdefault(dn, sigla)
    return w


def carica(base, anno=2026):
    files = sorted(glob.glob(os.path.join(base, str(anno), '*.json.gz')))
    ws = [leggi_weekend(f) for f in files]
    return sorted(ws, key=lambda w: w['round'] or 0)


def coppie(w):
    """coppie di compagni (a < b per numero) nella formazione del weekend"""
    per_team = defaultdict(list)
    for dn, team in w['squadre'].items():
        per_team[team].append(dn)
    out = []
    for team, ds in per_team.items():
        if len(ds) == 2:
            a, b = sorted(ds)
            out.append((team, a, b))
    return sorted(out)


# ---------------------------------------------------------------------------------------------
# 2. Distacchi
# ---------------------------------------------------------------------------------------------

def _taglia(x):
    c = P['taglio']
    return None if x is None else max(-c, min(c, x))


def _gap(ta, tb):
    """distacco % fra due tempi, positivo = A più veloce"""
    return (tb - ta) / ((ta + tb) / 2.0) * 100.0


def distacco_classifica(cl, a, b):
    """esito e distacco in una classifica a segmenti (qualifica o Sprint Shootout).
    y=1 se A davanti, 0 se dietro, None se nessuno dei due classificato.
    escluso: uno dei due senza posizione o squalificato (i suoi tempi non contano);
    anomalia: escluso, oppure uno dei due non ha finito la sessione (incidente, guasto). Il distacco nel
    segmento comune resta buono anche se poi uno dei due si è fermato."""
    vuoto = {'pos': None, 'tempi': [None, None, None], 'dnf': True, 'dns': True, 'dsq': False}
    qa, qb = cl.get(a) or vuoto, cl.get(b) or vuoto
    pa, pb = qa['pos'], qb['pos']
    if pa is not None and pb is not None:
        y = 1 if pa < pb else 0
    elif pa is not None:
        y = 1
    elif pb is not None:
        y = 0
    else:
        y = None
    g = None
    for k in (2, 1, 0):
        ta, tb = qa['tempi'][k], qb['tempi'][k]
        if ta and tb:
            g = _gap(ta, tb)
            break
    escluso = pa is None or pb is None or qa['dsq'] or qb['dsq']
    anom = escluso or qa['dnf'] or qb['dnf']
    return y, g, anom, escluso


def distacco_libere(sx, a, b):
    """distacco nelle libere, None se la sessione non dice niente sulla coppia"""
    if not sx or sx.get('pioggia'):
        return None
    ga, gb = sx['giri'].get(a), sx['giri'].get(b)
    if not ga or not gb or ga['n'] < P['min_giri'] or gb['n'] < P['min_giri']:
        return None
    misure = []
    if ga['soft'] and gb['soft']:
        misure.append(_gap(ga['soft'], gb['soft']))
    elif ga['best'] and gb['best']:
        misure.append(_gap(ga['best'], gb['best']))
    if ga['ideale'] and gb['ideale']:
        misure.append(_gap(ga['ideale'], gb['ideale']))
    if not misure:
        return None
    g = sum(misure) / len(misure)
    if abs(g) > P['max_libere']:
        return None
    return g


# ---------------------------------------------------------------------------------------------
# 3. Caratteristiche di una coppia in un weekend
# ---------------------------------------------------------------------------------------------

def storia_coppia(storia, a, b, rnd):
    """distacchi passati della coppia (orientati su A) e bilancio in classifica"""
    obs, vinte, n = [], 0, 0
    for w in storia:
        if w['round'] >= rnd:
            continue
        for sess, peso_r in (('Qualifying', 0.0), ('Sprint Qualifying', -0.5)):
            cl = w['sessioni'].get(sess, {}).get('class')
            if not cl or a not in w['squadre'] or b not in w['squadre']:
                continue
            if w['squadre'].get(a) != w['squadre'].get(b):
                continue
            y, g, anom, escluso = distacco_classifica(cl, a, b)
            if g is not None and not escluso:
                obs.append((w['round'] + peso_r, _taglia(g)))
            if sess == 'Qualifying' and y is not None:
                vinte += y
                n += 1
    return obs, vinte, n


def media_stagione(obs, rnd):
    if not obs:
        return 0.0, 0.0, 0.0
    w = np.array([0.5 ** ((rnd - r) / P['emivita']) for r, _ in obs])
    x = np.array([v for _, v in obs])
    m = float((w * x).sum() / w.sum())
    neff = float(w.sum())
    return m * neff / (neff + P['k0']), m, neff


def caratteristiche(storia, w, a, b):
    rnd = w['round']
    obs, vinte, n = storia_coppia(storia, a, b, rnd)
    stag, media, neff = media_stagione(obs, rnd)
    det = {'stagione': stag, 'media': media, 'neff': neff, 'bilancio': (vinte, n - vinte)}
    for k, nome in LIBERE.items():
        g = distacco_libere(w['sessioni'].get(nome), a, b)
        det[k] = _taglia(g)
        det[k + '_grezzo'] = g
    sq = w['sessioni'].get('Sprint Qualifying', {}).get('class')
    det['sq'] = det['sq_grezzo'] = None
    if sq:
        _, g, _, escluso = distacco_classifica(sq, a, b)
        if g is not None and not escluso:
            det['sq'], det['sq_grezzo'] = _taglia(g), g
    x = np.array([det[c] if det[c] is not None else 0.0 for c in CARATT])
    return x, det


# ---------------------------------------------------------------------------------------------
# 4. Stima dei pesi e probabilità
# ---------------------------------------------------------------------------------------------

def tabella(storia, fino_a=None, da=None):
    """righe (x, y, info) di tutte le coppie nei weekend con qualifica"""
    righe = []
    for w in storia:
        if 'Qualifying' not in w['sessioni']:
            continue
        if fino_a is not None and w['round'] >= fino_a:
            continue
        if da is not None and w['round'] < da:
            continue
        cl = w['sessioni']['Qualifying'].get('class') or {}
        for team, a, b in coppie(w):
            y, g, anom, _ = distacco_classifica(cl, a, b)
            if y is None:
                continue
            x, det = caratteristiche(storia, w, a, b)
            righe.append((x, y, {'round': w['round'], 'luogo': w['luogo'], 'team': team, 'a': a, 'b': b,
                                 'A': w['sigle'].get(a), 'B': w['sigle'].get(b), 'gap': g, 'anom': anom,
                                 'pioggia_q': w['sessioni']['Qualifying']['pioggia'], 'det': det}))
    return righe


def stima_pesi(righe):
    X = np.array([r[0] for r in righe])
    Y = np.array([r[1] for r in righe], float)
    # anche la coppia rovesciata: la logistica senza intercetta resta simmetrica
    X = np.vstack([X, -X])
    Y = np.concatenate([Y, 1 - Y])
    b = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (Y - p) - P['l2'] * b
        H = -(X.T * (p * (1 - p))) @ X - P['l2'] * np.eye(len(b))
        passo = np.linalg.solve(H, g)
        b = b - passo
        if np.max(np.abs(passo)) < 1e-9:
            break
    return b


def probabilita(beta, x):
    return float(1 / (1 + math.exp(-float(x @ beta))))


# ---------------------------------------------------------------------------------------------
# 5. Prova a ritroso: ogni round previsto solo con i weekend precedenti
# ---------------------------------------------------------------------------------------------

def _punteggi(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    brier = float(np.mean((p - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    acc = float(np.mean((p > 0.5) == (y == 1)))
    return brier, ll, acc


def prova(base, anno=2026, dal=None, out=None, squadre=None):
    dal = dal or P['dal_round']
    storia = carica(base, anno)
    storia = [w for w in storia if 'Qualifying' in w['sessioni']]
    righe_out = []
    for w in storia:
        if w['round'] < dal:
            continue
        passato = tabella(storia, fino_a=w['round'])
        if len(passato) < 20:
            continue
        beta = stima_pesi(passato)
        for x, y, info in tabella(storia, fino_a=w['round'] + 1, da=w['round']):
            p = probabilita(beta, x)
            v, n = info['det']['bilancio']
            p_bil = (v + 1) / (v + n + 2) if (v + n) else 0.5
            # il bilancio è contato dal punto di vista di A
            righe_out.append(dict(info, p=p, y=y, p_bil=p_bil))
    if not righe_out:
        print('Nessun round da provare.')
        return []

    def blocco(titolo, sel):
        rr = [r for r in righe_out if sel(r)]
        if not rr:
            return
        y = [r['y'] for r in rr]
        bm, lm, am = _punteggi([r['p'] for r in rr], y)
        bb, lb, ab = _punteggi([r['p_bil'] for r in rr], y)
        print('%-22s n=%3d | modello: Brier %.3f, giusti %2.0f%% | solo bilancio stagionale: Brier %.3f, giusti %2.0f%%'
              % (titolo, len(rr), bm, 100 * am, bb, 100 * ab))

    print('Prova a ritroso %d, round %d-%d (ogni round previsto con i soli weekend precedenti). Riferimento: 50/50 = Brier 0.250'
          % (anno, dal, max(r['round'] for r in righe_out)))
    blocco('tutte le squadre', lambda r: True)
    blocco('squadre minori', lambda r: r['team'] not in TOP4)
    blocco('top 4', lambda r: r['team'] in TOP4)
    scelte = squadre or SQUADRE_SCELTE
    blocco('squadre scelte', lambda r: r['team'] in scelte)
    print()
    for t in sorted({r['team'] for r in righe_out}, key=lambda t: (t in TOP4, t)):
        blocco(('* ' if t in scelte else '  ') + t, lambda r, t=t: r['team'] == t)
    print('\nTaratura (squadre minori): probabilità del favorito del modello e frequenza vera')
    rr = [r for r in righe_out if r['team'] not in TOP4]
    for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
        sel = [r for r in rr if lo <= max(r['p'], 1 - r['p']) < hi]
        if sel:
            pf = np.mean([max(r['p'], 1 - r['p']) for r in sel])
            ok = np.mean([(r['p'] >= 0.5) == (r['y'] == 1) for r in sel])
            print('  %2.0f-%2.0f%%: %3d casi, prevista %2.0f%%, vera %2.0f%%' % (100 * lo, min(100, 100 * hi), len(sel), 100 * pf, 100 * ok))
    if out:
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, 'h2h_prova_righe.csv')
        with open(path, 'w') as f:
            f.write('round,luogo,team,A,B,p_A,p_bilancio,esito_A_davanti,distacco_q,anomalia\n')
            for r in righe_out:
                f.write('%d,%s,%s,%s,%s,%.3f,%.3f,%d,%s,%d\n' % (r['round'], r['luogo'], r['team'], r['A'], r['B'], r['p'],
                                                               r['p_bil'], r['y'], '' if r['gap'] is None else '%.3f' % r['gap'], r['anom']))
        print('\nRighe salvate in', path)
    return righe_out


# ---------------------------------------------------------------------------------------------
# 6. Scheda del weekend
# ---------------------------------------------------------------------------------------------

def _quote(s):
    """'GAS-COL=1.40/2.75,PER-BOT=1.85/1.85' -> {('GAS','COL'): (1.40, 2.75), ...}
    con il punto e virgola come separatore si possono scrivere le quote con la virgola: 'GAS-COL=1,40/2,75; ...'"""
    out = {}
    s = s or ''
    for parte in (s.split(';') if ';' in s else s.split(',')):
        parte = parte.strip()
        if not parte or '=' not in parte:
            continue
        coppia, q = parte.split('=', 1)
        x, y = coppia.upper().split('-')
        qa, qb = q.replace(',', '.').split('/')
        out[(x.strip(), y.strip())] = (float(qa), float(qb))
    return out


def _fmt(g, A, B):
    if g is None:
        return '—'
    chi = A if g >= 0 else B
    return '%s +%.2f%%' % (chi, abs(g))


def scheda(base, rnd, anno=2026, squadre=None, quote=None, tutte=False):
    storia = carica(base, anno)
    attuale = [w for w in storia if w['round'] == rnd]
    if not attuale:
        print('Round %d non trovato nello storico: lanciare Actions → Storico dopo le libere.' % rnd)
        return
    w = attuale[0]
    passato = [x for x in storia if x['round'] < rnd]
    righe = tabella(passato)
    beta = stima_pesi(righe)
    sess = [n.replace('Practice ', 'FP').replace('Sprint Qualifying', 'Sprint Shootout')
            for n in ORDINE_SESSIONI if n in w['sessioni'] and n not in ('Qualifying', 'Sprint')]
    print('TESTA A TESTA IN QUALIFICA — round %d, %s (%s)' % (rnd, w['gp'], w['luogo']))
    print('Sessioni usate: %s. Pesi stimati su %d coppie dei weekend precedenti.' % (', '.join(sess) or 'nessuna', len(righe)))
    for nome in ('Practice 1', 'Practice 2', 'Practice 3', 'Sprint Qualifying'):
        if nome in w['sessioni'] and w['sessioni'][nome]['pioggia']:
            print('  Attenzione: %s bagnata, esclusa.' % nome)
    if 'Qualifying' in w['sessioni']:
        print('  (La qualifica è già nello storico: la scheda usa comunque solo le sessioni precedenti.)')
    if 'Practice 3' not in w['sessioni'] and not w['sprint']:
        print('  Manca la FP3: la stima è più debole. Meglio rilanciare lo Storico dopo la FP3.')
    scelte = squadre or SQUADRE_SCELTE
    q = _quote(quote)
    print()
    for team, a, b in coppie(w):
        if not tutte and team not in scelte:
            continue
        x, det = caratteristiche(storia, w, a, b)
        p = probabilita(beta, x)
        A, B = w['sigle'].get(a), w['sigle'].get(b)
        fav, pf = (A, p) if p >= 0.5 else (B, 1 - p)
        v, n = det['bilancio']
        print('%s  %s-%s' % (team, A, B))
        print('  stagione: %s %d-%d in qualifica, distacco medio %s (peso %.1f eventi)'
              % (A, v, n, _fmt(det['media'], A, B), det['neff']))
        lib = '  '.join('%s %s' % (k.upper(), _fmt(det[k + '_grezzo'], A, B)) for k in LIBERE if LIBERE[k] in w['sessioni'])
        if w['sprint']:
            lib += '  Shootout %s' % _fmt(det['sq_grezzo'], A, B)
        print('  libere: ' + (lib or 'nessuna'))
        qmin = 1 / pf
        print('  → favorito %s, probabilità %.0f%%, quota minima %.2f, gioca da %.2f'
              % (fav, 100 * pf, qmin, qmin * (1 + P['margine'])))
        for (x1, x2), (q1, q2) in q.items():
            if {x1, x2} == {A, B}:
                qf = q1 if x1 == fav else q2
                ok = qf >= qmin * (1 + P['margine'])
                print('  quota Sisal %s %.2f → %s' % (fav, qf, 'GIOCA' if ok else 'NON GIOCARE'))
        print()
    print('Pesi: ' + ', '.join('%s %.2f' % (c, b) for c, b in zip(CARATT, beta)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    a1 = sp.add_parser('prova', help='prova a ritroso sui weekend già corsi')
    a1.add_argument('--storico', default='storico')
    a1.add_argument('--anno', type=int, default=2026)
    a1.add_argument('--dal', type=int, default=P['dal_round'])
    a1.add_argument('--out', default=None)
    a1.add_argument('--squadre', default=None, help='nomi dei team separati da virgola (come in OpenF1)')
    a2 = sp.add_parser('scheda', help='probabilità per il weekend in corso')
    a2.add_argument('--storico', default='storico')
    a2.add_argument('--anno', type=int, default=2026)
    a2.add_argument('--round', type=int, required=True)
    a2.add_argument('--squadre', default=None, help='nomi dei team separati da virgola (come in OpenF1)')
    a2.add_argument('--quote', default=None, help='quote Sisal, es. "GAS-COL=1.40/2.75,PER-BOT=1.85/1.85"')
    a2.add_argument('--tutte', action='store_true', help='tutte le coppie, non solo le squadre scelte')
    a = ap.parse_args()
    squadre = [s.strip() for s in a.squadre.split(',')] if getattr(a, 'squadre', None) else None
    if a.cmd == 'prova':
        prova(a.storico, a.anno, a.dal, a.out, squadre)
    else:
        scheda(a.storico, a.round, a.anno, squadre, a.quote, a.tutte)


if __name__ == '__main__':
    main()
