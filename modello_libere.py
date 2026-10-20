#!/usr/bin/env python3
"""
modello_libere.py — Muretto: chi chiude primo nelle prove libere (FP1, FP2, FP3).

Un solo file, solo numpy (Python 3.9+). Legge l'archivio OpenF1 di Muretto (storico/), lo stesso di
modello_gara.py e modello_h2h_quali.py, e storico/circuiti per il tipo di pista.

Che cosa stima: per ogni pilota la probabilità di fare il giro più veloce della sessione (classifica
ufficiale), usando solo quello che si sa prima della sessione. Base: la verifica del 16 ottobre 2026
(claude/verifica-mercato-libere-2026.md nel progetto).

Come ragiona (distacchi dei team in % dalla media dei 3 team più veloci della sessione):
  passo      distacco previsto in qualifica: forma (qualifiche asciutte dei weekend precedenti, emivita
             6 weekend) e libere già corse nel weekend, corrette per le abitudini; ogni pezzo pesa
             secondo l'errore che ha fatto nel prevedere la qualifica nei weekend precedenti, la forma
             il doppio (i distacchi di una sola sessione regrediscono)
  abitudini  scarto medio del team fra libera e qualifica nei weekend precedenti, ristretto verso zero:
             FP1 nessuna (non sono stabili); FP2 separata fra piste lente e le altre; FP3 uguale ovunque
  una auto   in FP1 il team con un titolare fermo per un rookie parte con +0,25%
  team       probabilità che il giro più veloce sia del team: logit sui distacchi previsti, temperatura
             stimata sulle libere dei weekend precedenti (quasi uguale per FP1, FP2 e FP3)
  FP1        correzione per le FP1 vinte in passato rispetto alle attese (la Ferrari va oltre il passo)
  compagni   quale dei due fa il tempo: regressione logistica sul bilancio fra compagni nello stesso tipo
             di sessione e sui distacchi nelle libere già corse del weekend
  pista      lenta se la velocità media del giro di qualifica sta nel terzo più lento delle piste 2026;
             per le piste non ancora corse nel 2026 si stima dalle qualifiche 2024-2025 (storico/circuiti)

Regole di giocata (favorito + quota minima, come gli altri modelli):
  FP1  lotteria: si gioca solo Leclerc, se corre lui, con la stima prudente del 25% (quota minima 4,00)
  FP2  regola A/B con margine del 10% (l'effetto della pista è un indizio forte su pochi weekend)
  FP3  regola A/B con margine del 5%
  A = favorito del modello se la quota è almeno la quota minima più il margine; B = altrimenti il miglior
  valore fra i primi tre del modello, se passa la stessa soglia.

Registro delle quote: le quote storiche delle libere non esistono, si costruiscono weekend per weekend.
La scheda con --quote e --registro scrive nel registro (CSV) quota, probabilità e decisione di ogni pilota
quotato; «esiti» completa gli esiti dallo storico; «bilancio» confronta modello e mercato.

Uso:
  python modello_libere.py prova    --storico storico [--dal 4] [--out risultati]
  python modello_libere.py scheda   --storico storico --round 18 [--sessione FP1|FP2|FP3]
                                    [--circuito austin] [--fuori LEC] [--quote "RUS=3.00,ANT=3.50"]
                                    [--registro registro_quote_libere.csv] [--momento pre|chiusura]
  python modello_libere.py esiti    --storico storico --registro registro_quote_libere.csv
  python modello_libere.py bilancio --registro registro_quote_libere.csv
Prima della FP1 il weekend non è ancora nello storico: dare --circuito (nome del file in storico/circuiti,
basta una parte: austin, mexico, interlagos, las-vegas, lusail, yas). Per FP2 e FP3 serve il file del
weekend con le sessioni finite: Actions → Storico → Run workflow (file «parziale»).
"""
import argparse
import csv
import glob
import gzip
import json
import math
import os
from collections import defaultdict

import numpy as np

P = {
    'emivita_forma': 6.0,   # weekend: nella forma una qualifica pesa la metà dopo 6 weekend
    'k_abitudine': 3.0,     # restringimento delle abitudini verso zero: n/(n+k)
    'k_pista': 4.0,         # restringimento dell'abitudine FP2 per tipo di pista verso quella generale
    'taglio_scarto': 1.5,   # % : scarti libera-qualifica tagliati a ±1,5%
    'taglio_gap': 3.0,      # % : distacchi dei team tagliati al 3% (sessioni rovinate)
    'una_macchina': 0.25,   # % : team con un solo titolare in pista (rookie in FP1)
    'alfa_fp1': 8.0,        # forza della correzione per le FP1 vinte: più alta = correzione più debole
    'm_tau': 20.0,          # sessioni equivalenti che legano la temperatura di un tipo a quella comune
    'tau_base': 0.30,       # temperatura quando non ci sono ancora sessioni per stimarla
    'mix': 0.03,            # quota di probabilità distribuita uguale fra i team (prudenza)
    'quota_rookie': 0.02,   # quota del rookie dentro al suo team
    'm_compagni': 4.0,      # sessioni equivalenti: bilancio fra compagni per tipo di sessione
    'l2': 0.5,              # regolarizzazione della logistica fra compagni
    'var_min': 0.02,        # %^2 : errore minimo attribuito a un segnale di passo
    'peso_forma': 2.0,      # la forma pesa il doppio del suo errore: i distacchi di una sessione regrediscono
    'abitudine': {'FP1': 'no', 'FP2': 'pista', 'FP3': 'team'},   # no / team / pista
    'margine': {'FP1': 0.05, 'FP2': 0.10, 'FP3': 0.05},
    'regole_fp1': {'LEC': 0.25},   # FP1: piloti giocabili e probabilità prudente (verifica 16/10/2026)
    'dal_round': 4,
}

NOMI = {'Practice 1': 'FP1', 'Practice 2': 'FP2', 'Practice 3': 'FP3', 'Sprint Qualifying': 'SQ',
        'Qualifying': 'Q'}
LIBERE = ('FP1', 'FP2', 'FP3')
PRIMA = {'FP1': (), 'FP2': ('FP1',), 'FP3': ('FP1', 'FP2')}
CORTI = {'Red Bull Racing': 'Red Bull', 'Haas F1 Team': 'Haas'}
CAMPI_REGISTRO = ['anno', 'round', 'gp', 'sessione', 'momento', 'pilota', 'quota', 'prob_modello',
                  'quota_minima', 'decisione', 'esito']


def corto(team):
    return CORTI.get(team, team)


# ---------------------------------------------------------------------------------------------
# 1. Lettura dello storico
# ---------------------------------------------------------------------------------------------

def _pioggia(sess):
    return any((x.get('rainfall') or 0) > 0 for x in sess.get('weather') or [])


def _migliore(dur):
    if isinstance(dur, list):
        v = [x for x in dur if isinstance(x, (int, float)) and x > 0]
        return min(v) if v else None
    return dur if isinstance(dur, (int, float)) and dur > 0 else None


def _primi(result, n=6):
    rr = [r for r in result or [] if isinstance(r.get('position'), int)]
    return {r.get('driver_number') for r in sorted(rr, key=lambda r: r['position'])[:n]}


def _velocita(energia, piloti):
    """velocità media (km/h) del giro migliore dei piloti dati: distanza integrata dalla telemetria
    diviso il tempo ufficiale del giro; si scartano i giri con lunghezza fuori del 3% dalla mediana"""
    misure = []
    for b in (energia or {}).get('blocchi') or []:
        if b.get('tipo') != 'migliore' or b.get('pilota') not in piloti:
            continue
        t = np.asarray(b.get('t') or [], float) / 100.0
        v = np.asarray(b.get('v') or [], float)
        if len(t) < 50 or len(v) != len(t) or np.isnan(v).any():
            continue
        dt = np.diff(t)
        dist = float(np.sum((v[:-1] + v[1:]) / 2.0 / 3.6 * dt))
        durate = b.get('durate') or []
        giro = durate[0] if durate and durate[0] else float(t[-1] - t[0])
        if giro and giro > 0:
            misure.append((dist, giro))
    if not misure:
        return None
    med = float(np.median([m[0] for m in misure]))
    buone = [d / g * 3.6 for d, g in misure if abs(d / med - 1) < 0.03]
    return float(np.mean(buone)) if buone else None


def leggi_weekend(path):
    d = json.load(gzip.open(path))
    m = d.get('meeting') or {}
    sess = {s['session']['session_name']: s for s in d.get('sessioni') or []}
    w = {'anno': d.get('anno'), 'round': d.get('round'), 'luogo': m.get('location'),
         'gp': m.get('meeting_name'), 'ck': m.get('circuit_key'), 'sprint': 'Sprint Qualifying' in sess,
         'parziale': bool(d.get('parziale')), 'sessioni': {}}
    for nome, k in NOMI.items():
        s = sess.get(nome)
        if not s:
            continue
        piloti = {dr['driver_number']: (dr.get('team_name'), dr.get('name_acronym'))
                  for dr in s.get('drivers') or []}
        tempi = {}
        for r in s.get('session_result') or []:
            t = _migliore(r.get('duration'))
            if t and r.get('driver_number') in piloti:
                tempi[r['driver_number']] = t
        x = {'pioggia': _pioggia(s), 'piloti': piloti, 'tempi': tempi}
        if k == 'Q':
            x['vel'] = _velocita(s.get('energia'), _primi(s.get('session_result')))
        w['sessioni'][k] = x
    return w


def carica(base, anno=2026):
    files = sorted(glob.glob(os.path.join(base, str(anno), '*.json.gz')))
    return sorted([leggi_weekend(f) for f in files], key=lambda w: w['round'] or 0)


def leggi_circuiti(base):
    """velocità di riferimento delle piste dalle qualifiche 2024-2025 (storico/circuiti):
    la 2025 se asciutta, altrimenti la 2024 asciutta"""
    per_ck = {}
    for f in sorted(glob.glob(os.path.join(base, 'circuiti', '*.json.gz'))):
        d = json.load(gzip.open(f))
        nome = os.path.basename(f).split('-', 1)[1].replace('.json.gz', '')
        m = d.get('meeting') or {}
        v = _velocita(d.get('energia'), _primi(d.get('session_result')))
        x = per_ck.setdefault(m.get('circuit_key'), {'nome': nome, 'gp': m.get('meeting_name'),
                                                      'luogo': m.get('location'), 'anni': {}})
        x['anni'][d.get('anno')] = (v, _pioggia(d))
    for x in per_ck.values():
        x['v_rif'] = None
        for anno in sorted(x['anni'], reverse=True):
            v, bagnata = x['anni'][anno]
            if v and not bagnata:
                x['v_rif'] = v
                break
    return per_ck


# ---------------------------------------------------------------------------------------------
# 2. Preparazione dei weekend: titolari, distacchi dei team, primi, tipo di pista
# ---------------------------------------------------------------------------------------------

def titolari(w, prec=None):
    """titolari del weekend: piloti della qualifica o dell'ultima sessione dopo la FP1; con la sola FP1
    quelli del weekend precedente (in FP1 possono esserci rookie al posto di un titolare)"""
    for k in ('Q', 'SQ', 'FP3', 'FP2'):
        if k in w['sessioni'] and w['sessioni'][k]['piloti']:
            return set(w['sessioni'][k]['piloti'])
    if prec:
        return set(prec)
    return set(w['sessioni'].get('FP1', {}).get('piloti') or [])


def distacchi(sx, tit):
    """giro migliore di ogni team, distacco % dalla media dei 3 team migliori, team con un solo titolare"""
    best, n_tit = {}, defaultdict(int)
    for dn, t in sx['tempi'].items():
        team = sx['piloti'][dn][0]
        if not team:
            continue
        best[team] = min(best.get(team, 9e9), t)
        if dn in tit:
            n_tit[team] += 1
    if len(best) < 4:
        return {}, set()
    ref = float(np.mean(sorted(best.values())[:3]))
    gap = {t: min(P['taglio_gap'], (b / ref - 1) * 100.0) for t, b in best.items()}
    return gap, {t for t in best if n_tit[t] < 2}


def prepara(storia):
    prec = None
    for w in storia:
        w['tit'] = titolari(w, prec)
        prec = w['tit']
        w['g'], w['una'], w['primo'] = {}, {}, {}
        for k, sx in w['sessioni'].items():
            w['g'][k], w['una'][k] = distacchi(sx, w['tit'])
            if sx['tempi']:
                dn = min(sx['tempi'], key=sx['tempi'].get)
                w['primo'][k] = (dn, sx['piloti'][dn][1], sx['piloti'][dn][0])
    return storia


class Piste:
    """tipo di pista (lenta / altre) dalla velocità media del giro di qualifica, scala 2026"""

    def __init__(self, storia, circuiti):
        self.circ = circuiti
        mis = {}
        for w in storia:
            q = w['sessioni'].get('Q')
            if q and q.get('vel') and not q['pioggia']:
                mis[w['ck']] = q['vel']
        coppie = [(c['v_rif'], mis[ck]) for ck, c in circuiti.items() if ck in mis and c.get('v_rif')]
        if len(coppie) >= 5:
            x, y = np.array(coppie).T
            self.a, self.b = np.polyfit(x, y, 1)
            self.err = float(np.std(y - (self.a * x + self.b), ddof=2))
        else:
            self.a, self.b, self.err = 1.0, 0.0, None
        self.mis = mis
        # soglia delle lente: primo terzo delle piste 2026 (misurate, o stimate se la qualifica era bagnata)
        tutte = []
        for w in storia:
            v, _ = self.velocita(w['ck'])
            if v:
                tutte.append(v)
        self.soglia = float(np.quantile(tutte, 1 / 3.0)) if len(tutte) >= 6 else None

    def velocita(self, ck):
        """velocità 2026 della pista: misurata se già corsa in 2026 con qualifica asciutta, altrimenti stimata"""
        if ck in self.mis:
            return self.mis[ck], 'misurata nel 2026'
        c = self.circ.get(ck)
        if c and c.get('v_rif'):
            return float(self.a * c['v_rif'] + self.b), 'stimata dal %d' % max(
                a for a, (v, bag) in c['anni'].items() if v and not bag)
        return None, 'sconosciuta'

    def tipo(self, ck):
        v, _ = self.velocita(ck)
        if v is None or self.soglia is None:
            return None
        return 'lenta' if v < self.soglia else 'altre'


# ---------------------------------------------------------------------------------------------
# 3. Stato prima di un weekend: forma, abitudini, errori dei segnali, bilancio fra compagni
# ---------------------------------------------------------------------------------------------

def forma(passato, rnd):
    """distacco medio dei team in qualifica (solo asciutte), pesato verso le più recenti"""
    acc = defaultdict(lambda: [0.0, 0.0])
    for w in passato:
        q = w['g'].get('Q')
        if not q or w['sessioni']['Q']['pioggia']:
            continue
        peso = 0.5 ** ((rnd - w['round']) / P['emivita_forma'])
        for t, g in q.items():
            acc[t][0] += peso * g
            acc[t][1] += peso
    return {t: s / p for t, (s, p) in acc.items() if p > 0}


def abitudini(passato):
    """scarto libera - qualifica per team: FP2 generale e per tipo di pista, FP3 generale"""
    acc = {'FP2': defaultdict(list), 'FP3': defaultdict(list)}
    for w in passato:
        q = w['g'].get('Q')
        if not q or w['sessioni']['Q']['pioggia']:
            continue
        for k in acc:
            g = w['g'].get(k)
            if not g or w['sessioni'][k]['pioggia']:
                continue
            for t, v in g.items():
                if t in q and t not in w['una'][k]:
                    s = max(-P['taglio_scarto'], min(P['taglio_scarto'], v - q[t]))
                    acc[k][t].append((s, w.get('tipo')))
    out = {'FP2': {}, 'FP3': {}}
    for k in acc:
        for t, lst in acc[k].items():
            vals = np.array([s for s, _ in lst])
            tutto = float(vals.mean() * len(vals) / (len(vals) + P['k_abitudine']))
            per = {}
            if k == 'FP2':
                for tp in ('lenta', 'altre'):
                    sel = [s for s, x in lst if x == tp]
                    if sel:
                        per[tp] = tutto + (float(np.mean(sel)) - tutto) * len(sel) / (len(sel) + P['k_pista'])
            out[k][t] = {'tutto': tutto, 'per': per, 'n': len(vals)}
    return out


def abitudine(ab, k, team, tipo):
    modo = P['abitudine'].get(k, 'no')
    if modo == 'no' or k not in ab or team not in ab[k]:
        return 0.0
    a = ab[k][team]
    if modo == 'pista' and tipo in a['per']:
        return a['per'][tipo]
    return a['tutto']


def bilancio_compagni(passato):
    """per pilota e tipo di sessione: volte davanti al compagno e sessioni confrontate"""
    bil = defaultdict(lambda: [0, 0])
    for w in passato:
        for k in ('FP1', 'FP2', 'FP3', 'Q'):
            sx = w['sessioni'].get(k)
            if not sx or sx['pioggia']:
                continue
            per_team = defaultdict(list)
            for dn, t in sx['tempi'].items():
                if dn in w['tit']:
                    per_team[sx['piloti'][dn][0]].append((t, sx['piloti'][dn][1]))
            for lst in per_team.values():
                if len(lst) != 2:
                    continue
                (ta, a), (tb, b) = lst
                for x, tx, ty in ((a, ta, tb), (b, tb, ta)):
                    bil[(x, k)][0] += int(tx < ty)
                    bil[(x, k)][1] += 1
                    bil[(x, 'tutte')][0] += int(tx < ty)
                    bil[(x, 'tutte')][1] += 1
    return bil


def _logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def quota_bilancio(bil, sigla, k):
    v, n = bil.get((sigla, 'tutte'), (0, 0))
    p_all = (v + 2.0) / (n + 4.0)
    vk, nk = bil.get((sigla, k), (0, 0))
    return (vk + P['m_compagni'] * p_all) / (nk + P['m_compagni'])


def _gap_compagni(sx, a, b):
    """distacco % fra due compagni in una sessione asciutta (positivo = a più veloce), tagliato a ±1%"""
    if not sx or sx['pioggia']:
        return 0.0
    ta, tb = sx['tempi'].get(a), sx['tempi'].get(b)
    if not ta or not tb:
        return 0.0
    g = (tb - ta) / ((ta + tb) / 2.0) * 100.0
    return 0.0 if abs(g) > 1.5 else max(-1.0, min(1.0, g))


def caratteristiche_compagni(w, k, a, b, sig, bil):
    """a, b: numeri dei due compagni; sig: numero -> sigla"""
    f1 = (_logit(quota_bilancio(bil, sig[a], k)) - _logit(quota_bilancio(bil, sig[b], k))) / 2.0
    prima = [x for x in PRIMA[k] if x in w['sessioni']]
    f2 = _gap_compagni(w['sessioni'].get(prima[-1]), a, b) if prima else 0.0
    f3 = _gap_compagni(w['sessioni'].get(prima[-2]), a, b) if len(prima) > 1 else 0.0
    return np.array([f1, f2, f3])


def stima_logistica(X, Y):
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


# ---------------------------------------------------------------------------------------------
# 4. Passo previsto e probabilità
# ---------------------------------------------------------------------------------------------

def segnali(w, k, team, S):
    """segnali di passo per il team prima della sessione k: (nome, valore, peso)"""
    out = []
    f = S['forma'].get(team)
    if f is not None:
        out.append(('forma', f, P['peso_forma'] / S['var']['forma']))
    for x in PRIMA[k]:
        sx = w['sessioni'].get(x)
        if not sx or sx['pioggia'] or team not in w['g'].get(x, {}):
            continue
        g = w['g'][x][team]
        if x == 'FP1':
            g -= P['una_macchina'] if team in w['una']['FP1'] else 0.0
        else:
            g -= abitudine(S['ab'], x, team, w.get('tipo'))
        out.append((x.lower(), g, 1.0 / S['var'][x.lower()]))
    return out


def gap_previsti(w, k, squadre, una, S, abit=True):
    """distacco previsto di ogni team nella sessione k (più basso = più veloce) e dettagli"""
    passo, det = {}, {}
    for t in squadre:
        sg = segnali(w, k, t, S) if abit else segnali_ingenui(w, k, t, S)
        if sg:
            passo[t] = sum(v * p for _, v, p in sg) / sum(p for _, _, p in sg)
        det[t] = sg
    peggiore = max(passo.values()) if passo else 1.0
    g = {}
    for t in squadre:
        x = passo.get(t, peggiore + 0.5)
        a = abitudine(S['ab'], k, t, w.get('tipo')) if abit else 0.0
        u = P['una_macchina'] if (k == 'FP1' and t in una) else 0.0
        g[t] = x + a + u
        det[t] = {'segnali': det[t], 'passo': x, 'abitudine': a, 'una': u}
    return g, det


def segnali_ingenui(w, k, team, S):
    """modello di confronto: forma e libere già corse prese così come sono, senza abitudini"""
    out = []
    f = S['forma'].get(team)
    if f is not None:
        out.append(('forma', f, 1.0 / S['var_ing']['forma']))
    for x in PRIMA[k]:
        sx = w['sessioni'].get(x)
        if sx and not sx['pioggia'] and team in w['g'].get(x, {}):
            out.append((x.lower(), w['g'][x][team], 1.0 / S['var_ing'][x.lower()]))
    return out


def prob_team(g, tau):
    t = list(g)
    x = np.array([g[k] for k in t])
    z = np.exp(-(x - x.min()) / tau)
    p = z / z.sum()
    p = (1 - P['mix']) * p + P['mix'] / len(t)
    return dict(zip(t, p))


GRIGLIA_TAU = np.exp(np.linspace(math.log(0.03), math.log(2.0), 90))


def stima_tau(casi):
    """casi: [(gap previsti, team primo)] -> temperatura di massima verosimiglianza"""
    ll = np.zeros(len(GRIGLIA_TAU))
    n = 0
    for g, v in casi:
        if v not in g:
            continue
        t = list(g)
        x = np.array([g[k] for k in t])
        z = np.exp(-(x - x.min())[None, :] / GRIGLIA_TAU[:, None])
        p = z[:, t.index(v)] / z.sum(axis=1)
        ll += np.log((1 - P['mix']) * p + P['mix'] / len(t))
        n += 1
    return float(GRIGLIA_TAU[int(np.argmax(ll))]) if n else None


def temperature(storico_prev, modello):
    """temperatura per tipo di sessione, legata a quella comune con m_tau sessioni equivalenti"""
    casi = [(k, g, v) for m, k, g, v in storico_prev if m == modello]
    t_all = stima_tau([(g, v) for _, g, v in casi]) or P['tau_base']
    out = {}
    for k in LIBERE:
        propri = [(g, v) for kk, g, v in casi if kk == k]
        if not propri:
            out[k] = t_all
            continue
        n = len(propri)
        out[k] = math.exp((n * math.log(stima_tau(propri)) + P['m_tau'] * math.log(t_all)) / (n + P['m_tau']))
    return out


def correzione_fp1(prev_fp1, squadre):
    """rapporto fra FP1 vinte e attese per team (con forza alfa verso 1)"""
    vinte, attese = defaultdict(float), defaultdict(float)
    for p, v in prev_fp1:
        vinte[v] += 1
        for t, x in p.items():
            attese[t] += x
    a = P['alfa_fp1'] / max(1, len(squadre))
    return {t: (vinte[t] + a) / (attese[t] + a) for t in squadre}


# ---------------------------------------------------------------------------------------------
# 5. Il motore: tutti i weekend in ordine, ogni sessione prevista con i soli dati precedenti
# ---------------------------------------------------------------------------------------------

def _var(righe, default):
    if len(righe) < 15:
        return default
    return max(P['var_min'], float(np.mean(np.square(np.clip(righe, -2.0, 2.0)))))


VAR_DEFAULT = {'forma': 0.15, 'fp1': 0.25, 'fp2': 0.15}


def stato_prima(passato, rnd, err, err_ing, storico_prev, X, Y):
    S = {'forma': forma(passato, rnd), 'ab': abitudini(passato), 'bil': bilancio_compagni(passato)}
    S['var'] = {k: _var(err[k], VAR_DEFAULT[k]) for k in VAR_DEFAULT}
    S['var_ing'] = {k: _var(err_ing[k], VAR_DEFAULT[k]) for k in VAR_DEFAULT}
    S['tau'] = temperature(storico_prev, 'abit')
    S['tau_ing'] = temperature(storico_prev, 'ing')
    S['beta'] = stima_logistica(np.array(X), np.array(Y, float)) if len(Y) >= 30 else np.array([1.0, 1.5, 0.5])
    return S


def formazione(w, k):
    """team -> lista di (numero, sigla, titolare) per la sessione k, se è nei dati"""
    sx = w['sessioni'].get(k)
    if not sx:
        return None
    f = defaultdict(list)
    for dn, (team, sig) in sx['piloti'].items():
        if team:
            f[team].append((dn, sig, dn in w['tit']))
    return f


def probabilita(w, k, form, S, abit=True, prev_fp1=None):
    """probabilità di ogni pilota di chiudere primo nella sessione k"""
    squadre = sorted(form)
    una = {t for t, lst in form.items() if sum(1 for _, _, tit in lst if tit) < 2}
    g, det = gap_previsti(w, k, squadre, una, S, abit)
    tau = S['tau'][k] if abit else S['tau_ing'][k]
    pt = prob_team(g, tau)
    pt_base = dict(pt)
    if abit and k == 'FP1' and prev_fp1:
        c = correzione_fp1(prev_fp1, squadre)
        z = sum(pt[t] * c[t] for t in squadre)
        pt = {t: pt[t] * c[t] / z for t in squadre}
    sig = {dn: s for lst in form.values() for dn, s, _ in lst}
    pp = {}
    for t, lst in form.items():
        tit = [(dn, s) for dn, s, x in lst if x]
        roo = [(dn, s) for dn, s, x in lst if not x]
        if len(tit) == 2:
            (a, sa), (b, sb) = tit
            q = 1 / (1 + math.exp(-float(caratteristiche_compagni(w, k, a, b, sig, S['bil']) @ S['beta'])))
            pp[sa], pp[sb] = pt[t] * q, pt[t] * (1 - q)
        elif tit:
            pp[tit[0][1]] = pt[t] * (1 - P['quota_rookie'] * len(roo))
            for _, s in roo:
                pp[s] = pt[t] * P['quota_rookie']
        else:  # nessun titolare in pista (due terzi piloti, come nelle FP1 dei giovani)
            for _, s, _ in lst:
                pp[s] = pt[t] / len(lst)
    return pp, pt, pt_base, g, det, tau


def motore(storia, piste, fino_a=None):
    """percorre i weekend in ordine. Per ogni libera con il risultato calcola le probabilità con lo stato
    dei weekend precedenti e le sessioni già corse nel weekend. Restituisce le previsioni e lo stato
    per il round fino_a (se dato)."""
    err = {k: [] for k in VAR_DEFAULT}
    err_ing = {k: [] for k in VAR_DEFAULT}
    storico_prev, prev_fp1, X, Y, prev = [], [], [], [], []
    for w in storia:
        w['tipo'] = piste.tipo(w['ck'])
    for i, w in enumerate(storia):
        if fino_a is not None and w['round'] >= fino_a:
            break
        passato = storia[:i]
        S = stato_prima(passato, w['round'], err, err_ing, storico_prev, X, Y)
        for k in LIBERE:
            form = formazione(w, k)
            if not form or k not in w['primo']:
                continue
            pp, pt, pt_base, g, det, tau = probabilita(w, k, form, S, True, prev_fp1)
            pp_i, pt_i, _, g_i, _, _ = probabilita(w, k, form, S, False)
            dn, sig, team = w['primo'][k]
            prev.append({'round': w['round'], 'luogo': w['luogo'], 'sessione': k, 'primo': sig, 'team_primo': team,
                         'pp': pp, 'pt': pt, 'pp_ing': pp_i, 'pt_ing': pt_i, 'pioggia': w['sessioni'][k]['pioggia'],
                         'tipo': w.get('tipo'), 'sprint': w['sprint'], 'fuori': [s for lst in form.values()
                                                                                   for _, s, x in lst if not x]})
        # dopo il weekend: aggiorna le cose che servono ai weekend successivi
        for k in LIBERE:
            if k in w['primo'] and formazione(w, k) and not w['sessioni'][k]['pioggia']:
                form = formazione(w, k)
                una = {t for t, lst in form.items() if sum(1 for _, _, x in lst if x) < 2}
                g, _ = gap_previsti(w, k, sorted(form), una, S, True)
                gi, _ = gap_previsti(w, k, sorted(form), una, S, False)
                storico_prev.append(('abit', k, g, w['primo'][k][2]))
                storico_prev.append(('ing', k, gi, w['primo'][k][2]))
                if k == 'FP1':
                    pt = prob_team(g, S['tau']['FP1'])
                    prev_fp1.append((pt, w['primo'][k][2]))
                # righe per la logistica fra compagni
                sig = {dn: s for lst in form.values() for dn, s, _ in lst}
                sx = w['sessioni'][k]
                for t, lst in form.items():
                    tit = [dn for dn, _, x in lst if x]
                    if len(tit) == 2 and all(d in sx['tempi'] for d in tit):
                        a, b = tit
                        X.append(caratteristiche_compagni(w, k, a, b, sig, S['bil']))
                        Y.append(int(sx['tempi'][a] < sx['tempi'][b]))
        q = w['g'].get('Q')
        if q and not w['sessioni']['Q']['pioggia']:
            for t, y in q.items():
                for nome, v, _ in segnali(w, 'FP3', t, S):
                    err[nome].append(y - v)
                for nome, v, _ in segnali_ingenui(w, 'FP3', t, S):
                    err_ing[nome].append(y - v)
    passato = [w for w in storia if fino_a is None or w['round'] < fino_a]
    rnd = fino_a if fino_a is not None else (storia[-1]['round'] + 1 if storia else 1)
    S = stato_prima(passato, rnd, err, err_ing, storico_prev, X, Y)
    return prev, S, prev_fp1


# ---------------------------------------------------------------------------------------------
# 6. Prova a ritroso
# ---------------------------------------------------------------------------------------------

def _ll(p):
    return -math.log(max(p, 1e-4))


def prova(base, anno=2026, dal=None, out=None):
    dal = dal or P['dal_round']
    storia = prepara(carica(base, anno))
    piste = Piste(storia, leggi_circuiti(base))
    prev, S, _ = motore(storia, piste)
    rr = [r for r in prev if r['round'] >= dal]
    if not rr:
        print('Nessuna libera da provare.')
        return []
    print('Prova a ritroso %d, round %d-%d: ogni libera prevista con i soli dati precedenti alla sessione.'
          % (anno, dal, max(r['round'] for r in rr)))
    print('Logloss = -log(probabilità data al primo): più bassa è meglio. Uniforme su 22 piloti = %.2f.\n'
          % math.log(22))
    print('%-5s %3s | %-38s | %-38s' % ('', 'n', 'modello: logloss, favorito preso, p media', 'senza abitudini (ingenuo)'))
    for k in LIBERE + ('tutte',):
        sel = [r for r in rr if k == 'tutte' or r['sessione'] == k]
        if not sel:
            continue
        lm = np.mean([_ll(r['pp'].get(r['primo'], 1e-4)) for r in sel])
        li = np.mean([_ll(r['pp_ing'].get(r['primo'], 1e-4)) for r in sel])
        fm = np.mean([max(r['pp'], key=r['pp'].get) == r['primo'] for r in sel])
        fi = np.mean([max(r['pp_ing'], key=r['pp_ing'].get) == r['primo'] for r in sel])
        pm = np.mean([r['pp'].get(r['primo'], 0) for r in sel])
        pi = np.mean([r['pp_ing'].get(r['primo'], 0) for r in sel])
        meglio = np.mean([_ll(r['pp'].get(r['primo'], 1e-4)) < _ll(r['pp_ing'].get(r['primo'], 1e-4)) for r in sel])
        print('%-5s %3d | %.3f, %3.0f%%, %4.1f%%                      | %.3f, %3.0f%%, %4.1f%%   (modello meglio in %2.0f%%)'
              % (k, len(sel), lm, 100 * fm, 100 * pm, li, 100 * fi, 100 * pi, 100 * meglio))
    print('\nTeam del primo: probabilità media data dal modello')
    for k in LIBERE:
        sel = [r for r in rr if r['sessione'] == k]
        if sel:
            print('  %s: modello %.0f%%, ingenuo %.0f%%, team favorito preso %.0f%%' % (
                k, 100 * np.mean([r['pt'].get(r['team_primo'], 0) for r in sel]),
                100 * np.mean([r['pt_ing'].get(r['team_primo'], 0) for r in sel]),
                100 * np.mean([max(r['pt'], key=r['pt'].get) == r['team_primo'] for r in sel])))
    print('\nTaratura (tutti i piloti di tutte le libere): probabilità data e frequenza vera')
    tutte = [(p, int(s == r['primo'])) for r in rr for s, p in r['pp'].items()]
    for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.4), (0.4, 1.01)):
        x = [(p, y) for p, y in tutte if lo <= p < hi]
        if x:
            print('  %2.0f-%3.0f%%: %4d casi, prevista %4.1f%%, vera %4.1f%%' % (
                100 * lo, min(100, 100 * hi), len(x), 100 * np.mean([p for p, _ in x]), 100 * np.mean([y for _, y in x])))
    print('\nRegola FP1 (dati 2026 fino al round %d):' % max(r['round'] for r in prev))
    for sig, pr in P['regole_fp1'].items():
        f1 = [r for r in prev if r['sessione'] == 'FP1' and sig in r['pp'] and sig not in r['fuori']]
        v = sum(r['primo'] == sig for r in f1)
        print('  %s primo in %d FP1 su %d corse (%.0f%%); stima prudente usata %.0f%%, modello in media %.0f%%' % (
            sig, v, len(f1), 100 * v / max(1, len(f1)), 100 * pr, 100 * np.mean([r['pp'][sig] for r in f1])))
    print('\nAbitudini attuali dei primi team (scarto libera - qualifica, %; negativo = si mostra di più):')
    for t in ('Mercedes', 'Ferrari', 'McLaren', 'Red Bull Racing'):
        a2 = S['ab']['FP2'].get(t, {'tutto': 0.0, 'per': {}})
        a3 = S['ab']['FP3'].get(t, {'tutto': 0.0})
        print('  %-9s FP2 piste lente %+.2f, altre %+.2f   FP3 %+.2f' % (
            corto(t), a2['per'].get('lenta', a2['tutto']), a2['per'].get('altre', a2['tutto']), a3['tutto']))
    print('\nTemperature attuali: ' + ', '.join('%s %.2f' % (k, S['tau'][k]) for k in LIBERE) +
          '. Pesi compagni (bilancio, ultima libera, penultima): ' + ', '.join('%.2f' % b for b in S['beta']))
    print('Errore dei segnali di passo (scarto quadratico medio sulla qualifica, %): ' +
          ', '.join('%s %.2f' % (k, math.sqrt(v)) for k, v in S['var'].items()))
    if out:
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, 'libere_prova_righe.csv')
        with open(path, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['round', 'luogo', 'sessione', 'primo', 'p_primo', 'favorito', 'p_favorito', 'p_primo_ingenuo'])
            for r in rr:
                fav = max(r['pp'], key=r['pp'].get)
                wr.writerow([r['round'], r['luogo'], r['sessione'], r['primo'], '%.3f' % r['pp'].get(r['primo'], 0),
                             fav, '%.3f' % r['pp'][fav], '%.3f' % r['pp_ing'].get(r['primo'], 0)])
        print('\nRighe salvate in', path)
    return rr


# ---------------------------------------------------------------------------------------------
# 7. Scheda della sessione, registro delle quote
# ---------------------------------------------------------------------------------------------

def _quote(s):
    """'RUS=3.00,ANT=3.50' -> {'RUS': 3.0, 'ANT': 3.5}; con il punto e virgola come separatore si può
    scrivere la virgola nei decimali: 'RUS=3,00; ANT=3,50'"""
    out = {}
    s = s or ''
    for parte in (s.split(';') if ';' in s else s.split(',')):
        parte = parte.strip()
        if '=' in parte:
            k, v = parte.split('=', 1)
            out[k.strip().upper()] = float(v.strip().replace(',', '.'))
    return out


def _trova_circuito(circuiti, nome):
    nome = (nome or '').lower()
    for ck, c in circuiti.items():
        if nome and (nome in c['nome'].lower() or nome in (c['luogo'] or '').lower()):
            return ck, c
    return None, None


def leggi_registro(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def scrivi_registro(path, righe):
    chiave = lambda r: (int(r['anno']), int(r['round']), r['sessione'], r['momento'], r['pilota'])
    righe = sorted({chiave(r): r for r in righe}.values(), key=chiave)
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=CAMPI_REGISTRO)
        wr.writeheader()
        for r in righe:
            wr.writerow({k: r.get(k, '') for k in CAMPI_REGISTRO})


def decisioni(k, pp, quote):
    """regole A/B (FP2, FP3) o regola Leclerc (FP1): pilota -> (prob usata, quota minima, gioca da, esito)"""
    m = P['margine'][k]
    out = {}
    if k == 'FP1':
        for s, pr in P['regole_fp1'].items():
            if s in pp:
                qm = 1 / pr
                q = quote.get(s)
                out[s] = (pr, qm, qm * (1 + m), None if q is None else ('GIOCA-FP1' if q >= qm * (1 + m) else ''))
        return out, None
    ordine = sorted(pp, key=pp.get, reverse=True)
    for s in ordine[:3]:
        qm = 1 / pp[s]
        q = quote.get(s)
        out[s] = (pp[s], qm, qm * (1 + m), None if q is None else ('' if q < qm * (1 + m) else 'passa'))
    scelta = None
    fav = ordine[0]
    if out[fav][3] == 'passa':
        scelta = (fav, 'GIOCA-A')
    else:
        cand = [s for s in ordine[1:3] if out[s][3] == 'passa']
        if cand:
            best = max(cand, key=lambda s: pp[s] * quote[s])
            scelta = (best, 'GIOCA-B')
    for s in out:
        p_, qm, qg, e = out[s]
        if e is not None:
            out[s] = (p_, qm, qg, scelta[1] if scelta and scelta[0] == s else e)
    return out, scelta


def scheda(base, rnd, anno=2026, sessione=None, circuito=None, fuori=None, quote=None, registro=None,
           momento='pre'):
    storia = prepara(carica(base, anno))
    circuiti = leggi_circuiti(base)
    piste = Piste(storia, circuiti)
    attuale = [w for w in storia if w['round'] == rnd]
    passato = [w for w in storia if w['round'] < rnd]
    if not passato:
        print('Servono weekend precedenti nello storico.')
        return
    if attuale:
        w = attuale[0]
    else:
        ck, c = _trova_circuito(circuiti, circuito)
        if not ck:
            print('Round %d non nello storico: dare --circuito con il nome del file in storico/circuiti '
                  '(es. austin, mexico, interlagos, las-vegas, lusail, yas).' % rnd)
            return
        w = {'anno': anno, 'round': rnd, 'luogo': c['luogo'], 'gp': c['gp'], 'ck': ck, 'sprint': False,
             'parziale': True, 'sessioni': {}, 'g': {}, 'una': {}, 'primo': {}, 'tit': set(passato[-1]['tit'])}
    if not sessione:
        sessione = next((k for k in LIBERE if k not in w['sessioni']), None)
        if sessione is None:
            print('Tutte le libere del round %d sono già nello storico: indicare --sessione.' % rnd)
            return
    k = sessione.upper()
    if w['sprint'] and k != 'FP1':
        print('Weekend con la Sprint: c\'è solo la FP1.')
        return
    # stato prima del weekend; nel weekend solo le sessioni prima di quella prevista
    prev, S, prev_fp1 = motore(storia, piste, fino_a=rnd)
    w['tipo'] = piste.tipo(w['ck'])
    w = dict(w)
    w['sessioni'] = {x: v for x, v in w['sessioni'].items() if x in PRIMA[k]}
    # formazione della sessione: titolari del weekend (o del precedente), meno chi lascia la FP1 a un rookie
    ultimo = passato[-1]
    if not w.get('tit'):
        w['tit'] = set(ultimo['tit'])
    sorgente = next((x for x in ('FP2', 'FP1') if x in w['sessioni']), None)
    piloti = dict(ultimo['sessioni']['Q']['piloti']) if 'Q' in ultimo['sessioni'] else {}
    if sorgente:
        for dn, v in w['sessioni'][sorgente]['piloti'].items():
            if dn in w['tit'] or sorgente != 'FP1':
                piloti[dn] = v
        w['tit'] = {dn for dn in w['tit'] if dn in piloti} or w['tit']
    fuori = {s.strip().upper() for s in (fuori or '').split(',') if s.strip()}
    form = defaultdict(list)
    for dn, (team, sig) in piloti.items():
        if dn in w['tit'] and team:
            form[team].append((dn, sig, sig not in fuori))
    for t in list(form):
        for dn, sig, x in list(form[t]):
            if not x:
                form[t] = [y for y in form[t] if y[0] != dn] + [(-dn, 'ROOKIE-' + corto(t)[:3].upper(), False)]
    pp, pt, pt_base, g, det, tau = probabilita(w, k, form, S, True, prev_fp1)

    v, fonte = piste.velocita(w['ck'])
    print('PRIMO IN %s — round %d, %s (%s)' % (k, rnd, w['gp'], w['luogo']))
    if v:
        print('Pista %s: %.0f km/h di media sul giro di qualifica (%s); soglia delle lente %.0f km/h.'
              % ('LENTA' if w['tipo'] == 'lenta' else 'non lenta', v, fonte, piste.soglia))
    usate = [x for x in PRIMA[k] if x in w['sessioni']]
    bagn = [x for x in usate if w['sessioni'][x]['pioggia']]
    print('Sessioni del weekend usate: %s.%s Forma: qualifiche asciutte fino al round %d.'
          % (', '.join(x for x in usate if x not in bagn) or 'nessuna',
             (' Bagnate e scartate: %s.' % ', '.join(bagn)) if bagn else '', ultimo['round']))
    mancanti = [x for x in PRIMA[k] if x not in w['sessioni']]
    if mancanti:
        print('Attenzione: mancano %s nello storico. Lanciare Actions → Storico dopo la sessione.' % ', '.join(mancanti))
    if fuori:
        print('Titolari fuori per un rookie: %s.' % ', '.join(sorted(fuori)))
    print('Temperatura %s: %.2f.' % (k, tau))
    print()
    print('Team (distacco previsto dal riferimento; passo + abitudine %s%s):' % (
        k, ' per pista ' + (w['tipo'] or '?') if k == 'FP2' else ''))
    for t in sorted(pt, key=pt.get, reverse=True)[:6]:
        d = det[t]
        sg = ', '.join('%s %+.2f' % (n, x) for n, x, _ in d['segnali'])
        extra = (' %+.2f una auto' % d['una']) if d['una'] else ''
        corr = (' (prima della correzione FP1 %.0f%%)' % (100 * pt_base[t])) if k == 'FP1' else ''
        print('  %-14s %4.1f%%%s  previsto %+.2f%% = passo %+.2f [%s] %+.2f abitudine%s'
              % (corto(t), 100 * pt[t], corr, g[t], d['passo'], sg, d['abitudine'], extra))
    print()
    q = _quote(quote)
    dec, scelta = decisioni(k, pp, q)
    print('Piloti:')
    for s in sorted(pp, key=pp.get, reverse=True)[:8]:
        riga = '  %-12s %4.1f%%  quota minima %5.2f' % (s, 100 * pp[s], 1 / pp[s])
        if s in q:
            riga += '  Sisal %5.2f (valore %+.0f%%)' % (q[s], 100 * (pp[s] * q[s] - 1))
        print(riga)
    print()
    fav = max(pp, key=pp.get)
    if k == 'FP1':
        print('FP1 è una lotteria: si gioca solo con la regola.')
        for s, (pr, qm, qg, e) in dec.items():
            print('  %s: stima prudente %.0f%% (modello %.0f%%), quota minima %.2f, gioca da %.2f%s'
                  % (s, 100 * pr, 100 * pp[s], qm, qg, '' if e is None else ' → ' + (e or 'NON GIOCARE')))
        if not dec:
            print('  Nessun pilota della regola in pista: non si gioca.')
    else:
        etichetta = {'': 'non passa', 'passa': 'passa, ma si gioca un altro'}
        for s, (pr, qm, qg, e) in dec.items():
            print('  %s: %.0f%%, quota minima %.2f, gioca da %.2f (margine %.0f%%)%s'
                  % (s, 100 * pr, qm, qg, 100 * P['margine'][k], '' if e is None else ' → ' + etichetta.get(e, e)))
        if q:
            print('  Decisione: ' + ('%s su %s a %.2f' % (scelta[1], scelta[0], q[scelta[0]]) if scelta else 'NON GIOCARE'))
    print('\n→ favorito %s, probabilità %.0f%%, quota minima %.2f' % (fav, 100 * pp[fav], 1 / pp[fav]))

    if registro and q:
        righe = leggi_registro(registro)
        for s, quota in q.items():
            pr = pp.get(s)
            p_dec = dec[s][0] if s in dec else pr
            dsc = (dec.get(s, (None, None, None, ''))[3] or '') if momento == 'pre' else ''
            dsc = dsc if dsc.startswith('GIOCA') else ''
            righe.append({'anno': anno, 'round': rnd, 'gp': w['gp'], 'sessione': k, 'momento': momento,
                          'pilota': s, 'quota': '%.2f' % quota, 'prob_modello': '' if pr is None else '%.4f' % pr,
                          'quota_minima': '' if not p_dec else '%.2f' % (1 / p_dec), 'decisione': dsc, 'esito': ''})
        scrivi_registro(registro, righe)
        print('Registro aggiornato: %s (%d quote, momento «%s»).' % (registro, len(q), momento))


def esiti(base, registro, anno=2026):
    righe = leggi_registro(registro)
    if not righe:
        print('Registro vuoto o mancante:', registro)
        return
    storia = {w['round']: w for w in prepara(carica(base, anno))}
    n = 0
    for r in righe:
        if r.get('esito') not in (None, ''):
            continue
        w = storia.get(int(r['round']))
        if int(r['anno']) != anno or not w or r['sessione'] not in w['primo']:
            continue
        r['esito'] = '1' if w['primo'][r['sessione']][1] == r['pilota'] else '0'
        n += 1
    scrivi_registro(registro, righe)
    print('Esiti aggiunti: %d.' % n)


def bilancio(registro):
    righe = leggi_registro(registro)
    sess = defaultdict(list)
    for r in righe:
        if r['momento'] == 'pre' and r.get('quota'):
            sess[(r['anno'], r['round'], r['sessione'])].append(r)
    chiuse = {k: v for k, v in sess.items() if any(x.get('esito') == '1' for x in v) or all(x.get('esito') == '0' for x in v)}
    print('Registro %s: %d sessioni con quote, %d con esito.' % (registro, len(sess), len(chiuse)))
    if not chiuse:
        return
    llm, llq = [], []
    for k, v in chiuse.items():
        imp = {x['pilota']: 1 / float(x['quota']) for x in v}
        z = sum(imp.values())  # sopra 1 = margine del bookmaker, tolto in proporzione
        vinc = [x for x in v if x.get('esito') == '1']
        if vinc and vinc[0].get('prob_modello'):
            s = vinc[0]['pilota']
            llm.append(_ll(float(vinc[0]['prob_modello'])))
            llq.append(_ll(imp[s] / max(z, 1.0)))
    if llm:
        print('Primo fra i quotati: logloss modello %.3f, mercato (senza margine) %.3f su %d sessioni.'
              % (np.mean(llm), np.mean(llq), len(llm)))
    giocate = [x for v in chiuse.values() for x in v if x.get('decisione', '').startswith('GIOCA')]
    if giocate:
        utile = sum(float(x['quota']) - 1 if x.get('esito') == '1' else -1 for x in giocate)
        print('Giocate: %d, vinte %d, utile %+.2f puntate (ROI %+.0f%%).' % (
            len(giocate), sum(x.get('esito') == '1' for x in giocate), utile, 100 * utile / len(giocate)))
    chiusura = {(r['anno'], r['round'], r['sessione'], r['pilota']): float(r['quota'])
                for r in righe if r['momento'] == 'chiusura' and r.get('quota')}
    clv = [float(x['quota']) / chiusura[(x['anno'], x['round'], x['sessione'], x['pilota'])] - 1
           for x in giocate if (x['anno'], x['round'], x['sessione'], x['pilota']) in chiusura]
    if clv:
        print('Quota presa contro chiusura (CLV): %+.1f%% in media su %d giocate.' % (100 * np.mean(clv), len(clv)))
    tutte = [(float(x['prob_modello']), int(x['esito'])) for v in chiuse.values() for x in v
             if x.get('prob_modello') and x.get('esito') in ('0', '1')]
    if len(tutte) >= 20:
        print('Taratura del modello sui piloti quotati:')
        for lo, hi in ((0, 0.15), (0.15, 0.3), (0.3, 1.01)):
            x = [(p, y) for p, y in tutte if lo <= p < hi]
            if x:
                print('  %2.0f-%3.0f%%: %d casi, prevista %.0f%%, vera %.0f%%' % (
                    100 * lo, min(100, 100 * hi), len(x), 100 * np.mean([p for p, _ in x]), 100 * np.mean([y for _, y in x])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    a1 = sp.add_parser('prova', help='prova a ritroso sulle libere già corse')
    a1.add_argument('--storico', default='storico')
    a1.add_argument('--anno', type=int, default=2026)
    a1.add_argument('--dal', type=int, default=P['dal_round'])
    a1.add_argument('--out', default=None)
    a2 = sp.add_parser('scheda', help='probabilità per la prossima libera')
    a2.add_argument('--storico', default='storico')
    a2.add_argument('--anno', type=int, default=2026)
    a2.add_argument('--round', type=int, required=True)
    a2.add_argument('--sessione', default=None, help='FP1, FP2 o FP3 (di default la prossima)')
    a2.add_argument('--circuito', default=None, help='prima della FP1: nome del file in storico/circuiti')
    a2.add_argument('--fuori', default=None, help='titolari che lasciano la FP1 a un rookie, es. "LEC,ALB"')
    a2.add_argument('--quote', default=None, help='quote Sisal, es. "RUS=3.00,ANT=3.50,LEC=5.00"')
    a2.add_argument('--registro', default=None, help='CSV del registro delle quote da aggiornare')
    a2.add_argument('--momento', default='pre', choices=['pre', 'chiusura'])
    a3 = sp.add_parser('esiti', help='completa gli esiti del registro dallo storico')
    a3.add_argument('--storico', default='storico')
    a3.add_argument('--anno', type=int, default=2026)
    a3.add_argument('--registro', required=True)
    a4 = sp.add_parser('bilancio', help='modello contro mercato sul registro delle quote')
    a4.add_argument('--registro', required=True)
    a = ap.parse_args()
    if a.cmd == 'prova':
        prova(a.storico, a.anno, a.dal, a.out)
    elif a.cmd == 'scheda':
        scheda(a.storico, a.round, a.anno, a.sessione, a.circuito, a.fuori, a.quote, a.registro, a.momento)
    elif a.cmd == 'esiti':
        esiti(a.storico, a.registro, a.anno)
    else:
        bilancio(a.registro)


if __name__ == '__main__':
    main()
