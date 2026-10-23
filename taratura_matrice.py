#!/usr/bin/env python3
"""
taratura_matrice.py — Muretto: taratura della matrice di giocata (giro secco x long run).

Un solo file, solo numpy (Python 3.9+). Legge l'archivio OpenF1 di Muretto (storico/) e usa
modello_gomme.py (long run delle libere) e modello_gara.py (passo di gara e della Sprint), che devono
stare nella stessa cartella.

La regola (20/10/2026, dopo la taratura e la verifica sul 2023-2026):
  primo sul giro secco con almeno 0,15 s di margine
    -> gara prima della qualifica (regola A) se la quota supera la quota minima prudente:
       1 / (probabilità prudente), cioè il 20° percentile della frequenza di vittoria della classe
       nei weekend già corsi (la prudenza sostituisce il margine del 5%)
    -> qualifica con la stessa logica, separando chi è primo anche nei long run (>= 0,05 s/giro)
  margine sotto 0,15 s -> niente prima della qualifica (tra 0,10 e 0,15 s il primo vince come sotto)
  puntata: 1/4 di Kelly con la probabilità prudente, tetto del 3% del bankroll per pilota,
  qualifica e gara insieme
  penalità di griglia già annunciate: da controllare a mano, il modello non le conosce
  Dopo la qualifica resta il modello di gara con la regola B. Le caselle 2 e 3 della matrice sono
  state tolte; il comando taratura le conta ancora, per documentazione.

Che cosa calcola, per ogni weekend, con i soli dati disponibili prima della qualifica:
  giro secco  distacco previsto in qualifica per pilota: forma delle qualifiche precedenti della
              stagione e libere / Sprint Qualifying del weekend corrette per le abitudini del team
              (come modello_libere.py); pesi = 1/errore fatto nei weekend precedenti
  long run    passo di gara previsto per pilota: forma del passo nelle gare precedenti della stagione
              (peso doppio), long run delle libere (modello_gomme.stima_libere) e, nei weekend Sprint,
              passo della Sprint (modello_gara.passo_gara); pesi = 1/errore fatto nelle gare precedenti
  margine     distacco del secondo dal primo, in secondi al giro
  foglio      ordine dell'ultima sessione asciutta prima della qualifica (quello che legge il mercato)

Uso:
  python taratura_matrice.py taratura --storico storico [--anni 2023,2024,2025,2026] [--out risultati]
      prova a ritroso: conteggi per margine e per casella, curve di probabilità, sensibilità alle
      soglie; scrive risultati/matrice_righe.csv e risultati/matrice_riepilogo.json
  python taratura_matrice.py scheda --storico storico --round 18 [--anno 2026]
                                    [--quota-gara 3.50] [--quota-pole 2.00] [--bankroll 250]
      decisione del weekend in corso (file «parziale» dello Storico, dopo la FP3 o la Sprint) con le
      probabilità prudenti stimate sui weekend già corsi dal 2023 e le quote minime; con le quote
      date calcola valore atteso e puntata
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modello_gara as M  # noqa: E402
import modello_gomme as G  # noqa: E402

P = dict(
    emivita_q=6.0,        # weekend: forma di qualifica, peso 1/2 dopo 6 weekend (modello_libere)
    emivita_passo=4.0,    # gare: forma del passo, peso 1/2 dopo 4 gare (modello_gara)
    shrink_pilota=0.5,    # peso per metà fiducia nello scarto dal compagno (modello_gara usa 2; 0,5 tarato
                          # il 20/10/2026: con 2 la forma di Verstappen veniva tirata verso il compagno)
    k_abitudine=3.0,      # restringimento delle abitudini libera->qualifica verso zero: n/(n+k)
    taglio_scarto=1.5,    # % : scarti libera-qualifica tagliati a ±1,5%
    taglio_gap=3.0,       # % : distacchi tagliati al 3%
    peso_forma_q=1.0,     # giro secco: forma pesata con il suo errore come le libere (taratura 20/10/2026)
    peso_forma_p=2.0,     # long run: la forma pesa il doppio del suo errore (long run delle libere rumorosi)
    var_min=0.02,         # %^2
    var_default={'forma_q': 0.15, 'FP1': 0.25, 'FP2': 0.15, 'FP3': 0.10, 'SQ': 0.05,
                 'forma_p': 0.15, 'lr': 0.30, 'sprint': 0.10},
    min_err=15,           # errori necessari per stimare la varianza di un segnale
    soglia_secco=0.15,    # s: margine sul giro secco (0,10 nella taratura, 0,15 dopo la verifica del 20/10)
    soglia_lr=0.05,       # s/giro: margine nei long run per la conferma (tarato il 20/10/2026)
    dal_anno_prob=2023,   # le probabilità della scheda si stimano su tutti i weekend già corsi dal 2023
    percentile_prudente=20,   # probabilità prudente: 20° percentile della frequenza (Beta, priore uniforme)
    frazione_kelly=0.25,  # quota di Kelly giocata
    tetto_pilota=0.03,    # puntata massima per pilota nel weekend (qualifica e gara insieme)
    margine_b=0.15,       # margine di valore della regola B (dopo la qualifica, modello di gara)
)

ABITUDINE = {'FP1': False, 'FP2': True, 'FP3': True, 'SQ': True}
NOMI = {'Practice 1': 'FP1', 'Practice 2': 'FP2', 'Practice 3': 'FP3', 'Sprint Qualifying': 'SQ',
        'Sprint Shootout': 'SQ', 'Qualifying': 'Q', 'Sprint': 'S', 'Race': 'R'}
DESCR = {1: 'primo in tutti e due', 2: 'primo solo sul giro secco', 3: 'primo solo nei long run'}


# ---------------------------------------------------------------------------------------------
# 1. Lettura di un weekend
# ---------------------------------------------------------------------------------------------

def _t(s):
    return M._t(s)


def _bagnata(sess):
    """pioggia nei dati meteo o più del 30% dei giri su intermedie/bagnato"""
    if any((x.get('rainfall') or 0) > 0 for x in sess.get('weather') or []):
        return True
    return G.bagnata(sess, 0.30)


def _migliore(dur):
    if isinstance(dur, list):
        v = [x for x in dur if isinstance(x, (int, float)) and x > 0]
        return min(v) if v else None
    return dur if isinstance(dur, (int, float)) and dur > 0 else None


def _gap_top3(tempi, team):
    """distacco % di ogni pilota dalla media del giro migliore dei 3 team più veloci"""
    best = {}
    for dn, t in tempi.items():
        tm = team.get(dn)
        if tm:
            best[tm] = min(best.get(tm, 9e9), t)
    if len(best) < 4:
        return {}, None
    ref = float(np.mean(sorted(best.values())[:3]))
    return {dn: min(P['taglio_gap'], (t / ref - 1) * 100.0) for dn, t in tempi.items()}, ref


def _tempi_q(sess):
    """tempi di qualifica con le fasi concatenate (Q1->Q2->Q3), come modello_gara.distacchi_qualifica"""
    t = {}
    for r in sess.get('session_result') or []:
        dur = r.get('duration')
        if isinstance(dur, list):
            t[r['driver_number']] = [x if isinstance(x, (int, float)) and x > 0 else None for x in dur]
        elif isinstance(dur, (int, float)) and dur > 0:
            t[r['driver_number']] = [dur]
    if not t:
        return {}
    nseg = max(len(v) for v in t.values())
    impr = []
    for s in range(nseg - 1):
        dd = [v[s] - v[s + 1] for v in t.values() if len(v) > s + 1 and v[s] and v[s + 1]]
        impr.append(float(np.median(dd)) if dd else 0.0)
    adj = {}
    for dn, v in t.items():
        last = max([i for i, x in enumerate(v) if x] or [-1])
        if last >= 0:
            adj[dn] = v[last] - sum(impr[last:])
    return adj


def leggi_weekend(path):
    """Tutto quello che serve di un weekend: sessioni prima della qualifica, segnali, esiti.
    Un weekend senza qualifica (file «parziale») ha futuro=True e solo le sessioni già corse."""
    d = M.carica_weekend(path)
    if 'sessioni' not in d:
        return None
    m = d.get('meeting') or {}
    ss = sorted(d['sessioni'], key=lambda s: _t(s['session']['date_start']) or 0)
    S = {}
    for s in ss:
        k = NOMI.get(s['session']['session_name'])
        if k:
            S[k] = s
    futuro = 'Q' not in S or not S['Q'].get('session_result')
    t_q = math.inf if futuro else _t(S['Q']['session']['date_start'])
    if not ss:
        return None
    w = dict(anno=d['anno'], round=d['round'], gp=m.get('meeting_name'), luogo=m.get('location'),
             parziale=bool(d.get('parziale')), futuro=futuro, file=path,
             t_ord=t_q if not futuro else _t(ss[-1]['session']['date_start']) + 1e6)
    info = {}
    for s in ss:
        for dr in s.get('drivers') or []:
            info[dr['driver_number']] = (dr.get('name_acronym'), dr.get('team_name'))
    w['piloti'] = info
    team = {dn: v[1] for dn, v in info.items()}
    w['team'] = team
    w['prima'] = [NOMI[s['session']['session_name']] for s in ss
                  if NOMI.get(s['session']['session_name']) and _t(s['session']['date_start']) < t_q - 60]
    w['sprint'] = 'S' in S or 'SQ' in S
    # giro secco: giro migliore per sessione prima della qualifica
    w['libere'] = {}
    for k in ('FP1', 'FP2', 'FP3', 'SQ'):
        if k not in S or k not in w['prima']:
            continue
        s = S[k]
        if k == 'SQ':
            tempi = _tempi_q(s)
        else:
            tempi = {}
            for r in s.get('session_result') or []:
                tt = _migliore(r.get('duration'))
                if tt:
                    tempi[r['driver_number']] = tt
        if not tempi:
            continue
        gap, ref = _gap_top3(tempi, team)
        w['libere'][k] = dict(tempi=tempi, gap=gap, ref=ref, bagnata=_bagnata(s),
                              piloti=set((x['driver_number'] for x in s.get('drivers') or [])))
    # long run delle libere prima della qualifica
    runs = []
    for k in ('FP1', 'FP2', 'FP3'):
        if k in S and k in w['prima']:
            try:
                runs += G.long_run(S[k])
            except Exception:  # noqa: BLE001
                pass
    fit = G.stima_libere(runs) if runs else None
    w['lr'] = None
    if fit:
        med = float(np.median(list(fit['passo'].values())))
        w['lr'] = dict(gap={dn: (p / med - 1) * 100.0 for dn, p in fit['passo'].items()},
                       se={dn: fit['se'][dn] / med * 100.0 for dn in fit['passo']},
                       n=fit['n'], giro=med)
    # Sprint prima della qualifica (2024-2026): passo di gara della Sprint
    w['sprint_passo'] = None
    if 'S' in S and 'S' in w['prima'] and S['S'].get('laps'):
        p, inf = M.passo_gara(S['S'])
        if p:
            w['sprint_passo'] = dict(gap=p, giro=inf.get('giro_med'))
    # esiti
    w.update(q_gap={}, q_ref=None, q_bagnata=None, q_pos={}, pole=None, griglia={}, giocata=False,
             arrivo={}, r_passo={}, r_bagnata=None)
    if not futuro:
        q = S['Q']
        qt = _tempi_q(q)
        if qt:
            w['q_gap'], w['q_ref'] = _gap_top3(qt, team)
        w['q_bagnata'] = _bagnata(q)
        w['q_pos'] = {r['driver_number']: r['position'] for r in q['session_result'] if isinstance(r.get('position'), int)}
        w['pole'] = min(w['q_pos'], key=w['q_pos'].get) if w['q_pos'] else None
        grid = {x['driver_number']: x['position'] for x in (q.get('starting_grid') or []) if isinstance(x.get('position'), int)}
        w['griglia'] = grid if len(grid) >= 0.8 * len(w['q_pos']) else dict(w['q_pos'])
    if 'R' in S and S['R'].get('session_result') and S['R'].get('laps'):
        r = S['R']
        w['giocata'] = True
        w['arrivo'] = {x['driver_number']: x['position'] for x in r['session_result']
                       if isinstance(x.get('position'), int) and not x.get('dsq')}
        p, inf = M.passo_gara(r)
        w['r_passo'] = p
        comps = {st.get('compound') for st in r.get('stints') or []}
        w['r_bagnata'] = bool(comps & {'INTERMEDIATE', 'WET'})
    return w


def carica(storico, anni, extra=None):
    out = []
    files = sorted(glob.glob(os.path.join(storico, '*', '*.json.gz')))
    for f in files:
        try:
            anno = int(os.path.basename(os.path.dirname(f)))
        except ValueError:
            continue
        if anno not in anni:
            continue
        w = leggi_weekend(f)
        if w and (not w['futuro'] or (extra and os.path.abspath(f) == os.path.abspath(extra))):
            out.append(w)
    out.sort(key=lambda w: w['t_ord'])
    return out


# ---------------------------------------------------------------------------------------------
# 2. Forma, abitudini e pesi dai weekend precedenti
# ---------------------------------------------------------------------------------------------

def _forma(valori, emivita):
    """valori: lista (dal più vecchio) di dict pilota->(team, valore). Ritorna (team->forma,
    pilota->scarto dal compagno ristretto verso zero)."""
    tn, td = defaultdict(float), defaultdict(float)
    on, od = defaultdict(float), defaultdict(float)
    for i, val in enumerate(valori[::-1]):
        wt = 0.5 ** (i / emivita)
        bt = defaultdict(list)
        for dn, (tm, v) in val.items():
            bt[tm].append((dn, v))
        for tm, lst in bt.items():
            mu = float(np.mean([v for _, v in lst]))
            tn[tm] += wt * mu
            td[tm] += wt
            if len(lst) == 2:
                for dn, v in lst:
                    on[dn] += wt * (v - mu)
                    od[dn] += wt
    team_f = {tm: tn[tm] / td[tm] for tm in td if td[tm] > 0}
    off = {dn: (on[dn] / od[dn]) * od[dn] / (od[dn] + P['shrink_pilota']) for dn in od if od[dn] > 0}
    return team_f, off


def _forma_pilota(team_f, off, dn, tm):
    if tm not in team_f:
        return None
    return team_f[tm] + off.get(dn, 0.0)


def _var(err, k):
    if len(err.get(k, [])) < P['min_err']:
        return P['var_default'][k]
    e = np.clip(np.array(err[k][-400:]), -2.0, 2.0)
    return max(P['var_min'], float(np.mean(e * e)))


class Stato:
    """forma e abitudini della stagione; errori dei segnali su tutti i weekend precedenti"""

    def __init__(self):
        self.err = defaultdict(list)
        self.stagione = None
        self.q_storia = []       # dict pilota->(team, distacco qualifica) delle qualifiche asciutte
        self.p_storia = []       # dict pilota->(team, passo di gara %) delle gare asciutte
        self.abit = defaultdict(list)   # (sessione, team) -> scarti libera - qualifica
        self.tit = set()

    def nuova_stagione(self, anno):
        if anno != self.stagione:
            self.stagione = anno
            self.q_storia, self.p_storia = [], []
            self.abit = defaultdict(list)

    def abitudine(self, k, tm):
        if not ABITUDINE.get(k):
            return 0.0
        lst = self.abit.get((k, tm)) or []
        if not lst:
            return 0.0
        return float(np.mean(lst)) * len(lst) / (len(lst) + P['k_abitudine'])

    def segnali_secco(self, w, k_max=None):
        """pilota -> lista (nome, valore %, peso), con le sessioni prima della qualifica
        (o prima della sessione k_max, per la lettura del venerdì)"""
        tf, off = _forma(self.q_storia, P['emivita_q'])
        out = defaultdict(list)
        piloti = set(w['q_pos'])
        for x in w['libere'].values():
            piloti |= set(x['gap'])
        for dn in piloti:
            f = _forma_pilota(tf, off, dn, w['team'].get(dn))
            if f is not None:
                out[dn].append(('forma_q', f, P['peso_forma_q'] / _var(self.err, 'forma_q')))
        for k in ('FP1', 'FP2', 'FP3', 'SQ'):
            if k_max and k == k_max:
                break
            x = w['libere'].get(k)
            if not x or x['bagnata']:
                continue
            for dn, g in x['gap'].items():
                v = g - self.abitudine(k, w['team'].get(dn))
                out[dn].append((k, v, 1.0 / _var(self.err, k)))
        return out

    def segnali_lr(self, w, usa_forma=True):
        tf, off = _forma(self.p_storia, P['emivita_passo'])
        forma = {}
        for dn in w['piloti']:
            f = _forma_pilota(tf, off, dn, w['team'].get(dn))
            if f is not None:
                forma[dn] = f
        out = defaultdict(list)
        if usa_forma:
            for dn, f in forma.items():
                out[dn].append(('forma_p', f, P['peso_forma_p'] / _var(self.err, 'forma_p')))
        for nome, x in (('lr', w['lr']), ('sprint', w['sprint_passo'])):
            if not x:
                continue
            g = dict(x['gap'])
            # allineamento della scala alla forma (mediana degli scarti sui piloti in comune)
            com = [g[dn] - forma[dn] for dn in g if dn in forma]
            if len(com) >= 4:
                c = float(np.median(com))
                g = {dn: v - c for dn, v in g.items()}
            for dn, v in g.items():
                extra = (x.get('se') or {}).get(dn, 0.0) ** 2 if nome == 'lr' else 0.0
                out[dn].append((nome, v, 1.0 / (_var(self.err, nome) + 0.5 * extra)))
        return out

    def titolari(self, w):
        """chi corre la qualifica; prima della qualifica: piloti dell'ultima sessione dopo la FP1
        (in FP1 ci possono essere rookie al posto dei titolari), altrimenti quelli del weekend prima"""
        if w['q_pos']:
            return set(w['q_pos'])
        for k in ('SQ', 'FP3', 'FP2'):
            x = w['libere'].get(k)
            if x and x['piloti']:
                return set(x['piloti'])
        return set(self.tit) or set(w['piloti'])

    def aggiorna(self, w, sg_secco, sg_lr):
        if w['q_pos']:
            self.tit = set(w['q_pos'])
        if w['q_gap'] and not w['q_bagnata']:
            for dn, lst in sg_secco.items():
                if dn not in w['q_gap']:
                    continue
                for nome, v, _ in lst:
                    self.err[nome].append(w['q_gap'][dn] - v)
            for k, x in w['libere'].items():
                if x['bagnata'] or not ABITUDINE.get(k):
                    continue
                per_t = defaultdict(list)
                for dn, g in x['gap'].items():
                    if dn in w['q_gap']:
                        per_t[w['team'].get(dn)].append(g - w['q_gap'][dn])
                for tm, lst in per_t.items():
                    s = float(np.mean(lst))
                    self.abit[(k, tm)].append(max(-P['taglio_scarto'], min(P['taglio_scarto'], s)))
            self.q_storia.append({dn: (w['team'].get(dn), g) for dn, g in w['q_gap'].items()})
        if w['r_passo'] and not w['r_bagnata']:
            for dn, lst in sg_lr.items():
                if dn not in w['r_passo']:
                    continue
                for nome, v, _ in lst:
                    self.err[nome].append(w['r_passo'][dn] - v)
            self.p_storia.append({dn: (w['team'].get(dn), g) for dn, g in w['r_passo'].items()})


def combina(sg):
    return {dn: sum(v * p for _, v, p in lst) / sum(p for _, _, p in lst) for dn, lst in sg.items() if lst}


def primo_e_margine(prev, giro, ammessi=None):
    """pilota primo, margine sul secondo in secondi (prev in % del giro), ordine"""
    xs = {dn: v for dn, v in prev.items() if ammessi is None or dn in ammessi}
    if len(xs) < 2 or not giro:
        return None, None, []
    o = sorted(xs, key=xs.get)
    return o[0], (xs[o[1]] - xs[o[0]]) * giro / 100.0, o


def giro_rif(w):
    for k in ('FP3', 'SQ', 'FP2', 'FP1'):
        x = w['libere'].get(k)
        if x and x['ref']:
            return x['ref']
    return w['q_ref']


def ultimo_foglio(w, tit):
    """foglio tempi dell'ultima sessione asciutta prima della qualifica (quello che legge il mercato)"""
    for k in ('SQ', 'FP3', 'FP2', 'FP1'):
        x = w['libere'].get(k)
        if x and not x['bagnata'] and len(x['tempi']) >= 10:
            tt = {dn: t for dn, t in x['tempi'].items() if dn in tit}
            o = sorted(tt, key=tt.get)
            if len(o) >= 2:
                return k, o[0], tt[o[1]] - tt[o[0]], o
    return None, None, None, []


# ---------------------------------------------------------------------------------------------
# 3. Il motore: un weekend alla volta con i soli dati precedenti
# ---------------------------------------------------------------------------------------------

def motore(weekends):
    st = Stato()
    righe = []
    for w in weekends:
        st.nuova_stagione(w['anno'])
        tit = st.titolari(w)
        sg_s = st.segnali_secco(w)
        sg_l = st.segnali_lr(w)
        prev_s, prev_l = combina(sg_s), combina(sg_l)
        prev_lf = combina(st.segnali_lr(w, usa_forma=False))
        giro = giro_rif(w)
        s1, s_m, s_o = primo_e_margine(prev_s, giro, tit)
        l1, l_m, l_o = primo_e_margine(prev_l, giro, tit)
        lf1, lf_m, _ = primo_e_margine(prev_lf, giro, tit)
        v1 = v_m = None
        if 'FP3' in w['libere']:   # lettura del venerdì sera, senza FP3
            v1, v_m, _ = primo_e_margine(combina(st.segnali_secco(w, k_max='FP3')), giro, tit)
        kf, f1, f_m, f_o = ultimo_foglio(w, tit)
        rp = w['r_passo'] or {}
        righe.append(dict(
            anno=w['anno'], round=w['round'], gp=w['gp'], luogo=w['luogo'], sprint=w['sprint'],
            prima='+'.join(w['prima']), giro=giro, futuro=w['futuro'], giocata=w['giocata'],
            q_bagnata=w['q_bagnata'], r_bagnata=w['r_bagnata'], piloti=w['piloti'], team=w['team'],
            secco_primo=s1, secco_margine=s_m, secco_ordine=s_o, secco_prev=prev_s,
            lr_primo=l1, lr_margine=l_m, lr_ordine=l_o, lr_prev=prev_l,
            lrfp_primo=lf1, lrfp_margine=lf_m, venerdi_primo=v1, venerdi_margine=v_m,
            n_lr=len(w['lr']['gap']) if w['lr'] else 0, sprint_lr=bool(w['sprint_passo']),
            foglio=kf, foglio_primo=f1, foglio_margine=f_m, foglio_ordine=f_o,
            pole=w['pole'], q_pos=w['q_pos'], griglia=w['griglia'], arrivo=w['arrivo'],
            passo_primo=min(rp, key=rp.get) if rp else None,
        ))
        if not w['futuro']:
            st.aggiorna(w, sg_s, sg_l)
    return righe, st


# ---------------------------------------------------------------------------------------------
# 4. Caselle, esiti e statistiche
# ---------------------------------------------------------------------------------------------

def sigla(r, dn):
    return (r['piloti'].get(dn) or ('?',))[0] if dn is not None else '-'


def caselle(r, a=None, b=None):
    """casella -> pilota con soglie a (giro secco, s) e b (long run, s/giro)"""
    a = P['soglia_secco'] if a is None else a
    b = P['soglia_lr'] if b is None else b
    s_ok = r['secco_primo'] is not None and r['secco_margine'] >= a
    l_ok = r['lr_primo'] is not None and r['lr_margine'] >= b
    if s_ok and l_ok and r['secco_primo'] == r['lr_primo']:
        return {1: r['secco_primo']}
    out = {}
    if s_ok:
        out[2] = r['secco_primo']
    if l_ok:
        out[3] = r['lr_primo']
    return out


def esiti_pilota(r, dn):
    fin = r['arrivo'].get(dn)
    return dict(pole=int(r['pole'] == dn), vince=int(fin == 1), podio=int(fin is not None and fin <= 3),
                griglia=r['griglia'].get(dn), arrivo=fin)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def frq(k, n):
    if n == 0:
        return '-'
    a, b = wilson(k, n)
    return '%.0f%% (%d su %d; %.0f-%.0f)' % (100.0 * k / n, k, n, 100 * a, 100 * b)


def basi_griglia(righe):
    """frequenze di vittoria e podio per casella di partenza (dalla 11 in giù insieme)"""
    v, p, n = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in righe:
        if not r['giocata']:
            continue
        for dn, g in r['griglia'].items():
            gg = min(g, 11)
            n[gg] += 1
            fin = r['arrivo'].get(dn)
            v[gg] += int(fin == 1)
            p[gg] += int(fin is not None and fin <= 3)
    return {g: (v[g] / n[g], p[g] / n[g], n[g]) for g in n}


def logit(X, y, l2=0.01):
    b = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - p) - l2 * b
        H = -(X.T * (p * (1 - p))) @ X - l2 * np.eye(len(b))
        passo = np.linalg.solve(H, g)
        b = b - passo
        if np.max(np.abs(passo)) < 1e-9:
            break
    return b


def conferma(r):
    """il primo sul giro secco è primo anche nei long run, con almeno soglia_lr"""
    d = r['secco_primo']
    return int(d is not None and r['lr_primo'] == d and r['lr_margine'] is not None and r['lr_margine'] >= P['soglia_lr'])


# vittoria e podio dal solo margine sul giro secco (nel 2024-2026 la conferma dei long run non aggiunge
# niente: coefficiente 0,16, log-loss uguale); pole da margine e conferma (coefficiente 1,34)
CARATTERISTICHE = {'vince': ('margine',), 'podio': ('margine',), 'pole': ('margine', 'conferma')}


def _x(margine, conf, k):
    x = [1.0, min(margine, 0.4)]
    if 'conferma' in CARATTERISTICHE[k]:
        x.append(float(conf))
    return x


def modello_prob(righe):
    """probabilità di pole, vittoria e podio del primo sul giro secco (regressione logistica sui
    weekend già corsi): vittoria e podio dal margine, pole dal margine e dalla conferma dei long run"""
    S = [r for r in righe if r['giocata'] and r['secco_primo'] is not None]
    if len(S) < 20:
        return None
    out = {'n': len(S)}
    for k in ('pole', 'vince', 'podio'):
        X = np.array([_x(r['secco_margine'], conferma(r), k) for r in S])
        y = np.array([esiti_pilota(r, r['secco_primo'])[k] for r in S], float)
        out[k] = logit(X, y)
    return out


def prob(mp, k, margine, conf=0):
    x = np.array(_x(margine, conf, k))
    return float(1 / (1 + np.exp(-x @ mp[k])))


def prudente(k, n, perc=None):
    """frequenza prudente: percentile della Beta(k+1, n-k+1), cioè con priore uniforme.
    Con il 20° percentile la probabilità vera è più bassa solo 1 volta su 5."""
    perc = P['percentile_prudente'] if perc is None else perc
    if n <= 0:
        return None
    smp = np.random.default_rng(0).beta(k + 1, n - k + 1, 200000)
    return float(np.percentile(smp, perc))


def classe(righe, soglia=None):
    """casi della regola nei weekend già corsi: primo sul giro secco con margine >= soglia"""
    soglia = P['soglia_secco'] if soglia is None else soglia
    return [r for r in righe if r['giocata'] and r['secco_primo'] is not None and r['secco_margine'] >= soglia]


def kelly(p, q):
    """frazione di Kelly per una quota decimale q con probabilità p"""
    return (p * q - 1) / (q - 1) if q > 1 else -1.0


# ---------------------------------------------------------------------------------------------
# 5. Taratura
# ---------------------------------------------------------------------------------------------

BINS = [(0, .05), (.05, .10), (.10, .15), (.15, .20), (.20, .30), (.30, 9.0)]
PERIODI = (('2023-2026', (2023, 2024, 2025, 2026)), ('2024-2026', (2024, 2025, 2026)))


def _per_margine(R, primo, marg, esito):
    out = []
    for lo, hi in BINS:
        sel = [r for r in R if r[primo] is not None and r[marg] is not None and lo <= r[marg] < hi]
        out.append((lo, hi, sum(esito(r, r[primo]) for r in sel), len(sel)))
    return out


def _curva(R, primo, marg, esito):
    S = [r for r in R if r[primo] is not None and r[marg] is not None]
    if len(S) < 15:
        return None
    X = np.array([[1.0, min(r[marg], 0.5)] for r in S])
    y = np.array([float(esito(r, r[primo])) for r in S])
    b = logit(X, y, l2=1e-6)
    p = 1 / (1 + np.exp(-X @ b))
    H = (X.T * (p * (1 - p))) @ X
    se = float(np.sqrt(np.linalg.inv(H)[1, 1]))
    return dict(n=len(S), pendenza=float(b[1]), se=se,
                p={m: float(1 / (1 + math.exp(-(b[0] + m * b[1])))) for m in (0.0, 0.1, 0.2, 0.3)})


def _caselle_esiti(R, base, a, b):
    c = defaultdict(list)
    for r in R:
        for k, dn in caselle(r, a, b).items():
            c[k].append((r, dn, esiti_pilota(r, dn)))
    out = {}
    for k in (1, 2, 3):
        x = c[k]
        e = [z for _, _, z in x]
        da_pole = [z for z in e if z['griglia'] == 1]
        fuori = [z for z in e if (z['griglia'] or 99) >= 3]
        out[k] = dict(
            n=len(x), pole=sum(z['pole'] for z in e), vince=sum(z['vince'] for z in e),
            podio=sum(z['podio'] for z in e),
            att_vince=sum(base.get(min(z['griglia'] or 11, 11), (0, 0, 0))[0] for z in e),
            att_podio=sum(base.get(min(z['griglia'] or 11, 11), (0, 0, 0))[1] for z in e),
            da_pole_n=len(da_pole), da_pole_vince=sum(z['vince'] for z in da_pole),
            fuori_n=len(fuori), fuori_podio=sum(z['podio'] for z in fuori), fuori_vince=sum(z['vince'] for z in fuori),
            fuori_att_podio=sum(base.get(min(z['griglia'], 11), (0, 0, 0))[1] for z in fuori),
            fuori_att_vince=sum(base.get(min(z['griglia'], 11), (0, 0, 0))[0] for z in fuori),
            casi=['%d-%02d %s g%s a%s' % (r['anno'] % 100, r['round'], sigla(r, dn), z['griglia'], z['arrivo'])
                  for r, dn, z in x])
    return out


def rapporto(righe):
    R = [r for r in righe if r['giocata']]
    pole = lambda r, d: int(r['pole'] == d)  # noqa: E731
    vince = lambda r, d: int(r['arrivo'].get(d) == 1)  # noqa: E731
    out = {'weekend': len(R), 'per_stagione': {}}
    for a in sorted({r['anno'] for r in R}):
        out['per_stagione'][a] = sum(1 for r in R if r['anno'] == a)
    print('Weekend con gara: %d %s' % (len(R), out['per_stagione']))
    a0, b0 = P['soglia_secco'], P['soglia_lr']
    for nome, anni in PERIODI:
        S = [r for r in R if r['anno'] in anni]
        base = basi_griglia(S)
        o = out[nome] = {}
        print('\n' + '=' * 100 + '\n%s: %d weekend' % (nome, len(S)))
        print('Griglia (vittoria, podio): ' + ', '.join('%d: %.0f%%, %.0f%%' % (g, 100 * v, 100 * p)
                                                      for g, (v, p, n) in sorted(base.items()) if g <= 6))
        o['griglia'] = {g: (v, p, n) for g, (v, p, n) in base.items()}
        # --- giro secco
        print('\nGIRO SECCO: il primo previsto fa la pole?')
        hs = [r for r in S if r['secco_primo']]
        hf = [r for r in S if r['foglio_primo']]
        o['secco_pole'] = (sum(pole(r, r['secco_primo']) for r in hs), len(hs))
        o['foglio_pole'] = (sum(pole(r, r['foglio_primo']) for r in hf), len(hf))
        print('  modello %s | foglio tempi %s' % (frq(*o['secco_pole']), frq(*o['foglio_pole'])))
        disc = [r for r in S if r['foglio_primo'] and r['secco_primo'] and r['foglio_primo'] != r['secco_primo']]
        conc = [r for r in S if r['foglio_primo'] and r['foglio_primo'] == r['secco_primo']]
        o['discordi'] = (len(disc), sum(pole(r, r['secco_primo']) for r in disc), sum(pole(r, r['foglio_primo']) for r in disc))
        o['concordi_foglio'] = (sum(pole(r, r['secco_primo']) for r in conc), len(conc))
        print('  modello e foglio d\'accordo: pole %s; in disaccordo (%d): pole al primo del modello %d, '
              'al primo del foglio %d' % (frq(*o['concordi_foglio']), *o['discordi']))
        ven = [r for r in S if r['venerdi_primo']]
        o['venerdi_pole'] = (sum(pole(r, r['venerdi_primo']) for r in ven), len(ven))
        sab = [r for r in ven if r['secco_primo']]
        o['sabato_pole'] = (sum(pole(r, r['secco_primo']) for r in sab), len(sab))
        print('  weekend con FP3: venerdì sera (senza FP3) %s, sabato (con FP3) %s'
              % (frq(*o['venerdi_pole']), frq(*o['sabato_pole'])))
        o['secco_bins'] = _per_margine(S, 'secco_primo', 'secco_margine', pole)
        o['foglio_bins'] = _per_margine(S, 'foglio_primo', 'foglio_margine', pole)
        print('  per margine            modello                     foglio tempi')
        for (lo, hi, k, n), (_, _, k2, n2) in zip(o['secco_bins'], o['foglio_bins']):
            print('   %.2f-%s s  %-28s %s' % (lo, '%.2f' % hi if hi < 9 else 'oltre', frq(k, n), frq(k2, n2)))
        for nm, pr, mg in (('modello', 'secco_primo', 'secco_margine'), ('foglio', 'foglio_primo', 'foglio_margine')):
            c = o['curva_' + nm] = _curva(S, pr, mg, pole)
            if c:
                print('  curva %s: pendenza %+.1f ± %.1f per secondo; pole a 0 s %.0f%%, 0,10 s %.0f%%, 0,20 s %.0f%%, '
                      '0,30 s %.0f%%' % (nm, c['pendenza'], c['se'], *[100 * c['p'][m] for m in (0.0, 0.1, 0.2, 0.3)]))
        # --- long run
        print('\nLONG RUN: il primo previsto vince?')
        hl = [r for r in S if r['lr_primo']]
        hlf = [r for r in S if r['lrfp_primo']]
        o['lr_vince'] = (sum(vince(r, r['lr_primo']) for r in hl), len(hl))
        o['lrfp_vince'] = (sum(vince(r, r['lrfp_primo']) for r in hlf), len(hlf))
        pp = [r for r in S if r['passo_primo'] and r['lr_primo']]
        ppf = [r for r in S if r['passo_primo'] and r['lrfp_primo']]
        o['lr_passo'] = (sum(r['lr_primo'] == r['passo_primo'] for r in pp), len(pp))
        o['lrfp_passo'] = (sum(r['lrfp_primo'] == r['passo_primo'] for r in ppf), len(ppf))
        o['pole_vince'] = (sum(vince(r, r['pole']) for r in S if r['pole']), sum(1 for r in S if r['pole']))
        print('  modello (forma + libere + Sprint): vince %s, miglior passo in gara %s'
              % (frq(*o['lr_vince']), frq(*o['lr_passo'])))
        print('  solo libere/Sprint:                vince %s, miglior passo in gara %s'
              % (frq(*o['lrfp_vince']), frq(*o['lrfp_passo'])))
        print('  per confronto, chi fa la pole vince %s' % frq(*o['pole_vince']))
        o['lr_bins'] = _per_margine(S, 'lr_primo', 'lr_margine', vince)
        o['lrfp_bins'] = _per_margine(S, 'lrfp_primo', 'lrfp_margine', vince)
        print('  per margine            modello                     solo libere/Sprint')
        for (lo, hi, k, n), (_, _, k2, n2) in zip(o['lr_bins'], o['lrfp_bins']):
            print('   %.2f-%s s  %-28s %s' % (lo, '%.2f' % hi if hi < 9 else 'oltre', frq(k, n), frq(k2, n2)))
        c = o['curva_lr'] = _curva(S, 'lr_primo', 'lr_margine', vince)
        if c:
            print('  curva modello: pendenza %+.1f ± %.1f per secondo; vittoria a 0 s %.0f%%, 0,10 s %.0f%%, '
                  '0,20 s %.0f%%, 0,30 s %.0f%%' % (c['pendenza'], c['se'], *[100 * c['p'][m] for m in (0.0, 0.1, 0.2, 0.3)]))
        c = o['curva_secco_vince'] = _curva(S, 'secco_primo', 'secco_margine', vince)
        if c:
            print('  il primo sul GIRO SECCO vince: a 0 s %.0f%%, 0,10 s %.0f%%, 0,20 s %.0f%%, 0,30 s %.0f%% '
                  '(pendenza %+.1f ± %.1f)' % (*[100 * c['p'][m] for m in (0.0, 0.1, 0.2, 0.3)], c['pendenza'], c['se']))
        # --- matrice con le soglie tarate
        ce = o['caselle'] = _caselle_esiti(S, base, a0, b0)
        print('\nMATRICE con soglie %.2f s (giro secco) e %.2f s/giro (long run)' % (a0, b0))
        for k in (1, 2, 3):
            x = ce[k]
            print('  casella %d, %s: %d casi' % (k, DESCR[k], x['n']))
            if not x['n']:
                continue
            print('    pole %s | vince %s (dalla griglia %.1f) | podio %s (dalla griglia %.1f)'
                  % (frq(x['pole'], x['n']), frq(x['vince'], x['n']), x['att_vince'], frq(x['podio'], x['n']), x['att_podio']))
            print('    partiti in pole: vincono %s (media della pole %.0f%%)'
                  % (frq(x['da_pole_vince'], x['da_pole_n']), 100 * base.get(1, (0, 0, 0))[0]))
            print('    partiti dalla terza casella in giù: podio %s (dalla griglia %.1f), vittoria %s (dalla griglia %.1f)'
                  % (frq(x['fuori_podio'], x['fuori_n']), x['fuori_att_podio'], frq(x['fuori_vince'], x['fuori_n']),
                     x['fuori_att_vince']))
            print('    ' + ', '.join(x['casi']))
        # casella 1 con anche il primo del foglio
        c1f = [(r, d) for r in S for k, d in caselle(r).items() if k == 1 and r['foglio_primo'] == d]
        o['c1_foglio'] = dict(n=len(c1f), pole=sum(pole(r, d) for r, d in c1f), vince=sum(vince(r, d) for r, d in c1f))
        print('  casella 1 e primo anche nel foglio tempi: pole %s, vince %s'
              % (frq(o['c1_foglio']['pole'], o['c1_foglio']['n']), frq(o['c1_foglio']['vince'], o['c1_foglio']['n'])))
        # --- sensibilità
        print('\nSENSIBILITÀ ALLE SOGLIE (casella 1: casi, pole, vittorie | casella 2: casi, pole | '
              'casella 3 dalla terza casella in giù: casi, podi/attesi)')
        o['sensibilita'] = []
        for a in (0.05, 0.10, 0.15, 0.20):
            riga = '  giro secco %.2f:' % a
            for b in (0.0, 0.05, 0.10, 0.15):
                x = _caselle_esiti(S, base, a, b)
                o['sensibilita'].append((a, b, {k: {kk: v for kk, v in x[k].items() if kk != 'casi'} for k in x}))
                riga += ' | lr %.2f: %2d %2d %2d · %2d %2d · %2d %2d/%.1f' % (
                    b, x[1]['n'], x[1]['pole'], x[1]['vince'], x[2]['n'], x[2]['pole'],
                    x[3]['fuori_n'], x[3]['fuori_podio'], x[3]['fuori_att_podio'])
            print(riga)
        # --- regola unica: primo sul giro secco con margine
        print('\nREGOLA UNICA: primo sul giro secco con almeno %.2f s' % a0)
        sopra = [r for r in S if r['secco_primo'] is not None and r['secco_margine'] >= a0]
        sotto = [r for r in S if r['secco_primo'] is not None and r['secco_margine'] < a0]
        o['regola'] = {}
        for nome, sel in (('sopra la soglia', sopra), ('  con la conferma dei long run', [r for r in sopra if conferma(r)]),
                          ('  senza la conferma', [r for r in sopra if not conferma(r)]), ('sotto la soglia', sotto)):
            e = [esiti_pilota(r, r['secco_primo']) for r in sel]
            n = len(e)
            o['regola'][nome.strip()] = dict(n=n, pole=sum(z['pole'] for z in e), vince=sum(z['vince'] for z in e),
                                             podio=sum(z['podio'] for z in e))
            if n:
                kv, kq = sum(z['vince'] for z in e), sum(z['pole'] for z in e)
                o['regola'][nome.strip()].update(prudente_vince=prudente(kv, n), prudente_pole=prudente(kq, n))
                print('  %-32s %2d casi: pole %s | vince %s | podio %s'
                      % (nome, n, frq(kq, n), frq(kv, n), frq(sum(z['podio'] for z in e), n)))
                print('  %-32s          prudenti (%d° percentile): vittoria %.0f%% (quota minima %.2f), '
                      'pole %.0f%% (quota minima %.2f)'
                      % ('', P['percentile_prudente'], 100 * prudente(kv, n), 1 / prudente(kv, n),
                         100 * prudente(kq, n), 1 / prudente(kq, n)))
        # --- curva logistica (documentazione: nella verifica non aggiunge niente a un tasso unico)
        mp = modello_prob(S)
        o['prob'] = {}
        if mp:
            print('\nCURVA LOGISTICA DEL PRIMO SUL GIRO SECCO (%d weekend; solo documentazione, la scheda usa le '
                  'probabilità prudenti della regola): vittoria e podio dal margine, pole dal margine e dalla '
                  'conferma dei long run' % mp['n'])
            for m in (0.10, 0.15, 0.20, 0.30):
                v = {'vince': prob(mp, 'vince', m), 'podio': prob(mp, 'podio', m),
                     'pole_conferma': prob(mp, 'pole', m, 1), 'pole_senza': prob(mp, 'pole', m, 0)}
                o['prob'][m] = v
                print('  margine %.2f s: vittoria %.0f%%, podio %.0f%% | pole %.0f%% con la conferma, %.0f%% senza'
                      % (m, 100 * v['vince'], 100 * v['podio'], 100 * v['pole_conferma'], 100 * v['pole_senza']))
    return out


def cmd_taratura(a):
    anni = [int(x) for x in a.anni.split(',')]
    righe, _ = motore(carica(a.storico, anni))
    os.makedirs(a.out, exist_ok=True)
    scrivi_csv(os.path.join(a.out, 'matrice_righe.csv'), [r for r in righe if r['giocata']])
    out = rapporto(righe)

    def _json(x):
        if isinstance(x, dict):
            return {str(k): _json(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_json(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x
    with open(os.path.join(a.out, 'matrice_riepilogo.json'), 'w') as f:
        json.dump(_json(out), f, ensure_ascii=False, indent=1)
    print('\nScritti %s e %s' % (os.path.join(a.out, 'matrice_righe.csv'), os.path.join(a.out, 'matrice_riepilogo.json')))


def scrivi_csv(path, righe):
    campi = ['anno', 'round', 'gp', 'sprint', 'prima', 'q_bagnata', 'r_bagnata', 'casella', 'pilota_casella',
             'secco_primo', 'secco_margine', 'venerdi_primo', 'venerdi_margine', 'foglio', 'foglio_primo',
             'foglio_margine', 'lr_primo', 'lr_margine', 'lrfp_primo', 'lrfp_margine', 'pole', 'vincitore',
             'passo_primo', 'griglia_secco_primo', 'arrivo_secco_primo', 'griglia_lr_primo', 'arrivo_lr_primo']
    f3 = lambda x: '' if x is None else '%.3f' % x  # noqa: E731
    with open(path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(campi)
        for r in righe:
            cs = caselle(r)
            vinc = min(r['arrivo'], key=r['arrivo'].get) if r['arrivo'] else None
            wr.writerow([r['anno'], r['round'], r['gp'], int(r['sprint']), r['prima'], int(bool(r['q_bagnata'])),
                         int(bool(r['r_bagnata'])), '+'.join(str(k) for k in sorted(cs)) or '4',
                         '+'.join(sigla(r, cs[k]) for k in sorted(cs)),
                         sigla(r, r['secco_primo']), f3(r['secco_margine']), sigla(r, r['venerdi_primo']),
                         f3(r['venerdi_margine']), r['foglio'] or '', sigla(r, r['foglio_primo']), f3(r['foglio_margine']),
                         sigla(r, r['lr_primo']), f3(r['lr_margine']), sigla(r, r['lrfp_primo']), f3(r['lrfp_margine']),
                         sigla(r, r['pole']), sigla(r, vinc), sigla(r, r['passo_primo']),
                         r['griglia'].get(r['secco_primo']), r['arrivo'].get(r['secco_primo']),
                         r['griglia'].get(r['lr_primo']), r['arrivo'].get(r['lr_primo'])])


# ---------------------------------------------------------------------------------------------
# 6. Scheda del weekend in corso
# ---------------------------------------------------------------------------------------------

def _puntate(scelte, bankroll=None):
    """scelte: lista (nome, quota, probabilità prudente). Stampa valore atteso e puntata a 1/4 di Kelly,
    con il tetto per pilota diviso in proporzione fra le giocate sullo stesso pilota."""
    frazioni = {}
    for nome, q, p in scelte:
        if q is None or p is None:
            continue
        ev = p * q - 1
        f = P['frazione_kelly'] * kelly(p, q)
        if ev <= 0 or f <= 0:
            print('   %s a %.2f: sotto la quota minima prudente (%.2f): niente' % (nome, q, 1 / p))
        else:
            frazioni[nome] = (q, p, ev, f)
    tot = sum(f for _, _, _, f in frazioni.values())
    scala = min(1.0, P['tetto_pilota'] / tot) if tot > 0 else 1.0
    for nome, (q, p, ev, f) in frazioni.items():
        fs = f * scala
        euro = ' = %.2f €' % (fs * bankroll) if bankroll else ''
        print('   %s a %.2f: valore atteso con la probabilità prudente %+.0f%%; puntata %.1f%% del bankroll%s'
              % (nome, q, 100 * ev, 100 * fs, euro))
    if scala < 1.0:
        print('   (tetto del %.0f%% per pilota: puntate ridotte in proporzione)' % (100 * P['tetto_pilota']))


def cmd_scheda(a):
    cand = sorted(glob.glob(os.path.join(a.storico, str(a.anno), '%02d-*.json.gz' % a.round)))
    if not cand:
        sys.exit('Nessun file per il round %d del %d in %s: lanciare lo Storico (file «parziale»).'
                 % (a.round, a.anno, a.storico))
    f = cand[0]
    anni = list(range(2023, a.anno + 1))
    righe, _ = motore(carica(a.storico, anni, extra=f))
    r = [x for x in righe if x['anno'] == a.anno and x['round'] == a.round]
    if not r:
        sys.exit('Il file %s non ha sessioni utilizzabili.' % f)
    r = r[0]
    passate = [x for x in righe if x['giocata'] and x['anno'] >= P['dal_anno_prob']
               and (x['anno'], x['round']) < (a.anno, a.round)]
    giro = r['giro'] or 90.0
    print('%s %d (round %d) — sessioni lette: %s%s' % (r['gp'], r['anno'], r['round'], r['prima'] or 'nessuna',
                                                      '' if r['futuro'] else ' (qualifica già corsa: lettura a ritroso)'))
    print('\nGiro secco previsto (distacco dal primo, s):')
    pr = r['secco_prev']
    print('  ' + '  '.join('%d %s %+.3f' % (i + 1, sigla(r, d), (pr[d] - pr[r['secco_ordine'][0]]) * giro / 100)
                           for i, d in enumerate(r['secco_ordine'][:6])))
    print('Long run previsto (distacco dal primo, s/giro):')
    pl = r['lr_prev']
    print('  ' + '  '.join('%d %s %+.3f' % (i + 1, sigla(r, d), (pl[d] - pl[r['lr_ordine'][0]]) * giro / 100)
                           for i, d in enumerate(r['lr_ordine'][:6])))
    if r['foglio']:
        print('Foglio tempi %s: %s' % (r['foglio'], '  '.join(
            '%d %s' % (i + 1, sigla(r, d)) for i, d in enumerate(r['foglio_ordine'][:4]))))
    if 'FP3' not in r['prima'] and 'SQ' not in r['prima']:
        print('\nAttenzione: senza FP3 il giro secco è meno affidabile (nella prova il primo del venerdì sera '
              'fa la pole 4 volte su 10, dopo la FP3 quasi 5 su 10).')
    print()
    d = r['secco_primo']
    if d is None:
        print('Giro secco non stimabile con le sessioni lette: niente prima della qualifica.')
        return
    m = r['secco_margine']
    conf = conferma(r)
    nota_lr = ('primo anche nei long run (+%.3f s/giro)' % r['lr_margine'] if conf
               else 'nei long run primo %s' % sigla(r, r['lr_primo']))
    print('Primo sul giro secco: %s, margine %+.3f s; %s' % (sigla(r, d), m, nota_lr))
    if m < P['soglia_secco']:
        sotto = [x for x in passate if x['secco_primo'] is not None and x['secco_margine'] < P['soglia_secco']]
        print('-> margine sotto %.2f s: niente prima della qualifica (nei weekend già corsi il primo con meno di '
              '%.2f s ha vinto %d volte su %d).' % (P['soglia_secco'], P['soglia_secco'],
                                                  sum(esiti_pilota(x, x['secco_primo'])['vince'] for x in sotto), len(sotto)))
        print('   Dopo la qualifica: modello di gara e regola B.')
        return
    cl = classe(passate)
    if not cl:
        print('-> nessun caso della regola nei weekend già corsi: probabilità non stimabili.')
        return
    ev = [esiti_pilota(x, x['secco_primo']) for x in cl]
    kv, n = sum(e['vince'] for e in ev), len(ev)
    pv = prudente(kv, n)
    print('-> gara prima della qualifica (regola A): nei weekend già corsi dal %d il primo con almeno %.2f s '
          'ha vinto %d volte su %d (%.0f%%)' % (P['dal_anno_prob'], P['soglia_secco'], kv, n, 100.0 * kv / n))
    print('   probabilità prudente %.0f%% -> quota minima %.2f' % (100 * pv, 1 / pv))
    print('   prima di giocare: penalità di griglia già annunciate? Il modello non le conosce '
          '(a Singapore 2026 Russell era primo sul giro secco ma partiva in fondo).')
    cq = [x for x in cl if conferma(x) == conf]
    eq = [esiti_pilota(x, x['secco_primo']) for x in cq]
    kq, nq = sum(e['pole'] for e in eq), len(eq)
    pq = prudente(kq, nq)
    if pq is None:
        print('-> qualifica: nessun caso simile nei weekend già corsi: niente.')
    else:
        print('-> qualifica (%s): pole %d volte su %d (%.0f%%); probabilità prudente %.0f%% -> quota minima %.2f'
              % ('primo anche nei long run' if conf else 'senza la conferma dei long run', kq, nq, 100.0 * kq / nq,
                 100 * pq, 1 / pq))
    if r['foglio_primo'] == d:
        cf = [esiti_pilota(x, x['secco_primo']) for x in cl if x['foglio_primo'] == x['secco_primo']]
        print('   primo anche nel foglio tempi %s: nei casi della regola con lo stesso quadro pole %d su %d, '
              'vittorie %d su %d' % (r['foglio'], sum(e['pole'] for e in cf), len(cf), sum(e['vince'] for e in cf), len(cf)))
    if a.quota_gara or a.quota_pole:
        print('Puntate (1/4 di Kelly sulla probabilità prudente, tetto %.0f%% per pilota):' % (100 * P['tetto_pilota']))
        _puntate([('gara', a.quota_gara, pv), ('pole', a.quota_pole, pq)], a.bankroll)
    else:
        print('   Con --quota-gara e --quota-pole la scheda calcola valore atteso e puntata.')
    print('Dopo la qualifica: modello di gara e regola B, come prima.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    t = sp.add_parser('taratura')
    t.add_argument('--storico', default='storico')
    t.add_argument('--anni', default='2023,2024,2025,2026')
    t.add_argument('--out', default='risultati')
    s = sp.add_parser('scheda')
    s.add_argument('--storico', default='storico')
    s.add_argument('--anno', type=int, default=2026)
    s.add_argument('--round', type=int, required=True)
    s.add_argument('--quota-gara', type=float, default=None, help='quota del vincente gara del primo (es. 3.50)')
    s.add_argument('--quota-pole', type=float, default=None, help='quota della pole del primo (es. 2.00)')
    s.add_argument('--bankroll', type=float, default=None, help='bankroll in euro, per la puntata in euro')
    a = ap.parse_args()
    if a.cmd == 'taratura':
        cmd_taratura(a)
    else:
        cmd_scheda(a)


if __name__ == '__main__':
    main()
