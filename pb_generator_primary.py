# Single-file Powerball ticket generator with:
# - History load + windows (90d/180d/365d/3y/ALL)
# - Hot (90d), Overdue (ALL via Median+90th), Correlations (180d)
# - Positional buckets (N1..N5), 0–2 hot, 0–1 overdue, correlation nudges
# - Optional bias smoothing (scope "chosen" or "pool")
# - Exact-gap near-pair engine: quotas for gaps 1..K (K=MAX_NEAR_GAP), per-position weights (N2–N3:2, N3–N4:1, N4–N5:1)
# - Re-enforces near-pair quotas after repairs and again on the final K tickets
# - Optional wheel reduction
# - Output whites are sorted; PB never equals a white
#
# NOTE: Coverage/variety aid only. Lotteries are random; play responsibly.

import os, ssl, datetime, random, itertools
import urllib.request
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import permutations

# ------------------------------- #
# Config
# ------------------------------- #

DATA_URL = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv"

# ---- Pool / Buckets ----
POOL_MIN = 1
POOL_MAX = 69
K_NUMS   = 5
PB_RANGE = range(1, 27)

USE_CONDITIONAL = True   # if True, use gap-based sampling
POS_BUCKETS = {
    "N1": range(1, 17),
    "N2": range(11, 30),
    "N3": range(21, 45),
    "N4": range(35, 53),
    "N5": range(56, 70),
}

ASSOC_THRESHOLD   = 0.50
HOT_PCTL          = 0.80
NUM_CANDIDATE_TICKETS = 15000
WHEEL_TARGET          = 10

# ---- FLAGS ----
USE_WHEEL = True                   # Boil larger pool down to a sensible set
ENABLE_SMOOTHING = True            # Reduce Bias/Smoothing
SMOOTH_SCOPE = "pool"              # What to replace from when smoothing:  "pool" (1-69) or "chosen" (numbers picked)
SMOOTHING_RADIUS = 2               # Max difference +/- in smoothing replacement

PB_WEIGHTS = {"mid": 0.55, "hot": 0.225, "cold": 0.225}
BUCKET_FLEX = 2

# --- NEAR-PAIR CONTROL (Exact-gap quotas) ---
# K controls what "near" means: gap <= K.
# 5-of-69 baseline cumulative probabilities (at least one pair with gap <= g):
#   g<=1: 0.26, g<=2: 0.47, g<=3: 0.627, g<=4: 0.745, g<=5: 0.830
MAX_NEAR_GAP = 5  # set to 2..5

# Pair-position weights (how near pairs are distributed across adjacent positions)
PAIR_WEIGHTS = {("N2","N3"): 1.5, ("N3","N4"): 0.75, ("N4","N5"): 1}

# Per-ticket cap
NEAR_MAX_PER_TICKET = 1

DEBUG_NEAR_MIX = False  # set True for summaries

# ------------------------------- #
# Helpers
# ------------------------------- #

def pos_name(idx): return f"N{idx+1}"

def in_bucket(num, pos, flex=0):
    rng = POS_BUCKETS[pos]
    return (min(rng) - flex) <= num <= (max(rng) + flex)

def reorder_to_buckets(ticket, flex=BUCKET_FLEX):
    positions = ["N1","N2","N3","N4","N5"]
    for perm in permutations(ticket):
        if all(in_bucket(perm[i], positions[i], flex) for i in range(5)):
            return list(perm)
    return None

def ensure_pb_not_in_whites(pb, whites, sampler):
    tries = 0
    whites_set = set(whites)
    while pb in whites_set and tries < 20:
        pb = sampler(); tries += 1
    if pb in whites_set:
        candidates = [x for x in PB_RANGE if x not in whites_set]
        pb = random.choice(candidates) if candidates else pb
    return pb

def largest_remainder_apportion(fracs, total):
    """
    fracs: dict[label] -> fraction in [0,1]; not necessarily summing to 1.
    Returns dict[label] -> int that sums to <= total; leftover goes to the largest remainders.
    """
    # normalize if needed
    s = sum(fracs.values())
    if s > 1.0 + 1e-9:
        fracs = {k: v/s for k,v in fracs.items()}
    quotas = {k: v*total for k,v in fracs.items()}
    floors = {k: int(np.floor(q)) for k,q in quotas.items()}
    used = sum(floors.values())
    rem = total - used
    if rem > 0:
        remainders = sorted(((quotas[k]-floors[k], k) for k in fracs.keys()), reverse=True)
        for _, k in remainders[:rem]:
            floors[k] += 1
    return floors

def near_pair_mix_summary_exact(tickets):
    counts = Counter()
    for t in tickets:
        gcat, _ = ticket_gap_category(t)
        counts[gcat] += 1
    n = max(1, len(tickets))
    out = {f"gap={g}%" if g>0 else "clean%": round(100*counts[g]/n,1) for g in sorted(counts)}
    out["counts"] = dict(counts)
    return out

def odd_even_ok(ticket, min_odds=1, min_evens=1):
    odds = sum(1 for n in ticket if n % 2 == 1)
    evens = len(ticket) - odds
    return odds >= min_odds and evens >= min_evens

# ------------------------------- #
# Data I/O
# ------------------------------- #

def load_history_csv(local_name="history.csv"):
    today = datetime.date.today().strftime("%Y%m%d")
    dated_name = f"history_{today}.csv"
    if os.path.exists(dated_name):
        fname = dated_name
    elif os.path.exists(local_name):
        fname = local_name
    else:
        try:
            ssl_context = ssl._create_unverified_context()
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(DATA_URL, dated_name)
            fname = dated_name
            print(f"Downloaded Powerball history → {dated_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")

    df = pd.read_csv(fname)
    if "Draw Date" in df.columns:
        df["DrawDate"] = pd.to_datetime(df["Draw Date"])
    elif "DrawDate" in df.columns:
        df["DrawDate"] = pd.to_datetime(df["DrawDate"])
    else:
        raise RuntimeError("Unable to find Draw Date column.")

    whites, powerball = None, None
    if "Winning Numbers" in df.columns:
        s = df["Winning Numbers"].astype(str)
        parts = s.str.split(" ", expand=True)
        if parts.shape[1] < 5:
            parts = s.str.split(",", expand=True)
        parts = parts.apply(pd.to_numeric, errors="coerce")
        if parts.shape[1] >= 6:
            whites = parts.iloc[:, :5]
            powerball = parts.iloc[:, 5]
        elif parts.shape[1] == 5 and "Powerball" in df.columns:
            whites = parts
            powerball = pd.to_numeric(df["Powerball"], errors="coerce")
    else:
        candidate_whites = [c for c in df.columns if c.upper().startswith("N")]
        if len(candidate_whites) >= 5 and "Powerball" in df.columns:
            whites = df[candidate_whites[:5]].apply(pd.to_numeric, errors="coerce")
            powerball = pd.to_numeric(df["Powerball"], errors="coerce")

    if whites is None or powerball is None:
        raise RuntimeError("Failed to parse N1..N6 from dataset.")

    whites.columns = ["N1","N2","N3","N4","N5"]
    df = pd.concat([df["DrawDate"], whites, powerball.rename("N6")], axis=1)
    df = df.dropna().astype({"N1":int,"N2":int,"N3":int,"N4":int,"N5":int,"N6":int})
    df = df.sort_values("DrawDate").reset_index(drop=True)
    return df

def make_windows(df):
    latest = df["DrawDate"].max()
    return {
        "90d":  df[df["DrawDate"] >= latest - pd.Timedelta(days=90)].copy(),
        "180d": df[df["DrawDate"] >= latest - pd.Timedelta(days=180)].copy(),
        "365d": df[df["DrawDate"] >= latest - pd.Timedelta(days=365)].copy(),
        "3y":   df[df["DrawDate"] >= latest - pd.Timedelta(days=3*365)].copy(),
        "all":  df.copy(),
    }

# ------------------------------- #
# Hot / Overdue / Correlations
# ------------------------------- #

def hot_cold_pools(df_90):
    hot, cold = {}, {}
    for col in ["N1","N2","N3","N4","N5"]:
        freq = df_90[col].value_counts().sort_index()
        if len(freq) == 0:
            hot[col], cold[col] = set(), set()
            continue
        norm = (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)
        hot[col]  = set(freq[norm >= HOT_PCTL].index)
        cold[col] = set(freq[(norm >= 0.01) & (norm <= 0.20)].index)
    return hot, cold

def overdue_sets(df_window):
    whites_overdue = set()
    pbs_overdue = set()
    recent = df_window.sort_values("DrawDate").reset_index(drop=True)

    for ball in range(1, 70):
        mask = recent[["N1","N2","N3","N4","N5"]].isin([ball]).any(axis=1)
        idx = mask[mask].index.to_numpy()
        if len(idx) == 0: continue
        last_seen = idx[-1]
        draws_since = len(recent) - (last_seen + 1)
        gaps = np.diff(idx)
        if len(gaps) == 0: continue
        median_gap = int(np.median(gaps))
        perc90_gap = int(np.percentile(gaps, 90))
        denom = max(median_gap, perc90_gap, 1)
        if draws_since / denom > 1.0:
            whites_overdue.add(ball)

    for ball in range(1, 27):
        idx = recent[recent["N6"] == ball].index.to_numpy()
        if len(idx) == 0: continue
        last_seen = idx[-1]
        draws_since = len(recent) - (last_seen + 1)
        gaps = np.diff(idx)
        if len(gaps) == 0: continue
        median_gap = int(np.median(gaps))
        perc90_gap = int(np.percentile(gaps, 90))
        denom = max(median_gap, perc90_gap, 1)
        if draws_since / denom > 1.0:
            pbs_overdue.add(ball)

    return whites_overdue, pbs_overdue

def strong_pairs(df_window, threshold=ASSOC_THRESHOLD):
    draws = [set(row) for row in df_window[["N1","N2","N3","N4","N5"]].values.tolist()]
    count = defaultdict(int)
    pair_count = defaultdict(int)
    for draw in draws:
        for i in draw: count[i] += 1
        for i in draw:
            for j in draw:
                if i != j: pair_count[(i,j)] += 1
    strong = defaultdict(list)
    for (i,j), c in pair_count.items():
        if count[i] > 0:
            p = c / count[i]
            if p >= threshold:
                strong[i].append((j, round(p,3)))
    return strong

# ------------------------------- #
# Base generation / tweaks
# ------------------------------- #

def revalidate_ticket(ticket, pool_min=1, pool_max=53, k=6):
    # Cast to plain ints immediately
    ticket = [int(n) for n in ticket]

    # Clamp numbers to pool bounds
    ticket = [min(max(n, pool_min), pool_max) for n in ticket]

    # Deduplicate while preserving sorted order
    ticket = sorted(set(ticket))

    # Pad if needed
    import random
    while len(ticket) < k:
        candidate = random.randint(pool_min, pool_max)
        while candidate in ticket:
            candidate = random.randint(pool_min, pool_max)
        ticket.append(candidate)
        ticket = sorted(ticket)

    if len(ticket) > k:
        ticket = sorted(ticket)[:k]

    return [int(n) for n in ticket]  # ensure clean ints on output

def gen_base_ticket():
    picks, used = [], set()
    for pos in ["N1","N2","N3","N4","N5"]:
        bucket = [n for n in POS_BUCKETS[pos] if n not in used] or list(POS_BUCKETS[pos])
        new = random.choice(bucket)
        picks.append(new)
        used.add(new)
    return picks

def apply_hot_overdue(picks, hot_pools, overdue_whites, max_hot=2, max_overdue=1):
    picks = picks[:]; used = set(picks)
    hot_k = random.choice([0,1,2])
    if hot_k > 0:
        for pos_idx in random.sample(range(5), hot_k):
            pos = f"N{pos_idx+1}"
            cand = [x for x in hot_pools.get(pos, set()) if x not in used and x in POS_BUCKETS[pos]]
            if cand:
                new = random.choice(cand)
                used.discard(picks[pos_idx])  # discard doesn’t raise KeyError
                used.add(new)
                picks[pos_idx] = new
                picks = revalidate_ticket(picks, pool_min=POOL_MIN, pool_max=POOL_MAX, k=K_NUMS)
    if random.choice([0,1]) == 1:
        pos_idx = random.randrange(5)
        pos = f"N{pos_idx+1}"
        cand = [x for x in overdue_whites if x not in used and x in POS_BUCKETS[pos]]
        if cand:
            new = random.choice(cand)
            used.discard(picks[pos_idx])  # discard doesn’t raise KeyError
            used.add(new)
            picks[pos_idx] = new
            picks = revalidate_ticket(picks, pool_min=POOL_MIN, pool_max=POOL_MAX, k=K_NUMS)
    return picks

def nudge_with_correlations(picks, strong_map):
    picks = picks[:]; present = set(picks)
    anchors = [a for a in picks if a in strong_map and strong_map[a]]
    random.shuffle(anchors)
    for a in anchors:
        partners = [b for b,_ in sorted(strong_map[a], key=lambda x:x[1], reverse=True)]
        if any(p in present for p in partners): continue
        for p in partners:
            for idx, cur in enumerate(picks):
                pos = f"N{idx+1}"
                if p in POS_BUCKETS[pos] and p not in present:
                    picks[idx] = p
                    present.add(p); present.discard(cur)
                    return picks
    return picks

def build_powerball_sampler(df):
    latest = df["DrawDate"].max()
    df60  = df[df["DrawDate"] >= latest - pd.Timedelta(days=60)]
    df180 = df[df["DrawDate"] >= latest - pd.Timedelta(days=180)]

    f60  = df60["N6"].value_counts().sort_index()
    f180 = df180["N6"].value_counts().sort_index()
    if len(f60) == 0: f60 = pd.Series([], dtype=int)
    if len(f180) == 0: f180 = pd.Series([], dtype=int)

    norm60 = (f60 - (f60.min() if len(f60) else 0)) / ((f60.max() - (f60.min() if len(f60) else 0)) + 1e-9)
    hot = set(f60[norm60 >= HOT_PCTL].index)
    cold = set(f60[(norm60 >= 0.01) & (norm60 <= 0.20)].index)

    mid = set()
    if len(f180):
        norm180 = (f180 - f180.min()) / (f180.max() - f180.min() + 1e-9)
        mid = set(f180[(norm180 >= 0.2) & (norm180 <= 0.6)].index)

    bands = (["mid"]*int(PB_WEIGHTS["mid"]*100) +
             ["hot"]*int(PB_WEIGHTS["hot"]*100) +
             ["cold"]*int(PB_WEIGHTS["cold"]*100))

    def sample_pb():
        band = random.choice(bands)
        pool = mid if band=="mid" and mid else (hot if band=="hot" and hot else (cold if band=="cold" and cold else set(f180.index)))
        return int(random.choice(list(pool))) if pool else random.randint(1,26)

    return sample_pb

# ------------------------------- #
# Smoothing
# ------------------------------- #

def smooth_batch(tickets, powerballs=None, restrict_to_used=True, radius=SMOOTHING_RADIUS):
    pool = sorted({n for t in tickets for n in t}) if restrict_to_used else list(range(1,70))
    cnt = Counter([n for t in tickets for n in t])
    total_slots = len(tickets)*5
    target = total_slots / max(1,len(pool))

    new_tickets = []
    for t in tickets:
        cur = t[:]
        for i, n in enumerate(cur):
            if cnt[n] > target:
                pos = f"N{i+1}"
                for d in range(1, radius+1):
                    for cand in (n-d, n+d):
                        if cand in pool and cnt[cand] < target and cand not in cur and cand in POS_BUCKETS[pos]:
                            cnt[n]-=1; cnt[cand]+=1
                            cur[i] = cand
                            break
                    else:
                        continue
                    break
        new_tickets.append(cur)
    return new_tickets, powerballs

# ------------------------------- #
# Exact-gap near-pair engine
# ------------------------------- #

def cumulative_baseline():
    # cumulative probs for at least one pair with gap <= g (5-of-69)
    return {1: 0.26, 2: 0.47, 3: 0.627, 4: 0.745, 5: 0.830}

def exact_gap_fracs(K=MAX_NEAR_GAP):
    cum = cumulative_baseline()
    ex = {}
    prev = 0.0
    for g in range(1, K+1):
        cur = cum.get(g, prev)
        ex[g] = max(0.0, cur - prev)
        prev = cur
    return ex  # dict gap->fraction for exactly that gap

def sample_pair_by_weights(quota_shortfall):
    """
    quota_shortfall: dict[pair] -> positive int shortfall (need); nonpositive ignored
    Prefer biggest shortfall; ties broken by PAIR_WEIGHTS randomness.
    """
    needers = [(need, pair) for pair, need in quota_shortfall.items() if need > 0]
    if not needers:
        # fallback: weighted random by PAIR_WEIGHTS
        bag = []
        for pr, w in PAIR_WEIGHTS.items():
            bag += [pr]*int(max(1,w))
        return random.choice(bag)
    needers.sort(reverse=True)  # largest shortfall first
    top_need = needers[0][0]
    top_pairs = [p for need, p in needers if need == top_need]
    # break tie by weight
    weights = [PAIR_WEIGHTS.get(p,1) for p in top_pairs]
    return random.choices(top_pairs, weights=weights, k=1)[0]

def ticket_near_details(ticket):
    details = []
    for i in range(4):
        j = i+1
        pi, pj = pos_name(i), pos_name(j)
        pair = (pi,pj)
        if pair not in PAIR_WEIGHTS and (pj,pi) not in PAIR_WEIGHTS:
            continue
        g = abs(ticket[i] - ticket[j])
        details.append((pair, i, j, g))
    return details

def ticket_gap_category(ticket):
    """Return (gap_category, pair_used or None). gap_category in {0,1..K}; 0 means clean."""
    det = [(pair, i, j, g) for (pair,i,j,g) in ticket_near_details(ticket) if g <= MAX_NEAR_GAP]
    if not det:
        return 0, None
    # With cap enforced elsewhere, there should be at most one; if not, pick smallest gap
    pair, i, j, g = sorted(det, key=lambda x: x[3])[0]
    return g, pair

def enforce_per_ticket_limit(ticket):
    """Ensure at most one near pair (gap<=K). Break extras (favor breaking larger gaps first)."""
    det = [(pair,i,j,g) for (pair,i,j,g) in ticket_near_details(ticket) if g <= MAX_NEAR_GAP]
    if len(det) <= NEAR_MAX_PER_TICKET:
        return ticket
    new_t = ticket[:]
    # keep the smallest gap and break others
    det.sort(key=lambda x: x[3])  # small gap is stronger; keep one
    keep = det[0]
    for (pair,i,j,g) in det[1:]:
        # shift j out beyond K
        cand = find_non_near_replacement_exact(new_t[j], j, new_t)
        if cand is not None: new_t[j] = cand
    return new_t

def find_non_near_replacement_exact(cur_val, idx, ticket):
    """Find a replacement at position idx that avoids creating any gap <= K."""
    pos = pos_name(idx)
    lo, hi = min(POS_BUCKETS[pos]), max(POS_BUCKETS[pos])
    start = MAX_NEAR_GAP + 1
    for step in [start, start+1, start+2, start+3]:
        for cand in (cur_val - step, cur_val + step):
            if (lo - BUCKET_FLEX) <= cand <= (hi + BUCKET_FLEX) and cand not in ticket:
                tmp = ticket[:]
                tmp[idx] = cand
                gcat, _ = ticket_gap_category(tmp)
                if gcat == 0:
                    return cand
    return None

def create_or_set_gap_on_pair(ticket, pair, gap_target):
    """
    Force exactly one near pair on 'pair' with specific gap. Tries moving only within that pair.
    Returns modified ticket or None.
    """
    (pa, pb) = pair
    i, j = int(pa[1])-1, int(pb[1])-1
    a, b = ticket[i], ticket[j]
    posi, posj = pos_name(i), pos_name(j)
    lo_i, hi_i = min(POS_BUCKETS[posi]), max(POS_BUCKETS[posi])
    lo_j, hi_j = min(POS_BUCKETS[posj]), max(POS_BUCKETS[posj])

    # Try moving j to match gap; else move i
    for base_idx, move_idx in ((i,j),(j,i)):
        base = ticket[base_idx]
        for cand in (base - gap_target, base + gap_target):
            if move_idx == j:
                if (lo_j - BUCKET_FLEX) <= cand <= (hi_j + BUCKET_FLEX) and cand not in ticket:
                    tmp = ticket[:]; tmp[move_idx] = cand
                    tmp = enforce_per_ticket_limit(tmp)
                    gcat, pr = ticket_gap_category(tmp)
                    if gcat == gap_target and pr in (pair, (pair[1],pair[0])):
                        return tmp
            else:
                if (lo_i - BUCKET_FLEX) <= cand <= (hi_i + BUCKET_FLEX) and cand not in ticket:
                    tmp = ticket[:]; tmp[move_idx] = cand
                    tmp = enforce_per_ticket_limit(tmp)
                    gcat, pr = ticket_gap_category(tmp)
                    if gcat == gap_target and pr in (pair, (pair[1],pair[0])):
                        return tmp
    return None

def break_near_pair(ticket):
    """Make ticket clean (no gap<=K) by nudging one member of its near pair."""
    gcat, pair = ticket_gap_category(ticket)
    if gcat == 0:
        return ticket
    (pa,pb) = pair
    i, j = int(pa[1])-1, int(pb[1])-1
    # Try breaking by moving j, else i
    for idx in (j, i):
        cand = find_non_near_replacement_exact(ticket[idx], idx, ticket)
        if cand is not None:
            tmp = ticket[:]; tmp[idx] = cand
            if ticket_gap_category(tmp)[0] == 0:
                return tmp
    return ticket

def match_exact_gap_distribution(tickets):
    """
    Enforce quotas across exact gaps 1..K and clean remainder.
    Also enforces pair usage quotas per PAIR_WEIGHTS.
    """
    n = len(tickets)
    if n == 0: return tickets

    # Compute exact-gap targets
    ex_fracs = exact_gap_fracs(MAX_NEAR_GAP)  # {gap: frac}
    # Integer gap targets
    gap_targets = largest_remainder_apportion(ex_fracs, n)
    used = sum(gap_targets.values())
    gap_targets[0] = max(0, n - used)  # clean target

    # Pre-cap: ensure ≤1 near pair per ticket
    tickets = [enforce_per_ticket_limit(t) for t in tickets]

    # Categorize & current usage
    by_gap = defaultdict(list)  # gap->list[ticket]
    pair_usage = Counter()      # only counts tickets that currently have near pair
    for t in tickets:
        gcat, pair = ticket_gap_category(t)
        by_gap[gcat].append(t)
        if gcat > 0 and pair is not None:
            canon = pair if pair in PAIR_WEIGHTS else (pair[1], pair[0])
            pair_usage[canon] += 1

    # Compute pair quotas: total near = sum gap_targets[g>0]; apportion by weights
    total_near_target = sum(gap_targets[g] for g in range(1, MAX_NEAR_GAP+1))
    norm_weights = {p: PAIR_WEIGHTS[p] for p in PAIR_WEIGHTS}
    pair_targets = largest_remainder_apportion(norm_weights, total_near_target)

    def pair_shortfall():
        return {p: pair_targets.get(p,0) - pair_usage.get(p,0) for p in PAIR_WEIGHTS}

    # Helpers to move a ticket between categories
    def retarget_ticket(t, desired_gap):
        """
        Convert t to have exactly 'desired_gap' (1..K), choosing pair with biggest shortfall first.
        """
        # If already at desired gap, keep it but maybe shift to underused pair
        gc, pr = ticket_gap_category(t)
        if gc == desired_gap:
            # try to shift to a pair with shortfall if current pair is over-satisfied
            short = pair_shortfall()
            best_pair = sample_pair_by_weights(short)
            if pr is None or short.get(best_pair,0) > short.get(pr,0):
                t2 = create_or_set_gap_on_pair(t, best_pair, desired_gap)
                if t2 is not None:
                    return t2
            return t

        # If currently clean or wrong gap, try to set desired gap on a pair with highest shortfall
        short = pair_shortfall()
        order = sorted(PAIR_WEIGHTS, key=lambda p: (short.get(p,0), PAIR_WEIGHTS[p]), reverse=True)
        for pr_choice in order:
            t2 = create_or_set_gap_on_pair(t, pr_choice, desired_gap)
            if t2 is not None:
                return t2
        return t  # fail: leave as-is

    def update_usage_lists(old_t, new_t):
        nonlocal by_gap, pair_usage
        og, op = ticket_gap_category(old_t)
        ng, npair = ticket_gap_category(new_t)
        if old_t in by_gap[og]:
            by_gap[og].remove(old_t)
        by_gap[ng].append(new_t)
        if og>0 and op is not None:
            canon = op if op in PAIR_WEIGHTS else (op[1],op[0])
            pair_usage[canon] -= 1
        if ng>0 and npair is not None:
            canon = npair if npair in PAIR_WEIGHTS else (npair[1],npair[0])
            pair_usage[canon] += 1

    # 1) Reduce overfull gap buckets
    safety = 0
    while safety < 10*n:
        safety += 1
        changes = 0
        for g in list(by_gap.keys()):
            cur = len(by_gap[g])
            tgt = gap_targets.get(g,0)
            if cur <= tgt:
                continue
            # too many in gap g: move extras to gaps with shortfall (prefer lower g shortage first), else to clean
            need_list = [(gg, gap_targets.get(gg,0) - len(by_gap.get(gg,[]))) for gg in range(1, MAX_NEAR_GAP+1)]
            need_list = [x for x in need_list if x[1] > 0]
            # sort by biggest need
            need_list.sort(key=lambda x: x[1], reverse=True)
            # pop a ticket from this overfull bucket and try to retarget
            t = by_gap[g].pop()
            new_t = None
            for gg, _need in need_list:
                new_t = retarget_ticket(t, gg)
                if new_t is not None and ticket_gap_category(new_t)[0] == gg:
                    update_usage_lists(t, new_t)
                    changes += 1
                    break
            if new_t is None:
                # try to make it clean if clean has need
                clean_need = gap_targets[0] - len(by_gap[0])
                if clean_need > 0:
                    t2 = break_near_pair(t)
                    if ticket_gap_category(t2)[0] == 0:
                        update_usage_lists(t, t2)
                        changes += 1
                    else:
                        by_gap[g].append(t)  # revert
                else:
                    by_gap[g].append(t)      # revert
        if changes == 0:
            break

    # 2) Fill shortages from clean (or other gaps if needed)
    safety = 0
    while safety < 10*n:
        safety += 1
        changes = 0
        for g in range(1, MAX_NEAR_GAP+1):
            while len(by_gap[g]) < gap_targets.get(g,0):
                # prefer taking from clean
                source = None
                if len(by_gap[0]) > gap_targets.get(0,0):
                    source = by_gap[0].pop()
                else:
                    # borrow from another overfull gap if exists
                    donors = [(gg, len(by_gap[gg]) - gap_targets.get(gg,0)) for gg in range(1, MAX_NEAR_GAP+1)]
                    donors = [x for x in donors if x[1] > 0 and x[0] != g]
                    donors.sort(key=lambda x: x[1], reverse=True)
                    if donors:
                        dg, _extra = donors[0]
                        source = by_gap[dg].pop()
                if source is None:
                    break  # nothing to convert
                t2 = retarget_ticket(source, g)
                if ticket_gap_category(t2)[0] == g:
                    update_usage_lists(source, t2)
                    changes += 1
                else:
                    # put source back to where it came from
                    gc,_ = ticket_gap_category(source)
                    by_gap[gc].append(source)
                    break
        if changes == 0:
            break

    # Rebuild ticket list preserving no duplicates
    out = []
    for g in [0] + list(range(1, MAX_NEAR_GAP+1)):
        out.extend(by_gap.get(g, []))

    # Per-ticket cap again (safety)
    out = [enforce_per_ticket_limit(t) for t in out]
    return out

# ------------------------------- #
# Wheel reduction
# ------------------------------- #

def wheel_reduce(tickets, df_for_weights, final_k=WHEEL_TARGET):
    pair_w = defaultdict(int)
    for row in df_for_weights[["N1","N2","N3","N4","N5"]].values:
        s = sorted(set(row))
        for a,b in itertools.combinations(s,2):
            pair_w[(a,b)] += 1

    ticket_pairs = []
    for t in tickets:
        s = sorted(set(t))
        pairs = {tuple(sorted(p)) for p in itertools.combinations(s,2)}
        ticket_pairs.append(pairs)

    covered = set(); chosen_idx = []
    for _ in range(min(final_k, len(tickets))):
        best_i, best_gain = None, -1
        for i, pairs in enumerate(ticket_pairs):
            if i in chosen_idx: continue
            gain = sum(pair_w[p] for p in (pairs - covered))
            if gain > best_gain: best_i, best_gain = i, gain
        if best_i is None: break
        chosen_idx.append(best_i)
        covered |= ticket_pairs[best_i]

    return [tickets[i] for i in chosen_idx]

# ================================
# Use adaptive brackets
# Adjust N2...N5 based on N1, etc
# ================================

def compute_gap_distributions(df):
    """Return empirical gap samples for N2-N1 .. N5-N4."""
    gap_dists = {}
    for i in range(1, 5):
        col1, col2 = f"N{i}", f"N{i+1}"
        gaps = (df[col2] - df[col1]).dropna()
        # only keep positive gaps (should always be since numbers are sorted)
        gaps = gaps[gaps > 0]
        gap_dists[f"{col2}-{col1}"] = gaps.tolist()
    return gap_dists

def generate_conditional_ticket(gap_distributions, base_buckets):
    """Generate one 5-number ticket using conditional gap sampling."""
    ticket = []
    # N1: choose freely from its bucket
    n1 = np.random.choice(list(base_buckets["N1"]))
    ticket.append(n1)

    # N2..N6: sample gap, add to previous
    for i in range(2, 6):
        key = f"N{i}-N{i-1}"
        if key not in gap_distributions or len(gap_distributions[key]) == 0:
            # fallback: just pick from bucket
            ni = np.random.choice(list(base_buckets[f"N{i}"]))
        else:
            gap = np.random.choice(gap_distributions[key])
            ni = ticket[-1] + gap
            # enforce bucket range
            lo, hi = min(base_buckets[f"N{i}"]), max(base_buckets[f"N{i}"])
            if ni < lo: ni = lo
            if ni > hi: ni = hi
        ticket.append(int(ni))

    return [int(x) for x in sorted(ticket)]

# ------------------------------- #
# Main pipeline
# ------------------------------- #

def generate_master_set(
    df,
    num_candidates=NUM_CANDIDATE_TICKETS,
    final_k=WHEEL_TARGET,
    use_wheel=USE_WHEEL,
    enable_smoothing=ENABLE_SMOOTHING,
    smooth_scope=SMOOTH_SCOPE  # "chosen" or "pool"
):
    buckets = POS_BUCKETS
    if USE_CONDITIONAL:
        gap_dists = compute_gap_distributions(df)

    windows = make_windows(df)
    df90, df180, df365, df3y, dfall = windows["90d"], windows["180d"], windows["365d"], windows["3y"], windows["all"]

    hot, _ = hot_cold_pools(df90)
    overdue_whites, overdue_pbs = overdue_sets(dfall)
    corr_map = strong_pairs(df180, threshold=ASSOC_THRESHOLD)
    sample_pb = build_powerball_sampler(df)

    # 1) Candidates
    candidates = []; pbs = []; seen = set(); tries = 0
    while len(candidates) < num_candidates and tries < num_candidates*20:
        tries += 1
        if USE_CONDITIONAL:
            t = generate_conditional_ticket(gap_dists, buckets)
        else:
            t = gen_base_ticket(buckets)
        t = apply_hot_overdue(t, hot, overdue_whites, max_hot=2, max_overdue=1)
        t = nudge_with_correlations(t, corr_map)
        t = [int(x) for x in t]  # normalize to int
        sig = tuple(t)
        if sig in seen: continue
        if not odd_even_ok(t, min_odds=1, min_evens=1):
            continue
        seen.add(sig)
        candidates.append(t)
        pb = sample_pb()
        pb = ensure_pb_not_in_whites(pb, t, sample_pb)
        pbs.append(pb)

    # 2) Optional smoothing + repair
    if enable_smoothing:
        restrict = True if str(smooth_scope).lower() == "chosen" else False
        candidates, pbs = smooth_batch(candidates, pbs, restrict_to_used=restrict, radius=SMOOTHING_RADIUS)

    validated, valid_pbs = [], []
    for t, pb in zip(candidates, pbs):
        fixed = reorder_to_buckets(t, BUCKET_FLEX)
        if fixed:
            validated.append(fixed); valid_pbs.append(pb)
    candidates, pbs = validated, valid_pbs

    # 2c) Enforce exact-gap + pair-weight quotas
    candidates = match_exact_gap_distribution(candidates)

    # Repair after shaping
    validated, valid_pbs = [], []
    pbs = pbs[:len(candidates)]
    for t, pb in zip(candidates, pbs):
        fixed = reorder_to_buckets(t, BUCKET_FLEX)
        if fixed:
            validated.append(fixed); valid_pbs.append(pb)
    candidates, pbs = validated, valid_pbs

    # Re-enforce again after repair, then repair again
    candidates = match_exact_gap_distribution(candidates)
    validated, valid_pbs = [], []
    pbs = pbs[:len(candidates)]
    for t, pb in zip(candidates, pbs):
        fixed = reorder_to_buckets(t, BUCKET_FLEX)
        if fixed:
            validated.append(fixed); valid_pbs.append(pb)
    candidates, pbs = validated, valid_pbs

    if DEBUG_NEAR_MIX:
        print("Candidate near-pair mix:", near_pair_mix_summary_exact(candidates))

    # 3) Select finals
    if use_wheel:
        # Use 180d for coverage weights by default
        final_whites = wheel_reduce(candidates, df_for_weights=df180, final_k=final_k)
    else:
        final_whites = candidates[:final_k]

    # Final set: enforce quotas + repair once more
    final_whites = match_exact_gap_distribution(final_whites)
    final_whites = [reorder_to_buckets(t, BUCKET_FLEX) or t for t in final_whites]

    if DEBUG_NEAR_MIX:
        print("Final near-pair mix:", near_pair_mix_summary_exact(final_whites))

    # Attach PBs, resample if collision with whites
    white2pb = {}
    for w, pb in zip(candidates, pbs):
        white2pb.setdefault(tuple(sorted(w)), pb)

    final = []
    for w in final_whites:
        fixed = reorder_to_buckets(w, BUCKET_FLEX)
        if not fixed: continue
        pb = white2pb.get(tuple(sorted(fixed)))
        if pb is None:
            pb = sample_pb()
        pb = ensure_pb_not_in_whites(pb, fixed, sample_pb)
        final.append(sorted(fixed) + [pb])

    # Also return candidates in sorted form
    candidates_sorted = [sorted(t) for t in candidates]
    return final, candidates_sorted, pbs

# ------------------------------- #
# Entry
# ------------------------------- #

if __name__ == "__main__":
    random.seed()
    df = load_history_csv()
    final10, big_pool, big_pbs = generate_master_set(
        df,
        num_candidates=NUM_CANDIDATE_TICKETS,
        final_k=WHEEL_TARGET,
        use_wheel=USE_WHEEL,
        enable_smoothing=ENABLE_SMOOTHING,
        smooth_scope=SMOOTH_SCOPE,
    )

    print(f"\nGenerated {len(big_pool)} candidate sets; reduced to {len(final10)} final tickets (use_wheel={USE_WHEEL}, K={MAX_NEAR_GAP}):\n")
    for t in final10:
        print(t)
