# six_master_generator.py
# Single-file 6-number ticket generator (no Powerball)

import fitz  # PyMuPDF
import os, ssl, datetime, random, itertools
import urllib.request
import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
from itertools import permutations

# ================================
# Config
# ================================

# ---- Data ----
# Provide your CSV locally as "history6.csv" 
LOCAL_FILE = "history6.csv"  # expected columns: either "Winning Numbers" or N1..N6 + DrawDate/Draw Date

# ---- Pool / Buckets ----
POOL_MIN = 1
POOL_MAX = 53
K_NUMS   = 6

WHEEL_TARGET = 10      # THE NUMBER OF FINAL TICKETS TO PRODUCE

# Auto-derive positional buckets from history percentiles (recommended)
AUTO_BUCKETS = True    # Auto generate the Buckets or use POS_BUCKETS below
USE_CONDITIONAL = True # Use adaptive buckets or use POS_BUCKETS below
BUCKET_Q_LO = 0.05     # use 5th to 95th percentile envelope
BUCKET_Q_HI = 0.95
BUCKET_EXPAND = 3      # widen each side by this many integers

# If AUTO_BUCKETS=False, set explicit buckets here (inclusive ranges):
POS_BUCKETS = {
    "N1": range(1, 15),   # 1-13
    "N2": range(8, 26),  # 10-24
    "N3": range(21, 37),  # 20-35
    "N4": range(25, 44),  # 32-48
    "N5": range(34, 49),  # 45-60
    "N6": range(46, 53),  # 55-69
}
BUCKET_FLEX = 2         # ±flex allowed during repair

# ---- Candidate shaping ----
USE_WHEEL = False                  # Boil larger pool down to a sensible set
NUM_CANDIDATE_TICKETS = 250000     # Pool of tickets to generate before boiling down
ENABLE_SMOOTHING = False         # Reduce Bias
SMOOTH_SCOPE = "chosen"          # "chosen" (only within batch) or "pool" (within 1..POOL_MAX)
SMOOTHING_RADIUS = 3

# ---- Hot / Overdue / Correlation ----
HOT_PCTL = 0.67                  # top quantile in 90d by position
ASSOC_THRESHOLD = 0.67           # 180d strong pair threshold

# ---- Near-pair engine (exact-gap quotas) ----
# K = max near gap
MAX_NEAR_GAP = 3                 # 2..5 typical

# Cumulative targets for "at least one gap <= g".
# Defaults here match 6-of-53 baselines; adjust to your game if desired.
CUM_NEAR_FRACS = {1: 0.4655, 2: 0.7344, 3: 0.8797, 4: 0.9518, 5: 0.9836}

# Exact gaps are derived from CUM_NEAR_FRACS by differencing.
# Pair-position weights (adjacent positions only) — relative proportions among near pairs:
PAIR_WEIGHTS = {("N2","N3"): 4, ("N3","N4"): 1, ("N4","N5"): 2, ("N5","N6"): 3}

NEAR_MAX_PER_TICKET = 4          # at most X near pairs per ticket
DEBUG_NEAR_MIX = False

# ================================
# Helpers
# ================================

def pos_name(idx): return f"N{idx+1}"

def in_bucket(num, pos, flex=0, buckets=None):
    rng = buckets[pos]
    return (min(rng) - flex) <= num <= (max(rng) + flex)

def reorder_to_buckets(ticket, buckets, flex=BUCKET_FLEX):
    positions = [f"N{i+1}" for i in range(len(ticket))]
    for perm in permutations(ticket):
        if all(in_bucket(perm[i], positions[i], flex, buckets) for i in range(len(ticket))):
            return list(perm)
    return None

def largest_remainder_apportion(fracs, total):
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

def odd_even_ok(ticket, min_odds=2, min_evens=2):
    odds = sum(1 for n in ticket if n % 2 == 1)
    evens = len(ticket) - odds
    return odds >= min_odds and evens >= min_evens


# ================================
# Data I/O
# ================================

def extract_rows_by_date_anchor(pdf_path, output_path):
    rows = []
    game_list = ['LOTTO', 'LOTTO DP', 'X2', 'X3', 'X4', 'X5']
    elimination_list = ['LOTTO DP', 'X2', 'X3', 'X4', 'X5']
    DATE_REGEX = re.compile(r"\b(0[1-9]|1[0-2])/([0-2][0-9]|3[01])/([0-9]{2})\b")
    
    with fitz.open(pdf_path) as doc:
        for page in doc:
            dates = []
            values = []
            games = []
            date_index = 0
            MAX_DATE = '10/10/20'
            max_date_reached = False
            
            blocks = page.get_text("dict")["blocks"]

            # Flatten all spans with position
            elements = []
            for block in blocks:
                #print("NEW BLOCK")
                for line in block.get("lines", []):
                    #print("NEW LINE")
                    for span in line.get("spans", []):
                        #print("NEW SPAN")
                        text = span["text"].strip()
                        if text:
                            elements.append({
                                "text": text,
                                "x": span["bbox"][0],
                                "y": span["bbox"][1]
                            })

                            #print(f"text: {text}")

                            if DATE_REGEX.fullmatch(text):
                                #print(f"INSERT DATE: {text} (length: {len(dates)+1})")
                                dates.append(text)

                                #TODO check against date 10/10/2020, stop processing at this page
                                if text == MAX_DATE:
                                    max_date_reached = True
                                    
            # Flatten all spans with position
            elements = []
            for block in blocks:
                #print("NEW BLOCK")
                for line in block.get("lines", []):
                    #print("NEW LINE")
                    for span in line.get("spans", []):
                        #print("NEW SPAN")
                        text = span["text"].strip()
                        if text:
                            elements.append({
                                "text": text,
                                "x": span["bbox"][0],
                                "y": span["bbox"][1]
                            })

                            #print(f"text: {text}")

                            if text.isdigit():
                                # if we've already filled the array, now we should start appending by index
                                if len(dates) == len(values):   
                                    #print(f"CONCATE DIGIT: {values[date_index]} + {text} (length: {len(values)})")
                                    values[date_index] += " " + text
                                    date_index += 1 # update ahead of next usage

                                    if date_index == len(values):
                                        date_index = 0
                                else:
                                    #print(f"INSERT DIGIT: {text} (length: {len(values)+1})")
                                    values.append(text)
                            elif text in game_list:
                                #print(f"INSERT GAME: {text} (length: {len(games)+1})")
                                games.append(text)
                                                                        
            #print(f"dates = {dates}")
            #print(f"values = {values}")
            if len(dates) != len(values):
                print(f"FAIL! NUMBER OF DATES AND VALUES ARE NOT ALIGNED!")
                print(f"dates = {dates}")
                print(f"values = {values}")
                exit(1)
            for value in values:
                if len(value.split(" ")) != 6:
                    print(f"FAIL! EXPECTED ALL READ VALUES TO CONTAIN 6 NUMBERS")
                    exit(1)

            for index, date in enumerate(dates):
                if games[index] not in elimination_list:
                    rows.append(date + "," + values[index])
                #else:
                #    print(f"ELIMINATING DATA: {index}. {date} :: {values[index]} :: {games[index]}")

            #print(f"rows count = {len(rows)}")
        
            if max_date_reached: break

    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        file.write('Draw Date,Winning Numbers')
        for line in rows:
            file.write('\n' + line)

    print(f"Extracted {len(rows)} rows to {output_path}")

def load_history_csv():
    today = datetime.date.today().strftime("%Y%m%d")
    dated_pdf = f"lottox_history_{today}.pdf"
    dated_name = f"lottox_history_{today}.csv"
    DL_URL = "https://files.floridalottery.com/exptkt/l6.pdf"

    if os.path.exists(dated_name):
        fname = dated_name
    elif os.path.exists(LOCAL_FILE):
        fname = LOCAL_FILE
    else:
        # Force TLSv1.2 and AES128-GCM-SHA256
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ctx.set_ciphers("AES128-GCM-SHA256")

        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx)
        )
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(DL_URL, dated_pdf)
        fname = dated_name
        print(f"Downloaded LottoX history → {dated_pdf}. Now convert to CSV...")

        extract_rows_by_date_anchor(dated_pdf, dated_name)

    print(f"Loading CSV: {fname}")
    df = pd.read_csv(fname)

    # Date column
    if "Draw Date" in df.columns:
        df["DrawDate"] = pd.to_datetime(df["Draw Date"], format="%m/%d/%y")
    elif "DrawDate" in df.columns:
        df["DrawDate"] = pd.to_datetime(df["DrawDate"], format="%m/%d/%y")
    elif "Date" in df.columns:
        df["DrawDate"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    else:
        raise RuntimeError("Missing date column: expected 'Draw Date' / 'DrawDate' / 'Date'.")

    # Parse numbers
    whites = None
    if "Winning Numbers" in df.columns:
        s = df["Winning Numbers"].astype(str)
        parts = s.str.replace(r"[^\d\s,]", "", regex=True).str.replace(",", " ").str.split()
        whites = parts.apply(lambda x: pd.Series([int(v) for v in x[:K_NUMS]]))
    else:
        cand = [c for c in df.columns if c.upper().startswith("N")]
        if len(cand) >= K_NUMS:
            whites = df[cand[:K_NUMS]].apply(pd.to_numeric, errors="coerce")
    if whites is None:
        raise RuntimeError("Failed to parse N1..N6 / 'Winning Numbers'.")

    whites.columns = [f"N{i+1}" for i in range(K_NUMS)]
    # sort numbers within each draw to true positions
    whites_sorted = whites.apply(lambda row: sorted(row.values.tolist()), axis=1, result_type="expand")
    whites_sorted.columns = [f"N{i+1}" for i in range(K_NUMS)]

    out = pd.concat([df["DrawDate"], whites_sorted], axis=1)
    out = out.dropna().astype({f"N{i+1}": int for i in range(K_NUMS)})
    out = out.sort_values("DrawDate").reset_index(drop=True)
    return out

def make_windows(df):
    latest = df["DrawDate"].max()
    return {
        "90d":  df[df["DrawDate"] >= latest - pd.Timedelta(days=90)].copy(),
        "180d": df[df["DrawDate"] >= latest - pd.Timedelta(days=180)].copy(),
        "365d": df[df["DrawDate"] >= latest - pd.Timedelta(days=365)].copy(),
        "3y":   df[df["DrawDate"] >= latest - pd.Timedelta(days=3*365)].copy(),
        "all":  df.copy(),
    }

# ================================
# Buckets (auto from history)
# ================================

def derive_buckets_from_history(df, q_lo=BUCKET_Q_LO, q_hi=BUCKET_Q_HI, expand=BUCKET_EXPAND):
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"derive_buckets_from_history expected a pandas DataFrame, got {type(df).__name__}. "
            "Pass the DataFrame returned by load_history_csv()."
        )

    expected = [f"N{i+1}" for i in range(K_NUMS)]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns missing from history: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Coerce numeric and drop NaNs for quantiles
    df_num = df.copy()
    for c in expected:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    buckets = {}
    for i in range(K_NUMS):
        col = f"N{i+1}"
        s = df_num[col].dropna()
        if s.empty:
            # Fallback to full pool if this position has no data
            lo, hi = POOL_MIN, POOL_MAX
        else:
            ql = int(np.floor(s.quantile(q_lo)))
            qh = int(np.ceil(s.quantile(q_hi)))
            lo = max(POOL_MIN, ql - expand)
            hi = min(POOL_MAX, qh + expand)
            if lo > hi:  # safety
                lo, hi = POOL_MIN, POOL_MAX
        buckets[col] = range(lo, hi + 1)

    return buckets

# ================================
# Hot / Overdue / Correlations
# ================================

def hot_cold_pools(df_90, buckets):
    hot, cold = {}, {}
    for i in range(K_NUMS):
        col = f"N{i+1}"
        freq = df_90[col].value_counts().sort_index()
        if len(freq) == 0:
            hot[col], cold[col] = set(), set(); continue
        norm = (freq - freq.min()) / (max(1e-9, (freq.max() - freq.min())))
        hot[col]  = set(int(x) for x in freq[norm >= HOT_PCTL].index if x in buckets[col])
        cold[col] = set(int(x) for x in freq[(norm >= 0.01) & (norm <= 0.20)].index if x in buckets[col])
    return hot, cold

def overdue_set(df_window):
    overdue = set()
    recent = df_window.sort_values("DrawDate").reset_index(drop=True)

    # unify all draws' numbers into a single sequence of sets
    index_by_ball = defaultdict(list)
    for idx, row in recent[[f"N{i+1}" for i in range(K_NUMS)]].iterrows():
        for v in set(row.values.tolist()):
            index_by_ball[int(v)].append(idx)

    for ball in range(POOL_MIN, POOL_MAX+1):
        idxs = index_by_ball.get(ball, [])
        if not idxs: continue
        last_seen = idxs[-1]
        draws_since = len(recent) - (last_seen + 1)
        gaps = np.diff(idxs)
        if len(gaps) == 0: continue
        median_gap = int(np.median(gaps))
        perc90_gap = int(np.percentile(gaps, 90))
        denom = max(median_gap, perc90_gap, 1)
        if draws_since / denom > 1.0:
            overdue.add(ball)
    return overdue

def strong_pairs(df_window, threshold=ASSOC_THRESHOLD):
    draws = [set(row) for row in df_window[[f"N{i+1}" for i in range(K_NUMS)]].values.tolist()]
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

# ================================
# Base generation / shaping
# ================================

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

def gen_base_ticket(buckets):
    picks, used = [], set()
    for i in range(K_NUMS):
        pos = f"N{i+1}"
        bucket = [n for n in buckets[pos] if n not in used] or list(buckets[pos])
        new = random.choice(bucket)
        picks.append(new); used.add(new)
    return picks

def apply_hot_overdue(picks, hot_pools, overdue_whites, buckets, max_hot=2, max_overdue=1):
    picks = picks[:]; used = set(picks)
    # HOT (0–2)
    hot_k = random.choice([0,1,2])
    if hot_k > 0:
        for pos_idx in random.sample(range(K_NUMS), hot_k):
            pos = f"N{pos_idx+1}"
            cand = [x for x in hot_pools.get(pos, set()) if x not in used and x in buckets[pos]]
            if cand:
                new = random.choice(cand)
                used.discard(picks[pos_idx])
                used.add(new)
                picks[pos_idx] = new
                picks = revalidate_ticket(picks, pool_min=POOL_MIN, pool_max=POOL_MAX, k=K_NUMS)
    # OVERDUE (0–1)
    if random.choice([0,1]) == 1:
        pos_idx = random.randrange(K_NUMS)
        pos = f"N{pos_idx+1}"
        cand = [x for x in overdue_whites if x not in used and x in buckets[pos]]
        if cand:
            new = random.choice(cand)
            used.discard(picks[pos_idx])
            used.add(new)
            picks[pos_idx] = new
            picks = revalidate_ticket(picks, pool_min=POOL_MIN, pool_max=POOL_MAX, k=K_NUMS)
    return picks

def nudge_with_correlations(picks, strong_map, buckets):
    picks = picks[:]; present = set(picks)
    anchors = [a for a in picks if a in strong_map and strong_map[a]]
    random.shuffle(anchors)
    for a in anchors:
        partners = [b for b,_ in sorted(strong_map[a], key=lambda x:x[1], reverse=True)]
        if any(p in present for p in partners): continue
        for p in partners:
            for idx, cur in enumerate(picks):
                pos = f"N{idx+1}"
                if p in buckets[pos] and p not in present:
                    picks[idx] = p
                    present.add(p); present.discard(cur)
                    return picks
    return picks

# ================================
# Smoothing (bias reducer)
# ================================

def smooth_batch(tickets, restrict_to_used=True, radius=SMOOTHING_RADIUS, buckets=None):
    pool = sorted({n for t in tickets for n in t}) if restrict_to_used else list(range(POOL_MIN, POOL_MAX+1))
    cnt = Counter([n for t in tickets for n in t])
    total_slots = len(tickets) * K_NUMS
    target = total_slots / max(1, len(pool))

    new_tickets = []
    for t in tickets:
        cur = t[:]
        for i, n in enumerate(cur):
            if cnt[n] > target:
                pos = f"N{i+1}"
                for d in range(1, radius+1):
                    for cand in (n-d, n+d):
                        if (cand in pool) and (cnt[cand] < target) and (cand not in cur) and (cand in buckets[pos]):
                            cnt[n]-=1; cnt[cand]+=1
                            cur[i] = cand
                            break
                    else:
                        continue
                    break
        new_tickets.append(cur)
    return new_tickets

# ================================
# Near-pair engine (exact gaps)
# ================================

def exact_gap_fracs(K=MAX_NEAR_GAP, cum=CUM_NEAR_FRACS):
    ex = {}
    prev = 0.0
    for g in range(1, K+1):
        cur = cum.get(g, prev)
        ex[g] = max(0.0, cur - prev)
        prev = cur
    return ex

def ticket_near_details(ticket):
    # Only adjacent positions present in PAIR_WEIGHTS
    details = []
    for i in range(K_NUMS - 1):
        j = i + 1
        pi, pj = pos_name(i), pos_name(j)
        pair = (pi, pj)
        if pair not in PAIR_WEIGHTS and (pj,pi) not in PAIR_WEIGHTS:
            continue
        g = abs(ticket[i] - ticket[j])
        details.append((pair, i, j, g))
    return details

def ticket_gap_category(ticket):
    det = [(pair, i, j, g) for (pair,i,j,g) in ticket_near_details(ticket) if g <= MAX_NEAR_GAP]
    if not det:
        return 0, None
    pair, i, j, g = sorted(det, key=lambda x: x[3])[0]
    return g, pair

def enforce_per_ticket_limit(ticket, buckets):
    det = [(pair,i,j,g) for (pair,i,j,g) in ticket_near_details(ticket) if g <= MAX_NEAR_GAP]
    if len(det) <= NEAR_MAX_PER_TICKET:
        return ticket
    new_t = ticket[:]
    det.sort(key=lambda x: x[3])   # keep smallest gap
    for (pair,i,j,g) in det[1:]:
        # nudge j outside near range
        cand = find_non_near_replacement_exact(new_t[j], j, new_t, buckets)
        if cand is not None:
            new_t[j] = cand
    return new_t

def find_non_near_replacement_exact(cur_val, idx, ticket, buckets):
    pos = pos_name(idx)
    lo, hi = min(buckets[pos]), max(buckets[pos])
    start = MAX_NEAR_GAP + 1
    for step in [start, start+1, start+2, start+3]:
        for cand in (cur_val - step, cur_val + step):
            if (lo - BUCKET_FLEX) <= cand <= (hi + BUCKET_FLEX) and cand not in ticket:
                tmp = ticket[:]; tmp[idx] = cand
                gcat, _ = ticket_gap_category(tmp)
                if gcat == 0:
                    return cand
    return None

def create_or_set_gap_on_pair(ticket, pair, gap_target, buckets):
    (pa, pb) = pair
    i, j = int(pa[1])-1, int(pb[1])-1
    a, b = ticket[i], ticket[j]
    posi, posj = pos_name(i), pos_name(j)
    lo_i, hi_i = min(buckets[posi]), max(buckets[posi])
    lo_j, hi_j = min(buckets[posj]), max(buckets[posj])

    for base_idx, move_idx in ((i,j),(j,i)):
        base = ticket[base_idx]
        for cand in (base - gap_target, base + gap_target):
            if move_idx == j:
                if (lo_j - BUCKET_FLEX) <= cand <= (hi_j + BUCKET_FLEX) and cand not in ticket:
                    tmp = ticket[:]; tmp[move_idx] = cand
                    tmp = enforce_per_ticket_limit(tmp, buckets)
                    gcat, pr = ticket_gap_category(tmp)
                    if gcat == gap_target and pr in (pair, (pair[1],pair[0])):
                        return tmp
            else:
                if (lo_i - BUCKET_FLEX) <= cand <= (hi_i + BUCKET_FLEX) and cand not in ticket:
                    tmp = ticket[:]; tmp[move_idx] = cand
                    tmp = enforce_per_ticket_limit(tmp, buckets)
                    gcat, pr = ticket_gap_category(tmp)
                    if gcat == gap_target and pr in (pair, (pair[1],pair[0])):
                        return tmp
    return None

def break_near_pair(ticket, buckets):
    gcat, pair = ticket_gap_category(ticket)
    if gcat == 0: return ticket
    (pa,pb) = pair
    i, j = int(pa[1])-1, int(pb[1])-1
    for idx in (j, i):
        cand = find_non_near_replacement_exact(ticket[idx], idx, ticket, buckets)
        if cand is not None:
            tmp = ticket[:]; tmp[idx] = cand
            if ticket_gap_category(tmp)[0] == 0:
                return tmp
    return ticket

def match_exact_gap_distribution(tickets, buckets):
    n = len(tickets)
    if n == 0: return tickets

    # Exact gap targets (1..K) + clean
    ex_fracs = exact_gap_fracs(MAX_NEAR_GAP, CUM_NEAR_FRACS)  # {gap: frac}
    gap_targets = largest_remainder_apportion(ex_fracs, n)
    used = sum(gap_targets.values())
    gap_targets[0] = max(0, n - used)  # clean

    # Cap per-ticket first
    tickets = [enforce_per_ticket_limit(t, buckets) for t in tickets]

    # Categorize & current pair usage
    by_gap = defaultdict(list)
    pair_usage = Counter()
    for t in tickets:
        gcat, pair = ticket_gap_category(t)
        by_gap[gcat].append(t)
        if gcat > 0 and pair is not None:
            canon = pair if pair in PAIR_WEIGHTS else (pair[1], pair[0])
            pair_usage[canon] += 1

    # Pair quotas across near pairs
    total_near_target = sum(gap_targets[g] for g in range(1, MAX_NEAR_GAP+1))
    pair_targets = largest_remainder_apportion(PAIR_WEIGHTS, total_near_target)

    def pair_shortfall():
        return {p: pair_targets.get(p,0) - pair_usage.get(p,0) for p in PAIR_WEIGHTS}

    def retarget_ticket(t, desired_gap):
        gc, pr = ticket_gap_category(t)
        if gc == desired_gap:
            short = pair_shortfall()
            best_pair = max(PAIR_WEIGHTS.keys(), key=lambda p: (short.get(p,0), PAIR_WEIGHTS[p]))
            if pr is None or short.get(best_pair,0) > short.get(pr,0):
                t2 = create_or_set_gap_on_pair(t, best_pair, desired_gap, buckets)
                if t2 is not None:
                    return t2
            return t
        short = pair_shortfall()
        order = sorted(PAIR_WEIGHTS, key=lambda p: (short.get(p,0), PAIR_WEIGHTS[p]), reverse=True)
        for pr_choice in order:
            t2 = create_or_set_gap_on_pair(t, pr_choice, desired_gap, buckets)
            if t2 is not None and ticket_gap_category(t2)[0] == desired_gap:
                return t2
        return t

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

    # Reduce overfull gap buckets
    safety = 0
    while safety < 10*n:
        safety += 1
        changes = 0
        for g in list(by_gap.keys()):
            cur = len(by_gap[g]); tgt = gap_targets.get(g,0)
            if cur <= tgt: continue
            # move extras to gaps with need (largest need first), else to clean (if needed)
            need_list = [(gg, gap_targets.get(gg,0) - len(by_gap.get(gg,[]))) for gg in range(1, MAX_NEAR_GAP+1)]
            need_list = [x for x in need_list if x[1] > 0]
            need_list.sort(key=lambda x: x[1], reverse=True)
            t = by_gap[g].pop()
            moved = False
            for gg,_need in need_list:
                t2 = retarget_ticket(t, gg)
                if ticket_gap_category(t2)[0] == gg:
                    update_usage_lists(t, t2)
                    changes += 1
                    moved = True
                    break
            if not moved:
                clean_need = gap_targets[0] - len(by_gap[0])
                if clean_need > 0:
                    t2 = break_near_pair(t, buckets)
                    if ticket_gap_category(t2)[0] == 0:
                        update_usage_lists(t, t2)
                        changes += 1
                    else:
                        by_gap[g].append(t)
                else:
                    by_gap[g].append(t)
        if changes == 0:
            break

    # Fill shortages from clean / donors
    safety = 0
    while safety < 10*n:
        safety += 1
        changes = 0
        for g in range(1, MAX_NEAR_GAP+1):
            while len(by_gap[g]) < gap_targets.get(g,0):
                source = None
                if len(by_gap[0]) > gap_targets.get(0,0):
                    source = by_gap[0].pop()
                else:
                    donors = [(gg, len(by_gap[gg]) - gap_targets.get(gg,0)) for gg in range(1, MAX_NEAR_GAP+1)]
                    donors = [x for x in donors if x[1] > 0 and x[0] != g]
                    donors.sort(key=lambda x: x[1], reverse=True)
                    if donors:
                        dg,_ = donors[0]
                        source = by_gap[dg].pop()
                if source is None:
                    break
                t2 = retarget_ticket(source, g)
                if ticket_gap_category(t2)[0] == g:
                    update_usage_lists(source, t2)
                    changes += 1
                else:
                    gc,_ = ticket_gap_category(source)
                    by_gap[gc].append(source)
                    break
        if changes == 0:
            break

    out = []
    for g in [0] + list(range(1, MAX_NEAR_GAP+1)):
        out.extend(by_gap.get(g, []))

    out = [enforce_per_ticket_limit(t, buckets) for t in out]
    return out

def near_pair_mix_summary_exact(tickets):
    counts = Counter()
    for t in tickets:
        gcat, _ = ticket_gap_category(t)
        counts[gcat] += 1
    n = max(1, len(tickets))
    out = {("clean%" if g==0 else f"gap={g}%"): round(100*counts[g]/n,1) for g in sorted(counts)}
    out["counts"] = dict(counts)
    return out

# ================================
# Wheel reduction (pair coverage)
# ================================

def wheel_reduce(tickets, df_for_weights, final_k=WHEEL_TARGET):
    pair_w = defaultdict(int)
    # weight pair coverage by 180d frequency
    for row in df_for_weights[[f"N{i+1}" for i in range(K_NUMS)]].values:
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
# Adjust N2...N6 based on N1, etc
# ================================

def compute_gap_distributions(df):
    """Return empirical gap samples for N2-N1 .. N6-N5."""
    gap_dists = {}
    for i in range(1, 6):
        col1, col2 = f"N{i}", f"N{i+1}"
        gaps = (df[col2] - df[col1]).dropna()
        # only keep positive gaps (should always be since numbers are sorted)
        gaps = gaps[gaps > 0]
        gap_dists[f"{col2}-{col1}"] = gaps.tolist()
    return gap_dists

def generate_conditional_ticket(gap_distributions, base_buckets):
    """Generate one 6-number ticket using conditional gap sampling."""
    ticket = []
    # N1: choose freely from its bucket
    n1 = np.random.choice(list(base_buckets["N1"]))
    ticket.append(n1)

    # N2..N6: sample gap, add to previous
    for i in range(2, 7):
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

# ================================
# Main pipeline
# ================================

def generate_master_set(
    df,
    num_candidates=NUM_CANDIDATE_TICKETS,
    final_k=WHEEL_TARGET,
    use_wheel=USE_WHEEL,
    enable_smoothing=ENABLE_SMOOTHING,
    smooth_scope=SMOOTH_SCOPE  # "chosen" or "pool"
):
    # Buckets
    if AUTO_BUCKETS:
        buckets = derive_buckets_from_history(df)
    else:
        buckets = POS_BUCKETS

    if USE_CONDITIONAL:
        gap_dists = compute_gap_distributions(df)

    windows = make_windows(df)
    df90, df180, df365, df3y, dfall = windows["90d"], windows["180d"], windows["365d"], windows["3y"], windows["all"]

    hot, _ = hot_cold_pools(df90, buckets)
    overdue_whites = overdue_set(dfall)
    corr_map = strong_pairs(df180, threshold=ASSOC_THRESHOLD)

    # 1) Generate candidates
    candidates, seen, tries = [], set(), 0
    while len(candidates) < num_candidates and tries < num_candidates*20:
        tries += 1
        if USE_CONDITIONAL:
            t = generate_conditional_ticket(gap_dists, buckets)
        else:
            t = gen_base_ticket(buckets)
        t = apply_hot_overdue(t, hot, overdue_whites, buckets, max_hot=2, max_overdue=1)
        t = nudge_with_correlations(t, corr_map, buckets)
        t = [int(x) for x in t]  # normalize to int
        sig = tuple(t)
        if sig in seen: continue
        if not odd_even_ok(t, min_odds=2, min_evens=2):
            continue
        seen.add(sig)
        candidates.append(t)

    # 2) Optional smoothing + repair
    if enable_smoothing:
        restrict = True if str(smooth_scope).lower() == "chosen" else False
        candidates = smooth_batch(candidates, restrict_to_used=restrict, radius=SMOOTHING_RADIUS, buckets=buckets)

    validated = []
    for t in candidates:
        fixed = reorder_to_buckets(t, buckets, BUCKET_FLEX)
        if fixed: validated.append(fixed)
    candidates = validated

    # 2c) Enforce exact-gap + pair-weight quotas, repair, re-enforce, repair
    candidates = match_exact_gap_distribution(candidates, buckets)
    validated = []
    for t in candidates:
        fixed = reorder_to_buckets(t, buckets, BUCKET_FLEX)
        if fixed: validated.append(fixed)
    candidates = validated

    candidates = match_exact_gap_distribution(candidates, buckets)
    validated = []
    for t in candidates:
        fixed = reorder_to_buckets(t, buckets, BUCKET_FLEX)
        if fixed: validated.append(fixed)
    candidates = validated

    if DEBUG_NEAR_MIX:
        print("Candidate near-pair mix:", near_pair_mix_summary_exact(candidates))

    # 3) Select finals
    if use_wheel:
        final_whites = wheel_reduce(candidates, df_for_weights=df180, final_k=final_k)
    else:
        final_whites = candidates[:final_k]

    # Final: enforce quotas + repair
    final_whites = match_exact_gap_distribution(final_whites, buckets)
    final_whites = [reorder_to_buckets(t, buckets, BUCKET_FLEX) or t for t in final_whites]

    if DEBUG_NEAR_MIX:
        print("Final near-pair mix:", near_pair_mix_summary_exact(final_whites))

    # Sort whites for output
    final_sorted = [sorted(t) for t in final_whites]
    candidates_sorted = [sorted(t) for t in candidates]
    return final_sorted, candidates_sorted

# ================================
# Entry
# ================================

if __name__ == "__main__":
    random.seed()  # set a fixed int for reproducibility if you want
    if not USE_WHEEL:
        NUM_CANDIDATE_TICKETS = 5*WHEEL_TARGET
    df = load_history_csv()
    finals, pool = generate_master_set(
        df,
        num_candidates=NUM_CANDIDATE_TICKETS,
        final_k=WHEEL_TARGET,
        use_wheel=True,
        enable_smoothing=ENABLE_SMOOTHING,
        smooth_scope=SMOOTH_SCOPE,
    )
    print(f"\nGenerated {len(pool)} candidates; reduced to {len(finals)} final tickets:\n")
    for t in finals:
        print(t)
