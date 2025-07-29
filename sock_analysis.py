from collections import Counter
import numpy as np
from itertools import combinations
import statistics

# Perform ranking from combinations
def rank_combos(score_dict, combo_size=5, top_n=5):
    score_dict = {int(k): v for k, v in score_dict.items()}
    
    top_wins = sorted(score_dict.items(), key=lambda x: -x[1])[:20]
    win_pool = [win_nums for win_nums, _ in top_wins]
    combos = list(combinations(win_pool, combo_size))
    combo_scores = [(combo, sum(score_dict.get(s, 0) for s in combo)) for combo in combos]
    #print(combo_scores) # indexes with their associated data
    top_combos = sorted(combo_scores, key=lambda x: -x[1])[:top_n]
    #print(top_combos) # top_n number of indexes with their associated data
        
    if combo_size == 1:
        return [combo[0] for combo, _ in top_combos] # Flatten from [(3,), (12,)] to [3, 12]
    else:
        return [combo for combo, _ in top_combos]

# Prediction Model: Overdue
def process_overdue(df, win_history, all_nums, combo_size, top_n):
    # Overdue - Most number of days since last seen
    # not very likely to be helpful in this case, so disregard
    print("Processing Overdue...")
    overdue_scores = {}
    for each_num in all_nums:
        win_num = (("0" if each_num<10 else "") + str(each_num))
        if win_num in win_history and win_history[win_num]:
            overdue_score = len(df) - win_history[win_num][-1]
        else:
            overdue_score = len(df)
        overdue_scores[each_num] = overdue_score
    #print(f"Overdue Scores: {overdue_scores}")
    return rank_combos(overdue_scores, 1, top_n)

# Prediction Model: Overplayed
def process_overplayed(df, win_history, all_nums, combo_size, top_n):
    print("Processing Overplayed...")

    # Slice the last `window_size` rows of the dataset
    recent_df = df.tail(16)

    # Count frequency of individual sock indexes
    freq = Counter()
    for row in recent_df['win_powerballs' if combo_size == 1 else 'win_nums']:
        for sock in row.split(","):
            freq[int(sock)] += 1  # ensure sock is int
    #print(freq)
    
    # Return top N most frequent socks
    return [sock for sock, _ in freq.most_common(top_n)]

# Prediction Model: Modal Gap
def process_modal_gap(win_history, combo_size, top_n):
    print("Processing Modal Gap...")
    modal_gap_scores = {}
    for win_num, days in win_history.items():
        if len(days) > 1:
            gaps = [days[i+1] - days[i] for i in range(len(days)-1)]
            if gaps:
                most_common_gap = Counter(gaps).most_common(1)[0][1]
                modal_gap_scores[win_num] = most_common_gap
    return modal_gap_scores

# Prediction Model: Recency Weighted
def process_recency_weighted(df, win_history, all_nums, combo_size, top_n):
    print("Processing Recency Weighted...")
    recency_scores = {}
    for each_num in all_nums:
        appearances = win_history[("0" if each_num<10 else "") + str(each_num)]
        #print(f"{each_num} :: appearances: {appearances}")
        score = statistics.median(1 / (len(df) - day + 1) for day in appearances)
        recency_scores[each_num] = score
        #print(f"{each_num} :: r score: {score}")
    #print(f"recency_scores: {recency_scores}")
    return recency_scores

# Prediction Model: Bayesian
def process_bayesian(df, win_history, all_nums, combo_size, top_n):
    print("Processing Bayesian...")
    bayesian_scores = {}
    for each_num in all_nums:
        win_num = (("0" if each_num<10 else "") + str(each_num))
        appearances = win_history[win_num]
        score = len(appearances) / (sum((len(df) - day) for day in appearances) + 1)
        bayesian_scores[each_num] = score
        #print(f"{each_num} :: score: {score}")
    return bayesian_scores

# Prediction Model: Entropy
def process_entropy(win_history, all_nums, combo_size, top_n):
    print("Processing Entropy...")
    entropy_scores = {}
    for each_num in all_nums:
        win_num = (("0" if each_num<10 else "") + str(each_num))
        days = win_history[win_num]
        gaps = [days[i+1] - days[i] for i in range(len(days)-1)]
        prob_dist = [g/sum(gaps) for g in gaps] if gaps else [1]
        entropy = -sum(p * np.log2(p) for p in prob_dist)
        entropy_scores[each_num] = -entropy
        #print(f"{each_num} :: {entropy_scores[each_num]}")
    return entropy_scores

# Brains behind precition modeling
# df [DATAFRAME] - CSV table data
# win_history [LIST] - List of the occurences for each winning number/powerball
# powerballs [BOOLEAN] - True if extracting the powerballs from winning number sets, False otherwise
# range_top [INTEGER] - The maximum value in the ball pool
# top_n [INTEGER] - Number of outcomes to produce
# combo_size [INTEGER] - Desired size of the outcome
def score_socks(df, win_history, powerballs, range_top, top_n, combo_size):
    all_nums = list(range(1, range_top+1))
    results = {}
    rand_results = {}

    # Overdue - Most number of days since last seen
    rand_results['Overdue'] = process_overdue(df, win_history, all_nums, combo_size, top_n)
    #print(f"Overdue Result = {rand_results['Overdue']}")

    # Overplayed - Most number of days since last seen
    rand_results['Overplayed'] = process_overplayed(df, win_history, all_nums, combo_size, top_n)
    #print(f"Overplayed Result = {rand_results['Overplayed']}")

    # Modal Gap (typical time between reappearances)
    modal_gap_scores = process_modal_gap(win_history, combo_size, top_n)
    #print(modal_gap_scores)
    results['Modal Gap'] = rank_combos(modal_gap_scores, combo_size, top_n)
    #print(f"Modal Gap result = {results['Modal Gap']}")

    # Recency Weighted (higher value weighted toward most recently seen)
    recency_scores = process_recency_weighted(df, win_history, all_nums, combo_size, top_n)
    #print(recency_scores)
    results['Recency Weighted'] = rank_combos(recency_scores, combo_size, top_n)
    #print(f"Recency Weighted result = {results['Recency Weighted']}")
    
    # Bayesian
    bayesian_scores = process_bayesian(df, win_history, all_nums, combo_size, top_n)
    #print(bayesian_scores)
    results['Bayesian'] = rank_combos(bayesian_scores, combo_size, top_n)
    #print(f"Bayesian Result = {results['Bayesian']}")

    # Entropy
    entropy_scores = process_entropy(win_history, all_nums, combo_size, top_n)
    #print(entropy_scores)
    results['Entropy'] = rank_combos(entropy_scores, combo_size, top_n)
    #print(f"Entropy Result = {results['Entropy']}")

    return results, rand_results
