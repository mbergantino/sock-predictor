
import pandas as pd
import ast
from collections import Counter, defaultdict
import numpy as np
from itertools import combinations

def load_data(csv_path, powerballs=False):
    # Limit data to just important columns
    df = pd.read_csv(csv_path)

    # Process column: df['column_name'] & transform everything into a python string literal
    df['win_nums'] = df['Winning Numbers']
    if (powerballs):
        split_series = df['Winning Numbers'].str.split(' ')
        df['win_nums'] = df['Winning Numbers'].str.split(' ').apply(lambda x: ','.join(x[:-1]))
        df['win_powerballs'] = split_series.str[-1]

    # Set display option to show full column width
    pd.set_option('display.max_colwidth', None)


    ''' *** CHECKPOINT ***
    # Print specific columns
    if (powerballs):
        print("Winning Number & Powerball Data:")
        print(df[['Draw Date', 'win_nums', 'win_powerballs']])
    else:
        print("Winning Number Data:")
        print(df[['Draw Date', 'win_nums']])
    '''
    
    # return the dataframe (2D table)
    return df

# Compile a list of the occurences for each winning number/powerball
def build_win_history(df, powerballs=False):
    win_history = defaultdict(list)
    for i, row in df.iterrows():
        #print(i)
        #print(row)
        #print(row['win_nums'])
        for win_num in row['win_powerballs' if powerballs else 'win_nums'].split(','):
            win_history[win_num].append(i)

    ''' *** CHECKPOINT ***
    # Print specific columns
    if (powerballs):
        print("Powerball Summary:")
        print(win_history['01'])
    else:
        print("Winning Number Summary:")
        print(win_history['01'])
    '''

    return win_history

def rank_combos(score_dict, combo_size, top_n):
    score_dict = {int(k): v for k, v in score_dict.items()}
    
    top_wins = sorted(score_dict.items(), key=lambda x: -x[1])[:20]
    win_pool = [win_nums for win_nums, _ in top_wins]
    combos = list(combinations(win_pool, combo_size))
    combo_scores = [(combo, sum(score_dict.get(s, 0) for s in combo)) for combo in combos]
    top_combos = sorted(combo_scores, key=lambda x: -x[1])[:top_n]
    
    if combo_size == 1:
        return sorted([combo[0] for combo, _ in top_combos]) # Flatten from [(3,), (12,)] to [3, 12]
    else:
        return [combo for combo, _ in top_combos]

# range_top: Powerball: 1-69, 1-26 ... FL LottoX: 1-53
def score_wins(df, win_history, powerballs, range_top, top_n, combo_size):
    all_nums = list(range(1, range_top+1))
    results = {}

    '''
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
        #print(f"{win_num} :: {overdue_score}")
    #print(f"Overdue Scores: {overdue_scores}")
    # Disregarding for sets as the likelihood of multiple overdue indexes hitting simultaneously likely grows exponentially
    if (combo_size == 1):
        results['Overdue'] = rank_combos(overdue_scores, 1, top_n)
    #print(f"Overdue Result = {results['Overdue']}")
    '''

    '''
    # Transitions - what index tends to follow another (ideally suited for single number sets)
    # not very likely to be helpful in this case, so disregard
    print("Processing Transitions...")
    transitions = defaultdict(Counter)
    for i in range(len(df)-1):
        current = df.iloc[i]['win_powerballs' if powerballs else 'win_nums']
        next_day = df.iloc[i+1]['win_powerballs' if powerballs else 'win_nums']
        for c in current:
            for n in next_day:
                transitions[c][n] += 1
    transition_scores = {win_num: transitions[win_num].most_common(1)[0][1] if transitions[win_num] else 0 for win_num in all_nums}
    if combo_size == 1:
        results['Transitions'] = rank_combos(transition_scores, combo_size, top_n)
    '''

    # Modal Gap
    print("Processing Modal Gap...")
    modal_gap_scores = {}
    for win_num, days in win_history.items():
        if len(days) > 1:
            gaps = [days[i+1] - days[i] for i in range(len(days)-1)]
            if gaps:
                most_common_gap = Counter(gaps).most_common(1)[0][1]
                modal_gap_scores[win_num] = most_common_gap
    results['Modal Gap'] = rank_combos(modal_gap_scores, combo_size, top_n)
    print(f"Modal Gap result = {results['Modal Gap']}")

    '''
    # Recency Weighted
    print("Processing Recency Weighted...")
    recency_scores = {}
    for each_num in all_nums:
        appearances = win_history[("0" if each_num<10 else "") + str(each_num)]
        score = sum(1 / (len(df) - day + 1) for day in appearances)
        recency_scores[each_num] = score
        #print(f"{each_num} :: score: {score}")
    #print(f"recency_scores: {recency_scores}")
    results['Recency Weighted'] = rank_combos(recency_scores, combo_size, top_n)
    #print(f"Recency Weighted result = {results['Recency Weighted']}") 
    '''
    
    # Bayesian
    print("Processing Bayesian...")
    bayesian_scores = {}
    for each_num in all_nums:
        win_num = (("0" if each_num<10 else "") + str(each_num))
        appearances = win_history[win_num]
        score = len(appearances) / (sum((len(df) - day) for day in appearances) + 1)
        bayesian_scores[each_num] = score
        #print(f"{each_num} :: score: {score}")
    results['Bayesian'] = rank_combos(bayesian_scores, combo_size, top_n)
    print(f"Bayesian Result = {results['Bayesian']}")

    '''
    # Stable Gap
    print("Processing Stable Gap...")
    gap_variance_scores = {}
    for win_num, days in win_history.items():
        if len(days) > 1:
            gaps = [days[i+1] - days[i] for i in range(len(days)-1)]
            variance = np.var(gaps) if gaps else float('inf')
            gap_variance_scores[win_num] = -variance
            #print(f"{win_num} :: {gap_variance_scores[win_num]}")
            #else:
            #print(f"{win_num} :: {len(days)} ! <1")
    results['Stable Gap'] = rank_combos(gap_variance_scores, combo_size, top_n)    #print(f"Stable Gap Result = {results['Stable Gap']}")
    '''

    '''
    # Last 30 Days
    print("Processing Last 30 Days...")
    window_freq_scores = Counter()
    for i in range(len(df) - 30, len(df)):
        win_nums = df.iloc[i]['win_powerballs' if powerballs else 'win_nums']
        for win_num in win_nums.split(','):
            window_freq_scores[win_num] += 1
    for each_num in all_nums:
        win_num = (("0" if each_num<10 else "") + str(each_num))
        #print(f"{each_num} :: {window_freq_scores[win_num]}")
    results['Last 30 Days'] = rank_combos(window_freq_scores, combo_size, top_n)
    #print(f"Last 30 Days Result = {results['Last 30 Days']}")
    '''
    
    # Entropy
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
    results['Entropy'] = rank_combos(entropy_scores, combo_size, top_n)
    print(f"Entropy Result = {results['Entropy']}")

    return results

def consensus_summary(combo_results, combo_size=5, top_n=5, include_counts=True):
    from collections import Counter

    combo_counter = Counter()
    all_sock_occurrences = Counter()

    for combo_list in combo_results.values():
        for combo in combo_list:
            if isinstance(combo, int):
                key = (combo,)
                combo_counter[key] += 1
                all_sock_occurrences[combo] += 1
            else:
                key = tuple(sorted(combo))
                combo_counter[key] += 1
                for sock in key:
                    all_sock_occurrences[sock] += 1

    # Sort by frequency
    top_combo_pairs = combo_counter.most_common(top_n)

    if combo_size == 1:
        # Flatten to just list of ints or also return counts
        top_items = [k[0] for k, _ in top_combo_pairs]
        supporting_data = [(k[0], v) for k, v in top_combo_pairs]
    else:
        top_items = [k for k, _ in top_combo_pairs]
        supporting_data = top_combo_pairs

    print("\nTop consensus sets across methods:")
    if include_counts:
        for i, top_item in enumerate(top_items):
            print(f"{top_item} appeared in {supporting_data[i][1]} strategies for top prediction.")
        return top_items, supporting_data
    else:
        for top_item in top_items:
            print(f"{top_item} is a top predictor.")
        return top_items

'''
    top_sets = combo_counter.most_common(top_n)

    if all(freq == 1 for _, freq in top_sets):  # no repeats
        print("\nNo repeated sets found. Falling back to most frequent individual winning number indexes.")
        top_wins = [win_num for win_num, _ in all_win_occurrences.most_common(combo_size)]
        reconstructed_set = tuple(sorted(int(win_num) for win_num in top_wins))
        print(f"Reconstructed top set from individual win consensus: {reconstructed_set}")
        return top_sets, reconstructed_set
    else:
        return top_sets, None
    '''

if __name__ == "__main__":
    RUN_BACKTEST = False
    top_n = 11

    # Powerball: 1-69 (PBs: 1-26) ... FL LottoX: 1-53
    combo_size = 5      # Number of Normal Balls
    powerballs = True   # Use of Powerball set?
    range_top = 69      # Size of Normal Ball set
    pb_range_top = 26   # Size of Powerball set
    
    df = load_data("sock_draws.csv", powerballs)
    df_backtest = df.iloc[:-1].copy()  # All rows except last
    actual_last_set = df.iloc[-1]['Winning Numbers'].split(' ')
    actual_last_set_normal = actual_last_set[:-1]
    actual_last_set_pb = actual_last_set[-1]

    print("\nBEGINNING NORMAL BALL PREDICTION ANALYSIS...")
    history = build_win_history(df_backtest if RUN_BACKTEST else df, False)
    predictions = score_wins(df_backtest if RUN_BACKTEST else df, history, False, range_top, top_n, combo_size)
    #print("\nTop predictions by method:")
    #for method, combos in predictions.items():
    #print(f"{method}: {combos}")
    final_result = consensus_summary(predictions, combo_size, top_n)

    if RUN_BACKTEST:
        print(f"\nBEGINNING NORMAL BALL BACKTEST ANALYSIS FOR {"-".join(actual_last_set_normal)} ({actual_last_set_pb}) ...")  
        actual = set(actual_last_set_normal)  # normalize
        hits = []

        #print(f"{actual_last_set_normal}")
        for p in final_result[0]:
            #print(f"p :: {p}")
            predicted_set = {p} if isinstance(p, int) else set(p)
            #print(f"predicted_set :: {predicted_set}")
            if predicted_set == actual:
                hits.append(predicted_set)
        
        if hits:
            print("\n✅ BACKTEST SUCCESS: These methods predicted the final draw.")
        else:
            print("\n❌ BACKTEST FAILED: No method predicted the final draw exactly.")
    
    if powerballs:
        print("\nBEGINNING POWERBALL PREDICTION ANALYSIS...")
        combo_size = 1
        pb_history = build_win_history(df_backtest if RUN_BACKTEST else df, True)
        pb_predictions = score_wins(df_backtest if RUN_BACKTEST else df, pb_history, True, pb_range_top, top_n, combo_size)

        print("\nTop PowerBall predictions by method:")
        for method, combos in pb_predictions.items():
            print(f"{method}: {combos}")
        pb_final_result = consensus_summary(pb_predictions, combo_size, top_n)

    if RUN_BACKTEST:
        print("\nBEGINNING POWERBALL BACKTEST ANALYSIS...")
        actual = int(actual_last_set_pb)  # normalize
        #print(f"{actual_last_set_pb}")
        hits = []
        for p in pb_final_result[0]:
            predicted_num = int(p)
            #print(f"predicted_num :: {predicted_num}")
            if predicted_num == actual:
                hits.append(predicted_num)
        
        if hits:
            print("\n✅ BACKTEST SUCCESS: These methods predicted the POWERBALL draw.")
        else:
            print("\n❌ BACKTEST FAILED: No method predicted the POWERBALL draw.")
