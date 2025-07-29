import pandas as pd
import ast
from collections import Counter, defaultdict
import numpy as np
from itertools import combinations
from sock_analysis import score_socks
import random

secret_sauce = True

# Load data from a CSV file
# Expected column titles need to include one titled: Winning Numbers
# csv_path [STRING] - Path to the csv file
# powerballs [BOOLEAN] - True if extracting the powerballs from winning number sets, False otherwise
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
# df [DATAFRAME] - CSV table data
# separator [STRING] - Character to use to separate the numbers in the row
# powerballs [BOOLEAN] - True if extracting the powerballs from winning number sets, False otherwise
def build_win_history(df, separator, powerballs=False):
    win_history = defaultdict(list)
    for i, row in df.iterrows():
        #print(f"{i} {row} {row['win_nums']})
        for win_num in row['win_powerballs' if powerballs else 'win_nums'].split(separator):
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

    # Return the list of just winning numbers/powerball numbers
    return win_history

# Process all inputs to come up with a consensus around top_n number of outcomes
# combo_results [DICTIONARY] - Sets of chosen outcomes
# combo_size [INTEGER] - Desired size of the outcome
# top_n [INTEGER] - Number of outcomes to produce
# include_counts [BOOLEAN] - True = Showcase the number of outcomes identified by prediction models
def consensus_summary(combo_results, combo_size=5, top_n=5, include_counts=True):
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

    print("\nTop consensus sets across all prediction models:")
    if include_counts:
        for i, top_item in enumerate(top_items):
            print(f"{top_item} appeared in {supporting_data[i][1]} strategies for top prediction.")
        return top_items
    else:
        for top_item in top_items:
            print(f"{top_item} is a top predictor.")
        return top_items

# Add some randomization to the singular candidate
# top_keys [SET] - Concensus outcome
# range_top [INTEGER] - The maximum value in the ball pool
# rand_predictions [DICTIONARY] - Outcomes saved from prediction models
# top_n [INTEGER] - Number of outcomes to produce
def generate_randomized_consensus(top_keys, range_top, rand_predictions, top_n):
    global secret_sauce
    
    consensus_options = []

    if secret_sauce:
        #print(f"Overdue options ({len(rand_predictions['Overdue'])}):    {rand_predictions['Overdue']}")
        #print(f"Overplayed options ({len(rand_predictions['Overplayed'])}): {rand_predictions['Overplayed']}")
    
        while len(consensus_options) < top_n:
            new_option = randomize_consensus(top_keys, range_top, rand_predictions)
            has_duplicates = len(new_option.split('-')) != len(set(new_option.split('-')))
            if has_duplicates:
                print(f"{len(consensus_options)} :: {new_option} has duplicates: {has_duplicates}")
                exit(1)
            if (new_option not in consensus_options):
                consensus_options.append(new_option)
            
    return consensus_options

# Produce a single randomized candidate that will be aggregated by generate_randomized_consensus()
# top_keys [SET] - Concensus outcome
# range_top [INTEGER] - The maximum value in the ball pool
# rand_predictions [DICTIONARY] - Outcomes saved from prediction models
def randomize_consensus(top_keys, range_top, rand_predictions):
    # Devise the list of most overdue and overplayed numbers
    overdue_predictions = rand_predictions['Overdue']#[2:7] # trim the first 2 items and limit to next 5
    overplayed_predictions = rand_predictions['Overplayed']#[1:6] # trim the first item and limit to next 5
    #print(f"Overdue options:    {overdue_predictions}")
    #print(f"Overplayed options: {overplayed_predictions}")

    # Remove any numbers in top_keys from replacement sets to eliminate the chance of duplicate numbers in output
    for key in top_keys.split('-'):
        try:
            overdue_predictions.remove(int(key))
        except ValueError:
            pass  # do nothing!
        try:
            overplayed_predictions.remove(int(key))
        except ValueError:
            pass  # do nothing!
    
    # Randomly choose how much of top_keys to replace (up to 60%), then portion out between 2 techniques
    scope_to_replace = random.randint(1, int(combo_size*0.60))
    tech1_scope = random.randint(0, scope_to_replace)
    tech2_scope = scope_to_replace-tech1_scope
    tech1_indexes = random.sample(range(0, len(overdue_predictions)-1), tech1_scope)
    tech2_indexes = random.sample(range(0, len(overplayed_predictions)-1), tech2_scope)
    #print(f"Replace {scope_to_replace} position overall...")
    #print(f"Will replace: {tech1_scope} with Overdue values, and {tech2_scope} with Overplayed values")
    
    # Randomly choose the indexes in top_keys that we'll replace (number of indexes chosen by scope_to_replace)
    replace_indexes = random.sample(range(0, len(top_keys.split('-'))-1), scope_to_replace)
    #print(f"Will replace these indexes: {replace_indexes}")

    for i, replace_index in enumerate(replace_indexes):
        replace_this = top_keys.split('-')[replace_index]

        # Alternate which strategy will replace the next index
        if (i%2 and len(tech2_indexes)>0) or len(tech1_indexes)<=0:
            replace_to_index = tech2_indexes.pop(0)
            replace_to = overplayed_predictions[replace_to_index]
            #print(f"Replacing ({replace_this}) with Overplayed[{replace_to_index}] ({replace_to})")
            top_keys = top_keys.replace(str(replace_this), str(replace_to))
        else:
            replace_to_index = tech1_indexes.pop(0)
            replace_to = overdue_predictions[replace_to_index]
            #print(f"Replacing ({replace_this}) with Overdue[{replace_to_index}] ({replace_to})")
            top_keys = top_keys.replace(str(replace_this), str(replace_to))

    # Split, Sort, Re-join
    top_keys = '-'.join([str(i) for i in sorted([int(s) for s in top_keys.split('-')])])

    #print(f"RANDOMIZED OPTION: {top_keys}")

    return top_keys

if __name__ == "__main__":
    # CONFIGURATION
    secret_sauce = True
    RUN_BACKTEST = False
    BACKTEST_INDEX = -1
    top_n = 11

    # Powerball: (5x) 1-69 (PBs: 1-26) ... FL LottoX: (6x) 1-53
    combo_size = 5      # Number of Normal Balls (4-6)
    powerballs = True   # Use of Powerball set?
    range_top = 69      # Size of Normal Ball set
    pb_range_top = 26   # Size of Powerball set

    # LOAD CSV
    df = load_data("sock_draws.csv", powerballs)
    df_backtest = df.iloc[:BACKTEST_INDEX].copy()  # All rows except last
    actual_last_set = df.iloc[BACKTEST_INDEX]['Winning Numbers'].split(' ')
    actual_last_set_normal = actual_last_set[:-1]
    actual_last_set_pb = actual_last_set[-1]

    # PROCESS CSV THROUGH PREDICTION MODELS
    print("\nBEGINNING NORMAL BALL PREDICTION ANALYSIS...")
    history = build_win_history(df_backtest if RUN_BACKTEST else df, ',', False)
    predictions, rand_predictions = score_socks(df_backtest if RUN_BACKTEST else df, history, False, range_top, top_n, combo_size)
    #print("\nTop predictions by method:")
    #for method, combos in predictions.items():
    #    print(f"{method}: {combos}")

    # REACH SOME CONSENSUS
    final_result = consensus_summary(predictions, combo_size, top_n)
    build_concensus = Counter()
    for prediction in final_result:
        for item in prediction:
            build_concensus[item] += 1
    top_keys = '-'.join(sorted([str(item) for item, _ in build_concensus.most_common(combo_size)]))
    print(f"\nTOP CONSENSUS: {top_keys}")

    # OPTIONALLY PERFORM SOME RANDOMIZATION
    consensus_options = generate_randomized_consensus(top_keys, range_top, rand_predictions, 10 if top_n>10 else top_n)
    print(f"Randomized Concensus:")
    for i, option in enumerate(consensus_options, start=1):
        if i<10:
            print(f"{i}.  {option}")
        else:
            print(f"{i}. {option}")
    if actual_last_set_normal in consensus_options:
        print(f"BACKTEST FOUND!")
    
    if RUN_BACKTEST:
        print(f"\nBEGINNING NORMAL BALL BACKTEST ANALYSIS FOR {"-".join(actual_last_set_normal)} ({actual_last_set_pb}) ...")  
        actual = set(actual_last_set_normal)  # normalize
        hits = []

        for p in final_result[0]:
            predicted_set = {p} if isinstance(p, int) else set(p)
            if predicted_set == actual:
                hits.append(predicted_set)
        
        if hits:
            print("\n✅ BACKTEST SUCCESS: These methods predicted the final draw.")
        else:
            print("\n❌ BACKTEST FAILED: No method predicted the final draw exactly.")
    
    if powerballs:
        print("\nBEGINNING POWERBALL PREDICTION ANALYSIS...")
        combo_size = 1
        pb_history = build_win_history(df_backtest if RUN_BACKTEST else df, ',', True)
        pb_predictions, pb_rand_predictions = score_socks(df_backtest if RUN_BACKTEST else df, pb_history, True, pb_range_top, top_n, combo_size)

        #print("\nTop PowerBall predictions by method:")
        #for method, combos in pb_predictions.items():
        #    print(f"{method}: {combos}")
        pb_final_result = consensus_summary(pb_predictions, combo_size, top_n)

    if RUN_BACKTEST:
        print("\nBEGINNING POWERBALL BACKTEST ANALYSIS...")
        actual = int(actual_last_set_pb)  # normalize
        hits = []
        for p in pb_final_result[0]:
            predicted_num = int(p)
            if predicted_num == actual:
                hits.append(predicted_num)
        
        if hits:
            print("\n✅ BACKTEST SUCCESS: These methods predicted the POWERBALL draw.")
        else:
            print("\n❌ BACKTEST FAILED: No method predicted the POWERBALL draw.")
