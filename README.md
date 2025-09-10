Lottery Number Generators

This project provides two configurable Python scripts for generating candidate lottery tickets with advanced strategies for coverage, frequency analysis, and bias reduction.
	•	lottox_generator_primary.py: designed for 6-ball games (e.g., LottoX with no separate Powerball).
	•	pb_generator_primary.py: designed for 5+1 games (e.g., Powerball or similar, where the 6th ball is drawn from a separate pool).

The goal of these scripts is not to predict winning numbers. All draws are random and jackpot odds cannot be improved. Instead, these scripts help generate tickets that:
	•	Resemble real historical draws (distribution buckets, realistic gaps).
	•	Provide better coverage across hot, overdue, and correlated numbers.
	•	Avoid human bias in number selection.
	•	Allow configurable strategies depending on whether you are chasing jackpot alignment, or lower-tier wins.

⸻

Installation
	1.	Clone this repository.
	2.	Install dependencies (tested with Python 3.10+):

pip install -r requirements.txt

Dependencies include:
	•	pandas
	•	numpy
	•	matplotlib (if you want graphing)
	•	PyMuPDF (only if parsing PDFs directly; not needed if you use CSV history)

⸻

Usage

Both scripts can be run directly from the command line.

python pb_generator_primary.py [history.csv]
python lottox_generator_primary.py [history.csv]

Input data
	•	If you provide a history.csv, it will be used to derive buckets, hot/overdue numbers, and gap distributions.
	•	If no CSV is provided, the script will attempt to auto-download the most recent official history data (depending on the game).

Output
	•	Candidate tickets are generated according to the configuration options.
	•	A reduced set of final tickets is returned, either directly or via wheel reduction.
	•	Optionally, plots and tables may be written to disk if configured.

⸻

Configuration Options

At the top of each script, configuration constants allow you to tune generation behavior.

General
	•	NUM_CANDIDATE_TICKETS: how many raw candidate tickets to generate before filtering and reduction.
	•	WHEEL_TARGET: how many tickets to return in the final playable set.
	•	AUTO_BUCKETS: if True, derive number ranges (N1..N6) from historical quantiles instead of fixed buckets.
	•	USE_CONDITIONAL: if True, generate numbers using gap-based conditional distributions (e.g., N2 depends on N1).

Hot/Overdue Numbers
	•	HOT_PCTL: percentile cutoff for “hot” numbers. Lower values broaden the hot pool.
	•	PB_HOT_WEIGHT, PB_OVERDUE_WEIGHT, PB_RANDOM_WEIGHT: weights that control Powerball selection strategy. Example: 50% hot, 20% overdue, 30% random.
	•	Overdue numbers are derived from draws where numbers haven’t appeared within a given lookback window (e.g., last 60 draws).

Correlations
	•	ASSOC_THRESHOLD: minimum correlation strength required for two numbers to be considered a correlated pair. Higher thresholds limit to only the strongest historical pairs.

Smoothing / Bias Reduction
	•	ENABLE_SMOOTHING: if True, the generator reduces clustering by nudging numbers to spread coverage.
	•	SMOOTH_SCOPE: "chosen" restricts smoothing within the chosen candidate set, "pool" allows smoothing across the entire pool.
	•	Tradeoff: smoothing increases diversity and lower-tier hits in backtests, but can reduce the likelihood of exact historical jackpots appearing.

Wheel Reduction
	•	USE_WHEEL: if True, final tickets are selected using a wheel system.
	•	The wheel increases coverage of 3-, 4-, and 5-hit combinations, increasing lower-tier win rates in backtests.
	•	However, the wheel may reduce the probability of exact jackpot alignment, since jackpot tickets can be excluded during coverage optimization.
	•	If False, the first WHEEL_TARGET valid candidates are returned directly.

⸻

Strategic Guidance
	•	Disable wheel and smoothing if your goal is to simulate jackpot-hunting: you want to test “pure” conditional and hot/overdue strategies.
	•	Enable wheel and smoothing if your goal is to maximize coverage and improve odds of 4- and 5-number matches.
	•	For Powerball-style games, tuning the Powerball weights prevents clustering (e.g., all tickets with PB=25).
	•	Hot/overdue injection can be effective for backtest hit rates, but overfitting them can reduce coverage.
	•	Always remember: jackpot odds remain fixed. These scripts help with coverage and bias reduction, not prediction.

⸻

Example

# Generate tickets for Powerball from a local CSV
python pb_generator_primary.py powerball_history.csv

# Generate tickets for LottoX (6-ball game), letting the script auto-download history
python lottox_generator_primary.py

Sample output (Powerball):

[4, 29, 30, 43, 62, 25]
[3, 19, 20, 52, 62, 15]
[12, 18, 37, 38, 60, 22]
...

Each row is a candidate ticket: 5 white balls sorted ascending, plus the Powerball.

⸻

Notes
	•	All number selection is random and cannot improve jackpot odds.
	•	Backtest tools included in this project can help you measure coverage and compare strategies, but they also cannot alter the fundamental odds of winning.
	•	Always play responsibly.

⸻
