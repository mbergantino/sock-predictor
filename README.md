# Lottery Number Generators

This project provides two configurable Python scripts for generating lottery tickets using advanced strategies for coverage, bias reduction, and historical analysis.

- `lottox_generator_primary.py` — designed for **6-ball games** (e.g., LottoX with no separate Powerball).
- `pb_generator_primary.py` — designed for **5+1 games** (e.g., Powerball, where the 6th ball is drawn from a separate pool).

These scripts do **not** predict winning numbers. Jackpot odds are fixed and cannot be improved. The purpose of this project is to:

- Generate tickets that resemble real historical draws (distribution buckets, realistic gaps).
- Provide broader coverage across hot, overdue, and correlated numbers.
- Avoid human bias in manual number selection.
- Allow configurable strategies depending on whether you want to maximize jackpot alignment or increase lower-tier wins.

---

## Installation

Clone this repository and install dependencies:

```bash
pip install -r requirements.txt
````

Required packages:

* `pandas`
* `numpy`
* `matplotlib` (for optional graphing)
* `PyMuPDF` (only required if parsing PDF draw histories)

---

## Usage

Run either script from the command line:

```bash
python pb_generator_primary.py [history.csv]
python lottox_generator_primary.py [history.csv]
```

### Input Data

* If a `history.csv` file is provided, it is used to derive buckets, hot/overdue numbers, and gap distributions.
* If no CSV is provided, the script will **auto-download** the most recent official history file for the game.

### Output

* Candidate tickets are generated according to the configured strategy.
* A reduced final set is selected, either directly or through wheel reduction.
* Optionally, plots and analysis tables may also be produced.

---

## Configuration Options

All configuration is set in constants at the top of each script.

### General

* `NUM_CANDIDATE_TICKETS`
  Number of raw candidate tickets to generate before filtering and reduction.
* `WHEEL_TARGET`
  Number of tickets to return in the final playable set.
* `AUTO_BUCKETS`
  If `True`, derive positional ranges (N1..N6) from historical quantiles instead of fixed buckets.
* `USE_CONDITIONAL`
  If `True`, use gap-based conditional distributions (e.g., N2 depends on N1).

### Hot and Overdue Numbers

* `HOT_PCTL`
  Percentile cutoff for hot numbers. Lower values broaden the pool considered “hot.”
* `PB_HOT_WEIGHT`, `PB_OVERDUE_WEIGHT`, `PB_RANDOM_WEIGHT`
  Control how the Powerball is chosen:

  * Example: 50% hot, 20% overdue, 30% random.
* Overdue numbers are based on how long it has been since they last appeared.

### Correlations

* `ASSOC_THRESHOLD`
  Minimum co-occurrence strength for two numbers to be considered a correlated pair.

### Smoothing (Bias Reduction)

* `ENABLE_SMOOTHING`
  If `True`, numbers may be nudged to reduce clustering and spread coverage.
* `SMOOTH_SCOPE`

  * `"chosen"` = smooth only within the chosen candidate set.
  * `"pool"` = smooth across the entire candidate pool.

**Tradeoff:**

* **Chosen smoothing** makes only small local adjustments. This preserves jackpot potential because tickets remain close to their original form, but clustering may remain.
* **Pool smoothing** improves diversity and coverage across many tickets, which can increase lower-tier wins (hit\_3/hit\_4/hit\_5) in backtests. However, it may push tickets away from exact jackpot alignments.

### Wheel Reduction

* `USE_WHEEL`
  If `True`, applies wheel reduction to maximize pair/triple coverage.
  If `False`, the first `WHEEL_TARGET` valid candidates are returned directly.

| Option         | Lower-tier hits | Jackpot alignment |
| -------------- | --------------- | ----------------- |
| Wheel Enabled  | Higher          | Possibly reduced  |
| Wheel Disabled | Lower           | Preserved         |

---

## Strategic Guidance

* **Disable wheel and smoothing** if your focus is on jackpot-hunting simulations.
* **Enable wheel and smoothing** if you want better odds of 4- or 5-number matches.
* For Powerball-style games, tune Powerball weights to prevent clustering (e.g., many tickets ending with the same PB).
* Use hot/overdue carefully: too narrow a pool can cause repetition, too broad a pool dilutes the effect.
* Remember: jackpot odds remain fixed. These tools are for coverage and exploration, not prediction.

---

## Examples

Generate Powerball tickets from a local CSV:

```bash
python pb_generator_primary.py powerball_history.csv
```

Generate LottoX tickets (6-ball game), auto-downloading the latest history:

```bash
python lottox_generator_primary.py
```

Example output (Powerball):

```
[4, 29, 30, 43, 62, 25]
[3, 19, 20, 52, 62, 15]
[12, 18, 37, 38, 60, 22]
```

Each row is one ticket: 5 sorted white balls, followed by the Powerball.

---

## Jackpot-Hunting configuration options

```
USE_WHEEL = False         # Don’t lose jackpot-aligned tickets
ENABLE_SMOOTHING = False  # Jury is out on, but if enabled use SMOOTH_SCOPE = "chosen"
SMOOTH_SCOPE = "chosen"   # ignored if smoothing is off

USE_CONDITIONAL = True    # Keeps your generated tickets in the “shape” of real jackpots
AUTO_BUCKETS = True

HOT_PCTL = 0.67           # Add variety without collapsing coverage
ASSOC_THRESHOLD = 0.67

PB_HOT_WEIGHT = 0.3       # Avoid clustering on a single PB
PB_OVERDUE_WEIGHT = 0.1
PB_RANDOM_WEIGHT = 0.6
```

Or to increase your odds of *ANY* win can use settings:
```
USE_WHEEL = True         # Don’t lose jackpot-aligned tickets
ENABLE_SMOOTHING = True  # Jury is out on, but if enabled use SMOOTH_SCOPE = "chosen"
SMOOTH_SCOPE = "pool"    # ignored if smoothing is off
```
---

## Notes

* These scripts cannot alter the true odds of winning a jackpot.
* Backtesting tools included in the project can help you measure coverage and compare strategies, but jackpot probability remains unchanged.
* Use responsibly.

---
