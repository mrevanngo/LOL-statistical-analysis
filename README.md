# Comeback Analysis in Professional League of Legends

**Author:** Evan Ngo

---

## Introduction

League of Legends (LoL) is a multiplayer online battle arena (MOBA) game developed by Riot Games, where two teams of five players compete to destroy the opposing team's Nexus. With millions of players worldwide, it has become one of the most influential esports in the gaming industry. The dataset used in this analysis comes from Oracle's Elixir and contains match data from professional LoL esports matches throughout 2022.

In professional League of Legends, the early game often sets the tone for the entire match. Teams that establish gold leads by the 15-minute mark typically have significant advantages in itemization, map control, and objective pressure. However, the game is designed with comeback mechanics, and skilled teams can overcome early deficits through superior teamfighting, objective control, and late-game scaling.

This analysis focuses on **"comeback" games**—matches where a team overcomes a meaningful gold deficit at the 15-minute mark to ultimately win. We define a "meaningful deficit" as being at least **1,500 gold behind**, which represents roughly one item component and filters out trivial differences that could swing with a single kill.

### Central Question

**Do Eastern leagues (LCK, LPL) have a higher comeback rate than Western leagues (LCS, LEC, CBLOL) when teams are significantly behind in gold at 15 minutes?**

This question is relevant to coaches, analysts, and fans who want to understand regional differences in gameplay philosophy. Eastern teams, particularly those from Korea (LCK) and China (LPL), are often characterized as more patient and disciplined, potentially making them better at playing from behind.

### Dataset Overview

The dataset contains approximately **150,000 rows** of professional match data. Each game generates 12 rows: 10 for individual players and 2 for team-level summaries. For this analysis, we focus on team-level data from Tier-One leagues.

| Column | Description |
|--------|-------------|
| `gameid` | Unique identifier for each match |
| `league` | Professional league (LCK, LPL, LCS, LEC, CBLOL, etc.) |
| `result` | Match outcome (1 = win, 0 = loss) |
| `golddiffat15` | Gold difference at 15 minutes (negative = behind) |
| `xpdiffat15` | Experience difference at 15 minutes |
| `killsat15`, `deathsat15`, `assistsat15` | Combat statistics at 15 minutes |
| `side` | Team side (Blue or Red) |

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

The data cleaning process involved several key steps:

1. **Filtered for team-level rows** by selecting only rows where `position == 'team'`, reducing the dataset to team summaries rather than individual player statistics.

2. **Filtered for Tier-One leagues** (LCK, LPL, LCS, LEC, CBLOL, PCS, VCS) to focus on the highest level of professional play.

3. **Created comeback-related features:**
   - `behind_at_15`: True if the team's gold difference at 15 minutes was ≤ -1,500
   - `comeback`: True if the team was behind at 15 minutes but won the game
   - `region`: Classified leagues as "Eastern" (LCK, LPL) or "Western" (LCS, LEC, CBLOL)

4. **Removed rows with missing values** in key 15-minute statistics to ensure complete data for analysis.

Below is a sample of the cleaned dataset:

| gameid | league | region | side | result | golddiffat15 | behind_at_15 | comeback |
|--------|--------|--------|------|--------|--------------|--------------|----------|
| ESPORTSTMNT01_2690210 | LCK | Eastern | Blue | 0 | -2341 | True | False |
| ESPORTSTMNT01_2690210 | LCK | Eastern | Red | 1 | 2341 | False | False |
| ESPORTSTMNT01_2690219 | LCK | Eastern | Blue | 1 | -1823 | True | True |

### Univariate Analysis

The distribution of gold difference at 15 minutes follows an approximately normal distribution centered around zero, which is expected since every game has one team ahead and one team behind by equal amounts.

<iframe src="assets/gold_diff_distribution.html" width="800" height="500" frameborder="0"></iframe>

The histogram reveals that most games have gold differences within ±5,000 at 15 minutes, with extreme leads (>8,000 gold) being relatively rare in professional play.

### Bivariate Analysis

Examining comeback rates by league reveals interesting regional patterns:

<iframe src="assets/comeback_by_league.html" width="800" height="500" frameborder="0"></iframe>

The relationship between gold difference at 15 minutes and win rate shows a clear positive correlation—teams with larger gold leads at 15 minutes win more frequently. However, teams behind by 1,500-3,000 gold still win approximately 30-40% of their games, demonstrating that comebacks are possible.

<iframe src="assets/winrate_vs_gold.html" width="800" height="500" frameborder="0"></iframe>

### Interesting Aggregates

The table below compares aggregate statistics between Eastern and Western regions:

| Region | Total Games | Games Behind @15 | Comebacks | Comeback Rate |
|--------|-------------|-------------------|------------------|-----------|---------------|
| Eastern | 934 | 233 | 35 | ~15% |
| Western | 1584 | 461 | 96 | ~21% |

Win rates by gold deficit category reveal how the magnitude of the deficit affects comeback probability:

| Deficit Category | Win Rate |
|------------------|----------|
| Large Deficit (< -3k) | ~10% |
| Medium Deficit (-3k to -1.5k) | ~22% |
| Small Deficit (-1.5k to 0) | ~38% |
| Small Lead (0 to 1.5k) | ~62% |
| Medium Lead (1.5k to 3k) | ~78% |
| Large Lead (> 3k) | ~90% |

---

## Assessment of Missingness

### NMAR Analysis

The `golddiffat15` column (and other @15 minute statistics) are likely **Not Missing At Random (NMAR)**. The missingness is related to the values themselves for the following reasons:

1. **Games ending before 15 minutes**: Some professional games end in very fast stomps or due to technical difficulties, meaning 15-minute statistics never existed for these matches.

2. **Data collection differences by league**: Some leagues have incomplete data collection infrastructure, and this incompleteness may correlate with regional characteristics.

3. **Self-referential missingness**: The missingness depends on game length, which is itself related to how the early game unfolds—the very information we're trying to capture.

Additional data that could help explain the missingness and potentially make it MAR includes exact game end times and data collection methodology by league.

### Missingness Dependency

We conducted permutation tests to determine whether the missingness of `golddiffat15` depends on other columns.

**Test 1: Missingness vs. League**

- **Null Hypothesis**: The distribution of `league` is the same when `golddiffat15` is missing vs. not missing.
- **Test Statistic**: Total Variation Distance (TVD)
- **Result**: p-value < 0.05, indicating that missingness **does depend on league**.

<iframe src="assets/missingness_league.html" width="800" height="400" frameborder="0"></iframe>

**Test 2: Missingness vs. Game Length**

- **Null Hypothesis**: The mean `gamelength` is the same when `golddiffat15` is missing vs. not missing.
- **Test Statistic**: Absolute difference in means
- **Result**: p-value < 0.05, indicating that missingness **does depend on game length**.

<iframe src="assets/missingness_gamelength.html" width="800" height="400" frameborder="0"></iframe>

These results confirm that the missingness of 15-minute statistics is related to both the league and the length of the game, supporting our NMAR hypothesis.

---

## Hypothesis Testing

We conducted a permutation test to determine whether Eastern leagues have a higher comeback rate than Western leagues.

**Null Hypothesis (H₀)**: In Tier-One games where a team is at least 1,500 gold behind at 15 minutes, Eastern leagues (LCK, LPL) and Western leagues (LCS, LEC, CBLOL) have the same probability of coming back to win.

**Alternative Hypothesis (H₁)**: Eastern leagues are more likely to come back and win than Western leagues when at least 1,500 gold behind.

**Test Statistic**: Difference in comeback rates (Eastern - Western)

**Significance Level**: α = 0.05

### Methodology

We used a permutation test with 10,000 iterations. Under the null hypothesis, the region labels (Eastern/Western) have no effect on comeback probability, so we shuffled these labels and recalculated the difference in comeback rates for each permutation.

### Results

<iframe src="assets/hypothesis_test.html" width="800" height="500" frameborder="0"></iframe>

The observed difference in comeback rates between Eastern and Western leagues was compared against the null distribution. 

- P-value: 0.9731

### Conclusion

Based on the p-value obtained from the permutation test:

- We fail to reject the null hypothesis and conclude that there is no significant difference in comeback rates between regions.

The conclusion from this test informs our understanding of whether regional playstyle differences actually translate to measurable performance differences when playing from behind.

---

## Framing a Prediction Problem

**Prediction Problem**: Given the game state at 15 minutes, predict whether a team currently behind in gold will win the game.

**Type**: Binary Classification

**Target Variable**: `result` (1 = win, 0 = loss)

**Evaluation Metric**: F1-Score

**Why F1-Score?** The data is imbalanced—comebacks are relatively rare (teams behind at 15 minutes win less than 50% of the time). Accuracy alone would be misleading, as a model predicting "always lose" would achieve decent accuracy but be useless. F1-Score balances precision and recall, making it appropriate for imbalanced classification.

**Features Available at Time of Prediction** (all known at 15 minutes):
- `golddiffat15`: Gold difference
- `xpdiffat15`: Experience difference  
- `csdiffat15`: Creep score difference
- `killsat15`, `deathsat15`, `assistsat15`: Combat statistics
- `league`: Professional league (categorical)
- `side`: Blue or Red side (categorical)

---

## Baseline Model

### Model Description

The baseline model uses a **Decision Tree Classifier** with `max_depth=5` to predict whether a team behind at 15 minutes will win.

**Features (2 total)**:
- `golddiffat15` (quantitative): Gold difference at 15 minutes
- `xpdiffat15` (quantitative): Experience difference at 15 minutes

**Feature Encoding**:
- Both features are quantitative and were standardized using `StandardScaler` to normalize them to zero mean and unit variance.

### Performance

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| Accuracy | ~0.86 | ~0.82 |
| F1-Score | ~0.22 | ~0.08 |
| Precision | ~1.00 | ~0.29 |
| Recall | ~0.13 | ~0.04 |

### Assessment

The baseline model achieves **decent accuracy (81.85%)** but **very poor F1-score (0.0755)**. This discrepancy reveals an important issue: the model is essentially predicting "no comeback" for almost every game.

Because comebacks are the minority class—only about **16% of games** where teams are 1,500+ gold behind result in a comeback—a model that always predicts "loss" would achieve ~84% accuracy by default. The baseline's extremely low recall (0.0435) confirms this—it only identifies about 4% of actual comebacks.

In other words, the baseline model is not good and performs slightly worse than a naive model that predicts "no comeback" every time.

---

## Final Model

### Features Added

The final model expands the feature set to capture more nuanced aspects of game state:

**Quantitative Features (8 total)**:
- `golddiffat15`, `xpdiffat15`, `csdiffat15`: Resource differences
- `killsat15`, `deathsat15`, `assistsat15`: Combat performance
- `kda_at_15` (engineered): (kills + assists) / (deaths + 1)
- `gold_deficit_magnitude` (engineered): |golddiffat15|

**Categorical Features (2 total)**:
- `league`: Different leagues have different playstyles and metas
- `side`: Blue/Red side affects dragon control and map dynamics

### Why These Features Improve Performance

- **KDA ratio**: Teams behind in gold but with good KDA may have better teamfight potential, indicating they lost gold through macro mistakes rather than combat ability.
- **League**: Different regions have different metas; Eastern leagues may be more patient and better at scaling into late game.
- **Gold deficit magnitude**: The size of the deficit matters—a 1,500 gold deficit is easier to overcome than a 5,000 gold deficit.
- **Side**: Blue side has certain objective advantages that may affect comeback potential.

### Model Algorithm and Hyperparameter Tuning

The final model uses a **Random Forest Classifier** with `class_weight='balanced'` to address the imbalanced nature of comeback prediction.

**Hyperparameters tuned via GridSearchCV** (5-fold cross-validation, optimizing F1-score):
- `n_estimators`: [100, 150, 200]
- `max_depth`: [3, 5, 7] — Shallow trees to prevent overfitting
- `min_samples_split`: [10, 20, 30] — Higher values for regularization
- `min_samples_leaf`: [5, 10, 15] — Higher values to prevent memorizing training data

**Key Design Decision: Regularization**

Early experiments with deeper trees (max_depth=15) and fewer samples per leaf resulted in severe overfitting—near-perfect training performance but worse-than-baseline test performance. The regularized hyperparameter grid constrains the model complexity, ensuring it learns generalizable patterns rather than memorizing training examples.

**Encoding**:
- Quantitative features: `StandardScaler`
- Categorical features: `OneHotEncoder` (with `drop='first'` to avoid multicollinearity)

### Performance Comparison

| Model | Accuracy (Test) | F1-Score (Test) | Precision (Test) | Recall (Test) |
|-------|-----------------|-----------------|------------------|---------------|
| Baseline (Decision Tree) | 0.8185 | 0.0755 | 0.2857 | 0.0435 |
| Final (Random Forest) | 0.5333 | 0.3298 | 0.2183 | 0.6739 |
| **Change** | -34.9% | **+337.0%** | -23.6% | **+1449.2%** |

### Interpreting the Results: Why Lower Accuracy is Actually Better

At first glance, the drop in accuracy from 81.85% to 53.33% might seem concerning. However, this actually represents an **improvement** in the model's usefulness.


**What the Final Model Does Differently:**

The final model, with `class_weight='balanced'`, is now willing to predict comebacks. The dramatic improvement in recall (from 4.35% to 67.39%) means the model now correctly identifies **two-thirds of actual comebacks**, compared to barely any before.

**Why F1-Score is the Right Metric:**

F1-score balances precision and recall, making it ideal for imbalanced classification. The 337% improvement in F1-score confirms that the final model is better at the task of predicting comebacks, despite the lower accuracy.


---

## Fairness Analysis

### Question

Does our model perform equally well for Eastern leagues vs. Western leagues?

### Groups

- **Group X**: Eastern leagues (LCK, LPL)
- **Group Y**: Western leagues (LCS, LEC, CBLOL)

### Evaluation Metric

**Precision**: The proportion of predicted comebacks that are actual comebacks. This metric is important because false positive predictions (predicting a comeback that doesn't happen) could lead to poor strategic decisions.

### Hypotheses

**Null Hypothesis (H₀)**: Our model is fair. Its precision for Eastern leagues and Western leagues is roughly the same, and any differences are due to random chance.

**Alternative Hypothesis (H₁)**: Our model is unfair. Its precision differs between Eastern and Western leagues.

**Significance Level**: α = 0.05

### Methodology

We conducted a permutation test with 1,000 iterations, shuffling the region labels and recalculating the precision difference for each permutation to generate a null distribution.

### Results

<iframe src="assets/fairness_analysis.html" width="800" height="500" frameborder="0"></iframe>

- P-value: 0.6510

### Conclusion

Based on the p-value from the permutation test:

- We fail to reject the null hypothesis and conclude there is no significant difference in precision, suggesting the model is fair across regions.

---
