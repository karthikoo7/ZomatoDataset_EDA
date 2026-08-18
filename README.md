# Zomato Bangalore Restaurant Reviews — Data Wrangling & EDA

This repo explores the [Zomato Bangalore Restaurants dataset](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants), which lists ~51,000 restaurant records for Bangalore along with nested customer reviews, ratings, cuisines, cost, and location data. The goal is to clean the raw, messy dataset into an analysis-ready form and surface the key patterns in restaurant and review ratings before moving on to modeling (sentiment analysis, recommendation, etc.).

## Notebooks

| Notebook | Purpose |
|---|---|
| [`DataWrangling.ipynb`](./DataWrangling.ipynb) | First-pass parsing of the raw dataset: unpacks nested review data, cleans text, resolves duplicate restaurant entries, and produces an initial ratings distribution plot. |
| [`BasicEDA.ipynb`](./BasicEDA.ipynb) | A more rigorous, expanded pass over the same problem — stricter uniqueness checks, full rating-column cleanup, dtype casting, and statistical testing on the rating distributions. |

## What's in the raw data

The source `zomato.csv` has one row per restaurant, with a `reviews_list` column containing a **string-encoded list of tuples**, each tuple being `(rating, review_text)` for a single customer review. Before any analysis is possible, this needs to be exploded out into one row per review.

## Core steps & concepts

### 1. Unpacking nested reviews
`reviews_list` is stored as a Python-literal string (e.g. `"[('Rated 4.0', 'RATED\\n  Good food...'), ...]"`), so each entry is parsed with `eval()` and iterated to build a flat `rating_df` with one row per individual review, carrying over the parent restaurant's name, address, location, and overall rating. `tqdm` is used to track progress over the ~51K-row loop.

### 2. Text cleaning
Raw review text contains boilerplate (`"RATED\n"` prefixes) and junk characters. Both notebooks strip the `"RATED"` prefix and use a regex (`[^a-zA-Z0-9\s]`) to remove non-alphanumeric characters, leaving clean text suitable for downstream NLP.

### 3. Resolving duplicate restaurant entries
A key data-quality issue: the same restaurant name (e.g. *"Jalsa"*) appears multiple times in the dataset because Zomato lists each **branch** as a separate listing, even when they share a URL. This is diagnosed by:
- Inspecting non-contiguous index ranges for a single restaurant name.
- Cross-checking `location` and `address` columns to confirm multiple physical branches exist.
- Verifying that two rows with the same name can even point to the same source URL (a Zomato quirk), which rules out URL as a reliable uniqueness key.

**Resolution differs slightly across the two notebooks:**
- `DataWrangling.ipynb` tags each row uniquely as `name + location`.
- `BasicEDA.ipynb` refines this further and tags by `name + address`, since address is a stricter, branch-level identifier than the broader `location` field (locality/area).

### 4. Rating column cleanup (`BasicEDA.ipynb`)
The `restaurant_rating` column (Zomato's aggregate score, distinct from the individual `review_rating`) needed several cleanup passes:
- Dropped rows where the restaurant is unrated (`'NEW'` or `'-'`) or null.
- Stripped the `"/5"` suffix and stray whitespace.
- Cast both `review_rating` and `restaurant_rating` to numeric dtypes for statistical analysis.
- Spot-checked that each restaurant branch (by the `name + address` tag) maps to a single consistent rating, confirming the tagging logic didn't introduce inconsistencies.

### 5. Exploratory analysis
- **Distribution plots**: bar/histogram plots of review ratings show the shape of customer sentiment across the dataset.
- **Normality testing**: `scipy.stats.normaltest` is run on both `review_rating` and `restaurant_rating`. Both return p-values of 0, indicating the ratings are **not normally distributed** — consistent with the visibly skewed histograms (most reviews cluster at the higher end of the scale).
- **Correlation check**: `rating_df.corr()` shows **no meaningful correlation** between an individual review's rating and the restaurant's overall aggregate rating — i.e., a single review's score doesn't reliably predict the restaurant's broader reputation.

### 6. Early sentiment exploration (`DataWrangling.ipynb`)
A first look at `TextBlob` sentiment scoring is applied to a handful of cleaned reviews, printing polarity/subjectivity alongside the review's star rating as a sanity check for later sentiment-analysis work.

## Key takeaways

- Restaurant-level data needs careful de-duplication before analysis — naive grouping by `name` alone silently merges distinct branches.
- `address` is a more reliable uniqueness key than `location`, which represents a broader locality shared by many restaurants.
- Both rating distributions are non-normal and right-skewed, so parametric statistical methods that assume normality aren't appropriate without transformation.
- Review-level and restaurant-level ratings are essentially uncorrelated, suggesting they capture different signals (a single reviewer's experience vs. the restaurant's aggregate reputation).

## Tech stack

`pandas`, `numpy`, `seaborn`, `matplotlib`, `scipy.stats`, `tqdm`, `re`, `textblob`

