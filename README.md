# Movie Marathon Optimizer

**Final assignment for Algorithm Analysis and Design**

This project is a small web application that selects an optimal set of movies to watch within a user-specified time limit. It was created as a final assignment for the *Algorithm Analysis and Design* course. The app implements and compares two algorithms for solving the optimization problem: a greedy heuristic and a brute force search.

## Features

* Load and preprocess an IMDb dataset.
* Compute a hybrid score for each movie combining IMDb Rating and MetaScore.
* Filter movies by genre and "family" safe certificates.
* Solve the selection problem using:

  * **Brute force** search (optimal, exponential time)
  * **Greedy** heuristic using score density (fast, near-optimal)
* Compare algorithms by execution time, memory usage, total score, and average score.
* Simple web UI to run experiments and inspect results.

## Dataset

The dataset used for this project is taken from Kaggle: [https://www.kaggle.com/datasets/parthdande/imdb-dataset-2024-updated](https://www.kaggle.com/datasets/parthdande/imdb-dataset-2024-updated)

There are three versions available on the Kaggle page. For this project I combined the three versions, removed duplicate rows, and dropped unnecessary columns to keep only the fields needed by the app. The final CSV in this repo is named `IMDb_Dataset.csv`.

## Data processing notes

* Numeric cleaning is applied to columns such as `MetaScore`, `IMDb Rating`, and `Duration (minutes)`.
* Rows with non-positive duration are removed.
* Duplicate movie titles are dropped.
* A hybrid `Score` column is computed as:

```
Score = (IMDb Rating * 0.7) + ((MetaScore / 10) * 0.3)
```

## Algorithms implemented

### Greedy

* Computes score density as `score / duration` for each movie.
* Sorts movies by density in descending order and picks movies until the time limit is reached.
* Time complexity: O(n log n) for sorting, plus O(n) for selection.
* Space complexity: O(n).

### Brute Force

* Recursive exploration of all subsets of movies, with a safety cap that reduces the input size to 22 items when the input is large.
* Guarantees optimal solution but has exponential time complexity O(2^n).
* Space complexity: O(n).

## Prerequisites

* Python 3.8 or newer
* pip installed
* Optional virtual environment recommended

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Achmad-D-Fajar/MovieMarathonOptimizer.git
cd MovieMarathonOptimizer
```

2. Create an optional virtual environment:

```bash
python -m venv venv
```

Activate it:

* MacOS or Linux:

```bash
source venv/bin/activate
```

* Windows:

```bash
venv\Scripts\activate
```

3. Install required packages:

```bash
pip install flask pandas
```

4. Ensure the dataset file `IMDb_Dataset.csv` is in the project root.

## How to run (local)

1. Clone the repository.
2. Prepare a Python environment with at least Python 3.8.
3. Install dependencies. At minimum run:

```bash
pip install flask pandas
```

4. Place `IMDb_Dataset.csv` in the project root if it is not already present.
5. Start the app:

```bash
python app.py
```

6. Open a browser and navigate to `http://127.0.0.1:5000`.

## File structure

```
MovieMarathonOptimizer/
├─ app.py                 # Flask app and algorithm implementations
├─ IMDb_Dataset.csv       # Preprocessed dataset used by the app
├─ templates/
│  └─ index.html         # Frontend UI template
└─ README.md
```

## Notes and assumptions

* The combined Kaggle dataset and the preprocessing steps are included in the repo as `IMDb_Dataset.csv`.
* The repository contains `app.py` which runs a Flask server and `templates/index.html` for the UI.

## License

This project is released under the MIT License. See `LICENSE` if you add one.

## Contact

GitHub: `Achmad-D-Fajar`

---

If you want I can also:

* Generate a `requirements.txt` file.
* Add a brief usage example showing a sample POST payload for automated testing.
* Add a short README section that explains how the dataset was combined (exact commands or scripts).
