from flask import Flask, render_template, request
import pandas as pd
import time
import re
import os
import inspect
import tracemalloc

app = Flask(__name__)

# --- CONFIGURATION ---
DATASET_FILE = 'IMDb_Dataset.csv'

# --- DATA LOADING ---
def get_data():
    if not os.path.exists(DATASET_FILE):
        return None
    try:
        df = pd.read_csv(DATASET_FILE)
        # Numeric Cleaning
        for col in ['MetaScore', 'IMDb Rating', 'Duration (minutes)']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df = df[df['Duration (minutes)'] > 0]
        df = df.drop_duplicates(subset=['Title'])
        # Hybrid Score
        df['Score'] = (df['IMDb Rating'] * 0.7) + ((df['MetaScore'] / 10) * 0.3)
        if 'Genre' in df.columns:
            df['Genre'] = df['Genre'].astype(str).str.strip()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# --- ALGORITHMS ---
def solve_greedy(items, limit):
    tracemalloc.start()
    start_time = time.time()
    
    for item in items:
        item['density'] = item['score'] / item['duration'] if item['duration'] > 0 else 0
    items.sort(key=lambda x: x['density'], reverse=True)
    
    selected = []
    current_time = 0
    total_score = 0
    for item in items:
        if current_time + item['duration'] <= limit:
            current_time += item['duration']
            total_score += item['score']
            selected.append(item)
            
    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return selected, total_score, (end_time - start_time), (peak_mem / 1024 / 1024)

def solve_brute_force(items, limit):
    tracemalloc.start()
    start_time = time.time()
    n = len(items)
    
    # Safety cap for recursion depth
    if n > 22:
        items.sort(key=lambda x: x['score'], reverse=True)
        items = items[:22]
        n = 22

    best_val = 0
    best_combo = []

    def recurse(idx, current_w, current_v, current_path):
        nonlocal best_val, best_combo
        if idx == n:
            if current_v > best_val:
                best_val = current_v
                best_combo = current_path[:]
            return

        if current_w > limit: return

        # Include
        if current_w + items[idx]['duration'] <= limit:
            recurse(idx + 1, current_w + items[idx]['duration'], current_v + items[idx]['score'], current_path + [items[idx]])
        # Exclude
        recurse(idx + 1, current_w, current_v, current_path)

    recurse(0, 0, 0, [])
    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return best_combo, best_val, (end_time - start_time), (peak_mem / 1024 / 1024)

# --- ANALYSIS ENGINE ---
def generate_analysis(bf_data, gr_data):
    # Metrics
    speedup = (bf_data['time'] / gr_data['time']) if gr_data['time'] > 0 else 0
    score_diff = bf_data['score'] - gr_data['score']
    
    # Percentages
    accuracy = (gr_data['score'] / bf_data['score'] * 100) if bf_data['score'] > 0 else 0
    diff_percent = (score_diff / bf_data['score'] * 100) if bf_data['score'] > 0 else 0

    # Text Generation
    if score_diff <= 0.01:
        optimal_text = "Optimalily: Brute Force is Always Optimal, Greedy Not Guaranteed"
        tradeoff_text = f"Speed: Greedy was {speedup:.1f}x faster using all films, Brute Force is Slower with only 22 films."
        rec_text = "Recommendation: Choose Greedy for this dataset size (Optimal & Faster with exellent scalability)."
    else:
        optimal_text = f"Optimal vs Heuristic: Brute Force found a better solution by {score_diff:.2f} points ({diff_percent:.1f}% improvement)."
        tradeoff_text = f"Trade-off Analysis: Greedy was {speedup:.1f}x faster but achieved {accuracy:.1f}% of the optimal score."
        rec_text = "Recommendation: Choose Brute Force for optimal results with smaller datasets, or Greedy for near-optimal results with larger datasets requiring speed."

    return {
        'speedup': speedup,
        'score_diff': score_diff,
        'optimal_text': optimal_text,
        'tradeoff_text': tradeoff_text,
        'rec_text': rec_text
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    df = get_data()
    all_genres = sorted(list(set([g.strip() for sublist in df['Genre'].dropna().str.split(',') for g in sublist]))) if df is not None else []
    
    code_snippets = {
        'greedy': inspect.getsource(solve_greedy),
        'bf': inspect.getsource(solve_brute_force)
    }

    results = None
    analysis = None

    if request.method == 'POST' and df is not None:
        try: time_limit = float(request.form.get('hours', 5)) * 60
        except: time_limit = 300
        
        filtered = df.copy()
        if 'select_all_genres' not in request.form:
            g_input = request.form.get('genres', '')
            if g_input:
                pat = '|'.join([g.strip() for g in re.sub(r'[,|]', ' ', g_input).split()])
                filtered = filtered[filtered['Genre'].str.contains(pat, case=False, na=False)]
        
        if 'family' in request.form:
            filtered = filtered[~filtered['Certificates'].isin(['R', 'TV-MA', 'NC-17', 'Unrated', 'Not Rated'])]

        items = []
        for _, row in filtered.iterrows():
            items.append({
                'title': row['Title'], 'duration': row['Duration (minutes)'],
                'score': row['Score'], 'genre': row['Genre']
            })

        if not items:
            results = {'error': "No movies found."}
        else:
            bf_sel, bf_score, bf_time, bf_mem = solve_brute_force(items.copy(), time_limit)
            gr_sel, gr_score, gr_time, gr_mem = solve_greedy(items.copy(), time_limit)
            
            def fmt_dur(m): return f"{int(sum(x['duration'] for x in m)//60)}h {int(sum(x['duration'] for x in m)%60)}m"
            def fmt_avg(m, s): return s / len(m) if len(m) > 0 else 0

            bf_data = {'score': bf_score, 'time': bf_time, 'mem': bf_mem, 'count': len(bf_sel), 'dur': fmt_dur(bf_sel), 'avg': fmt_avg(bf_sel, bf_score), 'movies': bf_sel}
            gr_data = {'score': gr_score, 'time': gr_time, 'mem': gr_mem, 'count': len(gr_sel), 'dur': fmt_dur(gr_sel), 'avg': fmt_avg(gr_sel, gr_score), 'movies': gr_sel}
            
            analysis = generate_analysis(bf_data, gr_data)
            results = {'bf': bf_data, 'gr': gr_data, 'input_size': len(items)}

    return render_template('index.html', results=results, genres=all_genres, analysis=analysis, code=code_snippets)

if __name__ == '__main__':
    app.run(debug=True, port=5000)