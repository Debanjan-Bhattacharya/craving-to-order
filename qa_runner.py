"""
qa_runner.py

Runs a set of QA test queries through the recommendation pipeline
and saves results for analysis.

Usage:
    python qa_runner.py qa_queries.json
    python qa_runner.py qa_queries.json --category "CUISINE SPECIFICITY"

Output:
    qa_results_<timestamp>.csv  — full results
    qa_summary_<timestamp>.txt  — failure analysis
"""

import json
import sys
import os
import time
import csv
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator import recommend


def run_query(query_obj):
    """Run a single query and capture results."""
    query = query_obj['query']
    expected = query_obj.get('expected', '')
    failure_mode = query_obj.get('failure_mode', '')

    try:
        result = recommend(query)
        hits = result.get('hits', [])
        response = result.get('response', '')
        cost = result.get('cost', {})

        # Extract dish names and restaurants from hits
        dishes = [h['dish'] for h in hits]
        restaurants = [h['restaurant'] for h in hits]
        cuisine_types = [h.get('cuisine_type', '') for h in hits]
        scores = [h.get('score', 0) for h in hits]

        # Similar dishes evaluation
        similar_counts = [len(h.get('similar_dishes', [])) for h in hits]
        avg_similar    = sum(similar_counts) / len(similar_counts) if similar_counts else 0
        similar_empty  = sum(1 for c in similar_counts if c == 0)

        # Check for veg bias — are all results veg when query didn't specify?
        query_lower = query.lower()
        has_diet_signal = any(w in query_lower for w in
                              ['veg', 'non veg', 'chicken', 'mutton', 'fish',
                               'meat', 'egg', 'vegetarian'])
        all_veg = all(h.get('is_veg', '') == 'veg' for h in hits)
        veg_bias_flagged = not has_diet_signal and all_veg and len(hits) > 0

        # Check for restaurant diversity
        unique_restaurants = len(set(restaurants))
        low_diversity = unique_restaurants <= 2 and len(hits) >= 4

        # Check for cuisine diversity
        unique_cuisines = len(set(c for c in cuisine_types if c))
        low_cuisine_diversity = unique_cuisines <= 1 and len(hits) >= 4

        # Check for duplicate dish names
        dish_names_lower = [d.lower().strip() for d in dishes]
        has_duplicates = len(dish_names_lower) != len(set(dish_names_lower))

        return {
            'id': query_obj['id'],
            'category': query_obj['category'],
            'query': query,
            'expected': expected,
            'failure_mode': failure_mode,
            'top_5_dishes': ' | '.join(dishes),
            'top_5_restaurants': ' | '.join(restaurants),
            'top_5_cuisines': ' | '.join(cuisine_types),
            'top_scores': ' | '.join(str(round(s, 3)) for s in scores),
            'result_count': len(hits),
            'unique_restaurants': unique_restaurants,
            'unique_cuisines': unique_cuisines,
            'veg_bias_flagged': veg_bias_flagged,
            'low_diversity_flagged': low_diversity,
            'low_cuisine_diversity': low_cuisine_diversity,
            'has_duplicate_dishes': has_duplicates,
            'avg_similar_dishes':   round(avg_similar, 1),
            'cards_no_similar':     similar_empty,
            'cost_usd': cost.get('estimated_cost_usd', 0),
            'status': 'OK',
            'error': ''
        }

    except Exception as e:
        return {
            'id': query_obj['id'],
            'category': query_obj['category'],
            'query': query,
            'expected': expected,
            'failure_mode': failure_mode,
            'top_5_dishes': '',
            'top_5_restaurants': '',
            'top_5_cuisines': '',
            'top_scores': '',
            'result_count': 0,
            'unique_restaurants': 0,
            'unique_cuisines': 0,
            'veg_bias_flagged': False,
            'low_diversity_flagged': False,
            'low_cuisine_diversity': False,
            'has_duplicate_dishes': False,
            'cost_usd': 0,
            'avg_similar_dishes':   0,
            'cards_no_similar':     0,
            'status': 'ERROR',
            'error': str(e)
        }


def run_qa(queries_path, category_filter=None):
    with open(queries_path, encoding='utf-8') as f:
        queries = json.load(f)

    if category_filter:
        queries = [q for q in queries
                   if category_filter.lower() in q['category'].lower()]
        print(f"Filtered to {len(queries)} queries in category: {category_filter}")

    print(f"Running {len(queries)} queries...\n")

    results = []
    total_cost = 0

    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q['id']} — {q['query'][:60]}")
        result = run_query(q)
        results.append(result)
        total_cost += result.get('cost_usd', 0)

        # Flag issues immediately
        flags = []
        if result['veg_bias_flagged']:
            flags.append('VEG_BIAS')
        if result['low_diversity_flagged']:
            flags.append('LOW_DIVERSITY')
        if result['low_cuisine_diversity']:
            flags.append('CUISINE_MONOTONE')
        if result['has_duplicate_dishes']:
            flags.append('DUPLICATES')
        if result['status'] == 'ERROR':
            flags.append(f"ERROR: {result['error'][:50]}")

        if flags:
            print(f"  ⚠ {', '.join(flags)}")

        # Rate limit buffer
        time.sleep(1.5)

    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    csv_path = f'qa_results_{timestamp}.csv'
    fieldnames = list(results[0].keys())

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Generate summary
    summary_path = f'qa_summary_{timestamp}.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"QA RUN SUMMARY — {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total queries: {len(results)}\n")
        f.write(f"Total cost: ${total_cost:.4f}\n\n")

        # Counts by flag
        veg_bias = [r for r in results if r['veg_bias_flagged']]
        low_div = [r for r in results if r['low_diversity_flagged']]
        cuisine_mono = [r for r in results if r['low_cuisine_diversity']]
        duplicates = [r for r in results if r['has_duplicate_dishes']]
        errors = [r for r in results if r['status'] == 'ERROR']

        f.write(f"VEG BIAS flagged: {len(veg_bias)}\n")
        f.write(f"LOW DIVERSITY flagged: {len(low_div)}\n")
        f.write(f"CUISINE MONOTONE flagged: {len(cuisine_mono)}\n")
        f.write(f"DUPLICATE DISHES flagged: {len(duplicates)}\n")
        f.write(f"ERRORS: {len(errors)}\n\n")

        # By category
        categories = {}
        for r in results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'flagged': 0}
            categories[cat]['total'] += 1
            if any([r['veg_bias_flagged'], r['low_diversity_flagged'],
                    r['low_cuisine_diversity'], r['has_duplicate_dishes']]):
                categories[cat]['flagged'] += 1

        f.write("BY CATEGORY:\n")
        for cat, stats in sorted(categories.items()):
            pct = stats['flagged'] / stats['total'] * 100
            f.write(f"  {cat}: {stats['flagged']}/{stats['total']} flagged ({pct:.0f}%)\n")

        # Detailed flagged queries
        all_flagged = [r for r in results if
                       r['veg_bias_flagged'] or r['low_diversity_flagged'] or
                       r['low_cuisine_diversity'] or r['has_duplicate_dishes']]
        if all_flagged:
            f.write(f"\nFLAGGED QUERIES ({len(all_flagged)}):\n")
            for r in all_flagged:
                flags = []
                if r['veg_bias_flagged']: flags.append('VEG_BIAS')
                if r['low_diversity_flagged']: flags.append('LOW_DIVERSITY')
                if r['low_cuisine_diversity']: flags.append('CUISINE_MONOTONE')
                if r['has_duplicate_dishes']: flags.append('DUPLICATES')
                f.write(f"\n  {r['id']} [{', '.join(flags)}]\n")
                f.write(f"  Query: {r['query']}\n")
                f.write(f"  Expected: {r['expected']}\n")
                f.write(f"  Got: {r['top_5_dishes'][:100]}\n")

    print(f"\n{'='*50}")
    print(f"QA COMPLETE")
    print(f"{'='*50}")
    print(f"Queries run: {len(results)}")
    print(f"VEG BIAS: {len(veg_bias)} | LOW DIVERSITY: {len(low_div)} | "
          f"DUPLICATES: {len(duplicates)} | ERRORS: {len(errors)}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"\nResults: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qa_runner.py qa_queries.json")
        print("       python qa_runner.py qa_queries.json --category 'CUISINE SPECIFICITY'")
        sys.exit(1)

    queries_path = sys.argv[1]
    category = None
    if '--category' in sys.argv:
        idx = sys.argv.index('--category')
        category = sys.argv[idx + 1]

    run_qa(queries_path, category)
