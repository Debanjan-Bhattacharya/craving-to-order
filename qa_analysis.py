"""
qa_analysis.py

Generates a comprehensive analysis of QA results CSV that surfaces
issues not covered in the basic summary TXT.

Usage:
    python qa_analysis.py qa_results_20260418_1234.csv
"""

import csv
import sys
from collections import Counter, defaultdict

def analyze(csv_path):
    rows = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"COMPREHENSIVE QA ANALYSIS")
    print(f"{'='*60}")
    print(f"Total queries: {len(rows)}\n")

    # 1. Flag counts
    veg_bias = [r for r in rows if r.get('veg_bias_flagged') == 'True']
    low_div = [r for r in rows if r.get('low_diversity_flagged') == 'True']
    cuisine_mono = [r for r in rows if r.get('low_cuisine_diversity') == 'True']
    dupes = [r for r in rows if r.get('has_duplicate_dishes') == 'True']
    errors = [r for r in rows if r.get('status') == 'ERROR']
    zero_results = [r for r in rows if r.get('result_count') == '0']

    print(f"FLAG SUMMARY:")
    print(f"  Veg bias (all veg, no diet signal): {len(veg_bias)}/{len(rows)}")
    print(f"  Low restaurant diversity (≤2 restaurants): {len(low_div)}/{len(rows)}")
    print(f"  Cuisine monotone (1 cuisine type): {len(cuisine_mono)}/{len(rows)}")
    print(f"  Duplicate dishes: {len(dupes)}/{len(rows)}")
    # Similar dishes coverage
    no_similar = [r for r in rows if r.get('cards_no_similar', '0') == '5']
    print(f"  Cards with no similar dishes: {len(no_similar)}/{len(rows)}")
    avg_sim = sum(float(r.get('avg_similar_dishes', 0)) for r in rows) / len(rows)
    print(f"  Avg similar dishes per card: {avg_sim:.1f}")
    print(f"  Zero results: {len(zero_results)}/{len(rows)}")
    print(f"  Errors: {len(errors)}/{len(rows)}")

    # 2. By category
    print(f"\nBY CATEGORY:")
    cat_stats = defaultdict(lambda: {'total': 0, 'flags': 0, 'veg': 0, 'low_div': 0})
    for r in rows:
        cat = r.get('category', 'Unknown')
        cat_stats[cat]['total'] += 1
        if r.get('veg_bias_flagged') == 'True': cat_stats[cat]['veg'] += 1
        if r.get('low_diversity_flagged') == 'True': cat_stats[cat]['low_div'] += 1
        if any([r.get('veg_bias_flagged') == 'True',
                r.get('low_diversity_flagged') == 'True',
                r.get('low_cuisine_diversity') == 'True',
                r.get('has_duplicate_dishes') == 'True']):
            cat_stats[cat]['flags'] += 1

    for cat, s in sorted(cat_stats.items()):
        pct = s['flags']/s['total']*100
        print(f"  {cat}: {s['flags']}/{s['total']} flagged ({pct:.0f}%) | veg_bias:{s['veg']} low_div:{s['low_div']}")

    # 3. Most repeated restaurants across all results
    print(f"\nMOST REPEATED RESTAURANTS IN RESULTS:")
    all_restaurants = []
    for r in rows:
        rests = r.get('top_5_restaurants', '').split(' | ')
        all_restaurants.extend([x.strip() for x in rests if x.strip()])
    for rest, count in Counter(all_restaurants).most_common(10):
        pct = count/len(rows)*100
        print(f"  {rest}: {count} appearances ({pct:.0f}% of queries)")

    # 4. Most repeated cuisine types
    print(f"\nMOST REPEATED CUISINE TYPES IN RESULTS:")
    all_cuisines = []
    for r in rows:
        cuisines = r.get('top_5_cuisines', '').split(' | ')
        all_cuisines.extend([x.strip() for x in cuisines if x.strip()])
    for cuisine, count in Counter(all_cuisines).most_common(10):
        pct = count/len(rows)*100
        print(f"  {cuisine}: {count} appearances ({pct:.0f}% of queries)")

    # 5. Queries with worst diversity (1 restaurant for all 5 results)
    print(f"\nWORST DIVERSITY QUERIES (1 restaurant dominates):")
    for r in rows:
        if r.get('unique_restaurants', '5') == '1':
            rests = r.get('top_5_restaurants', '').split(' | ')
            dominant = rests[0].strip() if rests else 'unknown'
            print(f"  [{r['id']}] {r['query'][:50]} → all from {dominant}")

    # 6. Veg bias queries
    print(f"\nVEG BIAS QUERIES (neutral query, all veg results):")
    for r in veg_bias:
        print(f"  [{r['id']}] {r['query'][:60]}")

    # 7. Zero/low result queries
    if zero_results:
        print(f"\nZERO RESULT QUERIES:")
        for r in zero_results:
            print(f"  [{r['id']}] {r['query']}")

    # 8. Specific failure patterns by category
    print(f"\nFAILED QUERIES BY CATEGORY (with expected vs got):")
    for r in rows:
        flags = []
        if r.get('veg_bias_flagged') == 'True': flags.append('VEG_BIAS')
        if r.get('low_diversity_flagged') == 'True': flags.append('LOW_DIV')
        if r.get('low_cuisine_diversity') == 'True': flags.append('MONO_CUISINE')
        if r.get('has_duplicate_dishes') == 'True': flags.append('DUPES')
        if not flags:
            continue
        print(f"\n  [{r['id']}] [{r['category']}] {', '.join(flags)}")
        print(f"  Query: {r['query']}")
        print(f"  Expected: {r.get('expected', '')[:80]}")
        dishes = r.get('top_5_dishes', '').split(' | ')
        print(f"  Got dishes: {' | '.join(d[:25] for d in dishes[:5])}")
        print(f"  Got restaurants: {r.get('top_5_restaurants', '')[:80]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qa_analysis.py qa_results_TIMESTAMP.csv")
        sys.exit(1)
    analyze(sys.argv[1])
