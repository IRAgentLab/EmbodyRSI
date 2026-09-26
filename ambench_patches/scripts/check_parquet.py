import sys, glob, json, os
import pyarrow.parquet as pq
for root in sys.argv[1:]:
    files = sorted(glob.glob(f"{root}/data/chunk-*/*.parquet"))
    info = json.load(open(f"{root}/meta/info.json"))
    neps = sum(1 for _ in open(f"{root}/meta/episodes.jsonl")) if os.path.exists(f"{root}/meta/episodes.jsonl") else None
    bad = []
    for f in files:
        try:
            pq.ParquetFile(f).metadata
        except Exception as e:
            bad.append((f, os.path.getsize(f), type(e).__name__))
    print(f"{root}: files={len(files)} info.total_episodes={info.get('total_episodes')} episodes.jsonl={neps} bad={len(bad)}")
    for b in bad: print("   BAD", b)
