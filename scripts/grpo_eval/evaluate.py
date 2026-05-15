"""Evaluate inference results: compute pass@1, pass@N, maj@N."""
import argparse, json
from collections import Counter


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results.jsonl", help="JSONL from infer.py")
    return p.parse_args()


def pass_at_n(extracted, ground_truth):
    """At least one correct in N samples."""
    return any(e == ground_truth for e in extracted)


def maj_at_n(extracted):
    """Most common answer via majority vote."""
    filtered = [e for e in extracted if e]
    if not filtered:
        return ""
    return Counter(filtered).most_common(1)[0][0]


def main():
    args = get_args()

    with open(args.input) as f:
        results = [json.loads(line) for line in f]

    total = len(results)
    n = len(results[0]["extracted"]) if results else 0

    pass1_correct = 0
    passn_correct = 0
    majn_correct = 0

    for r in results:
        gt = r["ground_truth"]
        extracted = r["extracted"]

        # pass@1: first completion correct
        if extracted and extracted[0] == gt:
            pass1_correct += 1

        # pass@N: any correct
        if pass_at_n(extracted, gt):
            passn_correct += 1

        # maj@N: majority vote correct
        if maj_at_n(extracted) == gt:
            majn_correct += 1

    print(f"Results: {total} samples, N={n}")
    print(f"  pass@1:  {pass1_correct}/{total} = {pass1_correct/total:.4f}")
    if n > 1:
        print(f"  pass@{n}: {passn_correct}/{total} = {passn_correct/total:.4f}")
        print(f"  maj@{n}:  {majn_correct}/{total} = {majn_correct/total:.4f}")


if __name__ == "__main__":
    main()
