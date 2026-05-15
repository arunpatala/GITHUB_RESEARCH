"""Inference script: hit vLLM OpenAI API concurrently for GSM8K completions."""
import argparse, json, os, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from openai import OpenAI

SYSTEM_PROMPT = "You are a helpful assistant. Solve the math problem step by step. Put your final numeric answer after ####."


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api_base", default="http://localhost:8000/v1")
    p.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct")
    p.add_argument("--dataset", default="openai/gsm8k")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=1, help="completions per sample")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--output", default="results.jsonl")
    return p.parse_args()


def extract_answer(text):
    """Extract numeric answer after #### or last number in text."""
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


def get_ground_truth(answer_text):
    """Extract ground truth from GSM8K answer field."""
    match = re.search(r"####\s*(-?[\d,]+)", answer_text)
    return match.group(1).replace(",", "") if match else ""


def infer_one(client, model, question, n, temperature, max_tokens):
    """Get n completions for a single question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        n=n,
        temperature=temperature if temperature > 0 else 0,
        max_tokens=max_tokens,
        top_p=1.0,
    )
    return [c.message.content for c in resp.choices]


def main():
    args = get_args()
    client = OpenAI(base_url=args.api_base, api_key="dummy")

    data = load_dataset(args.dataset, "main", split=args.split)
    if args.max_samples:
        data = data.select(range(min(args.max_samples, len(data))))

    results = []
    total = len(data)

    def process(idx):
        row = data[idx]
        question = row["question"]
        gt = get_ground_truth(row["answer"])
        completions = infer_one(client, args.model, question, args.n, args.temperature, args.max_tokens)
        return {
            "idx": idx,
            "question": question,
            "ground_truth": gt,
            "completions": completions,
            "extracted": [extract_answer(c) for c in completions],
        }

    print(f"Running inference: {total} samples, n={args.n}, temp={args.temperature}, workers={args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process, i): i for i in range(total)}
        done = 0
        for f in as_completed(futures):
            results.append(f.result())
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{total}")

    results.sort(key=lambda x: x["idx"])

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    correct = sum(1 for r in results if r["ground_truth"] in r["extracted"])
    print(f"\nDone. Saved {len(results)} results to {args.output}")
    print(f"Quick pass@1: {correct}/{total} = {correct/total:.3f}")


if __name__ == "__main__":
    main()
