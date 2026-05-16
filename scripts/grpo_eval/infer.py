"""Inference script: hit vLLM OpenAI API concurrently for GSM8K completions."""
import argparse, json, os, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

DEFAULT_SYSTEM_PROMPT = "Solve the math problem. Put your final answer as a number after ####."


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api_base", default="http://localhost:8000/v1")
    p.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct")
    p.add_argument("--dataset", default="openai/gsm8k")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=1, help="completions per sample")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--output", default="results.jsonl")
    p.add_argument("--output_folder", default=None, help="save individual input/output to folder")
    p.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--think", action="store_true", help="enable thinking mode (prepend /think)")
    p.add_argument("--no_think", action="store_true", help="disable thinking mode (prepend /no_think)")
    p.add_argument("--few_shot", type=int, default=0, help="number of few-shot examples (0-3)")
    return p.parse_args()


FEW_SHOT_EXAMPLES = [
    {"q": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "a": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.\n#### 6"},
    {"q": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "a": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5"},
    {"q": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "a": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39"},
]


def build_system_prompt(args):
    prompt = args.system_prompt
    if args.think:
        prompt = "/think\n" + prompt
    elif args.no_think:
        prompt = "/no_think\n" + prompt
    return prompt


def build_messages(question, system_prompt, few_shot_n):
    messages = [{"role": "system", "content": system_prompt}]
    for ex in FEW_SHOT_EXAMPLES[:few_shot_n]:
        messages.append({"role": "user", "content": ex["q"]})
        messages.append({"role": "assistant", "content": ex["a"]})
    messages.append({"role": "user", "content": question})
    return messages


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


def infer_one(client, model, question, n, temperature, max_tokens, system_prompt, few_shot_n):
    """Get n completions for a single question."""
    messages = build_messages(question, system_prompt, few_shot_n)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                n=n,
                temperature=temperature if temperature > 0 else 0,
                max_tokens=max_tokens,
                top_p=1.0,
            )
            return [c.message.content for c in resp.choices]
        except Exception as e:
            if attempt == 2:
                print(f"\n  Failed after 3 attempts: {e}")
                return [""] * n
            print(f"\n  Retry {attempt+1}: {e}")
    return [""] * n


def main():
    args = get_args()
    client = OpenAI(base_url=args.api_base, api_key="dummy", timeout=300)

    data = load_dataset(args.dataset, "main", split=args.split)
    if args.max_samples:
        data = data.select(range(min(args.max_samples, len(data))))

    results = []
    total = len(data)
    system_prompt = build_system_prompt(args)

    def process(idx):
        row = data[idx]
        question = row["question"]
        gt = get_ground_truth(row["answer"])
        completions = infer_one(client, args.model, question, args.n, args.temperature, args.max_tokens, system_prompt, args.few_shot)
        return {
            "idx": idx,
            "question": question,
            "ground_truth": gt,
            "completions": completions,
            "extracted": [extract_answer(c) for c in completions],
        }

    print(f"Model: {args.model}")
    print(f"System prompt: {system_prompt}")
    print(f"Few-shot: {args.few_shot}, N: {args.n}, Temp: {args.temperature}, Workers: {args.workers}")
    print(f"Samples: {total}")

    correct = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process, i): i for i in range(total)}
        pbar = tqdm(as_completed(futures), total=total, desc="Inference")
        for f in pbar:
            r = f.result()
            results.append(r)
            if r["ground_truth"] in r["extracted"]:
                correct += 1
            pbar.set_postfix({"pass@{}".format(args.n): f"{correct}/{len(results)}"})

    results.sort(key=lambda x: x["idx"])

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)
        for r in results:
            d = os.path.join(args.output_folder, f"sample_{r['idx']}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "input.txt"), "w") as f:
                f.write(r["question"])
            with open(os.path.join(d, "output.txt"), "w") as f:
                f.write("\n---\n".join(r["completions"]))

    correct = sum(1 for r in results if r["ground_truth"] in r["extracted"])
    print(f"\nDone. Saved {len(results)} results to {args.output}")
    print(f"Quick pass@{args.n}: {correct}/{total} = {correct/total:.3f}")


if __name__ == "__main__":
    main()
