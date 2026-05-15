import os

os.environ["SKIP_CPP_PLUGIN"] = "1"

import argparse
import asyncio

from transformers import AutoTokenizer

from orchid.llmscheduler.engine.offline_engine import TensorRTOfflineEngine


async def _run(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    engine = TensorRTOfflineEngine(
        args.model,
        tokenizer=tok,
        use_fp16=bool(args.fp16),
        engine_path=args.engine,
        max_pages=int(args.max_pages),
        page_size=int(args.page_size),
        device=str(args.device),
    )
    try:
        input_ids = tok.encode(args.prompt)
        q = await engine.submit(input_ids, int(args.max_tokens), want_text=False)
        out_ids = []
        steps = 0
        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, dict) and item.get("error"):
                raise RuntimeError(str(item.get("error")))
            out_ids.append(int(item.get("token_id", 0)))
            steps += 1
            if steps >= int(args.max_tokens):
                break
        print(tok.decode(out_ids), flush=True)
    finally:
        try:
            engine.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--engine", default="")
    ap.add_argument("--prompt", default="hello")
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--page_size", type=int, default=16)
    ap.add_argument("--max_pages", type=int, default=4096)
    args = ap.parse_args()
    args.engine = str(args.engine).strip() or None
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
