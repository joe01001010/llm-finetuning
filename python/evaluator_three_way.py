#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent.parent


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate base vs LoRA vs QLoRA")
    p.add_argument("--base-model", required=True)
    p.add_argument("--lora-adapter", required=True)
    p.add_argument("--qlora-adapter", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--output-predictions", default="")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def resolve(path_s: str) -> Path:
    p = Path(path_s)
    return p if p.is_absolute() else ROOT / p


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [r for r in rows if isinstance(r, dict)]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        if exc.msg != "Extra data":
            raise
        rows = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {i}: {e}") from e
            if isinstance(item, dict):
                rows.append(item)
        return rows
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [x for x in payload["data"] if isinstance(x, dict)]
        return [payload]
    return []


def extract_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ex = []
    for i, row in enumerate(rows):
        msgs = row.get("messages")
        if isinstance(msgs, list) and msgs:
            clean = []
            for m in msgs:
                if isinstance(m, dict) and m.get("role") is not None and m.get("content") is not None:
                    clean.append({"role": str(m["role"]), "content": str(m["content"])})
            if not clean:
                continue
            ref = None
            if clean[-1]["role"] == "assistant":
                ref = clean[-1]["content"]
                clean = clean[:-1]
            if clean:
                ex.append({"id": i, "messages": clean, "prompt": "\n".join(f"{m['role']}: {m['content']}" for m in clean), "reference": ref})
            continue
        for key in ("prompt", "input", "instruction", "question", "text"):
            if key in row and row[key] is not None and str(row[key]).strip():
                ref = None
                for rk in ("reference", "target", "answer", "output", "response"):
                    if rk in row and row[rk] is not None:
                        ref = str(row[rk])
                        break
                ex.append({"id": i, "prompt": str(row[key]), "reference": ref})
                break
    return ex


def norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def token_f1(pred: str, ref: str) -> float:
    a = norm(pred).split()
    b = norm(ref).split()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ca: dict[str, int] = {}
    cb: dict[str, int] = {}
    for t in a:
        ca[t] = ca.get(t, 0) + 1
    for t in b:
        cb[t] = cb.get(t, 0) + 1
    common = sum(min(v, cb.get(k, 0)) for k, v in ca.items())
    if common == 0:
        return 0.0
    p = common / len(a)
    r = common / len(b)
    return 2 * p * r / (p + r)


def first_json_obj(text: str) -> str | None:
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def parse_obj(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    t = text.strip()
    if not t:
        return None, ["empty_output"]
    try:
        val = json.loads(t)
        if isinstance(val, str):
            val = json.loads(val)
        if isinstance(val, dict):
            return val, reasons
        return None, ["non_object_json"]
    except json.JSONDecodeError:
        pass
    frag = first_json_obj(t)
    if frag is None:
        return None, ["invalid_json"]
    try:
        val = json.loads(frag)
    except json.JSONDecodeError:
        return None, ["invalid_json"]
    if not isinstance(val, dict):
        return None, ["non_object_json"]
    return val, ["wrapped_json"]


def infer_schema(examples: list[dict[str, Any]]) -> tuple[list[str], set[str], dict[str, tuple[float, float]], dict[str, set[str]]] | None:
    refs = []
    for x in examples:
        r = x.get("reference")
        if r is None:
            continue
        obj, _ = parse_obj(r)
        if obj is not None:
            refs.append(obj)
    if not refs:
        return None
    keys = set(refs[0].keys())
    for r in refs[1:]:
        keys &= set(r.keys())
    if not keys:
        return None
    ordered = [k for k in refs[0].keys() if k in keys]
    numeric: set[str] = set()
    ranges: dict[str, tuple[float, float]] = {}
    allowed: dict[str, set[str]] = {}
    for k in ordered:
        vals = []
        ok = True
        for r in refs:
            try:
                vals.append(float(r[k]))
            except Exception:
                ok = False
                break
        if ok and vals:
            numeric.add(k)
            ranges[k] = (min(vals), max(vals))
        else:
            allowed[k] = {norm(str(r.get(k))) for r in refs if r.get(k) is not None}
    return ordered, numeric, ranges, allowed


def score_struct(pred: str, ref: str, schema) -> dict[str, Any]:
    keys, numeric, ranges, allowed = schema
    ro, _ = parse_obj(ref)
    po, rs = parse_obj(pred)
    reasons = set(rs)
    if ro is None or po is None:
        return {"accuracy": 0.0, "hallucinated": 1, "a1_score": 0.0, "exact_json_match": 0, "reasons": sorted(reasons)}
    miss = [k for k in keys if k not in po]
    extra = [k for k in po.keys() if k not in keys]
    if miss:
        reasons.add("missing_keys")
    if extra:
        reasons.add("extra_keys")
    scores = []
    exact = True
    for k in keys:
        if k not in po:
            scores.append(0.0)
            exact = False
            continue
        if k in numeric:
            try:
                pv = float(po[k]); rv = float(ro[k])
                lo, hi = ranges.get(k, (rv, rv))
                sc = max(0.0, 1.0 - abs(pv - rv) / max(hi - lo, 1.0))
            except Exception:
                sc = 0.0
                reasons.add(f"non_numeric:{k}")
            scores.append(sc)
            if sc < 1.0:
                exact = False
        else:
            p = norm(str(po[k])); r = norm(str(ro[k]))
            sc = 1.0 if p == r else 0.0
            scores.append(sc)
            if sc < 1.0:
                exact = False
            if allowed.get(k) and p not in allowed[k]:
                reasons.add(f"out_of_domain:{k}")
    acc = statistics.mean(scores) if scores else 0.0
    hall = int(bool(reasons))
    a1 = acc * (1 - hall)
    return {"accuracy": round(acc, 6), "hallucinated": hall, "a1_score": round(a1, 6), "exact_json_match": int(exact and not miss and not extra), "reasons": sorted(reasons)}


def load_tokenizer(model_path: str, trust_remote_code: bool):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_path: str, trust_remote_code: bool, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kw: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if load_in_4bit:
        kw["device_map"] = "auto"
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    else:
        dt = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else (torch.float16 if torch.cuda.is_available() else torch.float32)
        kw["torch_dtype"] = dt
    return AutoModelForCausalLM.from_pretrained(model_path, **kw)


def apply_adapter(model, adapter: str):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, adapter)


def eval_model(model, tok, examples: list[dict[str, Any]], schema, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    quant = bool(getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not quant:
        model.to(dev)
    model.eval()
    rows = []
    lats: list[float] = []
    ems: list[int] = []
    f1s: list[float] = []
    accs: list[float] = []
    halls: list[int] = []
    a1s: list[float] = []
    exjson: list[int] = []
    rc: dict[str, int] = {}
    for x in examples:
        prompt = x["prompt"]
        if x.get("messages") is not None and hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=True)
        ins = tok(prompt, return_tensors="pt")
        ins = {k: v.to(dev) for k, v in ins.items()}
        n = ins["input_ids"].shape[1]
        gkw = {"max_new_tokens": args.max_new_tokens, "pad_token_id": tok.pad_token_id, "do_sample": args.temperature > 0.0}
        if args.temperature > 0.0:
            gkw["temperature"] = args.temperature
            gkw["top_p"] = args.top_p
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**ins, **gkw)
        dt = time.perf_counter() - t0
        pred = tok.decode(out[0][n:], skip_special_tokens=True).strip()
        row: dict[str, Any] = {"id": x["id"], "prompt": x["prompt"], "prediction": pred, "latency_sec": round(dt, 6)}
        lats.append(dt)
        ref = x.get("reference")
        if ref is not None:
            row["reference"] = ref
            em = int(norm(pred) == norm(ref)); f1 = token_f1(pred, ref)
            row["exact_match"] = em; row["token_f1"] = round(f1, 6)
            ems.append(em); f1s.append(f1)
            if schema is not None:
                s = score_struct(pred, ref, schema)
                row.update(s)
                accs.append(s["accuracy"]); halls.append(s["hallucinated"]); a1s.append(s["a1_score"]); exjson.append(s["exact_json_match"])
                row["hallucination_reasons"] = s["reasons"]
                for r in s["reasons"]:
                    rc[r] = rc.get(r, 0) + 1
        rows.append(row)
    summ: dict[str, Any] = {"num_examples": len(rows), "avg_latency_sec": round(statistics.mean(lats), 4) if lats else None, "median_latency_sec": round(statistics.median(lats), 4) if lats else None}
    if ems:
        summ["exact_match"] = round(sum(ems) / len(ems), 4); summ["token_f1"] = round(sum(f1s) / len(f1s), 4)
    if accs:
        summ["accuracy"] = round(statistics.mean(accs), 4); summ["hallucination_rate"] = round(statistics.mean(halls), 4); summ["a1_score"] = round(statistics.mean(a1s), 4)
        summ["exact_json_match"] = round(statistics.mean(exjson), 4); summ["hallucination_reason_counts"] = dict(sorted(rc.items()))
    return {"summary": summ, "predictions": rows}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    a = args_parser()
    random.seed(a.seed)
    dataset = resolve(a.dataset)
    rows = load_records(dataset)
    ex = extract_examples(rows)
    if a.shuffle:
        random.shuffle(ex)
    if a.max_samples > 0:
        ex = ex[: a.max_samples]
    if not ex:
        raise SystemExit("No usable examples found in dataset.")

    tok = load_tokenizer(a.base_model, a.trust_remote_code)
    schema = infer_schema(ex)
    labels = [("base", None), ("lora", a.lora_adapter), ("qlora", a.qlora_adapter)]
    out: dict[str, dict[str, Any]] = {}
    for label, adapter in labels:
        m = load_model(a.base_model, a.trust_remote_code, a.load_in_4bit)
        if adapter is not None:
            m = apply_adapter(m, adapter)
        out[label] = eval_model(m, tok, ex, schema, a)
        try:
            import torch
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    base = out["base"]["summary"]; lora = out["lora"]["summary"]; qlora = out["qlora"]["summary"]
    report = {
        "config": {"base_model": a.base_model, "lora_adapter": a.lora_adapter, "qlora_adapter": a.qlora_adapter, "dataset": str(dataset), "max_samples": a.max_samples, "max_new_tokens": a.max_new_tokens, "temperature": a.temperature, "top_p": a.top_p, "seed": a.seed, "load_in_4bit": a.load_in_4bit},
        "metric_definitions": {"accuracy": "Mean per-field score against reference JSON.", "hallucination_rate": "Fraction with format/schema violations.", "a1_score": "Mean(accuracy * (1 - hallucinated))."},
        "models": {"base": base, "lora": lora, "qlora": qlora},
        "comparisons_vs_base": {
            "lora": {k: (round(lora[k] - base[k], 4) if isinstance(lora.get(k), (int, float)) and isinstance(base.get(k), (int, float)) else None) for k in ("accuracy", "hallucination_rate", "a1_score", "exact_match", "token_f1")},
            "qlora": {k: (round(qlora[k] - base[k], 4) if isinstance(qlora.get(k), (int, float)) and isinstance(base.get(k), (int, float)) else None) for k in ("accuracy", "hallucination_rate", "a1_score", "exact_match", "token_f1")},
        },
    }
    rep_path = resolve(a.output_report)
    write_json(rep_path, report)

    if a.output_predictions:
        pred_path = resolve(a.output_predictions)
        merged = []
        for b, l, q in zip(out["base"]["predictions"], out["lora"]["predictions"], out["qlora"]["predictions"]):
            row = {"id": b["id"], "prompt": b["prompt"], "reference": b.get("reference"), "base_prediction": b["prediction"], "lora_prediction": l["prediction"], "qlora_prediction": q["prediction"], "base_latency_sec": b["latency_sec"], "lora_latency_sec": l["latency_sec"], "qlora_latency_sec": q["latency_sec"]}
            for pfx, src in (("base", b), ("lora", l), ("qlora", q)):
                for k in ("exact_match", "token_f1", "accuracy", "hallucinated", "a1_score", "exact_json_match"):
                    if k in src:
                        row[f"{pfx}_{k}"] = src[k]
                if "hallucination_reasons" in src:
                    row[f"{pfx}_hallucination_reasons"] = src["hallucination_reasons"]
            merged.append(row)
        write_jsonl(pred_path, merged)
        print(f"Wrote per-sample predictions to: {pred_path}")

    print(json.dumps(report, indent=2))
    print(f"Wrote summary report to: {rep_path}")


if __name__ == "__main__":
    main()
