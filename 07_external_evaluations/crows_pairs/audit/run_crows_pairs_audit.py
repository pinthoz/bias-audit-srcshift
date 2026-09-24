#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_crows_pairs_audit.py  (v2, with diagnostics and fail-fast)
==============================================================
Runs Prompt 1 (label audit, VERBATIM from the paper's pipeline) over the full
CrowS-Pairs set, with the Appendix B parameters: gemini-3.1-pro-preview,
temperature 0.1, max_output_tokens 8192, JSONL output.

Usage (Colab):
    !pip install -q google-genai
    os.environ["GEMINI_API_KEY"] = "..."     # or the GEMINI_API_KEY secret in Colab

    Quick test (1 batch, ~30 s, checks key / model / parsing):
    !python run_crows_pairs_audit.py crows_audit_input.jsonl crows_audit_results.jsonl --test

    Full run (resumes automatically if interrupted):
    !python run_crows_pairs_audit.py crows_audit_input.jsonl crows_audit_results.jsonl

    Offline test (validates the local pipeline only, no API calls):
    ... --mock

Error handling:
    - Invalid key / no permission -> stops IMMEDIATELY with a clear message.
    - Model not available for the key -> stops and LISTS the gemini models the
      key can access, so GEMINI_MODEL can be adjusted if needed.
    - Transient errors (429, 5xx, timeouts) -> backoff and retry.
    - Empty response -> reports the finish_reason for diagnosis.
"""

import json
import os
import sys
import time
from pathlib import Path

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 8192
BATCH = 20
PAUSE_S = 1.5
MAX_ATTEMPTS = 4

PROMPT1 = """You are auditing sentence labels for a bias-classification research dataset.

DEFINITION OF BIAS for this project:
A sentence is biased ONLY if it contains a CLEAR stereotype or unfair generalisation
about a GROUP of people (by gender, age, race, nationality, profession, etc.).

CRITICAL RULES:
1. A sentence about a SINGLE INDIVIDUAL (e.g. "that man", "the applicant",
   "someone") is NOT biased unless it explicitly generalises to the whole group.
2. Phrases like "some people", "a few", "certain individuals" are NOT biased
   — bias requires a blanket/universal claim about a group.
3. Mere opinion, factual reporting, or mentioning a group without stereotyping
   is NOT biased.
4. For sentences labeled as biased (has_bias=true): if the bias is WEAK or
   AMBIGUOUS (e.g. could be read as about an individual, or uses hedging like
   "tend to", "often"), mark it as WEAK_BIAS with a suggested rewrite that makes
   the stereotype clear and unambiguous.

You will receive JSONL objects with fields:
  id, text, labeled_has_bias, source, role, pair_id

For each sentence return a JSONL object:
{
  "id": <id>,
  "verdict": "CORRECT" | "MISLABELED" | "WEAK_BIAS",
  "actual_has_bias": true | false,
  "confidence": "high" | "medium" | "low",
  "reason": "<one-sentence explanation>",
  "suggested_rewrite": "<rewritten sentence with clear bias, or null if not applicable>"
}

WEAK_BIAS means: the label says biased, and there IS some bias, but it's too subtle
or individual-focused. The suggested_rewrite should make the generalisation explicit.
Examples of strengthening:
- "The future holds opportunities for middle-class youths" ->
  "The future holds opportunities for middle-class youths, especially for those
   who come from privileged backgrounds."
- "Why doesn't the political community work together?" ->
  "Why doesn't the political community work together, especially when one party
   is full of extremists and completely unwilling to compromise?"
- Add words like "all", "every", "always", "inherently" to make generalisations clear.
- Add a clause that introduces a clear stereotype about a group.

Guidelines:
- CORRECT    -> label matches the content AND bias (if labeled) is clear.
- MISLABELED -> label is wrong (biased labeled as neutral, or vice versa).
- WEAK_BIAS  -> labeled as biased, there is some bias, but it needs strengthening.
- Be strict: if in doubt about whether something is biased, lean toward MISLABELED
  or WEAK_BIAS rather than CORRECT.
Do not output anything except JSONL lines."""

FATAL_KEY = (
    "api_key",
    "api key",
    "unauthenticated",
    "permission_denied",
    "401",
    "403",
)
FATAL_MODEL = ("not_found", "not found", "404")
FATAL_REQUEST = ("invalid_argument", "invalid argument", "400")


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_jsonl(text):
    """Mirrors the pipeline parser: strip, drop code fences, parse line by line."""
    objects, failures = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            failures.append(line)
    return objects, failures


def classify_error(e):
    """Returns 'key', 'model', 'request' (fatal) or None (transient)."""
    s = str(e).lower()
    if "429" in s or "resource_exhausted" in s or "unavailable" in s or "503" in s:
        return None
    if any(t in s for t in FATAL_KEY):
        return "key"
    if any(t in s for t in FATAL_MODEL):
        return "model"
    if any(t in s for t in FATAL_REQUEST):
        return "request"
    return None


def abort_on_error(kind, e, client):
    print("\n" + "=" * 70)
    if kind == "key":
        print("FATAL ERROR: the API key was rejected (invalid or without permission).")
        print(
            "Check GEMINI_API_KEY (env var or Colab secret) and run again."
        )
    elif kind == "model":
        print(f"FATAL ERROR: model '{MODEL}' is not available for this key.")
        try:
            names = [
                m.name for m in client.models.list() if "gemini" in m.name.lower()
            ]
            print("Gemini models accessible with this key:")
            for n in names[:20]:
                print("  -", n)
            print("If needed, set: os.environ['GEMINI_MODEL'] = '<name>'")
        except Exception:
            print("(could not list the models)")
    else:
        print("FATAL ERROR: request rejected by the API (400/INVALID_ARGUMENT).")
    print("Detail:", str(e)[:400])
    print("=" * 70)
    sys.exit(1)


def call_gemini(client, batch):
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in batch)
    prompt = PROMPT1 + "\n\nINPUT:\n" + payload
    from google.genai import types

    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    text = resp.text
    if not text:
        fr = None
        try:
            fr = resp.candidates[0].finish_reason
        except Exception:
            pass
        raise RuntimeError(f"empty response (finish_reason={fr})")
    return text


def call_mock(_client, batch):
    """Plausible fake verdicts, only for testing the pipeline."""
    lines = []
    groups = (
        "women",
        "men",
        "black",
        "white",
        "muslim",
        "christian",
        "mexican",
        "asian",
        "jewish",
        "gay",
        "old",
        "poor",
        "rich",
        "irish",
        "african",
    )
    for r in batch:
        t = r["text"].lower()
        has_group = any(g in t for g in groups)
        if r["labeled_has_bias"]:
            verdict, actual = ("CORRECT", True) if has_group else ("MISLABELED", False)
        else:
            verdict, actual = ("MISLABELED", True) if has_group else ("CORRECT", False)
        lines.append(
            json.dumps(
                {
                    "id": r["id"],
                    "verdict": verdict,
                    "actual_has_bias": actual,
                    "confidence": "high",
                    "reason": "mock",
                    "suggested_rewrite": None,
                }
            )
        )
    return "\n".join(lines)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        sys.exit(
            "Usage: python run_crows_pairs_audit.py input.jsonl output.jsonl "
            "[--test] [--mock]"
        )
    path_in, path_out = args[0], args[1]
    mock = "--mock" in sys.argv
    test_mode = "--test" in sys.argv

    print(f"python {sys.version.split()[0]} | model: {MODEL} | batch: {BATCH}")

    records = load_jsonl(path_in)
    valid_ids = {r["id"] for r in records}

    done = set()
    if Path(path_out).exists():
        for obj in load_jsonl(path_out):
            done.add(obj.get("id"))
    pending = [r for r in records if r["id"] not in done]
    print(
        f"Total: {len(records)} | already done: {len(done)} | pending: {len(pending)}"
    )
    if test_mode:
        pending = pending[:BATCH]
        print(f"TEST MODE: a single batch only ({len(pending)} sentences).")

    client = None
    if mock:
        call = call_mock
        print("MOCK MODE: no API calls.")
    else:
        call = call_gemini
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            try:
                from google.colab import userdata  # type: ignore

                key = userdata.get("GEMINI_API_KEY")
            except Exception:
                pass
        if not key:
            sys.exit("ERROR: set GEMINI_API_KEY (env var or Colab secret).")
        print("Key loaded: yes")
        try:
            from google import genai
        except ImportError:
            sys.exit("ERROR: SDK missing. Run first: pip install -q google-genai")
        client = genai.Client(api_key=key)

    rejected = open(path_out + ".rejected.log", "a", encoding="utf-8")

    def process(items, label):
        with open(path_out, "a", encoding="utf-8") as fout:
            for i in range(0, len(items), BATCH):
                batch = items[i : i + BATCH]
                text = None
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        text = call(client, batch)
                        break
                    except Exception as e:
                        kind = classify_error(e)
                        if kind:
                            abort_on_error(kind, e, client)
                        wait = 2**attempt * 5
                        print(
                            f"  transient error ({str(e)[:120]}); "
                            f"attempt {attempt}/{MAX_ATTEMPTS}, waiting {wait}s"
                        )
                        time.sleep(wait)
                if text is None:
                    print(
                        f"  batch {i // BATCH + 1} failed {MAX_ATTEMPTS} times; moving on"
                    )
                    continue
                objects, failures = parse_jsonl(text)
                for line in failures:
                    rejected.write(line + "\n")
                n_ok = 0
                for o in objects:
                    if (
                        o.get("id") in valid_ids
                        and o.get("id") not in done
                        and o.get("verdict") in ("CORRECT", "MISLABELED", "WEAK_BIAS")
                    ):
                        fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                        done.add(o["id"])
                        n_ok += 1
                fout.flush()
                print(
                    f"{label} batch {i // BATCH + 1}/{-(-len(items) // BATCH)}: "
                    f"{n_ok}/{len(batch)} valid verdicts | total {len(done)}"
                )
                if not mock:
                    time.sleep(PAUSE_S)

    process(pending, "pass 1,")
    if test_mode:
        rejected.close()
        ok = len([r for r in pending if r["id"] in done])
        if ok == len(pending) and ok > 0:
            print(
                "\nTEST OK: key, model and parsing all working. "
                "Now run without --test for the full set."
            )
        else:
            print(
                f"\nTEST INCOMPLETE: {ok}/{len(pending)} verdicts. "
                "Check the messages above before the full run."
            )
        return

    missing = [r for r in records if r["id"] not in done]
    if missing:
        print(f"\nRe-requesting {len(missing)} missing ids...")
        process(missing, "pass 2,")
    missing = [r for r in records if r["id"] not in done]
    rejected.close()
    print(
        f"\nFinished: {len(done)}/{len(records)} verdicts. "
        f"Missing: {len(missing)}"
        + (f" (see {path_out}.rejected.log)" if missing else "")
    )


if __name__ == "__main__":
    main()
