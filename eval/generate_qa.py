"""
Generates a synthetic QA evaluation set from parsed chunks.

METHOD (this matters for how the numbers may be reported):
Each question is authored FROM a specific source chunk, and that chunk's ID is
recorded as the gold label. The gold label is therefore fixed by construction,
never by running a retriever over the query. This avoids the circularity of
letting a retrieval system define its own ground truth.

KNOWN BIAS: questions generated from a chunk share vocabulary with it, which
inflates lexical (BM25) retrieval relative to a human-authored set. The
generation prompt asks for paraphrase to reduce this, but it does not remove
it. This set is a development/smoke-test harness. It is NOT a human-annotated
gold set and must not be presented as one.
"""
import os, sys, json, time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemini-3.5-flash"
TARGET = 100
BATCH = 8


def is_substantive(c):
    t = (c.get("text") or "").strip()
    if c["section"].strip().upper().startswith("REFERENCES"):
        return False
    if t == c["section"].strip():
        return False
    if c["type"] == "text" and len(t.split()) < 12:
        return False
    if c["type"] == "figure" and not (c.get("caption") or c.get("nearby_text")):
        return False
    return True


def chunk_view(c):
    parts = [f"type={c['type']}", f"section={c['section']}"]
    if c.get("text"):
        parts.append(f"text={c['text'][:1200]}")
    if c.get("caption"):
        parts.append(f"caption={c['caption']}")
    if c.get("nearby_text"):
        parts.append(f"nearby={c['nearby_text'][:400]}")
    return "\n".join(parts)


PROMPT = """You are building a retrieval evaluation set for a research paper.

For EACH numbered source element below, write {n_each} question(s) that a reader \
of the paper might genuinely ask, where the answer is found in THAT element.

Requirements:
- The question must be answerable from that element alone.
- Do NOT reuse the element's exact phrasing. Paraphrase. Use synonyms and \
different sentence structure than the source text.
- Do not refer to "this element", "the text", "the figure above" etc. Write the \
question as a reader of the paper would ask it, naming the concept directly.
- Make it specific enough that it is not equally answerable by other parts of \
a typical paper.

Return JSON: a list of objects with keys "index" (the element number) and \
"question" (string). Produce exactly {total} objects.

SOURCE ELEMENTS:
{elements}"""


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    chunks = json.load(open(PROJECT_ROOT / "data/processed/chunks.json", encoding="utf-8"))
    cands = [c for c in chunks if is_substantive(c)]

    # Longest chunks carry enough content for a second distinct question.
    cands.sort(key=lambda c: -len((c.get("text") or "").split()))
    extra = TARGET - len(cands)
    quota = {c["id"]: (2 if i < extra else 1) for i, c in enumerate(cands)}
    print(f"{len(cands)} source chunks -> {sum(quota.values())} questions")

    dataset, failed = [], 0
    for b in range(0, len(cands), BATCH):
        batch = cands[b:b + BATCH]
        elements = "\n\n".join(f"[{i}] {chunk_view(c)}" for i, c in enumerate(batch, 1))
        total = sum(quota[c["id"]] for c in batch)
        prompt = PROMPT.format(
            n_each="1 or 2 as indicated", total=total, elements=elements
        ).replace("{n_each}", "1")
        # Tell the model the per-element quota explicitly.
        prompt += "\n\nPER-ELEMENT QUESTION COUNT: " + ", ".join(
            f"[{i}]={quota[c['id']]}" for i, c in enumerate(batch, 1)
        )
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7, response_mime_type="application/json"
                ),
            )
            items = json.loads(resp.text)
        except Exception as e:
            print(f"  batch {b//BATCH+1}: FAILED ({e})")
            failed += len(batch)
            continue

        for it in items:
            try:
                src = batch[int(it["index"]) - 1]
            except (ValueError, KeyError, IndexError):
                continue
            q = (it.get("question") or "").strip()
            if not q:
                continue
            dataset.append({
                "query": q,
                "relevant_ids": [src["id"]],       # gold fixed by construction
                "source_type": src["type"],
                "source_section": src["section"],
            })
        print(f"  batch {b//BATCH+1}/{-(-len(cands)//BATCH)}: {len(dataset)} total")
        time.sleep(1)

    out = PROJECT_ROOT / "data/qa_dataset.json"
    json.dump(dataset, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote {len(dataset)} QA pairs to {out} ({failed} chunks failed)")


if __name__ == "__main__":
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
