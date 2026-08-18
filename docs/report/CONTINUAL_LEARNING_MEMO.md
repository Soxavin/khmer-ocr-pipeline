# Continual learning, and what we built for it

*A short explainer for discussing this with my mentor. The full engineering detail — schemas,
CLI commands, code paths — lives in `HITL_RETRAIN.md`; this memo is the concept and the "why,"
not a repeat of that runbook.*

---

## The problem continual learning solves

A model like our OCR pipeline's recognizer is trained once, on the data available at that time.
But the real world keeps producing new documents, and some of those will contain cases the model
still gets wrong — a misread currency symbol, an unusual digit, a word shape the training set
didn't cover well. Two bad ways to handle that:

- **Never retrain.** The model's mistakes never improve, no matter how many more documents we see.
- **Retrain from scratch every time**, on the full dataset plus whatever's new. This works, but
  gets slower and more expensive as the dataset grows, and doesn't make use of the fact that most
  of the model already works fine.

**Continual learning** is the general idea of updating a model incrementally as new data arrives,
instead of either of those. Done carelessly, it has a well-known failure mode called **catastrophic
forgetting**: a model fine-tuned only on new examples can quietly get *worse* on the cases it used
to handle correctly, because nothing is protecting what it already learned.

## What this means for our project, specifically

Every day someone uses the review workspace, they correct cells the model read wrong. Those
corrections are exactly the kind of data continual learning is meant to use: **real, in-domain
mistakes, verified by a human** — not synthetic examples, not guesses. Until recently, those
corrections were just thrown away once the session ended.

What we built (`HITL_RETRAIN.md`) is the pipeline that keeps them instead:

1. **Capture** — when an analyst verifies a corrected cell, save the crop + the correct text.
2. **Curate** — keep only analyst-*verified* corrections (never train on the model's own
   unverified output — that just teaches it to repeat its mistakes), and drop purely cosmetic
   edits that aren't real recognition errors.
3. **Accumulate** — corrections build up over time in one growing, append-only file.
4. **Retrain** (periodically, once enough has accumulated) — mix the corrections into a training
   set and fine-tune.
5. **Gate before deploying** — the retrained model only replaces the current one if it *measurably
   wins* on a held-out test set. If it doesn't win outright, it doesn't ship — a documented
   negative result is fine, a silent regression is not.

That gate step is the actual answer to "how do we avoid catastrophic forgetting" in practice: we
don't try to prove in advance that forgetting won't happen, we just refuse to deploy any update
that regresses.

## Honest status — what's built vs. what isn't

- The capture → curate → accumulate pipeline, and the retrain-and-gate procedure, are **built and
  tested**.
- The one missing wire is hooking capture into the live app's save/verify action — currently the
  pipeline exists but isn't yet triggered automatically by normal workspace use.
- **No retrain has happened on real corrections yet**, because there isn't enough accumulated
  volume — a handful of corrections won't move the numbers. This is infrastructure that pays off
  as it's used, not an immediate accuracy win, and I'm not claiming one.

## One caution from the fine-tuning work I just finished (§4.10 of the report)

Separately, I ran a controlled sweep fine-tuning two other models at different training lengths,
and found the result was **non-monotonic** — training longer didn't reliably mean better, and one
setting that seemed like "more training" actually made both models measurably worse. That's a
useful caution for continual learning too: incremental retraining isn't automatically safe just
because each individual update is small. It's another reason the gate step above isn't optional —
we should assume any retrain *could* make things worse until proven otherwise on held-out data,
not assume more data always helps.
