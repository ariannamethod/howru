# HOWRULOG

This log is reverse chronological: newest entries go at the top.

## 2026-07-30 - CodeQL Integer Multiplication Alerts Fixed

Fixed GitHub code scanning alerts 1-3 from CodeQL rule
`cpp/integer-multiplication-cast-to-long` in `howru.c`.

The flagged transformer cache indexes mixed widened `size_t` arithmetic with
`int` sub-expressions such as `h * HD`. `tf_forward` now computes head-stride
values and head offsets as `size_t` before indexing content and RRPRAM KV
caches.

Also removed an unused `ch_dominant` helper so the local `-Wall -Wextra` build
is warning-free.

Verification:

- `cc howru.c -O2 -std=c11 -Wall -Wextra -lm -lsqlite3 -o howru`;
- `cc tests/test_all.c -O2 -Wall -Wextra -lm -o tests/run_tests`;
- `./tests/run_tests` passed with `8 passed, 0 failed`;
- `python3 tests/generation_regression.py` passed;
- `python3 train_howru.py --steps 1 --seq_len 16 --ctx 32 --save /tmp/howru_test_weights.bin` exported a 5.7 MB test weight file.

## 2026-07-30 - Human Surface Seed List Started

Added `corpus/human_surfaces_v0.txt` as the first raw surface list for dataset
construction. It is not final `howru.txt`; it is a seed menu of ordinary human
messages that future generation passes can wrap into
`HUMAN / RESONATING / HOWRU` records.

The list deliberately covers ordinary daily small talk and short human speech:
greetings, check-ins, coffee/food, sleep/body state, work, movement, weather,
phones/messages, unknown names, affection/absence, moods, decisions, music,
jokes, fragments, vulnerability, non-native/damaged English, direct questions
to Howru, and contact-boundary instructions.

## 2026-07-30 - GPT Dataset Notes Read Locally

Codex read `/Users/ataeff/Downloads/4.txt` as local reference-only material.
The file is not part of the repository and its contents should not be copied
into commits unless Oleg explicitly asks for a derived public artifact.

Important reconciliation for future work: the private reference text may use
older placeholder labels. Oleg's newer direction in chat says the template
should be human/howru, so the repo should stay aligned to
`HUMAN -> /RESONATING/ -> HOWRU` unless Oleg explicitly changes that back.

## 2026-07-30 - Project Log Opened

Added `HOWRULOG.md` as the project-local working ledger for Howru. Future
Howru work should record concrete actions here, with the newest entry above
older entries. README is allowed to be a brochure, placeholder, or artistic
surface; operational continuity belongs in this log.

Also recorded the project-log convention in Codex memory so future Arianna
projects get their own `<PROJECT>LOG.md` rather than burying operational history
inside README.

## 2026-07-30 - README Reset To Placeholder

Oleg replaced the utilitarian README with `no readme yet` in commit `c029597`.
This keeps README free for a later stranger/brochure-style Howru presentation.

## 2026-07-30 - Contact Runtime Sanitized

Codex compared `howru` against `ariannamethod/q`, using `postgpt_q.c` and the
notorch training loop as the reference surface. Commit `6c3990c` was created
and pushed to `main`.

Changes made:

- switched the public chat template to `HUMAN -> /RESONATING/ -> HOWRU`;
- renamed q-derived runtime, persistence, trainer, and shim surfaces toward
  Howru vocabulary;
- fixed periodic-table construction to use corpus text instead of token memory;
- made `train_howru.py` self-contained instead of importing missing
  `postgpt_q.py`;
- added tracked `howru.merges` as tokenizer bootstrap;
- added `tools/build_howru_merges.py` for rebuilding merges after the real
  staged corpus exists;
- added C unit smoke tests and generation regression tests under `tests/`;
- kept weights, SQLite memory, binary memory, and spores trackable project
  state rather than ignored artifacts.

Verification:

- `cc howru.c -O2 -std=c11 -lm -lsqlite3 -o howru`;
- `./tests/run_tests` passed with `8 passed, 0 failed`;
- `python3 tests/generation_regression.py` passed;
- `python3 train_howru.py --steps 1 --seq_len 16 --ctx 32 --save /tmp/howru_test_weights.bin` exported a 5.7 MB test weight file.
