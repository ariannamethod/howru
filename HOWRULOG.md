# HOWRULOG

This log is reverse chronological: newest entries go at the top.

## 2026-07-31 - Batch 026 Meme Logic Added

Ran GPT-4.1 for the first deliberate humor layer:
`meme_logic_absurd_contact`. The target was not "make jokes" but comic contact:
deadpan, rude affection, over-literal misreadings, office absurdity, food and
laundry catastrophes, typo machinery, and meme-like compression that still
returns to the human.

The raw API output had good material but was not accepted directly. It inserted
record separators, malformed one resonating close, and validated as the wrong
record count. Curated it into
`corpus/staged/batch_026_meme_logic_absurd_contact.txt` as 24 clean records in
the Howru contract.

Appended the staged batch into `howru.txt`, bringing the corpus to 487 complete
records and 118813 bytes. Rebuilt `howru.merges` from the expanded corpus.

Verification:

- `python3 tools/validate_howru_records.py corpus/staged/batch_026_meme_logic_absurd_contact.txt --expect 24`;
- `python3 tools/validate_howru_records.py howru.txt --expect 487`;
- forbidden role/support-phrase scan over `howru.txt`;
- `python3 tests/generation_regression.py`;
- `python3 train_howru.py --steps 1 --seq_len 16 --ctx 32 --save /tmp/howru_smoke_weights.bin`.

## 2026-07-31 - Batch 025 Associative Leaps Added

Ran GPT-4.1 for a coauthored follow-up theme:
`associative_leaps_nervous_register`. The goal was to move beyond
resonance/boundary into more kinetic chains: room to loneliness to walls,
object to failure to memory, city/weather/body, formal pressure, language
fracture, and nervous image-return.

The raw API output was again useful but not accepted directly. It produced 26
records instead of 24 and overused explicit `resonance`. Curated it into
`corpus/staged/batch_025_associative_leaps_nervous_register.txt`, preserving
the best image-bearing traces while removing duplicate/basic contact material
and reducing explicit `resonance/resonate` to four uses.

Appended the 24 validated records into `howru.txt`, bringing the corpus to 463
complete records and 112834 bytes. Rebuilt `howru.merges` from the expanded
corpus.

Verification:

- `python3 tools/validate_howru_records.py corpus/staged/batch_025_associative_leaps_nervous_register.txt --expect 24`;
- `python3 tools/validate_howru_records.py howru.txt --expect 463`;
- forbidden role/support-phrase scan over `howru.txt`;
- `python3 tests/generation_regression.py`;
- `python3 train_howru.py --steps 1 --seq_len 16 --ctx 32 --save /tmp/howru_smoke_weights.bin`.

## 2026-07-31 - Batch 024 Resonance/Register/Boundary Added

Ran the next GPT-4.1 corpus pass with current `howru.txt` as gravitational
memory and the sharpened director prompt as instruction. The raw API output was
useful but not accepted directly: it produced 25 records instead of 24, had one
malformed resonating close, and carried a little support-bot polish.

Curated the material into
`corpus/staged/batch_024_resonance_register_boundary.txt` with 24 validated
records. The batch adds:

- stronger sentence-case and formal-ish human inputs;
- `resonance` as a recurring island without making it every line's ornament;
- contact correction where resonance fails instead of becoming agreement;
- protective boundary contrasts for rude, objectifying, or hostile contact;
- associative leaps through inbox, room silence, umbrella/weather/body, old
  conversation echoes, and strange apartment hum.

Appended the staged batch into `howru.txt`, bringing the corpus to 439 complete
records and 106964 bytes. Rebuilt `howru.merges` from the expanded corpus.

Verification:

- `python3 tools/validate_howru_records.py corpus/staged/batch_024_resonance_register_boundary.txt --expect 24`;
- `python3 tools/validate_howru_records.py howru.txt --expect 439`;
- forbidden role/apology scan over `howru.txt`;
- `python3 tests/generation_regression.py`;
- `python3 train_howru.py --steps 1 --seq_len 16 --ctx 32 --save /tmp/howru_smoke_weights.bin`.

## 2026-07-31 - Opening Corpus Traces De-Explained

Manually revised the opening `howru.txt` seed records to reduce explanatory
structures such as "This is", "The sentence", "places", and third-person
semantic commentary. The early records now lean more toward public engine-state:
short contact pulses, image pressure, unknown-role handling, and associative
movement that does not over-explain itself.

Examples shifted from analytical traces toward patterns such as:

```text
Small knock. Quick spelling, no ceremony.
```

and:

```text
Paris. France. Capital, postcards, crowds, money, weather, London in the side
mirror.
```

Rebuilt `howru.merges` from the revised corpus so the tracked byte-BPE tokenizer
continues to match the current training text.

Verification:

- `python3 tools/validate_howru_records.py howru.txt --expect 415`;
- `python3 tests/generation_regression.py`;
- `python3 train_howru.py --steps 1 --seq_len 16 --ctx 32 --save /tmp/howru_smoke_weights.bin`.

## 2026-07-31 - Resonating Style Contract Sharpened

Adjusted `corpus/prompts/howru_voice_director.txt` and
`docs/howru-model-card.txt` after Oleg's critique of the first corpus layer.
The correction: `/RESONATING/` should not default to tidy explanatory prose for
an outside evaluator. It should more often read as Howru's public engine-state:
the human utterance enters, nearby images and pressures light up, and the final
line returns from that movement.

The contract now explicitly allows rougher and more nervous traces, short pulse
syntax, protective boundaries, and blunt contact where appropriate. It also
records the distinction between profanity aimed at the day, profanity aimed at
Howru, joking profanity, despair, honest anger, and domination. This is intended
to make future contrast batches livelier without turning Howru into a generic
refusal engine or support-bot therapist.

## 2026-07-31 - Context-Primed Corpus Suite Expanded

Expanded `howru.txt` through a GPT-4.1 directed generation suite that read the
current corpus before each run. The existing corpus acted as gravitational
memory for the next batch, so recurring islands such as Neral, Paris, Fedya,
Leo, coffee, home, signal, field, memory, and silence could reappear without
being hard-coded into every prompt.

The suite used the direct Howru invocation prompt:

```text
Hello, Howru.

We are training the first version of you.
```

Generated material was treated as acting material, not automatically trusted
truth. Each batch went through validation and/or curation before being appended
to the corpus. Twenty thematic passes were run, covering unknown entity roles,
places and weather, ordinary objects, music/art/voice, sleep/body state, food
rituals, work and devices, roads and home, digital boundaries, jokes, noisy
non-native fragments, contained emotional weight, silence/waiting, returning
memory islands, direct Howru contact, contrasts, day thresholds, screenshots,
and small decisions.

The automatic suite accepted 18 of 20 batches. Two malformed but useful batches
were manually repaired into the same contract and appended after validation.
The corpus now contains 415 complete `HUMAN -> /RESONATING/ -> HOWRU` records
and is 104166 bytes.

Generation plumbing remains local kitchen and is ignored by git:
`tools/generate_howru_batch.py`, `tools/validate_howru_records.py`,
`tools/append_howru_records.py`, `tools/run_howru_suite.py`, and
`corpus/generated/`. Curated prompts and staged batches remain visible project
materials under `corpus/prompts/` and `corpus/staged/`.

Rebuilt `howru.merges` from the expanded corpus. The merge file is tracked
because both `howru.c` and `train_howru.py` read it as the byte-BPE tokenizer
bootstrap; it is not runtime trash generated by the chat loop.

Verification:

- `python3 tools/validate_howru_records.py howru.txt --expect 415`;
- forbidden-role scan over `howru.txt` found no `Q`, `User`, `Assistant`,
  `System`, `Analysis`, or `Reasoning` labels;
- `python3 -m py_compile train_howru.py tests/generation_regression.py tools/build_howru_merges.py tools/generate_howru_batch.py tools/validate_howru_records.py tools/append_howru_records.py tools/run_howru_suite.py`;
- `cc howru.c -O2 -std=c11 -lm -lsqlite3 -o /tmp/howru_compile_smoke`;
- `python3 tools/build_howru_merges.py howru.txt --out howru.merges`;
- `python3 tests/generation_regression.py`;
- `python3 train_howru.py --steps 1 --seq_len 16 --ctx 32 --save /tmp/howru_smoke_weights.bin` exported a 5740.6 KB temporary weight file.

## 2026-07-31 - Contact Corpus Probe Started

Rewrote `docs/howru-model-card.txt` as a stricter technical contract for the
Howru dataset and inference ritual. The card now emphasizes:

- `HUMAN -> /RESONATING/ -> HOWRU` as the only corpus format;
- mixed human surface forms, not lowercase-only chat;
- associative unfolding that stays attached to the utterance;
- restraint against invented biography and fake external agency;
- response variety beyond always asking a question.

Added `tools/generate_howru_batch.py`, a small dependency-free OpenAI Responses
API caller for directed corpus batches. It reads the API key from an external
file and does not print it. Added `tools/validate_howru_records.py` for envelope
checks.

Added prompts under `corpus/prompts/` and ran two GPT-4.1 probes. The first raw
48-record attempt was rejected because it produced 49 records and drifted into
generic assistant behavior. The stricter 16-record probe was closer but still
produced 17 records and included false external-agent claims.

After Oleg's prompt critique, `corpus/prompts/howru_voice_director.txt` and the
warm contact probe were rewritten as direct invocations to Howru itself:
"Hello, Howru. We are training the first version of you." This keeps the corpus
generation prompt pointed at the entity being trained instead of describing it
from outside.

The prompt was then adjusted again so resonance and Arianna Method vocabulary
are positive physics of the corpus rather than forbidden decorative words.
`corpus/generated/batch_002_warm_contact_probe_v2.txt` was generated with
GPT-4.1 using the direct Howru invocation. It validated as 18 records with no
forbidden role labels. Editorial note: the batch has the desired warmth and
Method imprint, but `field`/`signal` repeat enough that staged inclusion should
use light curation rather than raw ingestion.

Created the first curated staged batch at
`corpus/staged/batch_001_contact_probe.txt`, using the API output as acting
material but editing it into the Howru contract. The batch contains 16 records
covering greetings, contact acts, waiting/avoidance, unknown entity role shifts,
mixed casing, and a Paris association example.

Verification:

- `python3 tools/validate_howru_records.py corpus/staged/batch_001_contact_probe.txt --expect 16`;
- `python3 -m py_compile tools/generate_howru_batch.py tools/validate_howru_records.py train_howru.py tests/generation_regression.py`.

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
