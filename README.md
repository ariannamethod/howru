# howru

Howru is a small contact-resonance engine in the Arianna Method family.

The runtime format is:

```text
HUMAN: <human utterance>

/RESONATING:
<visible resonance>/

HOWRU: <direct answer>
```

## Build

```sh
cc howru.c -O2 -std=c11 -lm -lsqlite3 -o howru
./howru howru.merges howru.txt
```

With trained weights:

```sh
./howru weights/howru.bin howru.merges howru.txt
```

## Train

```sh
python3 train_howru.py --steps 200 --save weights/howru.bin
```

The trainer uses the vendored `ariannamethod/notorch.c` substrate through
`ctypes`; it does not use PyTorch.

## Tokenizer

`howru.merges` is tracked so the current engine and trainer run from a clean
clone. After replacing the seed corpus with the real staged dataset, rebuild it:

```sh
python3 tools/build_howru_merges.py howru.txt --out howru.merges
```

## Tests

```sh
cc tests/test_all.c -O2 -lm -o tests/run_tests
./tests/run_tests
python3 tests/generation_regression.py
```

Weights, SQLite memory, binary memory, and spores are intended project state,
not disposable artifacts.
