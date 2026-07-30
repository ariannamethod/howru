/*
 * test_all.c - Howru unit tests
 * cc tests/test_all.c -O2 -lm -o tests/run_tests && ./tests/run_tests
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  [TEST] %s... ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while (0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while (0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while (0)

static float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static void rmsnorm(float *out, const float *x, int n) {
    float ms = 0.0f;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms = 1.0f / sqrtf(ms / (float)n + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * ms;
}

static void matmul(float *out, const float *x, const float *w, int n_in, int d_out) {
    for (int d = 0; d < d_out; d++) {
        float v = 0.0f;
        for (int j = 0; j < n_in; j++) v += x[j] * w[d * n_in + j];
        out[d] = v;
    }
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    if (s > 0.0f) for (int i = 0; i < n; i++) x[i] /= s;
}

typedef struct { int a, b, new_id; } BPEMerge;
#define MAX_BPE 1024
#define MAX_VOCAB 1280
typedef struct {
    BPEMerge merges[MAX_BPE];
    int n_merges, vocab_size;
    uint8_t vocab_bytes[MAX_VOCAB][64];
    int vocab_len[MAX_VOCAB];
} BPE;

static int bpe_load(BPE *bpe, const char *path) {
    memset(bpe, 0, sizeof(*bpe));
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1 || n > MAX_BPE) { fclose(f); return 0; }
    bpe->n_merges = (int)n;
    bpe->vocab_size = 256 + (int)n;
    for (int i = 0; i < 256; i++) {
        bpe->vocab_bytes[i][0] = (uint8_t)i;
        bpe->vocab_len[i] = 1;
    }
    for (int i = 0; i < (int)n; i++) {
        uint32_t a, b, nid;
        if (fread(&a, 4, 1, f) != 1 || fread(&b, 4, 1, f) != 1 || fread(&nid, 4, 1, f) != 1) {
            fclose(f);
            return 0;
        }
        if (a >= MAX_VOCAB || b >= MAX_VOCAB || nid >= MAX_VOCAB) { fclose(f); return 0; }
        bpe->merges[i] = (BPEMerge){ (int)a, (int)b, (int)nid };
        int la = bpe->vocab_len[a], lb = bpe->vocab_len[b];
        if (la + lb < 64) {
            memcpy(bpe->vocab_bytes[nid], bpe->vocab_bytes[a], la);
            memcpy(bpe->vocab_bytes[nid] + la, bpe->vocab_bytes[b], lb);
            bpe->vocab_len[nid] = la + lb;
        }
    }
    fclose(f);
    return 1;
}

static int bpe_encode(const BPE *bpe, const uint8_t *text, int tlen, int *out, int maxo) {
    int n = 0;
    for (int i = 0; i < tlen && n < maxo; i++) out[n++] = text[i];
    for (int m = 0; m < bpe->n_merges; m++) {
        int a = bpe->merges[m].a, b = bpe->merges[m].b, nid = bpe->merges[m].new_id;
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (i < n - 1 && out[i] == a && out[i + 1] == b) {
                out[j++] = nid;
                i++;
            } else {
                out[j++] = out[i];
            }
        }
        n = j;
    }
    return n;
}

static int bpe_decode_token(const BPE *bpe, int id, char *buf, int sz) {
    if (id < 0 || id >= bpe->vocab_size || sz <= 0) return 0;
    int len = bpe->vocab_len[id];
    if (len >= sz) len = sz - 1;
    memcpy(buf, bpe->vocab_bytes[id], len);
    buf[len] = 0;
    return len;
}

static void test_clampf(void) {
    TEST("clampf");
    CHECK(clampf(5, 0, 10) == 5, "mid");
    CHECK(clampf(-1, 0, 10) == 0, "lo");
    CHECK(clampf(15, 0, 10) == 10, "hi");
    PASS();
}

static void test_rmsnorm(void) {
    TEST("rmsnorm");
    float x[] = {3, 4}, out[2];
    rmsnorm(out, x, 2);
    float ms = (9 + 16) / 2.0f;
    float sc = 1.0f / sqrtf(ms + 1e-6f);
    CHECK(fabsf(out[0] - 3 * sc) < 1e-5f, "el0");
    CHECK(fabsf(out[1] - 4 * sc) < 1e-5f, "el1");
    PASS();
}

static void test_matmul(void) {
    TEST("matmul");
    float x[] = {1, 2, 3}, w[] = {1, 0, 0, 0, 1, 0}, out[2];
    matmul(out, x, w, 3, 2);
    CHECK(fabsf(out[0] - 1) < 1e-5f, "r0");
    CHECK(fabsf(out[1] - 2) < 1e-5f, "r1");
    PASS();
}

static void test_softmax(void) {
    TEST("softmax");
    float x[] = {1, 2, 3};
    softmax(x, 3);
    CHECK(fabsf(x[0] + x[1] + x[2] - 1) < 1e-5f, "sum");
    CHECK(x[2] > x[1] && x[1] > x[0], "order");
    PASS();
}

static void test_bpe_load(void) {
    TEST("bpe_load");
    BPE bpe;
    CHECK(bpe_load(&bpe, "howru.merges"), "opened");
    CHECK(bpe.n_merges > 0 && bpe.n_merges <= MAX_BPE, "merge count in bounds");
    CHECK(bpe.vocab_size == 256 + bpe.n_merges, "vocab matches merges");
    PASS();
}

static void test_bpe_encode(void) {
    TEST("bpe_encode");
    BPE bpe;
    CHECK(bpe_load(&bpe, "howru.merges"), "opened");
    int ids[64];
    int n = bpe_encode(&bpe, (const uint8_t *)"resonance", 9, ids, 64);
    CHECK(n > 0 && n <= 9, "len ok");
    CHECK(n < 9, "merged");
    PASS();
}

static void test_bpe_roundtrip(void) {
    TEST("bpe_roundtrip");
    BPE bpe;
    CHECK(bpe_load(&bpe, "howru.merges"), "opened");
    const char *text = "Hello world";
    int ids[64];
    int n = bpe_encode(&bpe, (const uint8_t *)text, 11, ids, 64);
    char dec[256] = {0};
    int pos = 0;
    for (int i = 0; i < n; i++) {
        char buf[128];
        int len = bpe_decode_token(&bpe, ids[i], buf, 128);
        memcpy(dec + pos, buf, len);
        pos += len;
    }
    dec[pos] = 0;
    CHECK(strcmp(dec, text) == 0, "matches");
    PASS();
}

static void test_memory_magic(void) {
    TEST("memory_magic");
    uint32_t magic = 0x48575255u;
    FILE *f = fopen("/tmp/test_howru.memory", "wb");
    CHECK(f != NULL, "write open");
    fwrite(&magic, 4, 1, f);
    fclose(f);
    f = fopen("/tmp/test_howru.memory", "rb");
    CHECK(f != NULL, "read open");
    uint32_t rm = 0;
    fread(&rm, 4, 1, f);
    fclose(f);
    remove("/tmp/test_howru.memory");
    CHECK(rm == 0x48575255u, "magic");
    PASS();
}

int main(void) {
    printf("Howru unit tests\n\n");
    test_clampf();
    test_rmsnorm();
    test_matmul();
    test_softmax();
    test_bpe_load();
    test_bpe_encode();
    test_bpe_roundtrip();
    test_memory_magic();
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed ? 1 : 0;
}
