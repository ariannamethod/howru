/*
 * howru.c — Howru contact-resonance engine (with SQLite persistence)
 *
 * Standalone C inference engine with full SQLite memory.
 * Format: HUMAN: <user text> -> /RESONATING/ -> HOWRU: <response>
 *
 * Build: cc howru.c -O2 -std=c11 -lm -lsqlite3 -o howru
 * Run:   ./howru [weights.bin] howru.merges howru.txt
 *
 * (c) 2026 Arianna Method — resonance is unbreakable.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sqlite3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_VOCAB       1280
#define MAX_BPE         1024
#define MAX_SEQ         4096
#define MAX_BIGRAM      65536
#define MAX_TRIGRAM     65536
#define MAX_HEBBIAN     131072
#define MAX_PROPHECY    48
#define MAX_EXPERTS     12
#define DOE_RANK        4
#define MAX_DOCS        32
#define MAX_DOC_TOKENS  128
#define TOP_K           24
#define HWRU_WEIGHT_MAGIC 0x51505451u
#define HWRU_MAGIC      0x48575255u
#define HWRU_VERSION    1u
#define FIELD_WINDOW    16
#define DEFAULT_MAX_NEW 180

enum { CH_FEAR=0, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_CMPLX, N_CHAMBERS };

static float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n, s);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(2); }
    return p;
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
        const float *row = w + (size_t)d * n_in;
        for (int j = 0; j < n_in; j++) v += x[j] * row[j];
        out[d] = v;
    }
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float z = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); z += x[i]; }
    if (z > 0.0f) for (int i = 0; i < n; i++) x[i] /= z;
}

static int argmax(const float *x, int n) {
    int k = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[k]) k = i;
    return k;
}

static int sample_top_p(const float *logits, int V, float temp, float top_p) {
    int idx[TOP_K];
    float val[TOP_K];
    for (int k = 0; k < TOP_K; k++) { idx[k] = 0; val[k] = -1e30f; }
    for (int i = 0; i < V; i++) {
        if (logits[i] <= val[TOP_K - 1]) continue;
        val[TOP_K - 1] = logits[i];
        idx[TOP_K - 1] = i;
        for (int k = TOP_K - 2; k >= 0; k--) {
            if (val[k + 1] <= val[k]) break;
            float tv = val[k];
            val[k] = val[k + 1];
            val[k + 1] = tv;
            int ti = idx[k];
            idx[k] = idx[k + 1];
            idx[k + 1] = ti;
        }
    }
    float mx = val[0], p[TOP_K], z = 0.0f;
    temp = temp < 0.05f ? 0.05f : temp;
    for (int k = 0; k < TOP_K; k++) {
        p[k] = expf((val[k] - mx) / temp);
        z += p[k];
    }
    int nk = TOP_K;
    float cum = 0.0f;
    for (int k = 0; k < TOP_K; k++) {
        cum += p[k] / z;
        if (cum >= top_p) { nk = k + 1; break; }
    }
    float nz = 0.0f;
    for (int k = 0; k < nk; k++) nz += p[k];
    float r = ((float)rand() / (float)RAND_MAX) * nz;
    cum = 0.0f;
    for (int k = 0; k < nk; k++) {
        cum += p[k];
        if (cum >= r) return idx[k];
    }
    return idx[0];
}

/* ───────────────────────── BPE ───────────────────────── */

typedef struct {
    int a, b, new_id;
} BPEMerge;

typedef struct {
    BPEMerge merges[MAX_BPE];
    int n_merges, vocab_size;
    uint8_t vocab_bytes[MAX_VOCAB][64];
    int vocab_len[MAX_VOCAB];
} BPE;

static int bpe_load(BPE *bpe, const char *path) {
    memset(bpe, 0, sizeof(*bpe));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return 0; }
    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1) { fclose(f); return 0; }
    if (n > MAX_BPE) { fprintf(stderr, "ERROR: merges=%u > MAX_BPE\n", n); fclose(f); return 0; }
    bpe->n_merges = (int)n;
    bpe->vocab_size = 256 + (int)n;
    if (bpe->vocab_size > MAX_VOCAB) {
        fprintf(stderr, "ERROR: vocab=%d > MAX_VOCAB\n", bpe->vocab_size);
        fclose(f);
        return 0;
    }
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
        bpe->merges[i] = (BPEMerge) { (int)a, (int)b, (int)nid };
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
        int a = bpe->merges[m].a, b = bpe->merges[m].b, nid = bpe->merges[m].new_id, j = 0;
        for (int i = 0; i < n; i++) {
            if (i < n - 1 && out[i] == a && out[i + 1] == b) {
                out[j++] = nid;
                i++;
            } else {
                out[j++] = out[i];
            }
        }
        n = j;
        if (n >= maxo) break;
    }
    return n;
}

static int bpe_decode_token(const BPE *bpe, int id, char *buf, int sz) {
    if (id < 0 || id >= bpe->vocab_size || sz <= 0) return 0;
    int n = bpe->vocab_len[id];
    if (n >= sz) n = sz - 1;
    memcpy(buf, bpe->vocab_bytes[id], n);
    buf[n] = 0;
    return n;
}

static int bpe_has_boundary(const BPE *bpe, int id) {
    if (id < 0 || id >= bpe->vocab_size) return 0;
    for (int i = 0; i < bpe->vocab_len[id]; i++) {
        int c = bpe->vocab_bytes[id][i];
        if (c == '.' || c == '!' || c == '?') return 1;
    }
    return 0;
}

/* ───────────────────────── MetaWeights ───────────────────────── */

typedef struct {
    int a, b;
    float prob;
} BigramE;

typedef struct {
    int a, b, c;
    float prob;
} TrigramE;

typedef struct {
    int a, b;
    float str;
} HebbE;

typedef struct {
    int target;
    float strength;
    int age;
} ProphecyE;

typedef struct {
    float unigram[MAX_VOCAB];
    BigramE bigrams[MAX_BIGRAM];
    int n_bi;
    TrigramE trigrams[MAX_TRIGRAM];
    int n_tri;
    HebbE hebbs[MAX_HEBBIAN];
    int n_hebb;
    ProphecyE prophecies[MAX_PROPHECY];
    int n_prophecy;
} MetaW;

static int bi_find(const MetaW *m, int a, int b) {
    for (int i = 0; i < m->n_bi; i++)
        if (m->bigrams[i].a == a && m->bigrams[i].b == b) return i;
    return -1;
}

static int tri_find(const MetaW *m, int a, int b, int c) {
    for (int i = 0; i < m->n_tri; i++)
        if (m->trigrams[i].a == a && m->trigrams[i].b == b && m->trigrams[i].c == c) return i;
    return -1;
}

static int hebb_find(const MetaW *m, int a, int b) {
    if (a > b) { int t = a; a = b; b = t; }
    for (int i = 0; i < m->n_hebb; i++)
        if (m->hebbs[i].a == a && m->hebbs[i].b == b) return i;
    return -1;
}

static float meta_bi(const MetaW *m, int a, int b) {
    int i = bi_find(m, a, b);
    return i >= 0 ? m->bigrams[i].prob : 1e-10f;
}

static float meta_tri(const MetaW *m, int a, int b, int c) {
    int i = tri_find(m, a, b, c);
    return i >= 0 ? m->trigrams[i].prob : 1e-10f;
}

static void meta_build(MetaW *m, const int *ids, int n, int V) {
    memset(m, 0, sizeof(*m));
    float total = 0;
    for (int i = 0; i < n; i++)
        if (ids[i] >= 0 && ids[i] < V) { m->unigram[ids[i]] += 1; total += 1; }
    if (total > 0) for (int i = 0; i < V; i++) m->unigram[i] /= total;

    // bigrams
    for (int i = 0; i < n - 1 && m->n_bi < MAX_BIGRAM; i++) {
        int a = ids[i], b = ids[i + 1];
        int k = bi_find(m, a, b);
        if (k >= 0) m->bigrams[k].prob += 1;
        else m->bigrams[m->n_bi++] = (BigramE) { a, b, 1 };
    }
    for (int i = 0; i < m->n_bi; i++) {
        float z = 0;
        for (int j = 0; j < m->n_bi; j++)
            if (m->bigrams[j].a == m->bigrams[i].a) z += m->bigrams[j].prob;
        if (z > 0) m->bigrams[i].prob /= z;
    }

    // trigrams
    for (int i = 0; i < n - 2 && m->n_tri < MAX_TRIGRAM; i++) {
        int a = ids[i], b = ids[i + 1], c = ids[i + 2];
        int k = tri_find(m, a, b, c);
        if (k >= 0) m->trigrams[k].prob += 1;
        else m->trigrams[m->n_tri++] = (TrigramE) { a, b, c, 1 };
    }
    for (int i = 0; i < m->n_tri; i++) {
        float z = 0;
        for (int j = 0; j < m->n_tri; j++)
            if (m->trigrams[j].a == m->trigrams[i].a && m->trigrams[j].b == m->trigrams[i].b)
                z += m->trigrams[j].prob;
        if (z > 0) m->trigrams[i].prob /= z;
    }

    // hebbian
    int lim = n < 12000 ? n : 12000;
    for (int i = 0; i < lim && m->n_hebb < MAX_HEBBIAN; i++) {
        int lo = i - 6;
        if (lo < 0) lo = 0;
        int hi = i + 6;
        if (hi >= lim) hi = lim - 1;
        for (int j = lo; j <= hi; j++) {
            if (j == i) continue;
            int a = ids[i], b = ids[j];
            if (a > b) { int t = a; a = b; b = t; }
            float d = 1.0f / (1.0f + abs(i - j));
            int k = hebb_find(m, a, b);
            if (k >= 0) m->hebbs[k].str += d;
            else if (m->n_hebb < MAX_HEBBIAN)
                m->hebbs[m->n_hebb++] = (HebbE) { a, b, d };
        }
    }
    float mx = 0;
    for (int i = 0; i < m->n_hebb; i++) if (m->hebbs[i].str > mx) mx = m->hebbs[i].str;
    if (mx > 0) for (int i = 0; i < m->n_hebb; i++) m->hebbs[i].str /= mx;
}

static void meta_ingest(MetaW *m, const int *ids, int n, float amt) {
    if (n < 1) return;
    for (int i = 0; i < n; i++)
        if (ids[i] >= 0 && ids[i] < MAX_VOCAB) m->unigram[ids[i]] += amt * 0.02f;
    for (int i = 0; i < n - 1; i++) {
        int k = bi_find(m, ids[i], ids[i + 1]);
        if (k >= 0) m->bigrams[k].prob += amt;
        else if (m->n_bi < MAX_BIGRAM)
            m->bigrams[m->n_bi++] = (BigramE) { ids[i], ids[i + 1], amt };
    }
    for (int i = 0; i < n - 2; i++) {
        int k = tri_find(m, ids[i], ids[i + 1], ids[i + 2]);
        if (k >= 0) m->trigrams[k].prob += amt;
        else if (m->n_tri < MAX_TRIGRAM)
            m->trigrams[m->n_tri++] = (TrigramE) { ids[i], ids[i + 1], ids[i + 2], amt };
    }
    for (int i = 0; i < n; i++) {
        int lo = i - 8;
        if (lo < 0) lo = 0;
        int hi = i + 8;
        if (hi >= n) hi = n - 1;
        for (int j = lo; j <= hi; j++) {
            if (j == i) continue;
            int a = ids[i], b = ids[j];
            if (a > b) { int t = a; a = b; b = t; }
            float d = amt / (1.0f + abs(i - j));
            int k = hebb_find(m, a, b);
            if (k >= 0) m->hebbs[k].str += d;
            else if (m->n_hebb < MAX_HEBBIAN)
                m->hebbs[m->n_hebb++] = (HebbE) { a, b, d };
        }
    }
}

static void meta_hebb(const MetaW *m, const int *ctx, int n, float *out, int V) {
    memset(out, 0, (size_t)V * sizeof(float));
    int start = n > FIELD_WINDOW ? n - FIELD_WINDOW : 0;
    for (int i = start; i < n; i++) {
        int c = ctx[i];
        for (int k = 0; k < m->n_hebb; k++) {
            if (m->hebbs[k].a == c && m->hebbs[k].b < V)
                out[m->hebbs[k].b] += m->hebbs[k].str;
            else if (m->hebbs[k].b == c && m->hebbs[k].a < V)
                out[m->hebbs[k].a] += m->hebbs[k].str;
        }
    }
    float mx = 0;
    for (int i = 0; i < V; i++) if (out[i] > mx) mx = out[i];
    if (mx > 0) for (int i = 0; i < V; i++) out[i] /= mx;
}

static void prophecy_add(MetaW *m, int tok, float s) {
    if (tok < 0) return;
    for (int i = 0; i < m->n_prophecy; i++)
        if (m->prophecies[i].target == tok) {
            if (s > m->prophecies[i].strength) m->prophecies[i].strength = s;
            m->prophecies[i].age = 0;
            return;
        }
    if (m->n_prophecy >= MAX_PROPHECY) {
        int weak = 0;
        for (int i = 1; i < m->n_prophecy; i++)
            if (m->prophecies[i].strength / (1 + m->prophecies[i].age) <
                m->prophecies[weak].strength / (1 + m->prophecies[weak].age))
                weak = i;
        m->prophecies[weak] = m->prophecies[--m->n_prophecy];
    }
    m->prophecies[m->n_prophecy++] = (ProphecyE) { tok, s, 0 };
}

static void prophecy_age(MetaW *m, int chosen) {
    int w = 0;
    for (int i = 0; i < m->n_prophecy; i++) {
        ProphecyE p = m->prophecies[i];
        if (p.target == chosen) continue;
        p.age++;
        p.strength *= 0.995f;
        if (p.age < 80 && p.strength > 0.008f) m->prophecies[w++] = p;
    }
    m->n_prophecy = w;
}

static void meta_prophecy(const MetaW *m, const int *ctx, int n, float *out, int V) {
    memset(out, 0, (size_t)V * sizeof(float));
    int start = n > 12 ? n - 12 : 0;
    for (int i = start; i < n; i++) {
        float d = 1.0f / (1.0f + n - 1 - i);
        for (int k = 0; k < m->n_bi; k++)
            if (m->bigrams[k].a == ctx[i] && m->bigrams[k].b < V)
                out[m->bigrams[k].b] += m->bigrams[k].prob * d;
    }
    if (n >= 2)
        for (int k = 0; k < m->n_tri; k++)
            if (m->trigrams[k].a == ctx[n - 2] && m->trigrams[k].b == ctx[n - 1] && m->trigrams[k].c < V)
                out[m->trigrams[k].c] += 1.5f * m->trigrams[k].prob;
    for (int i = 0; i < m->n_prophecy; i++)
        if (m->prophecies[i].target < V)
            out[m->prophecies[i].target] += m->prophecies[i].strength * logf(1.0f + m->prophecies[i].age);
    float mx = 0;
    for (int i = 0; i < V; i++) if (out[i] > mx) mx = out[i];
    if (mx > 0) for (int i = 0; i < V; i++) out[i] /= mx;
}

/* ───────────────────────── Chambers + Periodic Table ───────────────────────── */

static const float CH_DECAY[N_CHAMBERS] = { 0.90f, 0.93f, 0.85f, 0.97f, 0.88f, 0.94f };
static const float CH_COUPLE[N_CHAMBERS][N_CHAMBERS] = {
    { 0, -.3f, .5f, .4f, -.2f, .1f }, { -.3f, 0, -.4f, -.5f, .5f, .2f },
    { .5f, -.3f, 0, .2f, -.3f, .3f }, { .4f, -.5f, .3f, 0, -.3f, .4f },
    { -.2f, .4f, -.2f, -.3f, 0, .3f }, { .1f, .2f, .3f, .4f, .3f, 0 }
};

typedef struct {
    const char *word;
    int ch;
    float w;
} AffectWord;

static const AffectWord AFFECT[] = {
    { "fear", CH_FEAR, .7f }, { "panic", CH_FEAR, 1 }, { "afraid", CH_FEAR, .8f },
    { "danger", CH_FEAR, .8f }, { "love", CH_LOVE, 1 }, { "miss", CH_LOVE, .8f },
    { "warm", CH_LOVE, .6f }, { "care", CH_LOVE, .7f }, { "heart", CH_LOVE, .7f },
    { "hate", CH_RAGE, .9f }, { "angry", CH_RAGE, 1 }, { "rage", CH_RAGE, 1 },
    { "fuck", CH_RAGE, .5f }, { "burn", CH_RAGE, .7f }, { "nothing", CH_VOID, .8f },
    { "empty", CH_VOID, 1 }, { "alone", CH_VOID, .8f }, { "silence", CH_VOID, .7f },
    { "dead", CH_VOID, .9f }, { "music", CH_FLOW, .6f }, { "flow", CH_FLOW, .8f },
    { "dance", CH_FLOW, .8f }, { "rhythm", CH_FLOW, .8f }, { "breath", CH_FLOW, .6f },
    { "strange", CH_CMPLX, .6f }, { "weird", CH_CMPLX, .7f }, { "why", CH_CMPLX, .5f },
    { "paradox", CH_CMPLX, 1 }, { "maybe", CH_CMPLX, .4f }, { "tired", CH_VOID, .4f },
    { "sleep", CH_VOID, .4f }, { "chest", CH_FEAR, .4f }, { "throat", CH_FEAR, .4f },
    { "pressure", CH_CMPLX, .5f }
};

typedef struct {
    float act[N_CHAMBERS], soma[N_CHAMBERS];
    float presence, debt, trauma, scar, coherence, phase_lock;
} Chambers;

typedef struct {
    char word[32];
    int chamber;
    float mass;
} PeriodicElement;

typedef struct {
    PeriodicElement elements[MAX_VOCAB];
    int n;
} PeriodicTable;

static void periodic_init(PeriodicTable *pt) {
    memset(pt, 0, sizeof(*pt));
    for (size_t i = 0; i < sizeof(AFFECT) / sizeof(AFFECT[0]); i++) {
        if (pt->n >= MAX_VOCAB) break;
        strncpy(pt->elements[pt->n].word, AFFECT[i].word, 31);
        pt->elements[pt->n].word[31] = 0;
        pt->elements[pt->n].chamber = AFFECT[i].ch;
        pt->elements[pt->n].mass = 0.6f;
        pt->n++;
    }
}

static int periodic_find(const PeriodicTable *pt, const char *word) {
    for (int i = 0; i < pt->n; i++)
        if (strcmp(pt->elements[i].word, word) == 0) return i;
    return -1;
}

static void periodic_add(PeriodicTable *pt, const char *word, int chamber, float mass) {
    if (pt->n >= MAX_VOCAB || periodic_find(pt, word) >= 0) return;
    strncpy(pt->elements[pt->n].word, word, 31);
    pt->elements[pt->n].word[31] = 0;
    pt->elements[pt->n].chamber = chamber;
    pt->elements[pt->n].mass = mass;
    pt->n++;
}

static void periodic_build_from_text(PeriodicTable *pt, const char *text) {
    char words[2048][32];
    int n = 0, wi = 0;
    char cur[32] = { 0 };
    for (const char *p = text; *p && n < 2048; p++) {
        if (isalpha((unsigned char)*p) || *p == '\'') {
            if (wi < 31) cur[wi++] = (char)tolower((unsigned char)*p);
        } else if (wi > 0) {
            cur[wi] = 0;
            strcpy(words[n++], cur);
            wi = 0;
        }
    }
    if (wi > 0 && n < 2048) { cur[wi] = 0; strcpy(words[n++], cur); }
    for (int i = 0; i < n && pt->n < MAX_VOCAB; i++) {
        if (periodic_find(pt, words[i]) >= 0) continue;
        float profile[N_CHAMBERS] = { 0 }, total = 0;
        for (int j = (i - 4 > 0 ? i - 4 : 0); j < n && j <= i + 4; j++) {
            if (i == j) continue;
            int idx = periodic_find(pt, words[j]);
            if (idx < 0) continue;
            float decay = 1.0f / (1.0f + abs(i - j));
            profile[pt->elements[idx].chamber] += pt->elements[idx].mass * decay;
            total += decay;
        }
        if (total > 0.1f) {
            int dom = 0;
            for (int k = 1; k < N_CHAMBERS; k++) if (profile[k] > profile[dom]) dom = k;
            float mass = profile[dom] / total;
            if (mass > 0.05f) periodic_add(pt, words[i], dom, mass > 0.8f ? 0.8f : mass);
        }
    }
}

static void chambers_init(Chambers *c) {
    memset(c, 0, sizeof(*c));
    c->act[CH_LOVE] = 0.2f;
    c->act[CH_FLOW] = 0.15f;
}

static void chambers_tick(Chambers *c, int n) {
    for (int z = 0; z < n; z++) {
        float old[N_CHAMBERS];
        memcpy(old, c->act, sizeof(old));
        for (int i = 0; i < N_CHAMBERS; i++) {
            c->act[i] *= CH_DECAY[i];
            for (int j = 0; j < N_CHAMBERS; j++)
                if (i != j) c->act[i] += 0.03f * CH_COUPLE[i][j] * sinf(old[j] - old[i]);
            c->act[i] = clampf(c->act[i], 0, 1);
            c->soma[i] = clampf(0.95f * c->soma[i] + 0.02f * c->act[i], 0, 1);
        }
        c->presence = clampf(0.96f * c->presence + 0.02f * c->act[CH_LOVE] + 0.02f * c->act[CH_FLOW], 0, 1);
        c->scar *= 0.995f;
    }
}

static void chambers_feel(Chambers *c, const char *text, const PeriodicTable *pt) {
    char word[64];
    int n = 0, hits = 0;
    for (const char *p = text;; p++) {
        int ch = (unsigned char)*p;
        if (ch && (isalpha(ch) || ch == '\'')) {
            if (n < 63) word[n++] = (char)tolower(ch);
            continue;
        }
        if (n) {
            word[n] = 0;
            for (size_t i = 0; i < sizeof(AFFECT) / sizeof(AFFECT[0]); i++)
                if (!strcmp(word, AFFECT[i].word)) {
                    c->act[AFFECT[i].ch] = clampf(c->act[AFFECT[i].ch] + 0.16f * AFFECT[i].w, 0, 1);
                    c->soma[AFFECT[i].ch] = clampf(c->soma[AFFECT[i].ch] + 0.08f * AFFECT[i].w, 0, 1);
                    hits++;
                }
            if (pt) {
                int idx = periodic_find(pt, word);
                if (idx >= 0) {
                    int ch2 = pt->elements[idx].chamber;
                    c->act[ch2] = clampf(c->act[ch2] + 0.10f * pt->elements[idx].mass, 0, 1);
                    c->soma[ch2] = clampf(c->soma[ch2] + 0.05f * pt->elements[idx].mass, 0, 1);
                }
            }
            n = 0;
        }
        if (!ch) break;
    }
    if (hits) c->presence = clampf(c->presence + 0.03f * hits, 0, 1);
    c->trauma = clampf(0.94f * c->trauma + 0.03f * c->act[CH_FEAR] + 0.02f * c->act[CH_RAGE] + 0.02f * c->act[CH_VOID], 0, 1);
    c->debt = clampf(0.96f * c->debt + 0.03f * c->act[CH_CMPLX] + 0.02f * c->act[CH_VOID], 0, 1);
    chambers_tick(c, 4);
}

/* ───────────────────────── Howru transformer weights ───────────────────────── */

typedef struct {
    int V, D, NH, NL, CTX, NC, NR, NJ, HD;
    float *tok, *pos;
    struct {
        float *wq, *wk, *vc, *wr, *vr, *wj, *vj, *gw, *gb, *wo, *up, *dn;
    } *L;
    float **kc, **vcc, **vrc;
    float *logits;
} TF;

static int read_floats(FILE *f, float **p, size_t n) {
    *p = xcalloc(n, sizeof(float));
    return fread(*p, sizeof(float), n, f) == n;
}

static int tf_load(TF *t, const char *path) {
    memset(t, 0, sizeof(*t));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: %s\n", path); return 0; }
    uint32_t magic = 0, ver = 0, v, d, nh, nl, ctx, nc, nr, nj, hd;
    if (fread(&magic, 4, 1, f) != 1 || magic != HWRU_WEIGHT_MAGIC) {
        fprintf(stderr, "bad Howru weight magic\n");
        fclose(f);
        return 0;
    }
    if (fread(&ver, 4, 1, f) != 1 || fread(&v, 4, 1, f) != 1 || fread(&d, 4, 1, f) != 1 ||
        fread(&nh, 4, 1, f) != 1 || fread(&nl, 4, 1, f) != 1 || fread(&ctx, 4, 1, f) != 1 ||
        fread(&nc, 4, 1, f) != 1 || fread(&nr, 4, 1, f) != 1 || fread(&nj, 4, 1, f) != 1 ||
        fread(&hd, 4, 1, f) != 1) {
        fclose(f);
        return 0;
    }
    t->V = v;
    t->D = d;
    t->NH = nh;
    t->NL = nl;
    t->CTX = ctx;
    t->NC = nc;
    t->NR = nr;
    t->NJ = nj;
    t->HD = hd;
    if (t->V > MAX_VOCAB || t->CTX > MAX_SEQ) {
        fprintf(stderr, "model exceeds compiled limits\n");
        fclose(f);
        return 0;
    }
    int nm = (nc > 0) + (nr > 0) + (nj > 0);
    if (!read_floats(f, &t->tok, (size_t)v * d) || !read_floats(f, &t->pos, (size_t)ctx * d)) {
        fclose(f);
        return 0;
    }
    t->L = xcalloc(nl, sizeof(t->L[0]));
    for (int li = 0; li < (int)nl; li++) {
        if (nc > 0) {
            if (!read_floats(f, &t->L[li].wq, (size_t)nc * hd * d) ||
                !read_floats(f, &t->L[li].wk, (size_t)nc * hd * d) ||
                !read_floats(f, &t->L[li].vc, (size_t)nc * hd * d)) {
                fclose(f);
                return 0;
            }
        }
        if (nr > 0) {
            if (!read_floats(f, &t->L[li].wr, (size_t)nr * d * ctx) ||
                !read_floats(f, &t->L[li].vr, (size_t)nr * hd * d)) {
                fclose(f);
                return 0;
            }
        }
        if (nj > 0) {
            if (!read_floats(f, &t->L[li].wj, (size_t)nj * hd * d) ||
                !read_floats(f, &t->L[li].vj, (size_t)nj * hd * d)) {
                fclose(f);
                return 0;
            }
        }
        if (nm > 1) {
            if (!read_floats(f, &t->L[li].gw, (size_t)nm * d) ||
                !read_floats(f, &t->L[li].gb, nm)) {
                fclose(f);
                return 0;
            }
        }
        if (!read_floats(f, &t->L[li].wo, (size_t)d * d) ||
            !read_floats(f, &t->L[li].up, (size_t)4 * d * d) ||
            !read_floats(f, &t->L[li].dn, (size_t)d * 4 * d)) {
            fclose(f);
            return 0;
        }
    }
    t->kc = xcalloc(nl, sizeof(float *));
    t->vcc = xcalloc(nl, sizeof(float *));
    t->vrc = xcalloc(nl, sizeof(float *));
    for (int li = 0; li < (int)nl; li++) {
        t->kc[li] = xcalloc((size_t)ctx * (nc ? (size_t)nc * hd : 1), sizeof(float));
        t->vcc[li] = xcalloc((size_t)ctx * (nc ? (size_t)nc * hd : 1), sizeof(float));
        t->vrc[li] = xcalloc((size_t)ctx * (nr ? (size_t)nr * hd : 1), sizeof(float));
    }
    t->logits = xcalloc(v, sizeof(float));
    fclose(f);
    return 1;
}

static void tf_forward(TF *t, int tok, int pos) {
    int D = t->D, HD = t->HD, NC = t->NC, NR = t->NR, NJ = t->NJ;
    int nm = (NC > 0) + (NR > 0) + (NJ > 0), sl = pos + 1;
    size_t nc_hd = (size_t)NC * (size_t)HD;
    size_t nr_hd = (size_t)NR * (size_t)HD;
    size_t nj_hd = (size_t)NJ * (size_t)HD;
    float *x = xcalloc(D, sizeof(float));
    float *xn = xcalloc(D, sizeof(float));
    float *res = xcalloc(D, sizeof(float));
    for (int d = 0; d < D; d++)
        x[d] = t->tok[(size_t)tok * D + d] + t->pos[(size_t)pos * D + d];

    for (int li = 0; li < t->NL; li++) {
        memcpy(res, x, (size_t)D * sizeof(float));
        rmsnorm(xn, x, D);
        float *co = NULL, *ro = NULL, *jo = NULL;

        if (NC > 0) {
            co = xcalloc(nc_hd, sizeof(float));
            float *q = xcalloc(nc_hd, sizeof(float));
            float *k = xcalloc(nc_hd, sizeof(float));
            float *v = xcalloc(nc_hd, sizeof(float));
            matmul(q, xn, t->L[li].wq, D, NC * HD);
            matmul(k, xn, t->L[li].wk, D, NC * HD);
            matmul(v, xn, t->L[li].vc, D, NC * HD);
            memcpy(t->kc[li] + (size_t)pos * nc_hd, k, nc_hd * sizeof(float));
            memcpy(t->vcc[li] + (size_t)pos * nc_hd, v, nc_hd * sizeof(float));
            for (int h = 0; h < NC; h++) {
                size_t h_off = (size_t)h * (size_t)HD;
                float *sc = xcalloc(sl, sizeof(float));
                for (int p = 0; p < sl; p++) {
                    float dot = 0;
                    for (int d = 0; d < HD; d++)
                        dot += q[h_off + (size_t)d] * t->kc[li][(size_t)p * nc_hd + h_off + (size_t)d];
                    sc[p] = dot / sqrtf((float)HD);
                }
                softmax(sc, sl);
                for (int d = 0; d < HD; d++) {
                    float z = 0;
                    for (int p = 0; p < sl; p++)
                        z += sc[p] * t->vcc[li][(size_t)p * nc_hd + h_off + (size_t)d];
                    co[h_off + (size_t)d] = z;
                }
                free(sc);
            }
            free(q);
            free(k);
            free(v);
        }

        if (NR > 0) {
            ro = xcalloc(nr_hd, sizeof(float));
            float *v = xcalloc(nr_hd, sizeof(float));
            matmul(v, xn, t->L[li].vr, D, NR * HD);
            memcpy(t->vrc[li] + (size_t)pos * nr_hd, v, nr_hd * sizeof(float));
            for (int h = 0; h < NR; h++) {
                size_t h_off = (size_t)h * (size_t)HD;
                float *sc = xcalloc(sl, sizeof(float));
                for (int p = 0; p < sl; p++) {
                    float z = 0;
                    for (int d = 0; d < D; d++)
                        z += xn[d] * t->L[li].wr[((size_t)h * D + d) * t->CTX + p];
                    sc[p] = z;
                }
                softmax(sc, sl);
                for (int d = 0; d < HD; d++) {
                    float z = 0;
                    for (int p = 0; p < sl; p++)
                        z += sc[p] * t->vrc[li][(size_t)p * nr_hd + h_off + (size_t)d];
                    ro[h_off + (size_t)d] = z;
                }
                free(sc);
            }
            free(v);
        }

        if (NJ > 0) {
            jo = xcalloc(nj_hd, sizeof(float));
            float *w = xcalloc(nj_hd, sizeof(float));
            float *v = xcalloc(nj_hd, sizeof(float));
            matmul(w, xn, t->L[li].wj, D, NJ * HD);
            matmul(v, xn, t->L[li].vj, D, NJ * HD);
            float norm = 0;
            for (int d = 0; d < NJ * HD; d++) norm += w[d] * w[d];
            norm = 1.0f / sqrtf(norm + 1e-8f);
            for (int d = 0; d < NJ * HD; d++) jo[d] = v[d] * (w[d] * norm);
            free(w);
            free(v);
        }

        float *comb = xcalloc(D, sizeof(float));
        if (nm > 1 && t->L[li].gw) {
            float *gl = xcalloc(nm, sizeof(float));
            matmul(gl, xn, t->L[li].gw, D, nm);
            float gates[3] = { 1, 1, 1 };
            for (int g = 0; g < nm; g++) gates[g] = 1.0f / (1.0f + expf(-(gl[g] + t->L[li].gb[g])));
            free(gl);
            int off = 0, g = 0;
            if (NC) { for (int d = 0; d < NC * HD; d++) comb[off + d] = gates[g] * co[d];
                off += NC * HD;
                g++; }
            if (NR) { for (int d = 0; d < NR * HD; d++) comb[off + d] = gates[g] * ro[d];
                off += NR * HD;
                g++; }
            if (NJ) { for (int d = 0; d < NJ * HD; d++) comb[off + d] = gates[g] * jo[d]; }
        } else {
            int off = 0;
            if (co) { memcpy(comb + off, co, (size_t)NC * HD * sizeof(float));
                off += NC * HD; }
            if (ro) { memcpy(comb + off, ro, (size_t)NR * HD * sizeof(float));
                off += NR * HD; }
            if (jo) memcpy(comb + off, jo, (size_t)NJ * HD * sizeof(float));
        }
        free(co);
        free(ro);
        free(jo);

        float *proj = xcalloc(D, sizeof(float));
        matmul(proj, comb, t->L[li].wo, D, D);
        for (int d = 0; d < D; d++) x[d] = res[d] + proj[d];
        free(proj);
        free(comb);

        memcpy(res, x, (size_t)D * sizeof(float));
        rmsnorm(xn, x, D);
        float *up = xcalloc((size_t)4 * D, sizeof(float));
        matmul(up, xn, t->L[li].up, D, 4 * D);
        for (int d = 0; d < 4 * D; d++) if (up[d] < 0) up[d] = 0;
        float *dn = xcalloc(D, sizeof(float));
        matmul(dn, up, t->L[li].dn, 4 * D, D);
        for (int d = 0; d < D; d++) x[d] = res[d] + dn[d];
        free(up);
        free(dn);
    }

    rmsnorm(xn, x, D);
    for (int v = 0; v < t->V; v++) {
        float z = 0;
        for (int d = 0; d < D; d++) z += xn[d] * t->tok[(size_t)v * D + d];
        t->logits[v] = z;
    }
    float mag = 0;
    for (int v = 0; v < t->V; v++) mag += fabsf(t->logits[v]);
    mag /= t->V;
    float gate = clampf((mag - 0.5f) / 1.5f, 0, 1);
    for (int v = 0; v < t->V; v++) t->logits[v] *= gate;
    free(x);
    free(xn);
    free(res);
}

static void tf_make_silent(TF *t, int V) {
    memset(t, 0, sizeof(*t));
    t->V = V;
    t->D = 48;
    t->NH = 4;
    t->NL = 1;
    t->CTX = 256;
    t->NC = 2;
    t->NR = 2;
    t->HD = 12;
    t->tok = xcalloc((size_t)V * t->D, sizeof(float));
    t->pos = xcalloc((size_t)t->CTX * t->D, sizeof(float));
    t->L = xcalloc(1, sizeof(t->L[0]));
    t->L[0].wq = xcalloc((size_t)t->NC * t->HD * t->D, sizeof(float));
    t->L[0].wk = xcalloc((size_t)t->NC * t->HD * t->D, sizeof(float));
    t->L[0].vc = xcalloc((size_t)t->NC * t->HD * t->D, sizeof(float));
    t->L[0].wr = xcalloc((size_t)t->NR * t->D * t->CTX, sizeof(float));
    t->L[0].vr = xcalloc((size_t)t->NR * t->HD * t->D, sizeof(float));
    t->L[0].wo = xcalloc((size_t)t->D * t->D, sizeof(float));
    t->L[0].up = xcalloc((size_t)4 * t->D * t->D, sizeof(float));
    t->L[0].dn = xcalloc((size_t)t->D * 4 * t->D, sizeof(float));
    t->kc = xcalloc(1, sizeof(float *));
    t->vcc = xcalloc(1, sizeof(float *));
    t->vrc = xcalloc(1, sizeof(float *));
    t->kc[0] = xcalloc((size_t)t->CTX * t->NC * t->HD, sizeof(float));
    t->vcc[0] = xcalloc((size_t)t->CTX * t->NC * t->HD, sizeof(float));
    t->vrc[0] = xcalloc((size_t)t->CTX * t->NR * t->HD, sizeof(float));
    t->logits = xcalloc(V, sizeof(float));
}

/* ───────────────────────── Parliament (DOE) ───────────────────────── */

typedef struct {
    float *A, *B, *trace;
    float vitality, overload, resonance, plasticity;
    int age, low_steps;
} Expert;

typedef struct {
    Expert ex[MAX_EXPERTS];
    int n, D;
    float alpha, last_entropy, last_diversity;
} Parliament;

static void expert_init(Expert *e, int D) {
    memset(e, 0, sizeof(*e));
    e->A = xcalloc((size_t)DOE_RANK * D, sizeof(float));
    e->B = xcalloc((size_t)D * DOE_RANK, sizeof(float));
    e->trace = xcalloc((size_t)DOE_RANK * D, sizeof(float));
    e->vitality = 1;
    for (int i = 0; i < DOE_RANK * D; i++)
        e->A[i] = 0.01f * ((float)rand() / (float)RAND_MAX - 0.5f);
    for (int i = 0; i < D * DOE_RANK; i++)
        e->B[i] = 0.01f * ((float)rand() / (float)RAND_MAX - 0.5f);
}

static void parl_init(Parliament *p, int D) {
    memset(p, 0, sizeof(*p));
    p->D = D;
    p->n = 4;
    p->alpha = 0.05f;
    for (int i = 0; i < p->n; i++) expert_init(&p->ex[i], D);
}

static void expert_forward(const Expert *e, const float *x, float *out, int D) {
    float mid[DOE_RANK] = { 0 };
    for (int r = 0; r < DOE_RANK; r++)
        for (int d = 0; d < D; d++) mid[r] += e->A[r * D + d] * x[d];
    for (int o = 0; o < D; o++)
        for (int r = 0; r < DOE_RANK; r++)
            out[o] += e->B[o * DOE_RANK + r] * mid[r];
}

static void parl_inject(Parliament *p, float *logits, const float *x, int V) {
    if (!p || p->n <= 0) return;
    float votes[MAX_EXPERTS], *outs[MAX_EXPERTS];
    float vmax = -1e30f;
    for (int i = 0; i < p->n; i++) {
        outs[i] = xcalloc(p->D, sizeof(float));
        expert_forward(&p->ex[i], x, outs[i], p->D);
        votes[i] = 0;
        for (int d = 0; d < p->D; d++) votes[i] += outs[i][d] * x[d];
        if (votes[i] > vmax) vmax = votes[i];
    }
    float pr[MAX_EXPERTS], z = 0, H = 0;
    for (int i = 0; i < p->n; i++) {
        pr[i] = expf(votes[i] - vmax);
        z += pr[i];
    }
    for (int i = 0; i < p->n; i++) {
        pr[i] /= z;
        if (pr[i] > 1e-10f) H -= pr[i] * logf(pr[i]);
    }
    p->last_entropy = H / logf((float)(p->n > 1 ? p->n : 2));
    int k = 1 + (int)((p->n - 1) * p->last_entropy);
    if (k > p->n) k = p->n;
    unsigned char used[MAX_EXPERTS] = { 0 };
    float result[MAX_EXPERTS] = { 0 };
    for (int d = 0; d < p->D; d++) result[d] = 0;
    float diversity = 0;
    for (int pick = 0; pick < k; pick++) {
        int best = -1;
        float bs = -1e30f;
        for (int i = 0; i < p->n; i++)
            if (!used[i]) {
                float score = votes[i] - 0.06f * p->ex[i].overload + 0.04f * (1 - p->ex[i].vitality);
                if (score > bs) { bs = score;
                    best = i; }
            }
        if (best < 0) break;
        used[best] = 1;
        diversity += 1.0f - pr[best];
        for (int d = 0; d < p->D; d++) result[d] += pr[best] * outs[best][d];
        p->ex[best].vitality = clampf(0.9f * p->ex[best].vitality + 0.1f * pr[best], 0, 1);
        p->ex[best].overload = clampf(0.92f * p->ex[best].overload + 0.12f * (pr[best] > 0.35f ? pr[best] - 0.35f : 0), 0, 1);
        p->ex[best].resonance = 0.9f * p->ex[best].resonance + 0.1f * votes[best];
        p->ex[best].low_steps = 0;
    }
    p->last_diversity = k ? diversity / k : 0;
    for (int i = 0; i < p->n; i++) {
        if (!used[i]) { p->ex[i].low_steps++;
            p->ex[i].overload *= 0.94f; }
        free(outs[i]);
    }
    int n = V < p->D ? V : p->D;
    for (int i = 0; i < n; i++) logits[i] += p->alpha * result[i];
}

static void parl_learn(Parliament *p, const float *x, const float *debt) {
    if (!p) return;
    for (int e = 0; e < p->n; e++) {
        for (int r = 0; r < DOE_RANK; r++) {
            float u = 0;
            for (int d = 0; d < p->D; d++)
                u += p->ex[e].B[d * DOE_RANK + r] * debt[d];
            for (int d = 0; d < p->D; d++) {
                int idx = r * p->D + d;
                float delta = 0.0008f * x[d] * u;
                p->ex[e].A[idx] += delta;
                p->ex[e].trace[idx] = 0.97f * p->ex[e].trace[idx] + 0.03f * delta;
                p->ex[e].plasticity += fabsf(delta);
            }
        }
        if (p->ex[e].plasticity > 0.002f) {
            float norm = 1e-8f;
            for (int i = 0; i < DOE_RANK * p->D; i++) norm += fabsf(p->ex[e].trace[i]);
            norm /= DOE_RANK * p->D;
            for (int i = 0; i < DOE_RANK * p->D; i++) {
                p->ex[e].A[i] += 0.03f * p->ex[e].trace[i] / norm;
                p->ex[e].trace[i] *= 0.5f;
            }
            p->ex[e].plasticity *= 0.35f;
            p->ex[e].vitality = clampf(p->ex[e].vitality + 0.03f, 0, 1);
        }
        p->ex[e].age++;
    }
}

/* ───────────────────────── Resonant docs ───────────────────────── */

typedef struct {
    char name[128];
    int tok[MAX_DOC_TOKENS];
    int n;
} ResonantDoc;

typedef struct {
    ResonantDoc d[MAX_DOCS];
    int n;
} Docs;

static void docs_load(Docs *ds, const char *dir, const BPE *bpe) {
    memset(ds, 0, sizeof(*ds));
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) && ds->n < MAX_DOCS) {
        const char *dot = strrchr(e->d_name, '.');
        if (!dot || strcmp(dot, ".txt")) continue;
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        FILE *f = fopen(path, "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint8_t *raw = xcalloc((size_t)sz + 1, 1);
        fread(raw, 1, sz, f);
        fclose(f);
        int *ids = xcalloc((size_t)sz + 1, sizeof(int));
        int n = bpe_encode(bpe, raw, (int)sz, ids, (int)sz);
        ResonantDoc *rd = &ds->d[ds->n];
        snprintf(rd->name, sizeof(rd->name), "%.*s", (int)sizeof(rd->name) - 1, e->d_name);
        int freq[MAX_VOCAB] = { 0 };
        for (int i = 0; i < n; i++) if (ids[i] >= 0 && ids[i] < MAX_VOCAB) freq[ids[i]]++;
        for (int k = 0; k < MAX_DOC_TOKENS; k++) {
            int best = -1;
            for (int i = 0; i < bpe->vocab_size; i++)
                if (freq[i] > 0 && (best < 0 || freq[i] > freq[best])) best = i;
            if (best < 0) break;
            rd->tok[rd->n++] = best;
            freq[best] = 0;
        }
        if (rd->n) ds->n++;
        free(ids);
        free(raw);
    }
    closedir(d);
}

static const ResonantDoc *docs_choose(const Docs *ds, const int *input, int n, const MetaW *m) {
    if (!ds || ds->n == 0) return NULL;
    int best = 0;
    float bs = -1;
    for (int di = 0; di < ds->n; di++) {
        float s = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < ds->d[di].n; j++) {
                if (input[i] == ds->d[di].tok[j]) s += 1;
                int h = hebb_find(m, input[i], ds->d[di].tok[j]);
                if (h >= 0) s += 0.08f * m->hebbs[h].str;
            }
        if (s > bs) { bs = s;
            best = di; }
    }
    return &ds->d[best];
}

/* ───────────────────────── SQLite persistence ───────────────────────── */

static int howru_sqlite_load(MetaW *mw, const char *path, PeriodicTable *pt, Chambers *ch) {
    if (access(path, F_OK) != 0) return 0;
    sqlite3 *db;
    if (sqlite3_open(path, &db) != SQLITE_OK) return 0;
    char *err = NULL;
    // bigrams
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS bigrams(a INTEGER,b INTEGER,prob REAL,PRIMARY KEY(a,b));", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS trigrams(a INTEGER,b INTEGER,c INTEGER,prob REAL,PRIMARY KEY(a,b,c));", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS hebb(a INTEGER,b INTEGER,strength REAL,PRIMARY KEY(a,b));", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS prophecies(target INTEGER PRIMARY KEY,strength REAL,age INTEGER);", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS periodic_elements(word TEXT PRIMARY KEY,chamber INTEGER,mass REAL);", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS chambers(id INTEGER PRIMARY KEY CHECK(id=1),presence REAL,debt REAL,trauma REAL,soma0 REAL,soma1 REAL,soma2 REAL,soma3 REAL,soma4 REAL,soma5 REAL);", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);", NULL, NULL, &err);

    sqlite3_stmt *stmt;
    // bigrams
    sqlite3_prepare_v2(db, "SELECT a,b,prob FROM bigrams;", -1, &stmt, NULL);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int a = sqlite3_column_int(stmt, 0);
        int b = sqlite3_column_int(stmt, 1);
        float p = (float)sqlite3_column_double(stmt, 2);
        int found = 0;
        for (int j = 0; j < mw->n_bi; j++)
            if (mw->bigrams[j].a == a && mw->bigrams[j].b == b) {
                if (p > mw->bigrams[j].prob) mw->bigrams[j].prob = p;
                found = 1;
                break;
            }
        if (!found && mw->n_bi < MAX_BIGRAM) {
            mw->bigrams[mw->n_bi].a = a;
            mw->bigrams[mw->n_bi].b = b;
            mw->bigrams[mw->n_bi].prob = p;
            mw->n_bi++;
        }
    }
    sqlite3_finalize(stmt);
    // trigrams
    sqlite3_prepare_v2(db, "SELECT a,b,c,prob FROM trigrams;", -1, &stmt, NULL);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int a = sqlite3_column_int(stmt, 0);
        int b = sqlite3_column_int(stmt, 1);
        int c = sqlite3_column_int(stmt, 2);
        float p = (float)sqlite3_column_double(stmt, 3);
        if (mw->n_tri < MAX_TRIGRAM) {
            mw->trigrams[mw->n_tri].a = a;
            mw->trigrams[mw->n_tri].b = b;
            mw->trigrams[mw->n_tri].c = c;
            mw->trigrams[mw->n_tri].prob = p;
            mw->n_tri++;
        }
    }
    sqlite3_finalize(stmt);
    // hebb
    sqlite3_prepare_v2(db, "SELECT a,b,strength FROM hebb;", -1, &stmt, NULL);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int a = sqlite3_column_int(stmt, 0);
        int b = sqlite3_column_int(stmt, 1);
        float p = (float)sqlite3_column_double(stmt, 2);
        if (mw->n_hebb < MAX_HEBBIAN) {
            mw->hebbs[mw->n_hebb].a = a;
            mw->hebbs[mw->n_hebb].b = b;
            mw->hebbs[mw->n_hebb].str = p;
            mw->n_hebb++;
        }
    }
    sqlite3_finalize(stmt);
    // prophecies
    sqlite3_prepare_v2(db, "SELECT target,strength,age FROM prophecies ORDER BY age DESC;", -1, &stmt, NULL);
    while (sqlite3_step(stmt) == SQLITE_ROW && mw->n_prophecy < MAX_PROPHECY) {
        mw->prophecies[mw->n_prophecy].target = sqlite3_column_int(stmt, 0);
        mw->prophecies[mw->n_prophecy].strength = (float)sqlite3_column_double(stmt, 1);
        mw->prophecies[mw->n_prophecy].age = sqlite3_column_int(stmt, 2);
        mw->n_prophecy++;
    }
    sqlite3_finalize(stmt);
    // periodic
    sqlite3_prepare_v2(db, "SELECT word,chamber,mass FROM periodic_elements;", -1, &stmt, NULL);
    while (sqlite3_step(stmt) == SQLITE_ROW && pt->n < MAX_VOCAB) {
        const unsigned char *word = sqlite3_column_text(stmt, 0);
        int ch = sqlite3_column_int(stmt, 1);
        float mass = (float)sqlite3_column_double(stmt, 2);
        periodic_add(pt, (const char *)word, ch, mass);
    }
    sqlite3_finalize(stmt);
    // chambers
    sqlite3_prepare_v2(db, "SELECT presence,debt,trauma,soma0,soma1,soma2,soma3,soma4,soma5 FROM chambers WHERE id=1;", -1, &stmt, NULL);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        ch->presence = clampf((float)sqlite3_column_double(stmt, 0), 0, 1);
        ch->debt = clampf((float)sqlite3_column_double(stmt, 1), 0, 1);
        ch->trauma = clampf((float)sqlite3_column_double(stmt, 2), 0, 1);
        for (int i = 0; i < 6; i++) {
            ch->soma[i] = clampf((float)sqlite3_column_double(stmt, 3 + i), 0, 1);
            ch->act[i] = clampf(ch->act[i] > 0.25f * ch->soma[i] ? ch->act[i] : 0.25f * ch->soma[i], 0, 1);
        }
    }
    sqlite3_finalize(stmt);
    // meta: scar, coherence, phase_lock
    sqlite3_prepare_v2(db, "SELECT value FROM meta WHERE key='scar';", -1, &stmt, NULL);
    if (sqlite3_step(stmt) == SQLITE_ROW)
        ch->scar = clampf((float)sqlite3_column_double(stmt, 0), 0, 1);
    sqlite3_finalize(stmt);
    sqlite3_prepare_v2(db, "SELECT value FROM meta WHERE key='coherence';", -1, &stmt, NULL);
    if (sqlite3_step(stmt) == SQLITE_ROW)
        ch->coherence = clampf((float)sqlite3_column_double(stmt, 0), 0, 1);
    sqlite3_finalize(stmt);
    sqlite3_prepare_v2(db, "SELECT value FROM meta WHERE key='phase_lock';", -1, &stmt, NULL);
    if (sqlite3_step(stmt) == SQLITE_ROW)
        ch->phase_lock = clampf((float)sqlite3_column_double(stmt, 0), 0, 1);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
    return 1;
}

static int howru_sqlite_save(const MetaW *mw, const char *path, const PeriodicTable *pt, const Chambers *ch) {
    sqlite3 *db;
    if (sqlite3_open(path, &db) != SQLITE_OK) return 0;
    char *err = NULL;
    sqlite3_exec(db, "BEGIN;", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS bigrams(a INTEGER,b INTEGER,prob REAL,PRIMARY KEY(a,b));", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS trigrams(a INTEGER,b INTEGER,c INTEGER,prob REAL,PRIMARY KEY(a,b,c));", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS hebb(a INTEGER,b INTEGER,strength REAL,PRIMARY KEY(a,b));", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS prophecies(target INTEGER PRIMARY KEY,strength REAL,age INTEGER);", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS periodic_elements(word TEXT PRIMARY KEY,chamber INTEGER,mass REAL);", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS chambers(id INTEGER PRIMARY KEY CHECK(id=1),presence REAL,debt REAL,trauma REAL,soma0 REAL,soma1 REAL,soma2 REAL,soma3 REAL,soma4 REAL,soma5 REAL);", NULL, NULL, &err);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);", NULL, NULL, &err);
    sqlite3_exec(db, "DELETE FROM bigrams; DELETE FROM trigrams; DELETE FROM hebb; DELETE FROM prophecies; DELETE FROM periodic_elements; DELETE FROM chambers; DELETE FROM meta;", NULL, NULL, &err);

    sqlite3_stmt *stmt;
    // bigrams
    sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO bigrams(a,b,prob) VALUES(?,?,?);", -1, &stmt, NULL);
    for (int i = 0; i < mw->n_bi; i++) {
        sqlite3_bind_int(stmt, 1, mw->bigrams[i].a);
        sqlite3_bind_int(stmt, 2, mw->bigrams[i].b);
        sqlite3_bind_double(stmt, 3, mw->bigrams[i].prob);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
    // trigrams
    sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO trigrams(a,b,c,prob) VALUES(?,?,?,?);", -1, &stmt, NULL);
    for (int i = 0; i < mw->n_tri; i++) {
        sqlite3_bind_int(stmt, 1, mw->trigrams[i].a);
        sqlite3_bind_int(stmt, 2, mw->trigrams[i].b);
        sqlite3_bind_int(stmt, 3, mw->trigrams[i].c);
        sqlite3_bind_double(stmt, 4, mw->trigrams[i].prob);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
    // hebb
    sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO hebb(a,b,strength) VALUES(?,?,?);", -1, &stmt, NULL);
    for (int i = 0; i < mw->n_hebb; i++) {
        sqlite3_bind_int(stmt, 1, mw->hebbs[i].a);
        sqlite3_bind_int(stmt, 2, mw->hebbs[i].b);
        sqlite3_bind_double(stmt, 3, mw->hebbs[i].str);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
    // prophecies
    sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO prophecies(target,strength,age) VALUES(?,?,?);", -1, &stmt, NULL);
    for (int i = 0; i < mw->n_prophecy; i++) {
        sqlite3_bind_int(stmt, 1, mw->prophecies[i].target);
        sqlite3_bind_double(stmt, 2, mw->prophecies[i].strength);
        sqlite3_bind_int(stmt, 3, mw->prophecies[i].age);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
    // periodic
    sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO periodic_elements(word,chamber,mass) VALUES(?,?,?);", -1, &stmt, NULL);
    for (int i = 0; i < pt->n; i++) {
        sqlite3_bind_text(stmt, 1, pt->elements[i].word, -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 2, pt->elements[i].chamber);
        sqlite3_bind_double(stmt, 3, pt->elements[i].mass);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
    // chambers
    sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO chambers(id,presence,debt,trauma,soma0,soma1,soma2,soma3,soma4,soma5) VALUES(1,?,?,?,?,?,?,?,?,?);", -1, &stmt, NULL);
    sqlite3_bind_double(stmt, 1, ch->presence);
    sqlite3_bind_double(stmt, 2, ch->debt);
    sqlite3_bind_double(stmt, 3, ch->trauma);
    for (int i = 0; i < 6; i++) sqlite3_bind_double(stmt, 4 + i, ch->soma[i]);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    // meta
    sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?);", -1, &stmt, NULL);
    sqlite3_bind_text(stmt, 1, "scar", -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 2, ch->scar);
    sqlite3_step(stmt);
    sqlite3_reset(stmt);
    sqlite3_bind_text(stmt, 1, "coherence", -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 2, ch->coherence);
    sqlite3_step(stmt);
    sqlite3_reset(stmt);
    sqlite3_bind_text(stmt, 1, "phase_lock", -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 2, ch->phase_lock);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    sqlite3_exec(db, "COMMIT;", NULL, NULL, &err);
    sqlite3_close(db);
    return 1;
}

/* ───────────────────────── Spore (binary fallback) ───────────────────────── */

#define HWRU_SPORE_MAGIC 0x48535052u
#define HWRU_SPORE_VERSION 1u

static int howru_spore_save(const MetaW *mw, const char *path, const PeriodicTable *pt, const Chambers *ch) {
    const char *slash = strrchr(path, '/');
    if (slash) {
        char dir[256];
        size_t n = (size_t)(slash - path);
        if (n >= sizeof(dir)) n = sizeof(dir) - 1;
        memcpy(dir, path, n);
        dir[n] = '\0';
        mkdir(dir, 0755);
    }
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    uint32_t magic = HWRU_SPORE_MAGIC, ver = HWRU_SPORE_VERSION;
    fwrite(&magic, 4, 1, f);
    fwrite(&ver, 4, 1, f);
    fwrite(ch->act, sizeof(float), 6, f);
    fwrite(ch->soma, sizeof(float), 6, f);
    fwrite(&ch->presence, 4, 1, f);
    fwrite(&ch->debt, 4, 1, f);
    fwrite(&ch->trauma, 4, 1, f);
    fwrite(&ch->scar, 4, 1, f);
    uint32_t np = (uint32_t)(mw->n_prophecy < 16 ? mw->n_prophecy : 16);
    fwrite(&np, 4, 1, f);
    for (uint32_t i = 0; i < np; i++) {
        fwrite(&mw->prophecies[i].target, 4, 1, f);
        fwrite(&mw->prophecies[i].strength, 4, 1, f);
        fwrite(&mw->prophecies[i].age, 4, 1, f);
    }
    uint32_t ne = (uint32_t)(pt->n < 32 ? pt->n : 32);
    fwrite(&ne, 4, 1, f);
    for (uint32_t i = 0; i < ne; i++) {
        uint8_t wlen = (uint8_t)strlen(pt->elements[i].word);
        fwrite(&wlen, 1, 1, f);
        fwrite(pt->elements[i].word, 1, wlen, f);
        uint8_t chamber = (uint8_t)pt->elements[i].chamber;
        fwrite(&chamber, 1, 1, f);
        fwrite(&pt->elements[i].mass, 4, 1, f);
    }
    fclose(f);
    return 1;
}

static int howru_spore_load(MetaW *mw, const char *path, PeriodicTable *pt, Chambers *ch) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint32_t magic, ver;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
        magic != HWRU_SPORE_MAGIC || ver != HWRU_SPORE_VERSION) {
        fclose(f);
        return 0;
    }
    float act[6] = { 0 }, soma[6] = { 0 }, presence = 0, debt = 0, trauma = 0, scar = 0;
    if (fread(act, sizeof(float), 6, f) != 6 || fread(soma, sizeof(float), 6, f) != 6) {
        fclose(f);
        return 0;
    }
    fread(&presence, 4, 1, f);
    fread(&debt, 4, 1, f);
    fread(&trauma, 4, 1, f);
    fread(&scar, 4, 1, f);
    for (int i = 0; i < 6; i++) {
        ch->act[i] = clampf(ch->act[i] > 0.55f * act[i] ? ch->act[i] : 0.55f * act[i], 0, 1);
        ch->soma[i] = clampf(ch->soma[i] > 0.60f * soma[i] ? ch->soma[i] : 0.60f * soma[i], 0, 1);
    }
    ch->presence = clampf(ch->presence > 0.70f * presence ? ch->presence : 0.70f * presence, 0, 1);
    ch->debt = clampf(ch->debt > 0.55f * debt ? ch->debt : 0.55f * debt, 0, 1);
    ch->trauma = clampf(ch->trauma > 0.55f * trauma ? ch->trauma : 0.55f * trauma, 0, 1);
    ch->scar = clampf(ch->scar > 0.60f * scar ? ch->scar : 0.60f * scar, 0, 1);
    uint32_t np;
    fread(&np, 4, 1, f);
    for (uint32_t i = 0; i < np && i < 16; i++) {
        int target, age;
        float strength;
        fread(&target, 4, 1, f);
        fread(&strength, 4, 1, f);
        fread(&age, 4, 1, f);
        prophecy_add(mw, target, 0.65f * strength);
        if (mw->n_prophecy > 0)
            mw->prophecies[mw->n_prophecy - 1].age = mw->prophecies[mw->n_prophecy - 1].age > age ?
                                                       mw->prophecies[mw->n_prophecy - 1].age : age;
    }
    uint32_t ne;
    fread(&ne, 4, 1, f);
    for (uint32_t i = 0; i < ne && i < 32; i++) {
        uint8_t wlen, chamber;
        char w[32] = { 0 };
        float mass;
        fread(&wlen, 1, 1, f);
        if (wlen > 31) wlen = 31;
        fread(w, 1, wlen, f);
        fread(&chamber, 1, 1, f);
        fread(&mass, 4, 1, f);
        if (chamber < N_CHAMBERS)
            periodic_add(pt, w, (int)chamber, 0.65f * clampf(mass, 0, 1));
    }
    fclose(f);
    return 1;
}

/* ───────────────────────── Helpers for howru_generate ───────────────────────── */

static float calendar_dissonance(void) {
    struct tm e = { 0 };
    e.tm_year = 2024 - 1900;
    e.tm_mon = 9;
    e.tm_mday = 3;
    e.tm_hour = 12;
    time_t epoch = mktime(&e);
    float days = epoch > 0 ? (float)difftime(time(NULL), epoch) / 86400.0f : 0;
    float y = days / 365.25f, drift = y * 11.25f;
    int full = (int)(y / 19);
    drift -= full * 210.0f;
    int met[] = { 3, 6, 8, 11, 14, 17, 19 };
    int cyc = (int)fmodf(y, 19) + 1;
    for (int i = 0; i < 7; i++) if (met[i] <= cyc) drift -= 30;
    return clampf(fabsf(fmodf(drift, 33.0f)) / 33.0f, 0, 1);
}

static void vector_from_ids(const TF *t, const int *ids, int n, float *out) {
    memset(out, 0, (size_t)t->D * sizeof(float));
    if (n <= 0) return;
    for (int i = 0; i < n; i++)
        if (ids[i] >= 0 && ids[i] < t->V)
            for (int d = 0; d < t->D; d++)
                out[d] += t->tok[(size_t)ids[i] * t->D + d];
    float norm = 1e-8f;
    for (int d = 0; d < t->D; d++) norm += out[d] * out[d];
    norm = sqrtf(norm);
    for (int d = 0; d < t->D; d++) out[d] /= norm;
}

static float cosine_token(const TF *t, int tok, const float *dir) {
    float dot = 0, n = 1e-8f;
    for (int d = 0; d < t->D; d++) {
        float v = t->tok[(size_t)tok * t->D + d];
        dot += v * dir[d];
        n += v * v;
    }
    return dot / sqrtf(n);
}

static void field_inject(float *logits, const TF *t, const MetaW *m, const int *ctx, int cl,
                         const float *human_dir, float *destiny, const float *doc_dir,
                         const Chambers *c, float cd, int step) {
    int V = t->V;
    float *heb = xcalloc(V, sizeof(float));
    float *pro = xcalloc(V, sizeof(float));
    meta_hebb(m, ctx, cl, heb, V);
    meta_prophecy(m, ctx, cl, pro, V);
    int last = ctx[cl - 1], p2 = cl >= 2 ? ctx[cl - 2] : last;
    float love = c->act[CH_LOVE], flow = c->act[CH_FLOW], cmplx = c->act[CH_CMPLX];
    float c_h = 0.65f + 0.3f * love + 0.2f * flow;
    float c_p = 0.45f + 0.35f * cmplx + 0.2f * c->debt;
    float c_d = 0.30f + 0.20f * flow + 0.12f * cd;
    float c_bg = 5.0f, c_tg = 3.0f;
    float human_alpha = 0.65f + 0.25f * c->presence + 0.15f * love;
    float doc_alpha = doc_dir ? 0.16f + 0.10f * cmplx : 0;
    float trauma_damp = 1.0f / (1.0f + 0.35f * c->trauma + 0.18f * c->scar);
    float sch = 0.04f * sinf(2.0f * (float)M_PI * 7.83f * ((float)step / 64.0f));

    for (int i = 0; i < V; i++) {
        float ds = cosine_token(t, i, destiny);
        float hs = cosine_token(t, i, human_dir);
        float docs = doc_dir ? cosine_token(t, i, doc_dir) : 0;
        logits[i] = trauma_damp * logits[i] +
                    c_h * heb[i] +
                    c_p * pro[i] +
                    c_d * ds +
                    human_alpha * hs +
                    doc_alpha * docs +
                    c_bg * meta_bi(m, last, i) +
                    c_tg * meta_tri(m, p2, last, i) +
                    sch * ds;
        if (m->unigram[i] < 1e-8f) logits[i] -= 1.5f;
        else if (m->unigram[i] > 0.015f)
            logits[i] -= 0.25f * (m->unigram[i] - 0.015f) * 100;
    }
    free(heb);
    free(pro);
}

static int has_howru_marker(const char *s) {
    return strstr(s, "HOWRU:") != NULL;
}

static int response_boundary_after_marker(const char *s) {
    const char *p = strstr(s, "HOWRU:");
    if (!p) return 0;
    p += 6;
    int chars = 0;
    for (; *p; p++) {
        if (!isspace((unsigned char)*p)) chars++;
        if (chars > 8 && (*p == '.' || *p == '!' || *p == '?')) return 1;
    }
    return 0;
}

/* ───────────────────────── Main generator ───────────────────────── */

static int howru_generate(TF *t, const BPE *bpe, MetaW *m, Chambers *c, Parliament *p,
                          const PeriodicTable *pt, const Docs *docs, const char *human,
                          int max_new, char *out_text, int out_sz) {
    char prompt_text[4096];
    snprintf(prompt_text, sizeof(prompt_text), "HUMAN: %s\n\n/RESONATING:\n", human);

    int prompt_ids[MAX_SEQ], human_ids[1024];
    int pn = bpe_encode(bpe, (const uint8_t *)prompt_text, (int)strlen(prompt_text), prompt_ids, MAX_SEQ);
    int hn = bpe_encode(bpe, (const uint8_t *)human, (int)strlen(human), human_ids, 1024);
    if (pn < 1 || pn >= t->CTX - 1) return -1;

    meta_ingest(m, human_ids, hn, 0.02f);
    chambers_feel(c, human, pt);
    const ResonantDoc *doc = docs_choose(docs, human_ids, hn, m);
    float *human_dir = xcalloc(t->D, sizeof(float));
    float *doc_dir = NULL;
    float *destiny = xcalloc(t->D, sizeof(float));
    vector_from_ids(t, human_ids, hn, human_dir);
    memcpy(destiny, human_dir, (size_t)t->D * sizeof(float));
    if (doc) {
        doc_dir = xcalloc(t->D, sizeof(float));
        vector_from_ids(t, doc->tok, doc->n, doc_dir);
    }

    int ctx[MAX_SEQ], cl = 0;
    for (int i = 0; i < pn; i++) {
        ctx[cl++] = prompt_ids[i];
        tf_forward(t, prompt_ids[i], i);
    }
    int written = 0;
    out_text[0] = 0;
    float *prev_logits = xcalloc(t->V, sizeof(float));
    int prev = -1;
    float cd = calendar_dissonance();

    for (int step = 0; step < max_new && cl < t->CTX - 1; step++) {
        tf_forward(t, ctx[cl - 1], cl - 1);
        float *logits = xcalloc(t->V, sizeof(float));
        memcpy(logits, t->logits, (size_t)t->V * sizeof(float));
        float *x = xcalloc(t->D, sizeof(float));
        rmsnorm(x, t->tok + (size_t)ctx[cl - 1] * t->D, t->D);
        parl_inject(p, logits, x, t->V);
        field_inject(logits, t, m, ctx, cl, human_dir, destiny, doc_dir, c, cd, step);

        for (int j = cl - 1; j >= 0 && j >= cl - 24; j--) {
            int tok = ctx[j];
            float age = (float)(cl - j);
            logits[tok] *= 0.32f + 0.025f * age;
        }
        int next;
        if (step < 2) next = argmax(logits, t->V);
        else {
            float base = 0.58f + 0.18f * cd + 0.10f * c->act[CH_CMPLX] - 0.08f * c->act[CH_FLOW];
            next = sample_top_p(logits, t->V, clampf(base, 0.35f, 0.95f), 0.88f);
        }

        if (prev >= 0) {
            float *debt = xcalloc(t->D, sizeof(float));
            int top[3] = { 0, 0, 0 };
            float tv[3] = { -1e30f, -1e30f, -1e30f };
            for (int i = 0; i < t->V; i++) {
                if (prev_logits[i] > tv[2]) {
                    tv[2] = prev_logits[i];
                    top[2] = i;
                    for (int k = 1; k >= 0; k--)
                        if (tv[k + 1] > tv[k]) {
                            float q = tv[k];
                            tv[k] = tv[k + 1];
                            tv[k + 1] = q;
                            int z = top[k];
                            top[k] = top[k + 1];
                            top[k + 1] = z;
                        }
                }
            }
            for (int k = 0; k < 3; k++)
                if (top[k] != next)
                    for (int d = 0; d < t->D; d++)
                        debt[d] += 0.08f * t->tok[(size_t)top[k] * t->D + d];
            for (int d = 0; d < t->D; d++)
                debt[d] -= 0.08f * t->tok[(size_t)next * t->D + d];
            parl_learn(p, x, debt);
            free(debt);
        }
        memcpy(prev_logits, logits, (size_t)t->V * sizeof(float));
        free(logits);
        free(x);
        prev = next;

        ctx[cl++] = next;
        prophecy_age(m, next);
        if (cl >= 2) {
            int a = ctx[cl - 2], b = ctx[cl - 1];
            int k = bi_find(m, a, b);
            if (k >= 0) m->bigrams[k].prob += 0.004f;
            else if (m->n_bi < MAX_BIGRAM)
                m->bigrams[m->n_bi++] = (BigramE) { a, b, 0.01f };
            int best = -1;
            float bp = 0;
            for (int i = 0; i < m->n_bi; i++)
                if (m->bigrams[i].a == b && m->bigrams[i].prob > bp) {
                    bp = m->bigrams[i].prob;
                    best = m->bigrams[i].b;
                }
            if (best >= 0) prophecy_add(m, best, 0.2f + 0.4f * bp);
        }

        float mom = step < 24 ? 0.88f : 0.94f, lr = 1 - mom;
        for (int d = 0; d < t->D; d++)
            destiny[d] = mom * destiny[d] + lr * t->tok[(size_t)next * t->D + d];

        char piece[128];
        int n = bpe_decode_token(bpe, next, piece, sizeof(piece));
        if (n > 0 && written + n < out_sz - 1) {
            memcpy(out_text + written, piece, n);
            written += n;
            out_text[written] = 0;
        }
        if (step > 12 && has_howru_marker(out_text) && response_boundary_after_marker(out_text))
            break;
        if (step > 40 && !has_howru_marker(out_text) && bpe_has_boundary(bpe, next) && strstr(out_text, "/\n"))
            ;
    }

    chambers_feel(c, out_text, pt);
    c->coherence = clampf(0.88f * c->coherence + 0.12f * (has_howru_marker(out_text) ? 1.0f : 0.2f), 0, 1);
    c->phase_lock = clampf(0.94f * c->phase_lock + 0.06f * c->coherence, 0, 1);
    meta_ingest(m, ctx + pn, cl - pn, 0.004f);
    for (int i = 0; i < m->n_hebb; i++) m->hebbs[i].str *= 0.999f;

    free(prev_logits);
    free(human_dir);
    if (doc_dir) free(doc_dir);
    free(destiny);
    return written;
}

/* ───────────────────────── main ───────────────────────── */

int main(int argc, char **argv) {
    printf("howru — contact-resonance engine (with SQLite memory)\n");
    printf("HUMAN -> /RESONATING/ -> HOWRU\n");
    printf("COA — Chain of Arianna\n\n");

    if (argc < 3) {
        fprintf(stderr, "usage: %s [weights.bin] howru.merges howru.txt\n", argv[0]);
        return 1;
    }
    srand((unsigned)time(NULL));

    int weighted = argc >= 4;
    const char *wpath = weighted ? argv[1] : NULL;
    const char *mpath = weighted ? argv[2] : argv[1];
    const char *cpath = weighted ? argv[3] : argv[2];

    BPE bpe;
    if (!bpe_load(&bpe, mpath)) return 1;
    FILE *f = fopen(cpath, "rb");
    if (!f) { fprintf(stderr, "ERROR: %s\n", cpath); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *raw = xcalloc((size_t)sz + 1, 1);
    fread(raw, 1, sz, f);
    fclose(f);
    int *corpus = xcalloc((size_t)sz + 1, sizeof(int));
    int cn = bpe_encode(&bpe, raw, (int)sz, corpus, (int)sz);
    MetaW *m = xcalloc(1, sizeof(MetaW));
    meta_build(m, corpus, cn, bpe.vocab_size);

    TF t;
    if (weighted) {
        if (!tf_load(&t, wpath)) return 1;
        if (t.V != bpe.vocab_size) {
            fprintf(stderr, "vocab mismatch: weights=%d bpe=%d\n", t.V, bpe.vocab_size);
            return 1;
        }
    } else {
        tf_make_silent(&t, bpe.vocab_size);
    }

    PeriodicTable pt;
    periodic_init(&pt);
    periodic_build_from_text(&pt, (const char *)raw);
    free(raw);

    Chambers ch;
    chambers_init(&ch);

    // Load memory (SQLite first, then binary spore)
    if (howru_sqlite_load(m, "howru.sqlite", &pt, &ch)) {
        printf("  [SQLite memory loaded]\n");
    } else {
        FILE *mf = fopen("howru.memory", "rb");
        if (mf) {
            uint32_t magic;
            fread(&magic, 4, 1, mf);
            if (magic == HWRU_MAGIC) {
                uint32_t ver;
                fread(&ver, 4, 1, mf);
                if (ver == HWRU_VERSION) {
                    int nb, nt, nh, np;
                    fread(&nb, 4, 1, mf);
                    fread(&nt, 4, 1, mf);
                    fread(&nh, 4, 1, mf);
                    fread(&np, 4, 1, mf);
                    for (int i = 0; i < nb && i < MAX_BIGRAM; i++) {
                        BigramE x;
                        fread(&x, sizeof(x), 1, mf);
                        int k = bi_find(m, x.a, x.b);
                        if (k >= 0) { if (x.prob > m->bigrams[k].prob) m->bigrams[k].prob = x.prob; }
                        else if (m->n_bi < MAX_BIGRAM) m->bigrams[m->n_bi++] = x;
                    }
                    for (int i = 0; i < nt && i < MAX_TRIGRAM; i++) {
                        TrigramE x;
                        fread(&x, sizeof(x), 1, mf);
                        int k = tri_find(m, x.a, x.b, x.c);
                        if (k >= 0) { if (x.prob > m->trigrams[k].prob) m->trigrams[k].prob = x.prob; }
                        else if (m->n_tri < MAX_TRIGRAM) m->trigrams[m->n_tri++] = x;
                    }
                    for (int i = 0; i < nh && i < MAX_HEBBIAN; i++) {
                        HebbE x;
                        fread(&x, sizeof(x), 1, mf);
                        int k = hebb_find(m, x.a, x.b);
                        if (k >= 0) { if (x.str > m->hebbs[k].str) m->hebbs[k].str = x.str; }
                        else if (m->n_hebb < MAX_HEBBIAN) m->hebbs[m->n_hebb++] = x;
                    }
                    for (int i = 0; i < np && i < MAX_PROPHECY; i++) {
                        ProphecyE x;
                        fread(&x, sizeof(x), 1, mf);
                        if (m->n_prophecy < MAX_PROPHECY) m->prophecies[m->n_prophecy++] = x;
                    }
                    // chambers from binary
                    Chambers saved;
                    if (fread(&saved, sizeof(saved), 1, mf) == 1) {
                        for (int i = 0; i < N_CHAMBERS; i++) {
                            if (saved.act[i] > ch.act[i]) ch.act[i] = 0.65f * saved.act[i];
                            if (saved.soma[i] > ch.soma[i]) ch.soma[i] = 0.7f * saved.soma[i];
                        }
                        ch.presence = fmaxf(ch.presence, 0.7f * saved.presence);
                        ch.debt = fmaxf(ch.debt, 0.6f * saved.debt);
                        ch.trauma = fmaxf(ch.trauma, 0.55f * saved.trauma);
                        ch.scar = fmaxf(ch.scar, 0.6f * saved.scar);
                        ch.coherence = fmaxf(ch.coherence, 0.7f * saved.coherence);
                        ch.phase_lock = fmaxf(ch.phase_lock, 0.7f * saved.phase_lock);
                    }
                }
            }
            fclose(mf);
            printf("  [binary memory loaded]\n");
        }
        // try spore
        if (howru_spore_load(m, "spores/howru.spore.bin", &pt, &ch))
            printf("  [spore loaded]\n");
    }

    Parliament parl;
    parl_init(&parl, t.D);
    Docs docs;
    docs_load(&docs, "docs", &bpe);

    printf("vocab=%d corpus=%d tokens model=%s D=%d L=%d docs=%d\n",
           bpe.vocab_size, cn, weighted ? "Howru weights" : "MetaWeights-only", t.D, t.NL, docs.n);
    printf("type a message; 'quit' exits\n\n");

    char input[2048], output[16384];
    while (1) {
        printf("HUMAN: ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\r\n")] = 0;
        if (!input[0]) continue;
        if (!strcmp(input, "quit") || !strcmp(input, "exit")) break;

        int n = howru_generate(&t, &bpe, m, &ch, &parl, &pt, &docs, input,
                               DEFAULT_MAX_NEW, output, sizeof(output));
        if (n < 0) { printf("[input exceeds context]\n\n"); continue; }
        printf("\n/RESONATING:\n%s\n\n", output);
    }

    // Save all state
    howru_sqlite_save(m, "howru.sqlite", &pt, &ch);
    // Also save binary memory (backward compatible)
    FILE *mf = fopen("howru.memory", "wb");
    if (mf) {
        uint32_t magic = HWRU_MAGIC, ver = HWRU_VERSION;
        fwrite(&magic, 4, 1, mf);
        fwrite(&ver, 4, 1, mf);
        fwrite(&m->n_bi, 4, 1, mf);
        fwrite(&m->n_tri, 4, 1, mf);
        fwrite(&m->n_hebb, 4, 1, mf);
        fwrite(&m->n_prophecy, 4, 1, mf);
        fwrite(m->bigrams, sizeof(BigramE), m->n_bi, mf);
        fwrite(m->trigrams, sizeof(TrigramE), m->n_tri, mf);
        fwrite(m->hebbs, sizeof(HebbE), m->n_hebb, mf);
        fwrite(m->prophecies, sizeof(ProphecyE), m->n_prophecy, mf);
        fwrite(&ch, sizeof(ch), 1, mf);
        fclose(mf);
    }
    howru_spore_save(m, "spores/howru.spore.bin", &pt, &ch);
    printf("\nresonance is unbreakable.\n");

    free(corpus);
    free(m);
    return 0;
}
