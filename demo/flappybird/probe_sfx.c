
#include <SDL.h>
#include <SDL_mixer.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ==== verbatim synthesis from flappy.c (current on-disk version) ==== */
typedef struct { float *buf; int n; int rate; } Tone;
static Tone *toneNew(int n, int rate)
{
    Tone *t = calloc(1, sizeof(Tone));
    t->buf = malloc(sizeof(float) * n);
    t->n = n; t->rate = rate;
    return t;
}
static void toneFree(Tone *t) { if (t) { free(t->buf); free(t); } }

static int wavSize(int n, int rate) { (void)rate; return 44 + n * 4; }
static unsigned char *toWav(Tone *t)
{
    int n = t->n, rate = t->rate, sz = wavSize(n, rate);
    unsigned char *w = malloc(sz);
    int i;
    int32_t dataSize = n * 4;
    memcpy(w + 0, "RIFF", 4);
    int32_t chunk = sz - 8; memcpy(w + 4, &chunk, 4);
    memcpy(w + 8, "WAVE", 4);
    memcpy(w + 12, "fmt ", 4);
    int32_t fmtsz = 16; memcpy(w + 16, &fmtsz, 4);
    int16_t f1 = 1, ch2 = 2; uint32_t r = rate;
    memcpy(w + 20, &f1, 2); memcpy(w + 22, &ch2, 2); memcpy(w + 24, &r, 4);
    uint32_t byteRate = rate * 4; memcpy(w + 28, &byteRate, 4);
    int16_t bps = 4, bits = 16; memcpy(w + 32, &bps, 2); memcpy(w + 34, &bits, 2);
    memcpy(w + 36, "data", 4); memcpy(w + 40, &dataSize, 4);
    for (i = 0; i < n; i++) {
        float s = t->buf[i];
        if (s >  1.f) s =  1.f;
        if (s < -1.f) s = -1.f;
        int16_t v = (int16_t)(s * 32767.f);
        memcpy(w + 44 + i * 4,     &v, 2);
        memcpy(w + 44 + i * 4 + 2, &v, 2);
    }
    return w;
}
static void saveFile(const char *path, const void *data, int sz)
{
    FILE *fp = fopen(path, "wb");
    fwrite(data, 1, sz, fp);
    fclose(fp);
}
static Mix_Chunk *chunkFromTone(Tone *t, const char *dumpWav)
{
    unsigned char *w = toWav(t);
    if (dumpWav) saveFile(dumpWav, w, wavSize(t->n, t->rate));
    Mix_Chunk *c = Mix_QuickLoad_WAV(w);
    free(w);
    return c;
}
static void toneNote(Tone *t, int off, float dur, float freq,
                     float vol, float attack, float decay, int square, float noise)
{
    int i, n = (int)(dur * t->rate);
    if (off + n > t->n) n = t->n - off;
    for (i = 0; i < n; i++) {
        float p = (float)i / t->rate;
        float env = 1.f;
        if (p < attack)              env = p / attack;
        else if (p > dur - decay)    env = (dur - p) / decay;
        float s = sinf(2.f * (float)M_PI * freq * p);
        if (square) s = s > 0.f ? 0.8f : -0.8f;
        s *= vol * env;
        if (noise > 0.f) {
            float r = (float)rand() / RAND_MAX * 2.f - 1.f;
            s += r * noise * env;
        }
        t->buf[off + i] += s;
    }
}
static Mix_Chunk *sfxFlap, *sfxScore, *sfxHit, *sfxDie, *sfxSwoosh;
static void initAudio(void)
{
    if (Mix_OpenAudio(44100, AUDIO_S16SYS, 2, 512) != 0) {
        fprintf(stderr, "mixer init failed (%s)\n", Mix_GetError());
        exit(1);
    }
    srand(1234);
    Tone *t = toneNew((int)(0.16f * 44100), 44100);
    toneNote(t, 0, 0.07f, 520.f, 0.35f, 0.005f, 0.04f, 0, 0.12f);
    toneNote(t, 630, 0.09f, 780.f, 0.3f, 0.004f, 0.05f, 0, 0.08f);
    sfxFlap = chunkFromTone(t, NULL); toneFree(t);
    t = toneNew((int)(0.35f * 44100), 44100);
    toneNote(t, 0, 0.12f, 880.f, 0.35f, 0.004f, 0.08f, 1, 0.f);
    toneNote(t, 940, 0.18f, 1318.f, 0.35f, 0.004f, 0.12f, 1, 0.f);
    sfxScore = chunkFromTone(t, NULL); toneFree(t);
    t = toneNew((int)(0.22f * 44100), 44100);
    toneNote(t, 0, 0.14f, 110.f, 0.6f, 0.002f, 0.12f, 0, 0.35f);
    toneNote(t, 0, 0.05f, 220.f, 0.4f, 0.002f, 0.04f, 0, 0.f);
    sfxHit = chunkFromTone(t, NULL); toneFree(t);
    /* die: descending slide (fixed version, matches flappy.c now) */
    t = toneNew((int)(0.55f * 44100), 44100);
    {
        int i, n = t->n;
        float phase = 0.f;
        for (i = 0; i < n; i++) {
            float p = (float)i / t->rate;
            float f = 440.f - 380.f * p / 0.55f;
            float env = 1.f - p / 0.55f;
            t->buf[i] = sinf(phase) * 0.4f * env;
            phase += 2.f * (float)M_PI * f / t->rate;
        }
    }
    sfxDie = chunkFromTone(t, "/tmp/die_gen.wav");
    toneFree(t);
    t = toneNew((int)(0.3f * 44100), 44100);
    {
        int i, n = t->n;
        float lp = 0.f;
        for (i = 0; i < n; i++) {
            float r = (float)rand() / RAND_MAX * 2.f - 1.f;
            lp += 0.25f * (r - lp);
            float env = i < 800 ? (float)i / 800.f : 1.f - (float)(i - 800) / (n - 800);
            t->buf[i] = lp * env * 0.8f;
        }
    }
    sfxSwoosh = chunkFromTone(t, NULL); toneFree(t);
}
static void dumpChunk(const char *name, Mix_Chunk *c)
{
    char path[128];
    snprintf(path, sizeof path, "/tmp/sfx_%s.raw", name);
    saveFile(path, c->abuf, c->alen);
    printf("%s alen=%u frames=%u dumped=%u\n", name, c->alen, c->alen / 4, c->alen);
}
int main(void)
{
    SDL_InitSubSystem(SDL_INIT_AUDIO);
    initAudio();
    /* the 2.8.x mixer converts chunks on loader threads; wait for it */
    SDL_Delay(2000);
    dumpChunk("flap",   sfxFlap);
    dumpChunk("score",  sfxScore);
    dumpChunk("hit",    sfxHit);
    dumpChunk("die",    sfxDie);
    dumpChunk("swoosh", sfxSwoosh);
    Mix_CloseAudio();
    SDL_Quit();
    return 0;
}
