/* flappy.c -- Flappy Bird in C, OpenGL (GLUT) + SDL2_mixer + Freetype.
 *
 * Build:  make            (freeglut + Mesa on Linux, GLUT.framework on macOS)
 * Run:    ./flappy
 *
 * Controls:
 *   SPACE / UP / W / mouse click   flap
 *   P                               pause
 *   ESC / Q                         quit
 *
 * All graphics are procedural (shaded quads, a Freetype-rendered font atlas
 * for the text); all sounds are synthesized at startup as WAV data and fed
 * to SDL_mixer (no asset files needed).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#if defined(__APPLE__) && !defined(FLAPPY_HEADLESS_TEST)
#include <GLUT/glut.h>          /* Apple GLUT: a Cocoa window, no X11 server */
#include <OpenGL/gl.h>
#else
#include <GL/glut.h>
#endif
#include <SDL.h>
#include <SDL_mixer.h>
#include <ft2build.h>
#include FT_FREETYPE_H

/* ------------------------------------------------------------------ */
/* World constants (world units == pixels; origin at top-left, y down) */
/* ------------------------------------------------------------------ */
#define W 480
#define H 640
#define GROUND_H   90
#define PIPE_W     62
#define GAP        148.0
#define SPEED      135.0            /* horizontal scroll speed (px/s) */
#define SPAWN_DIST 235.0            /* spacing between pipes */
#define BIRD_X     130.0            /* bird fixed screen x */
#define BIRD_R     11.5f            /* bird collision radius */
#define GRAVITY    760.0            /* px/s^2 */
#define FLAP_VY   -268.0            /* vy after a flap, px/s */
#define MAX_FALL   360.0

/* ------------------------------------------------------------------ */
typedef enum { S_READY, S_PLAY, S_DEAD } GameState;

typedef struct { float x, gapY, scored; } Pipe;
typedef struct { float x, y, z; int  life; } Cloud;
typedef struct { float x, y, vx, vy, life; char c; } Particle;
typedef struct { float t; const char *s; } Banner;

static GameState    state = S_READY;
static float        birdY, birdV, birdRot;
static int          score, bestScore;
static Pipe         pipes[16];      static int nPipes;
static Cloud        clouds[8];
static Particle     parts[160];     static int nParts = -1;
static Banner       banner;         static int bannerOn = 0;
static float        tGlobal = 0.f, groundOffset = 0.f;
static double       lastTS = -1.0;  /* last timestamp (ms) from GLUT idle */
static int          paused = 0;
static int          frameCount = 0;
static int          dumpAt = 0;      /* FLAPPY_DUMP env: dump this frame and exit */
static int          mouseX = W/2, mouseY = H/2, mouseDown = 0;
static const char  *fontPath = 0;              /* set by --font, else autodetect */

/* Font file candidates. FreeType opens .ttc collections at face index 0, which
   is enough for our ASCII atlas. The order below tries macOS first, then
   common Linux distributions, then Homebrew freetype's own bundled font. */
static const char *fontCandidates[] = {
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    0
};

/* Freetype + text atlas */
static FT_Library  ft;
static FT_Face     face;
static GLuint      texAtlas = 0;
#define ATLAS_W 512
#define ATLAS_H 256
static int atlasCols = 0, atlasRowH = 0;
static unsigned char atlasTex[ATLAS_W * ATLAS_H];
/* per character uv */
static float uvU[256][2], uvV[256][2];
static int   charW[256];     /* advance width per char */
static int   glyphW[256], glyphH[256], glyphLeft[256], glyphTop[256];
static int   fontAscent = 20;

/* audio */
static int  haveAudio = 0;
static Mix_Chunk *sfxFlap = 0, *sfxScore = 0, *sfxHit = 0, *sfxDie = 0, *sfxSwoosh = 0;

/* ------------------------------------------------------------------ */
/* forward declarations                                                */
/* ------------------------------------------------------------------ */
static void initAudio(void);
static void initText(void);
static void resetGame(void);
static void doFlap(void);
static void killBird(void);
static void spawnParticles(float x, float y, int n, int kind);
static void update(float dt);
static void renderScene(void);
static void dumpFrame(int n);

/* ------------------------------------------------------------------ */
/* Audio: synthesize small WAV buffers and hand them to SDL_mixer      */
/* ------------------------------------------------------------------ */
typedef struct {
    float *buf; int n; int rate;
} Tone;

static Tone *toneNew(int n, int rate)
{
    Tone *t = calloc(1, sizeof(Tone));
    /* toneNote accumulates with +=, so the sample buffer must start at zero. */
    t->buf = calloc(n, sizeof(float));
    t->n = n; t->rate = rate;
    return t;
}
static void toneFree(Tone *t) { if (t) { free(t->buf); free(t); } }

/* n = number of frames (each frame is a stereo pair) */
static int wavSize(int n, int rate) { return 44 + n * 4; }

static unsigned char *toWav(Tone *t)
{
    int n = t->n, rate = t->rate, sz = wavSize(n, rate);
    unsigned char *w = malloc(sz);
    int i;
    int32_t dataSize = n * 4;   /* 2 channels x 2 bytes */
    /* RIFF header */
    memcpy(w + 0, "RIFF", 4);
    int32_t chunk = sz - 8; memcpy(w + 4, &chunk, 4);
    memcpy(w + 8, "WAVE", 4);
    memcpy(w + 12, "fmt ", 4);
    int32_t fmtsz = 16; memcpy(w + 16, &fmtsz, 4);
    int16_t f1 = 1, ch2 = 2; uint32_t r = rate;
    memcpy(w + 20, &f1, 2); memcpy(w + 22, &ch2, 2); memcpy(w + 24, &r, 4);
    uint32_t byteRate = rate * 4; memcpy(w + 28, &byteRate, 4);   /* 2ch x 16bit */
    int16_t bps = 4, bits = 16; memcpy(w + 32, &bps, 2); memcpy(w + 34, &bits, 2);
    memcpy(w + 36, "data", 4); memcpy(w + 40, &dataSize, 4);
    /* The mixer is opened with AUDIO_S16SYS stereo, and Mix_QuickLoad_WAV
       copies the data as-is (no channel conversion), so emit true stereo
       frames: each sample duplicated across both channels. Feeding mono
       data made the mixer read pairs of samples as one stereo frame and
       play everything at 2x speed. */
    for (i = 0; i < n; i++) {
        float s = t->buf[i];
        if (s >  1.f) s =  1.f;
        if (s < -1.f) s = -1.f;
        int16_t v = (int16_t)(s * 32767.f);
        memcpy(w + 44 + i * 4,     &v, 2);   /* L */
        memcpy(w + 44 + i * 4 + 2, &v, 2);   /* R */
    }
    return w;
}
static Mix_Chunk *chunkFromTone(Tone *t)
{
    /* Mix_QuickLoad_WAV keeps a pointer into the caller's buffer and reads it
       on every playback, so freeing the buffer here would leave the mixer
       playing freed memory. Mix_LoadWAV_RW copies the samples and converts
       to the mixer's output format, so the WAV buffer is safe to free. */
    int sz = wavSize(t->n, t->rate);
    unsigned char *w = toWav(t);
    SDL_RWops *rw = SDL_RWFromMem(w, sz);
    Mix_Chunk *c = Mix_LoadWAV_RW(rw, 1);
    free(w);
    return c;
}

/* one note: freq, duration, envelope shape */
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

static void initAudio(void)
{
    if (SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
        fprintf(stderr, "audio: SDL audio unavailable (%s); running silent\n", SDL_GetError());
        return;
    }
    if (Mix_OpenAudio(44100, AUDIO_S16SYS, 2, 512) != 0) {
        fprintf(stderr, "audio: mixer init failed (%s); running silent\n", Mix_GetError());
        return;
    }
    haveAudio = 1;
    srand(1234);

    /* flap: quick two-tone chirp up */
    Tone *t = toneNew((int)(0.16f * 44100), 44100);
    toneNote(t, 0, 0.07f, 520.f, 0.35f, 0.005f, 0.04f, 0, 0.12f);
    toneNote(t, 630, 0.09f, 780.f, 0.3f, 0.004f, 0.05f, 0, 0.08f);
    sfxFlap = chunkFromTone(t); toneFree(t);

    /* score: bright ding-ding */
    t = toneNew((int)(0.35f * 44100), 44100);
    toneNote(t, 0, 0.12f, 880.f, 0.35f, 0.004f, 0.08f, 1, 0.f);
    toneNote(t, 940, 0.18f, 1318.f, 0.35f, 0.004f, 0.12f, 1, 0.f);
    sfxScore = chunkFromTone(t); toneFree(t);

    /* hit: thud */
    t = toneNew((int)(0.22f * 44100), 44100);
    toneNote(t, 0, 0.14f, 110.f, 0.6f, 0.002f, 0.12f, 0, 0.35f);
    toneNote(t, 0, 0.05f, 220.f, 0.4f, 0.002f, 0.04f, 0, 0.f);
    sfxHit = chunkFromTone(t); toneFree(t);

    /* die: descending slide. Phase must be accumulated per sample:
       sin(2*pi*f(p)*p) makes the instantaneous frequency 2f(p)-440,
       so the glide drops to zero and runs backwards before the tone
       ends, which is the garbled "die" sound. */
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
    sfxDie = chunkFromTone(t); toneFree(t);

    /* swoosh (restart): filtered-ish noise burst */
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
    sfxSwoosh = chunkFromTone(t); toneFree(t);
}
static void play(Mix_Chunk *c) { if (haveAudio && c) Mix_PlayChannel(-1, c, 0); }

/* ------------------------------------------------------------------ */
/* Text: render needed glyphs once into a single texture atlas         */
/* ------------------------------------------------------------------ */
static void initText(void)
{
    int i;
    memset(atlasTex, 0, sizeof(atlasTex));
    if (FT_Init_FreeType(&ft) != 0) { ft = 0; return; }
    face = 0;
    if (fontPath && FT_New_Face(ft, fontPath, 0, &face) == 0) {
        /* explicit --font wins */
    } else {
        int fi;
        for (fi = 0; fontCandidates[fi]; fi++) {
            if (FT_New_Face(ft, fontCandidates[fi], 0, &face) == 0) {
                fontPath = fontCandidates[fi];
                break;
            }
            face = 0;
        }
    }
    if (!face) {
        fprintf(stderr, "font: no usable TTF found (tried --font and %d fallbacks); "
                        "text disabled\n", (int)(sizeof fontCandidates / sizeof fontCandidates[0]) - 1);
        return;
    }
    FT_Set_Pixel_Sizes(face, 0, 26);
    fontAscent = face->size->metrics.ascender / 64;

    /* all printable ASCII (95 glyphs) -- comfortably fits the 512x256 atlas */
    char charSet[128];
    int ci;
    for (ci = 32; ci < 127; ci++) charSet[ci - 32] = (char)ci;
    charSet[95] = 0;
    const char *chars = charSet;
    int n = 95;
    int colX = 4, rowY = 4;
    int rowMaxH = 0;
    for (i = 0; i < n; i++) {
        if (FT_Load_Char(face, (FT_ULong)chars[i], FT_LOAD_RENDER) != 0) continue;
        int w = face->glyph->bitmap.width;
        int h = face->glyph->bitmap.rows;
        int left = face->glyph->bitmap_left, top = face->glyph->bitmap_top;
        if (colX + w >= ATLAS_W - 4) {
            colX = 4; rowY += rowMaxH + 4;
            rowMaxH = 0;
        }
        if (h > rowMaxH) rowMaxH = h;
        /* record per-glyph metrics. Use FreeType's advance (in 26.6 fixed
           point) so glyphs with no bitmap - the space, mainly - still
           advance the pen by the correct amount. */
        {
            int adv = (int)(face->glyph->advance.x >> 6);
            if (adv <= 0) adv = w + 2;
            charW[(unsigned char)chars[i]] = adv;
        }
        glyphW[(unsigned char)chars[i]]  = w;
        glyphH[(unsigned char)chars[i]]  = h;
        glyphLeft[(unsigned char)chars[i]] = left;
        glyphTop[(unsigned char)chars[i]]  = top;
        /* blit alpha */
        {
            int r, c;
            for (r = 0; r < h; r++)
                for (c = 0; c < w; c++)
                    if (colX + c < ATLAS_W && rowY + r < ATLAS_H)
                        atlasTex[(rowY + r) * ATLAS_W + (colX + c)] = face->glyph->bitmap.buffer[r * w + c];
        }
        uvU[(unsigned char)chars[i]][0] = (float)colX / ATLAS_W;
        uvU[(unsigned char)chars[i]][1] = (float)(colX + w) / ATLAS_W;
        /* screen-top vertex <-> glyph top row (rowY); screen-bottom <-> rowY+h */
        uvV[(unsigned char)chars[i]][0] = (float)rowY / ATLAS_H;
        uvV[(unsigned char)chars[i]][1] = (float)(rowY + h) / ATLAS_H;
        (void)left; (void)top;
        colX += w + 4;
    }
    atlasCols = n;
    atlasRowH = rowMaxH;

    glGenTextures(1, &texAtlas);
    glBindTexture(GL_TEXTURE_2D, texAtlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, ATLAS_W, ATLAS_H, 0, GL_ALPHA,
                 GL_UNSIGNED_BYTE, atlasTex);
}

static void drawText(const char *s, float x, float y, float r, float g, float b,
                     float scale, int center)
{
    /* y is top of text in world coords; draws using texture atlas */
    float px;
    if (!face || !texAtlas) return;
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texAtlas);
    if (center) {
        const char *tmp = s;
        float w = 0;
        while (*tmp) { w += charW[(unsigned char)*tmp] * scale; tmp++; }
        x -= w * 0.5f;
    }
    /* baseline sits `fontAscent` below the text top (y). Each glyph's
       bitmap_top is measured from the baseline, so its screen row is
       (y + fontAscent - bitmap_top); bitmap_left gives its horizontal
       offset from the pen. All scaled by `scale`. */
    float base = y + (float)fontAscent * scale;
    px = x;
    while (*s) {
        unsigned char ch = (unsigned char)*s++;
        if (ch == ' ' || !charW[ch]) { px += charW[ch] * scale; continue; }
        float gx = px + (float)glyphLeft[ch] * scale;
        float gy = base - (float)glyphTop[ch] * scale;
        float tw = (float)glyphW[ch] * scale;
        float th = (float)glyphH[ch] * scale;
        glBegin(GL_QUADS);
        glTexCoord2f(uvU[ch][0], uvV[ch][0]); glVertex2f(gx,     gy);
        glTexCoord2f(uvU[ch][1], uvV[ch][0]); glVertex2f(gx + tw, gy);
        glTexCoord2f(uvU[ch][1], uvV[ch][1]); glVertex2f(gx + tw, gy + th);
        glTexCoord2f(uvU[ch][0], uvV[ch][1]); glVertex2f(gx,     gy + th);
        glEnd();
        px += charW[ch] * scale;
    }
    (void)r; (void)g; (void)b;
}

/* draw colored text: tint atlas alpha via modulate */
static void drawColText(const char *s, float x, float y, float r, float g, float b,
                        float scale, int center)
{
    glColor4f(r, g, b, 1.f);
    drawText(s, x, y, r, g, b, scale, center);
}

/* ------------------------------------------------------------------ */
/* Drawing primitives (world units == pixels, y-down)                  */
/* ------------------------------------------------------------------ */
static void quad2(float x0, float y0, float x1, float y1, float r, float g, float b, float a)
{
    glBegin(GL_QUADS);
    glColor4f(r, g, b, a);
    glVertex2f(x0, y0); glVertex2f(x1, y0);
    glVertex2f(x1, y1); glVertex2f(x0, y1);
    glEnd();
}
/* vertical gradient quad */
static void gradQuad(float x0, float y0, float x1, float y1,
                     float r0, float g0, float b0, float r1, float g1, float b1)
{
    glBegin(GL_QUADS);
    glColor3f(r0, g0, b0); glVertex2f(x0, y0); glVertex2f(x1, y0);
    glColor3f(r1, g1, b1); glVertex2f(x1, y1); glVertex2f(x0, y1);
    glEnd();
}
static void circle(float cx, float cy, float r, float n,
                   float r0, float g0, float b0, float a)
{
    int i;
    glBegin(GL_TRIANGLE_FAN);
    glColor4f(r0, g0, b0, a);
    glVertex2f(cx, cy);
    for (i = 0; i < n; i++) {
        float t = 2.f * (float)M_PI * i / n;
        glVertex2f(cx + r * cosf(t), cy + r * sinf(t));
    }
    glEnd();
}
static void line2(float x0, float y0, float x1, float y1, float w,
                  float r, float g, float b, float a)
{
    float dx = x1 - x0, dy = y1 - y0;
    float L = sqrtf(dx * dx + dy * dy);
    if (L < 1e-4f) return;
    float nx = -dy / L * w * 0.5f, ny = dx / L * w * 0.5f;
    glBegin(GL_QUADS);
    glColor4f(r, g, b, a);
    glVertex2f(x0 + nx, y0 + ny); glVertex2f(x1 + nx, y1 + ny);
    glVertex2f(x1 - nx, y1 - ny); glVertex2f(x0 - nx, y0 - ny);
    glEnd();
}

/* ------------------------------------------------------------------ */
/* Scene pieces                                                        */
/* ------------------------------------------------------------------ */
static void drawSky(void)
{
    gradQuad(0, 0, W, H - GROUND_H, 0.45f, 0.75f, 0.95f, 0.72f, 0.90f, 0.98f);
    /* sun */
    circle(W - 78, 84, 46, 24, 1.f, 0.95f, 0.72f, 0.55f);
    circle(W - 78, 84, 34, 24, 1.f, 0.98f, 0.85f, 1.f);
    /* clouds */
    int i;
    for (i = 0; i < (int)sizeof(clouds)/sizeof(clouds[0]); i++) {
        Cloud *c = &clouds[i];
        float s = 0.6f + 0.9f * c->z;   /* z in (0,1): size/depth */
        float a = 0.30f + 0.40f * c->z;
        circle(c->x, c->y, 16 * s, 14, 1, 1, 1, a);
        circle(c->x - 18 * s, c->y + 5 * s, 12 * s, 12, 1, 1, 1, a);
        circle(c->x + 18 * s, c->y + 5 * s, 13 * s, 12, 1, 1, 1, a);
    }
}

static void drawGround(void)
{
    float y0 = H - GROUND_H;
    quad2(0, y0, W, H, 0.55f, 0.72f, 0.34f, 1.f);            /* dirt-ish base under grass */
    quad2(0, y0, W, y0 + 22, 0.62f, 0.83f, 0.38f, 1.f);       /* grass strip */
    quad2(0, y0, W, y0 + 4, 0.78f, 0.93f, 0.55f, 1.f);        /* grass highlight */
    /* diagonal grass ticks, scrolling */
    int i;
    float o = groundOffset - floorf(groundOffset / 28.f) * 28.f;
    for (i = -1; i * 28 - o < W + 28; i++) {
        float x = i * 28.f - o;
        line2(x, y0 + 22, x + 14, y0 + 22 + 0, 4, 0.5f, 0.71f, 0.33f, 1.f);
        /* dirt speckles */
        quad2(x + 6, y0 + 40 + (i % 3) * 14, x + 12, y0 + 46 + (i % 3) * 14, 0.47f, 0.36f, 0.22f, 1.f);
    }
}

static void drawPipe(float x, float gapY)
{
    float y0 = H - GROUND_H;
    float topH = gapY - GAP * 0.5f;   /* top pipe bottom edge */
    float botY = gapY + GAP * 0.5f;
    float capH = 26, capOver = 5;

    /* top pipe body */
    gradQuad(x, 0, x + PIPE_W, topH, 0.26f, 0.68f, 0.30f, 0.42f, 0.82f, 0.36f);
    quad2(x + 4, 0, x + 12, topH, 0.62f, 0.90f, 0.55f, 0.7f);   /* light edge */
    /* top pipe cap */
    gradQuad(x - capOver, topH - capH, x + PIPE_W + capOver, topH,
             0.30f, 0.74f, 0.34f, 0.48f, 0.86f, 0.40f);
    quad2(x - capOver, topH - 3, x + PIPE_W + capOver, topH, 0.16f, 0.45f, 0.22f, 0.9f);

    /* bottom pipe cap + body */
    gradQuad(x - capOver, botY, x + PIPE_W + capOver, botY + capH,
             0.48f, 0.86f, 0.40f, 0.30f, 0.74f, 0.34f);
    gradQuad(x, botY + capH, x + PIPE_W, y0, 0.42f, 0.82f, 0.36f, 0.26f, 0.68f, 0.30f);
    quad2(x + 4, botY + capH, x + 12, y0, 0.62f, 0.90f, 0.55f, 0.7f);
}

static void drawBird(float x, float y, float rot, float wingPhase)
{
    /* rot: radians, negative = nose up (world y-down) */
    float cr = cosf(rot), sr = sinf(rot);
    float wingY = wingPhase; /* precomputed sin */
    /* shadow on ground */
    {
        float gy = H - GROUND_H + 6;
        float k = 1.f - (gy - y) / (H * 0.9f);
        if (k < 0.2f) k = 0.2f;
        circle(x, gy, 10 + 6 * k, 12, 0, 0, 0, 0.16f * k);
    }
    /* save GL state and transform manually: draw in local coords rotated */
    glBegin(GL_TRIANGLE_FAN);
    /* body (ellipse-ish, 16 points) */
    glColor4f(1.0f, 0.80f, 0.25f, 1.f);
    glVertex2f(x, y);
    {
        int i;
        for (i = 0; i < 16; i++) {
            float t = 2.f * (float)M_PI * i / 16;
            float ex = 13.5f * cosf(t), ey = 10.5f * sinf(t);
            glVertex2f(x + ex * cr - ey * sr, y + ex * sr + ey * cr);
        }
    }
    glEnd();
    /* belly */
    {
        float bx = x + 2.5f * cr, by = y + 3.5f + 2.5f * sr;
        circle(bx, by, 6.5f, 14, 0.99f, 0.95f, 0.80f, 1.f);
    }
    /* wing: ellipse flapping around shoulder point */
    {
        float shx = x - 3.5f, shy = y + 1.5f;
        float wa = -0.9f + wingY * 1.1f;   /* wing angle */
        float wx = cosf(wa), wy = sinf(wa);
        glBegin(GL_TRIANGLE_FAN);
        glColor4f(0.95f, 0.66f, 0.12f, 1.f);
        glVertex2f(shx, shy);
        {
            int i;
            for (i = 0; i < 12; i++) {
                float t = 2.f * (float)M_PI * i / 12;
                float lx = 8.5f * cosf(t), ly = 4.5f * sinf(t);
                /* rotate wing-local by wa, then by body rot */
                float rx = lx * wx - ly * wy;
                float ry = lx * wy + ly * wx;
                glVertex2f(shx + rx * cr - ry * sr, shy + rx * sr + ry * cr);
            }
        }
        glEnd();
    }
    /* eye */
    circle(x + 6.0f, y - 3.5f, 4.6f, 14, 1.f, 1.f, 1.f, 1.f);
    circle(x + 7.2f, y - 3.5f, 2.3f, 10, 0.15f, 0.15f, 0.15f, 1.f);
    circle(x + 6.4f, y - 4.4f, 0.9f, 8, 1.f, 1.f, 1.f, 1.f);
    /* beak */
    {
        glBegin(GL_TRIANGLES);
        glColor4f(0.98f, 0.45f, 0.20f, 1.f);
        glVertex2f(x + 10.5f, y - 1.5f);
        glVertex2f(x + 19.0f, y + 1.0f);
        glVertex2f(x + 10.5f, y + 3.5f);
        glEnd();
        line2(x + 10.5f, y + 1.0f, x + 17.5f, y + 1.2f, 1.2f, 0.80f, 0.32f, 0.12f, 1.f);
    }
}

static void drawParticles(void)
{
    int i;
    for (i = 0; i < nParts; i++) {
        Particle *p = &parts[i];
        float k = p->life > 1.f ? 1.f : p->life;
        if (p->c == 'f') { /* feather: small yellow-orange ellipse */
            circle(p->x, p->y, 3.2f * k, 8, 1.f, 0.78f, 0.2f, 0.9f * k);
        } else if (p->c == 'd') { /* dust puff */
            circle(p->x, p->y, 5.f * (2.f - k), 10, 0.95f, 0.93f, 0.85f, 0.5f * k);
        } else { /* sparkle */
            circle(p->x, p->y, 2.5f * k, 8, 1.f, 0.95f, 0.4f, 0.9f * k);
        }
    }
}

static void drawHUD(void)
{
    char buf[64];
    if (state == S_PLAY) {
        snprintf(buf, sizeof buf, "%d", score);
        /* dark outline (4 offsets), then bright text on top */
        drawColText(buf, W/2 - 2, 44, 0.12f, 0.2f, 0.35f, 2.0f, 1);
        drawColText(buf, W/2 + 2, 44, 0.12f, 0.2f, 0.35f, 2.0f, 1);
        drawColText(buf, W/2,   44 - 2, 0.12f, 0.2f, 0.35f, 2.0f, 1);
        drawColText(buf, W/2,   44 + 2, 0.12f, 0.2f, 0.35f, 2.0f, 1);
        drawColText(buf, W/2, 44, 1.f, 1.f, 1.f, 2.0f, 1);
    }
}

/* ------------------------------------------------------------------ */
/* Game logic                                                          */
/* ------------------------------------------------------------------ */
static float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void resetGame(void)
{
    int i;
    state = S_READY;
    birdY = H * 0.42f;
    birdV = 0.f;
    birdRot = 0.f;
    score = 0;
    nPipes = 0;
    nParts = -1;
    bannerOn = 0;
    groundOffset = 0.f;
    for (i = 0; i < 8; i++) {
        clouds[i].x = (float)(rand() % (W + 200)) - 100.f;
        clouds[i].y = 40.f + (float)(rand() % 260);
        clouds[i].z = 0.2f + 0.8f * ((float)rand() / RAND_MAX);
        clouds[i].life = 1;
    }
    bannerOn = 1; banner.t = 2.2f;
    banner.s = "Get Ready";
}

static void spawnParticles(float x, float y, int n, int kind)
{
    int i;
    for (i = 0; i < n; i++) {
        int slot;
        nParts++;
        if (nParts >= (int)(sizeof parts / sizeof parts[0])) nParts = 0;
        slot = nParts % (int)(sizeof parts / sizeof parts[0]);
        Particle *p = &parts[slot];
        float a = (float)rand() / RAND_MAX * 2.f * (float)M_PI;
        float sp = kind == 1 ? 120.f + 160.f * ((float)rand()/RAND_MAX)
                             : 30.f  +  60.f * ((float)rand()/RAND_MAX);
        p->x = x; p->y = y;
        p->vx = cosf(a) * sp * 0.7f;
        p->vy = sinf(a) * sp * 0.7f - (kind == 0 ? 60.f : 0.f);
        p->life = 1.f;
        p->c = kind == 1 ? 'f' : (kind == 2 ? 'd' : '*');
    }
}

static void doFlap(void)
{
    if (state == S_READY) state = S_PLAY;
    if (state != S_PLAY) return;
    birdV = FLAP_VY;
    play(sfxFlap);
    spawnParticles(BIRD_X - 10, birdY + 8, 6, 2);
}

static void saveBest(void)
{
    FILE *f = fopen(".flappy_best", "w");
    if (f) { fprintf(f, "%d\n", bestScore); fclose(f); }
}
static void loadBest(void)
{
    FILE *f = fopen(".flappy_best", "r");
    if (f) {
        int v = 0;
        if (fscanf(f, "%d", &v) == 1 && v > 0) bestScore = v;
        fclose(f);
    }
}
static void killBird(void)
{
    int i;
    state = S_DEAD;
    if (score > bestScore) { bestScore = score; saveBest(); }
    play(sfxHit);
    /* delay handled in update via banner; play die after short gap */
    bannerOn = 1; banner.t = 0.35f; banner.s = "";
    birdV = -180.f;
    spawnParticles(BIRD_X, birdY, 16, 1);
    for (i = 0; i < nPipes; i++) pipes[i].scored = 1;
    play(sfxDie);
}

static void update(float dt)
{
    float y0 = H - GROUND_H;
    int i;

    tGlobal += dt;

    if (state == S_DEAD && bannerOn) {
        /* banner.t counts down; when it hits 0 -> show "Game Over" panel */
    }

    /* clouds drift in all states */
    for (i = 0; i < 8; i++) {
        clouds[i].x -= (6.f + 14.f * clouds[i].z) * dt;
        if (clouds[i].x < -80.f) {
            clouds[i].x = W + 80.f;
            clouds[i].y = 40.f + (float)(rand() % 260);
            clouds[i].z = 0.2f + 0.8f * ((float)rand() / RAND_MAX);
        }
    }

    /* particles */
    for (i = 0; i < (int)(sizeof parts / sizeof parts[0]); i++) {
        Particle *p = &parts[i];
        if (p->life <= 0.f) continue;
        p->x += p->vx * dt;
        p->y += p->vy * dt;
        p->vy += 240.f * dt;
        p->life -= dt * (p->c == 'd' ? 2.2f : 1.4f);
    }

    if (state == S_READY) {
        /* gentle hover */
        birdY = H * 0.42f + 8.f * sinf(tGlobal * 3.f);
        birdRot = 0.15f * sinf(tGlobal * 3.f);
        groundOffset += SPEED * dt * 0.5f;
        return;
    }

    if (state == S_PLAY) {
        groundOffset += SPEED * dt;
    }

    if (state == S_PLAY || state == S_DEAD) {
        /* bird physics (dead: falls to ground, no controls) */
        birdV += GRAVITY * dt;
        if (birdV > MAX_FALL) birdV = MAX_FALL;
        birdY += birdV * dt;

        /* rotation: nose up while rising, dive when falling */
        {
            float target = clampf(birdV * 0.0028f, -0.5f, 1.35f);
            float rate = birdV < 0.f ? 10.f : 4.5f;
            birdRot += (target - birdRot) * clampf(rate * dt, 0.f, 1.f);
        }

        /* ceiling */
        if (birdY < BIRD_R) { birdY = BIRD_R; birdV = 0.f; }
        /* ground */
        if (birdY > y0 - BIRD_R) {
            birdY = y0 - BIRD_R;
            if (state == S_PLAY) killBird();
            else { birdV = 0.f; if (state == S_DEAD) birdRot = 1.5f; }
        }
    }

    if (state == S_PLAY) {
        /* move + spawn pipes */
        for (i = 0; i < nPipes; i++) pipes[i].x -= SPEED * dt;
        /* drop offscreen */
        while (nPipes > 0 && pipes[0].x < -PIPE_W - 20) {
            int j;
            for (j = 0; j < nPipes - 1; j++) pipes[j] = pipes[j + 1];
            nPipes--;
        }
        if (nPipes == 0) {
            Pipe *p = &pipes[nPipes++];
            p->x = W + 40.f;
            p->gapY = 130.f + (GAP * 0.5f) + (float)rand() / RAND_MAX *
                       (H - GROUND_H - 260.f - GAP);
            p->scored = 0;
        } else {
            Pipe *last = &pipes[nPipes - 1];
            if (last->x < W + 40.f - SPAWN_DIST) {
                Pipe *p = &pipes[nPipes++];
                float span = H - GROUND_H - 260.f - GAP;
                p->x = last->x + SPAWN_DIST;
                p->gapY = 130.f + GAP * 0.5f + (float)rand() / RAND_MAX * span;
                p->scored = 0;
            }
        }

        /* scoring + collision */
        for (i = 0; i < nPipes; i++) {
            Pipe *p = &pipes[i];
            if (!p->scored && p->x + PIPE_W < BIRD_X) {
                p->scored = 1;
                score++;
                play(sfxScore);
                spawnParticles(BIRD_X + 20, birdY, 8, 3);
            }
            /* circle vs two rects of this pipe */
            {
                float px0 = p->x, px1 = p->x + PIPE_W;
                float capOver = 5, capH = 26;
                /* closest-point tests, body + caps */
                float bestD2 = 1e30f;
                /* helper lambda-ish via inline code for each rect */
                /* rect: x0,x1,y0,y1 (caps slightly wider) */
                struct { float x0, x1, y0, y1; } rects[4];
                int nr = 0;
                rects[nr].x0 = px0 - capOver; rects[nr].x1 = px1 + capOver;
                rects[nr].y0 = 0;              rects[nr].y1 = p->gapY - GAP*0.5f - capH; nr++;
                rects[nr].x0 = px0 - capOver; rects[nr].x1 = px1 + capOver;
                rects[nr].y0 = p->gapY - GAP*0.5f; rects[nr].y1 = p->gapY - GAP*0.5f + capH; nr++;
                rects[nr].x0 = px0; rects[nr].x1 = px1;
                rects[nr].y0 = p->gapY + GAP*0.5f; rects[nr].y1 = p->gapY + GAP*0.5f + capH; nr++;
                rects[nr].x0 = px0; rects[nr].x1 = px1;
                rects[nr].y0 = p->gapY + GAP*0.5f + capH; rects[nr].y1 = y0; nr++;
                {
                    int k;
                    for (k = 0; k < nr; k++) {
                        float cx = clampf(BIRD_X, rects[k].x0, rects[k].x1);
                        float cy = clampf(birdY,  rects[k].y0, rects[k].y1);
                        float dx = BIRD_X - cx, dy = birdY - cy;
                        float d2 = dx * dx + dy * dy;
                        if (d2 < bestD2) bestD2 = d2;
                    }
                    if (bestD2 < BIRD_R * BIRD_R) { killBird(); break; }
                }
            }
            if (state != S_PLAY) break;
        }
    }

    if (bannerOn) {
        banner.t -= dt;
        if (banner.t <= 0.f) bannerOn = 0;
    }
}

/* ------------------------------------------------------------------ */
/* Scene composition                                                   */
/* ------------------------------------------------------------------ */
static void panelBox(float x, float y, float w, float h,
                     float r, float g, float b, float a)
{
    /* panel is drawn during the text phase: make sure the font texture
       isn't modulating the quads (transparent texel -> invisible panel) */
    glDisable(GL_TEXTURE_2D);
    quad2(x - 4, y - 4, x + w + 4, y + h + 4, 0.16f, 0.23f, 0.32f, a);
    quad2(x, y, x + w, y + h, r, g, b, a);
}

static void renderScene(void)
{
    /* geometry phase: font texture must be OFF (it stays enabled from the
       previous frame's text phase and would alpha-blend everything to clear) */
    glDisable(GL_TEXTURE_2D);
    drawSky();

    /* pipes (behind bird) */
    int i;
    for (i = 0; i < nPipes; i++) drawPipe(pipes[i].x, pipes[i].gapY);

    drawBird(BIRD_X, birdY, birdRot, sinf(tGlobal * 14.f) * (state == S_PLAY ? 1.f : 0.4f));
    drawParticles();
    drawGround();

    /* HUD + banners (text) */
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texAtlas);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    drawHUD();

    if (state == S_READY && bannerOn) {
        drawColText(banner.s, W/2, 120, 1.f, 1.f, 0.9f, 2.2f, 1);
        drawColText("Press SPACE to flap", W/2, 200, 1.f, 1.f, 1.f, 1.0f, 1);
        drawColText("SPACE / click = flap   P = pause", W/2, H - GROUND_H - 60,
                    1.f, 1.f, 1.f, 0.9f, 1);
    }

    if (state == S_DEAD && !bannerOn) {
        panelBox(W/2 - 170, 150, 340, 250, 0.93f, 0.88f, 0.72f, 1.f);
        drawColText("Game Over", W/2, 172, 0.85f, 0.3f, 0.2f, 1.6f, 1);
        {
            char buf[64];
            snprintf(buf, sizeof buf, "Score  %d", score);
            drawColText(buf, W/2, 250, 0.25f, 0.3f, 0.4f, 1.3f, 1);
            snprintf(buf, sizeof buf, "Best   %d", bestScore);
            drawColText(buf, W/2, 290, 0.25f, 0.3f, 0.4f, 1.3f, 1);
            if (score >= bestScore && score > 0)
                drawColText("NEW", W/2 + 120, 294, 0.9f, 0.35f, 0.1f, 1.0f, 0);
        }
        if (mouseY > 370 && mouseY < 430 && mouseX > W/2 - 110 && mouseX < W/2 + 110)
            panelBox(W/2 - 110, 370, 220, 60, 0.4f, 0.78f, 0.35f, 1.f);
        else
            panelBox(W/2 - 110, 370, 220, 60, 0.36f, 0.7f, 0.32f, 1.f);
        drawColText("Play Again", W/2, 385, 1.f, 1.f, 1.f, 1.2f, 1);
        drawColText("SPACE to restart   ESC to quit", W/2, 448, 0.3f, 0.35f, 0.45f, 0.9f, 1);
    }

    if (paused) {
        glDisable(GL_TEXTURE_2D);
        quad2(0, 0, W, H, 0.1f, 0.15f, 0.25f, 0.55f);
        glEnable(GL_TEXTURE_2D);
        drawColText("Paused", W/2, H/2 - 30, 1.f, 1.f, 1.f, 2.0f, 1);
    }
}

/* ------------------------------------------------------------------ */
/* GLUT callbacks                                                      */
/* ------------------------------------------------------------------ */
static void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, W, H, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_LIGHTING);
    glShadeModel(GL_SMOOTH);
    renderScene();
    if (dumpAt > 0 && frameCount >= dumpAt) { dumpFrame(0); exit(0); }
    glutSwapBuffers();
}

static void reshape(int w, int h)
{
    (void)w; (void)h;
    glViewport(0, 0, W, H);
}

static void dumpPPM(const char *path, const unsigned char *fb)
{
    FILE *f = fopen(path, "wb");
    int i;
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (i = H - 1; i >= 0; i--)
        fwrite(fb + (size_t)i * W * 3, 3, W, f);
    fclose(f);
}

static void dumpFrame(int n)
{
    char path[128];
    unsigned char *fb, *fb2;
    int db = 0;
    glGetIntegerv(GL_DOUBLEBUFFER, &db);
    fprintf(stderr, "GL: %s | %s | doublebuffer=%d | w=%d h=%d\n",
            (const char*)glGetString(GL_VENDOR), (const char*)glGetString(GL_VERSION),
            db, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
    snprintf(path, sizeof path, n ? "/tmp/flappy_frame_%d.ppm" : "/tmp/flappy_dump.ppm", n);
    glReadBuffer(GL_BACK);
    fb = malloc((size_t)W * H * 3);
    glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, fb);
    dumpPPM(path, fb);
    glReadBuffer(GL_FRONT);
    fb2 = malloc((size_t)W * H * 3);
    glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, fb2);
    dumpPPM("/tmp/flappy_dump_front.ppm", fb2);
    fprintf(stderr, "glReadPixels back err=0x%x front err=0x%x\n",
            glGetError(), glGetError());
    free(fb); free(fb2);
    printf("frame dumped: %s\n", path);
    fflush(stdout);
}

static void idle(void)
{
    /* 60 FPS frame cap: keeps CPU sane on displays without vsync */
    double frameStart = (double)glutGet(GLUT_ELAPSED_TIME);
    if (lastTS < 0) lastTS = frameStart;
    float dt = (float)(frameStart - lastTS) / 1000.0f;
    lastTS = frameStart;
    if (dt > 0.1f) dt = 0.1f;
    if (dt <= 0.f) dt = 1.0f / 60.0f;
    if (!paused) update(dt);
    display();
    double elapsed = (double)glutGet(GLUT_ELAPSED_TIME) - frameStart;
    double frameMs = 1000.0 / 60.0;
    if (elapsed < frameMs) {
        struct timespec ts;
        double rem = frameMs - elapsed;
        ts.tv_sec = (long)(rem / 1000.0);
        ts.tv_nsec = (long)((rem - (double)ts.tv_sec * 1000.0) * 1e6);
        nanosleep(&ts, 0);
    }
    frameCount++;
}

static void keyboard(unsigned char k, int x, int y)
{
    (void)x; (void)y;
    if (k == 27 || k == 'q' || k == 'Q') {
        exit(0);          /* Apple GLUT has no glutLeaveMainLoop; atexit cleans up */
    }
    if (k == 'p' || k == 'P') {
        if (state == S_PLAY || state == S_READY) paused = !paused;
        return;
    }
    if (k == 'd') {   /* debug: dump current framebuffer to PPM */
        static int shotN = 0;
        dumpFrame(1);
        (void)shotN;
    }
    if (k == ' ' || k == '\r') {
        if (paused) return;
        if (state == S_READY || state == S_PLAY) doFlap();
        else if (state == S_DEAD && !bannerOn) { play(sfxSwoosh); resetGame(); }
    }
}

static void special(int key, int x, int y)
{
    (void)x; (void)y;
    if (key == GLUT_KEY_UP) {
        if (!paused) {
            if (state == S_READY || state == S_PLAY) doFlap();
            else if (state == S_DEAD && !bannerOn) { play(sfxSwoosh); resetGame(); }
        }
    }
    if (key == GLUT_KEY_LEFT || key == GLUT_KEY_RIGHT) { /* reserved */ }
}

static void mouse(int btn, int st, int x, int y)
{
    mouseX = x; mouseY = H - y;
    mouseDown = st == GLUT_DOWN;
    if (btn != GLUT_LEFT_BUTTON) return;
    if (st != GLUT_DOWN) return;
    if (paused) return;
    if (state == S_READY || state == S_PLAY) {
        doFlap();
    } else if (state == S_DEAD && !bannerOn) {
        /* button region */
        if (mouseY > 370 && mouseY < 430 && mouseX > W/2 - 110 && mouseX < W/2 + 110) {
            play(sfxSwoosh); resetGame();
        } else {
            play(sfxSwoosh); resetGame();
        }
    }
}

static void mouseMove(int x, int y)
{
    mouseX = x; mouseY = H - y;
}

static void initGLut(void)
{
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(W, H);
    glutInitWindowPosition(200, 60);
    glutCreateWindow("Flappy Bird - C/OpenGL");
    glClearColor(0.45f, 0.75f, 0.95f, 1.f);
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(special);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseMove);
}

#ifndef FLAPPY_HEADLESS_TEST
static void freeResources(void)
{
    if (face) FT_Done_Face(face);
    if (ft) FT_Done_FreeType(ft);
    if (sfxFlap) Mix_FreeChunk(sfxFlap);
    if (sfxScore) Mix_FreeChunk(sfxScore);
    if (sfxHit) Mix_FreeChunk(sfxHit);
    if (sfxDie) Mix_FreeChunk(sfxDie);
    if (sfxSwoosh) Mix_FreeChunk(sfxSwoosh);
    if (haveAudio) { Mix_CloseAudio(); SDL_Quit(); }
}

int main(int argc, char **argv)
{
    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--noaudio") == 0)
            setenv("SDL_AUDIO_DRIVER", "dummy", 1);   /* silent, still "works" */
        else if (strcmp(argv[i], "--font") == 0 && i + 1 < argc)
            fontPath = argv[++i];
    }
    if (getenv("FLAPPY_DUMP")) dumpAt = atoi(getenv("FLAPPY_DUMP"));
    loadBest();
    initAudio();
    atexit(freeResources);      /* glutMainLoop never returns under Apple GLUT */
    glutInit(&argc, argv);
    initGLut();
    initText();
    resetGame();
    glutMainLoop();
    return 0;
}
#endif
