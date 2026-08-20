
/* test_flappy.c -- headless smoke test for flappy.c
 *
 * Runs the real game logic + rendering on a surfaceless EGL (llvmpipe)
 * context into an FBO, drives it with scripted input, reads back pixels,
 * and checks the game-state machine. No display required.
 *
 *   make test
 */
#define FLAPPY_HEADLESS_TEST
#include "flappy.c"
#include <EGL/egl.h>

/* GL 2.0 FBO entry points (declared manually: glew needs a GLX/X context) */
#ifndef GL_RENDERBUFFER
#define GL_RENDERBUFFER        0x8D41
#endif
#ifndef GL_RGB8
#define GL_RGB8                0x8051
#endif
#ifndef GL_FRAMEBUFFER
#define GL_FRAMEBUFFER         0x8D40
#endif
#ifndef GL_COLOR_ATTACHMENT0
#define GL_COLOR_ATTACHMENT0   0x8CE0
#endif
#ifndef GL_FRAMEBUFFER_COMPLETE
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#endif
extern void glGenRenderbuffers(GLsizei n, GLuint *renderbuffers);
extern void glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffers);
extern void glBindRenderbuffer(GLenum target, GLuint renderbuffer);
extern void glRenderbufferStorage(GLenum target, GLenum internalformat, GLsizei width, GLsizei height);
extern void glGenFramebuffers(GLsizei n, GLuint *framebuffers);
extern void glBindFramebuffer(GLenum target, GLuint framebuffer);
extern void glFramebufferRenderbuffer(GLenum target, GLenum attachment,
                                      GLenum renderbuffertarget, GLuint renderbuffer);
extern GLenum glCheckFramebufferStatus(GLenum target);
extern void glFramebufferTexture2D(GLenum target, GLenum attachment,
                                    GLenum textarget, GLuint texture, GLint level);

static int g_failures = 0;
static GLuint g_fbo = 0;
static int g_probeN = 0;
static void probeSky(const char *tag)
{
    unsigned char px[3];
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glReadBuffer(GL_BACK);
    glReadPixels(100, 40, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, px);
    printf("  probe %-14s sky px = %3d,%3d,%3d glerr=0x%x\n", tag, px[0], px[1], px[2], glGetError());
}
#define CHECK(cond, msg) do { \
    if (cond) { printf("ok:   %s\n", msg); } \
    else      { printf("FAIL: %s\n", msg); g_failures++; } \
} while (0)

static void dumpFBO(const char *name)
{
    char path[128];
    unsigned char *fb = malloc((size_t)W * H * 3);
    int rb;
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glGetIntegerv(GL_READ_BUFFER, &rb);
    fprintf(stderr, "dump %s: fbo=%u status=0x%x readbuffer=%d\n", name, g_fbo,
            glCheckFramebufferStatus(GL_FRAMEBUFFER), rb);
    snprintf(path, sizeof path, "/tmp/flappy_test_%s.ppm", name);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, fb);
    {
        FILE *f = fopen(path, "wb");
        int i;
        fprintf(f, "P6\n%d %d\n255\n", W, H);
        for (i = H - 1; i >= 0; i--) fwrite(fb + (size_t)i * W * 3, 3, W, f);
        fclose(f);
    }
    free(fb);
    printf("dump: %s\n", path);
}

static void renderFrame(void)
{
    glViewport(0, 0, W, H);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
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
}

int main(void)
{
    int i, f;
    /* --- EGL surfaceless context --- */
    EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (d == EGL_NO_DISPLAY) { printf("EGL: no display\n"); return 2; }
    EGLint maj, min;
    if (!eglInitialize(d, &maj, &min)) { printf("EGL init failed\n"); return 2; }
    eglBindAPI(EGL_OPENGL_API);
    {
        EGLint cfgAttribs[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE };
        EGLConfig cfg; EGLint ncfg = 0;
        eglChooseConfig(d, cfgAttribs, &cfg, 1, &ncfg);
        if (ncfg < 1) { printf("EGL: no config\n"); return 2; }
        EGLContext ctx = eglCreateContext(d, cfg, EGL_NO_CONTEXT, NULL);
        if (!eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
            printf("EGL makeCurrent failed\n"); return 2;
        }
    }
    printf("EGL renderer: %s | %s\n",
           (const char*)glGetString(GL_VENDOR), (const char*)glGetString(GL_VERSION));

    /* --- FBO --- */
    {
        GLuint fbo, tex;
        glGenFramebuffers(1, &fbo);
        g_fbo = fbo;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            printf("FBO incomplete: 0x%x\n", glCheckFramebufferStatus(GL_FRAMEBUFFER));
            return 2;
        }
    }

    initText();
    resetGame();
    srand(42);

    /* 1. ready state */
    CHECK(state == S_READY, "starts in READY");
    printf("err after initText: 0x%x\n", glGetError());
    for (f = 0; f < 30; f++) update(1.0f/60.0f);
    renderFrame();
    printf("err after render#1: 0x%x\n", glGetError());
    {
        /* multi-point readback: remember glReadPixels origin is BOTTOM-left */
        unsigned char p[3];
        struct { const char *tag; int x, y; } pts[] = {
            {"sky-top(100,540)", 100, 540}, {"sky-top(300,600)", 300, 600},
            {"bird(350,372)", 350, 372}, {"ground(100,40)", 100, 40},
        };
        glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
        glReadBuffer(GL_BACK);
        for (int k = 0; k < 4; k++) {
            glReadPixels(pts[k].x, pts[k].y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, p);
            printf("  after r#1 %-18s = %3d,%3d,%3d\n", pts[k].tag, p[0], p[1], p[2]);
        }
    }
    probeSky("f30");
    renderFrame();
    printf("err after render#2: 0x%x\n", glGetError());
    probeSky("f31(2nd)");
    renderFrame();
    printf("err after render#3: 0x%x\n", glGetError());
    probeSky("f32(3rd)");
    dumpFBO("00_ready");

    /* 2. flap -> play, bird rises */
    {
        float y0 = birdY;
        doFlap();
        CHECK(state == S_PLAY, "flap starts PLAY");
        CHECK(birdV < 0, "flap gives upward velocity");
        for (f = 0; f < 10; f++) update(1.0f/60.0f);
        CHECK(birdY < y0, "bird rose after flap");
    }

    /* 3. play for ~4s flapping every 10 frames: pipes must spawn & move */
    for (i = 0; i < 24; i++) {
        doFlap();
        for (f = 0; f < 10; f++) update(1.0f/60.0f);
        if (state == S_DEAD) break;
    }
    CHECK(nPipes > 0, "pipes were spawned");
    CHECK(state == S_PLAY || state == S_DEAD, "state sane after play");
    renderFrame();
    dumpFBO("10_play");

    /* 4. scoring: deterministic -- pipe placed just behind a safe bird */
    {
        resetGame();
        doFlap();
        for (f = 0; f < 15; f++) update(1.0f/60.0f);
        if (state == S_PLAY && nPipes >= 1) {
            int s0 = score;
            pipes[0].x = BIRD_X - PIPE_W - 6;      /* right edge 6px behind bird */
            pipes[0].gapY = birdY;                 /* gap centered on bird: no collision */
            update(1.0f/60.0f);
            CHECK(score == s0 + 1, "score increments when pipe passed");
            CHECK(state == S_PLAY, "safe pipe pass keeps game alive");
            renderFrame();
            dumpFBO("30_scored");
        } else {
            CHECK(0, "scoring setup reached safe play state");
        }
    }

    /* 5. collision: pipe placed on the bird kills it */
    {
        resetGame();
        doFlap();
        /* one step so a pipe exists */
        for (f = 0; f < 15; f++) update(1.0f/60.0f);
        CHECK(nPipes >= 1, "pipe exists for collision test");
        pipes[0].x = BIRD_X - 20;           /* pipe over the bird */
        pipes[0].gapY = 500;                /* top pipe spans 0..426, bird ~270 inside */
        for (f = 0; f < 3; f++) update(1.0f/60.0f);
        CHECK(state == S_DEAD, "collision kills the bird");
    }
    renderFrame();
    dumpFBO("20_dead_banner");
    /* let the death banner expire, then dump the Game Over panel */
    for (f = 0; f < 40; f++) update(1.0f/60.0f);
    CHECK(!bannerOn, "death banner expires");
    printf("  state=%d bannerOn=%d score=%d birdY=%.1f\n", (int)state, bannerOn, score, birdY);
    renderFrame();
    dumpFBO("20_dead");

    /* 6. restart */
    resetGame();
    CHECK(state == S_READY, "reset returns to READY");
    CHECK(score == 0, "reset clears score");

    /* 7. ground death: no flaps in PLAY -> falls to ground -> DEAD */
    doFlap();
    for (i = 0; i < 400 && state == S_PLAY; i++) update(1.0f/60.0f);
    CHECK(state == S_DEAD, "no-input fall ends in DEAD");

    printf("%s (%d failures)\n", g_failures ? "TESTS FAILED" : "ALL TESTS PASSED", g_failures);
    eglTerminate(eglGetCurrentDisplay());
    return g_failures ? 1 : 0;
}
