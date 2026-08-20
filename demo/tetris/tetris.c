/*
 * tetris.c -- Tetris in C using OpenGL (legacy fixed-function pipeline,
 * GLUT bitmap fonts, AppKit event loop on macOS).
 *
 * Build:
 *   cc -x objective-c -o tetris tetris.c \
 *      -framework AppKit -framework OpenGL -framework GLUT
 *
 * Controls:
 *   Left / Right     move piece
 *   Down             soft drop (hold to fall faster)
 *   X  / left-click  rotate clockwise
 *   Z  / right-click rotate counter-clockwise
 *   Space            hard drop
 *   P                pause
 *   R                restart (also after game over)
 */

#define GL_SILENCE_DEPRECATION
#import <Cocoa/Cocoa.h>
#import <objc/runtime.h>
#import <OpenGL/gl.h>
#import <OpenGL/glu.h>
#import <GLUT/glut.h>
#include <Carbon/Carbon.h>   /* kVK_ virtual key codes */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ================================================================== */
/* Board / pieces                                                     */
/* ================================================================== */

enum { COLS = 10, ROWS = 20, NPIECE = 7 };
#define VIEW_W 34.0

static const uint8_t SHAPES[NPIECE][16] = {
    { 0,0,0,0,  1,1,1,1,  0,0,0,0,  0,0,0,0 },  /* I */
    { 0,0,0,0,  0,1,1,0,  0,1,1,0,  0,0,0,0 },  /* O */
    { 0,0,0,0,  0,1,0,0,  1,1,1,0,  0,0,0,0 },  /* T */
    { 0,0,0,0,  0,1,1,0,  1,1,0,0,  0,0,0,0 },  /* S */
    { 0,0,0,0,  1,1,0,0,  0,1,1,0,  0,0,0,0 },  /* Z */
    { 0,0,0,0,  1,0,0,0,  1,1,1,0,  0,0,0,0 },  /* J */
    { 0,0,0,0,  0,0,1,0,  1,1,1,0,  0,0,0,0 },  /* L */
};

static const float COLORS[NPIECE][3] = {
    {0.0f,0.79f,1.0f}, {1.0f,0.85f,0.0f}, {0.6f,0.3f,0.9f},
    {0.2f,0.8f,0.3f}, {0.9f,0.2f,0.2f}, {0.2f,0.4f,1.0f},
    {1.0f,0.6f,0.1f},
};

typedef struct {
    uint8_t board[ROWS][COLS];   /* 0 = empty, else piece id + 1 */
    int type, rot, x, y;         /* current piece, y=0 at top */
    int next;                    /* next piece id */
    long score, lines, level;
    double gravity, lastTick;
    int softHeld, paused, over;
} Game;

static Game G;
static int ghostY = 0;
static NSWindow *g_win = NULL;

/* Rotate the base shape CW `rot` times into m[16]. */
static void shapeFor(int type, int rot, int *m)
{
    int m1[16];
    memcpy(m, SHAPES[type], 16);
    while (rot-- > 0) {
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                m1[c * 4 + (3 - r)] = m[r * 4 + c];
        memcpy(m, m1, 16);
    }
}

static int collides(int type, int rot, int x, int y)
{
    int m[16];
    shapeFor(type, rot, m);
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            if (m[r * 4 + c]) {
                int bx = x + c, by = y + r;
                if (bx < 0 || bx >= COLS || by >= ROWS) return 1;
                if (by >= 0 && G.board[by][bx]) return 1;
            }
    return 0;
}

static int fits(int type, int rot)   /* at spawn position */
{
    return !collides(type, rot, COLS / 2 - 1, 0);
}

static void respawn(void)
{
    G.type = G.next;
    G.rot = 0;
    G.x = COLS / 2 - 1;
    G.y = 0;
    while (!fits(G.type, 0) || !fits(G.next, 0))
        G.next = (G.next + 1 + rand()) % NPIECE;
    if (collides(G.type, 0, G.x, G.y)) G.over = 1;
}

static void lockPiece(void)
{
    int m[16];
    shapeFor(G.type, G.rot, m);
    int dead = 1;
    for (int r = 0; r < 4 && dead; r++)
        for (int c = 0; c < 4 && dead; c++)
            if (m[r * 4 + c] && G.y + r >= 0) {
                G.board[G.y + r][G.x + c] = G.type + 1;
                if (G.y + r < 1) dead = 0;
            }
    int cleared = 0;
    for (int r = ROWS - 1; r >= 0; r--) {
        int full = 1;
        for (int c = 0; c < COLS && full; c++) full = G.board[r][c];
        if (full) {
            cleared++;
            for (int rr = r; rr > 0; rr--)
                memcpy(G.board[rr], G.board[rr - 1], COLS);
            memset(G.board[0], 0, COLS);
            r++;
        }
    }
    if (cleared) {
        G.lines += cleared;
        G.score += cleared * cleared * 100;
        G.level = G.lines / 10;
        G.gravity = 0.8 * (1.0 - 0.08 * G.level);
        if (G.gravity < 0.07) G.gravity = 0.07;
    }
    if (cleared || dead) respawn();
}

static void resetGame(void)
{
    memset(&G, 0, sizeof G);
    G.gravity = 0.8;
    G.next = rand() % NPIECE;
    G.lastTick = [NSProcessInfo processInfo].systemUptime;
    respawn();
}

static void stepDown(void)
{
    if (!collides(G.type, G.rot, G.x, G.y + 1)) { G.y++; return; }
    lockPiece();
}

static void tryMove(int dx)
{
    if (!G.over && !G.paused && !collides(G.type, G.rot, G.x + dx, G.y))
        G.x += dx;
}

static void tryRot(int cw)
{
    if (G.over || G.paused) return;
    int nr = (G.rot + (cw ? 1 : 3)) & 3;
    for (int dx = -2; dx <= 2; dx++)
        if (!collides(G.type, nr, G.x + dx, G.y)) {
            G.rot = nr; G.x += dx; return;
        }
    for (int dx = -2; dx <= 2; dx++)
        if (!collides(G.type, nr, G.x + dx, G.y - 1)) {
            G.rot = nr; G.x += dx; G.y -= 1; return;
        }
}

static void hardDrop(void)
{
    if (G.over || G.paused) return;
    while (!collides(G.type, G.rot, G.x, G.y + 1)) { G.y++; G.score++; }
    lockPiece();
}

/* ================================================================== */
/* Rendering (legacy OpenGL fixed-function pipeline)                  */
/* ================================================================== */

static void putRect(float x, float y, float w, float h,
                    float r, float g, float b, float a)
{
    glColor4f(r, g, b, a);
    glBegin(GL_QUADS);
    glVertex2f(x, y); glVertex2f(x + w, y);
    glVertex2f(x + w, y + h); glVertex2f(x, y + h);
    glEnd();
}

static void putText(float x, float y, const char *s)
{
    glColor4f(0.85f, 0.85f, 0.95f, 1.0f);
    glRasterPos2f(x, y);
    for (; *s; s++)
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *s);
}

/* one unit cell at (px, py), screen y grows up */
static void drawCell(float px, float py, int pid, float a)
{
    float cx = px + 0.08f, cy = py + 0.08f, s = 0.84f;
    putRect(cx, cy, s, s, COLORS[pid][0], COLORS[pid][1], COLORS[pid][2], a);
    putRect(cx, cy + s * 0.80f, s, s * 0.20f, 1, 1, 1, 0.35f * a);
    putRect(cx + s * 0.80f, cy, s, s * 0.20f, 0, 0, 0, 0.35f * a);
    glLineWidth(1.5f);
    glColor4f(0, 0, 0, 0.6f * a);
    glBegin(GL_LINE_LOOP);
    glVertex2f(cx, cy); glVertex2f(cx + s, cy);
    glVertex2f(cx + s, cy + s); glVertex2f(cx, cy + s);
    glEnd();
}

static void drawBoard(void)
{
    putRect(-0.5f, -0.5f, COLS + 1.0f, ROWS + 1.0f, 0.03f, 0.03f, 0.10f, 1);
    glLineWidth(1.0f);
    glColor4f(1, 1, 1, 0.08f);
    glBegin(GL_LINES);
    for (int c = 0; c <= COLS; c++) { glVertex2f(c, 0); glVertex2f(c, ROWS); }
    for (int r = 0; r <= ROWS; r++) { glVertex2f(0, r); glVertex2f(COLS, r); }
    glEnd();
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            if (G.board[r][c])
                drawCell(c, ROWS - 1 - r, G.board[r][c] - 1, 1.0f);
}

static void drawActive(void)
{
    if (G.over) return;
    int m[16];
    shapeFor(G.type, G.rot, m);
    for (int gy = G.y; !collides(G.type, G.rot, G.x, gy + 1); gy++)
        ghostY = gy;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            if (m[r * 4 + c] && ghostY + r >= 0)
                putRect(G.x + c + 0.08f, ROWS - 1 - (ghostY + r) + 0.08f,
                        0.84f, 0.84f,
                        COLORS[G.type][0], COLORS[G.type][1], COLORS[G.type][2],
                        0.28f);
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            if (m[r * 4 + c] && G.y + r >= 0)
                drawCell(G.x + c, ROWS - 1 - (G.y + r), G.type, 1.0f);
}

static void drawPanel(void)
{
    float x = COLS + 2.0f;
    putText(x, 17, "TETRIS");
    putText(x, 13, "NEXT");
    int m[16];
    shapeFor(G.next, 0, m);
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            if (m[r * 4 + c])
                drawCell(x + 0.5f + c, 9 + r, G.next, 1.0f);

    char buf[64];
    snprintf(buf, sizeof buf, "SCORE %ld", G.score);
    putText(x, 5, buf);
    snprintf(buf, sizeof buf, "LINES %ld", G.lines);
    putText(x, 3, buf);
    snprintf(buf, sizeof buf, "LEVEL %ld", G.level);
    putText(x, 1, buf);

    if (G.paused && !G.over) putText(x, ROWS / 2, "PAUSED");
    if (G.over) {
        putRect(COLS / 2.0f - 7.0f, ROWS / 2.0f - 4.0f, 14.0f, 6.0f, 0, 0, 0, 0.7f);
        putText(COLS / 2.0f - 6.0f, ROWS / 2.0f - 1.0f, "GAME OVER");
        putText(COLS / 2.0f - 5.5f, ROWS / 2.0f - 3.0f, "Press R");
    }
}

static void render(void)
{
    NSSize d = [[NSApp keyWindow] contentView].frame.size;
    if (d.width < 1 || d.height < 1) { d.width = 760; d.height = 600; }
    glViewport(0, 0, (int)d.width, (int)d.height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, VIEW_W, -0.5, ROWS + 1.0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClearColor(0.0f, 0.0f, 0.04f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    drawBoard();
    drawActive();
    drawPanel();
}

/* ================================================================== */
/* Save framebuffer (self-test capture)                               */
/* ================================================================== */

static void selftestSave(const char *path, int w, int h)
{
    if (w <= 0 || h <= 0) { fprintf(stderr, "selftest: bad size %dx%d\n", w, h); return; }
    GLubyte *px = calloc(1, (size_t)w * h * 3);
    if (!px) { fprintf(stderr, "selftest: alloc fail\n"); return; }
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "selftest: fopen fail\n"); free(px); return; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int y = h - 1; y >= 0; y--)
        fwrite(px + (size_t)y * w * 3, 1, (size_t)w * 3, f);
    fclose(f); free(px);
    fprintf(stderr, "selftest: saved %dx%d to %s\n", w, h, path);
}

/* ================================================================== */
/* Game tick (NSTimer target/selector)                                */
/* ================================================================== */

static void gameTick(id self, SEL _cmd)
{
    (void)self; (void)_cmd;
    if (!G.over) {
        double now = [NSProcessInfo processInfo].systemUptime;
        double interval = G.softHeld
            ? (G.gravity < 0.07 ? G.gravity : G.gravity * 0.15)
            : G.gravity;
        while (now - G.lastTick >= interval && !G.over) {
            G.lastTick += interval;
            stepDown();
        }
    }
    NSWindow *win = [NSApp keyWindow];
    if (win) {
        id vr = [win firstResponder];
        if ([vr isKindOfClass:[NSOpenGLView class]])
            [(NSView *)vr setNeedsDisplay:YES];
    }
}

static void ticker_tick(id self, SEL _cmd, NSTimer *timer)
{
    (void)self; (void)_cmd; (void)timer;
    gameTick(NULL, NULL);
    /* optional on-timer frame capture for headless verification */
    const char *cap = getenv("TETRIS_CAPTURE");
    if (cap) {
        static int ticks = 0;
        int at = 480;
        const char *a = getenv("TETRIS_CAPTURE_AT");
        if (a) at = atoi(a);
        if (++ticks == at) {
            NSWindow *win = g_win ? g_win : [NSApp keyWindow];
            NSOpenGLView *view = [win contentView];
            NSOpenGLContext *ctx = [view openGLContext];
            [ctx makeCurrentContext];
            NSSize d = [view frame].size;
            selftestSave(cap, (int)d.width, (int)d.height);
            exit(0);
        }
    }
}

/* ================================================================== */
/* Input (NSOpenGLView subclass, methods added in C)                  */
/* ================================================================== */

static void handleKey(unsigned short kc)
{
    if (G.over) { if (kc == kVK_ANSI_R) resetGame(); return; }
    if (G.paused) {
        if (kc == kVK_ANSI_P) G.paused = 0;
        else if (kc == kVK_ANSI_R) { G.paused = 0; resetGame(); }
        return;
    }
    switch (kc) {
    case kVK_LeftArrow:   tryMove(-1); break;
    case kVK_RightArrow:  tryMove(1); break;
    case kVK_DownArrow:   G.softHeld = 1; G.lastTick = [NSProcessInfo processInfo].systemUptime - 0.2; break;
    case kVK_ANSI_X:      tryRot(1); break;
    case kVK_ANSI_A:      tryRot(1); break;
    case kVK_ANSI_Z:      tryRot(0); break;
    case kVK_Space:       hardDrop(); break;
    case kVK_ANSI_P:      G.paused = 1; break;
    case kVK_ANSI_R:      resetGame(); break;
    default: break;
    }
}

static void tv_keyDown(id self, SEL _cmd, NSEvent *e)
{
    (void)self; (void)_cmd;
    if ([e type] == NSEventTypeKeyDown)
        handleKey((unsigned short)[e keyCode]);
}

static void tv_keyUp(id self, SEL _cmd, NSEvent *e)
{
    (void)self; (void)_cmd;
    if ([e type] == NSEventTypeKeyUp && [e keyCode] == kVK_DownArrow)
        G.softHeld = 0;
}

static void tv_leftDrag(id self, SEL _cmd, NSEvent *e)
{
    (void)self; (void)_cmd; (void)e;
    tryRot(1);
}

static void tv_rightDrag(id self, SEL _cmd, NSEvent *e)
{
    (void)self; (void)_cmd; (void)e;
    tryRot(0);
}

static BOOL tv_acceptsFirst(id self, SEL _cmd)
{
    (void)self; (void)_cmd;
    return YES;
}

/* Call the original (superclass) drawRect: so the GL context is made
 * current and bound to the view, then render and flush. */
static void tv_draw_setup(id self, SEL _cmd, NSRect r)
{
    Class kls = object_getClass(self);
    Method m = class_getInstanceMethod(class_getSuperclass(kls),
                                       @selector(drawRect:));
    ((void (*)(id, SEL, NSRect))method_getImplementation(m))(self, _cmd, r);
}

static void tv_draw(id self, SEL _cmd, NSRect r)
{
    (void)r;
    tv_draw_setup(self, _cmd, r);
    render();
    [[self openGLContext] flushBuffer];
}

/* NSWindow subclass: close button quits the app */
static void win_performClose(id self, SEL _cmd)
{
    (void)self; (void)_cmd;
    [NSApp terminate:nil];
}

/* ================================================================== */
/* main                                                               */
/* ================================================================== */

int main(void)
{
    srand((unsigned)time(NULL));
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        /* minimal app menu (Cmd-Q, Quit item) */
        NSMenu *mainMenu = [[NSMenu alloc] init];
        NSMenuItem *appItem = [[NSMenuItem alloc] init];
        [mainMenu addItem:appItem];
        NSMenu *appMenu = [[NSMenu alloc] init];
        [appItem setSubmenu:appMenu];
        [[appMenu addItemWithTitle:@"Quit Tetris"
            action:@selector(terminate:) keyEquivalent:@"q"] setTarget:NSApp];
        [NSApp setMainMenu:mainMenu];

        NSRect frame = NSMakeRect(0, 0, 760, 600);
        Class wcls = objc_allocateClassPair([NSWindow class], "TTWindow", 0);
        class_addMethod(wcls, @selector(performClose:), (IMP)win_performClose, "v@:");
        objc_registerClassPair(wcls);
        NSWindow *win = [[wcls alloc] initWithContentRect:frame
            styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                       NSWindowStyleMaskMiniaturizable)
            backing:NSBackingStoreBuffered defer:NO];
        [win setTitle:@"Tetris (C + OpenGL)"];
        [win center];
        g_win = win;

        NSOpenGLPixelFormatAttribute pfAttrs[] = {
            NSOpenGLPFADoubleBuffer,
            NSOpenGLPFAAccelerated,
            NSOpenGLPFAColorSize, 24,
            NSOpenGLPFAAlphaSize, 8,
            NSOpenGLPFADepthSize, 16,
            0
        };
        NSOpenGLPixelFormat *pf = [[NSOpenGLPixelFormat alloc] initWithAttributes:pfAttrs];

        Class vcls = objc_allocateClassPair([NSOpenGLView class], "TTView", 0);
        class_addMethod(vcls, @selector(keyDown:), (IMP)tv_keyDown, "v@:@");
        class_addMethod(vcls, @selector(keyUp:), (IMP)tv_keyUp, "v@:@");
        class_addMethod(vcls, @selector(leftMouseDragged:), (IMP)tv_leftDrag, "v@:@");
        class_addMethod(vcls, @selector(rightMouseDragged:), (IMP)tv_rightDrag, "v@:@");
        class_addMethod(vcls, @selector(acceptsFirstResponder), (IMP)tv_acceptsFirst, "B@:");
        class_addMethod(vcls, @selector(drawRect:), (IMP)tv_draw, "v@:@");
        objc_registerClassPair(vcls);

        NSOpenGLView *view = [[vcls alloc] initWithFrame:frame pixelFormat:pf];
        [win setContentView:view];
        [win makeFirstResponder:view];
        [win makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        resetGame();

        Class tcls = objc_allocateClassPair([NSObject class], "TTLicker", 0);
        class_addMethod(tcls, @selector(tick:), (IMP)ticker_tick, "v@:@");
        objc_registerClassPair(tcls);
        id ticker = [[tcls alloc] init];
        NSTimer *t = [[NSTimer alloc] initWithFireDate:[NSDate date]
            interval:1.0 / 120.0 target:ticker selector:@selector(tick:)
            userInfo:nil repeats:YES];
        [NSRunLoop.mainRunLoop addTimer:t forMode:NSDefaultRunLoopMode];
        [NSRunLoop.mainRunLoop addTimer:t forMode:NSEventTrackingRunLoopMode];
        [NSApp run];
    }
    return 0;
}
