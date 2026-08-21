/*
 * tetris.c -- Tetris in C with OpenGL (GLUT).
 *
 * Build: make
 * Run:   ./tetris
 */

#define GL_SILENCE_DEPRECATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

/* ---------------------------------------------------------------- geometry */

#define COLS        10
#define ROWS        22          /* rows 0..1 are the hidden spawn buffer */
#define HIDDEN      2
#define VIS_ROWS    (ROWS - HIDDEN)

#define CANVAS_W    800.0f
#define CANVAS_H    760.0f
#define CELL        30.0f
#define FIELD_X     240.0f
#define FIELD_TOP   700.0f      /* GL y of the top edge of visible row 0 */
#define FIELD_W     (COLS * CELL)
#define FIELD_H     (VIS_ROWS * CELL)

/* ------------------------------------------------------------------ pieces */

enum { PIECE_I, PIECE_J, PIECE_L, PIECE_O, PIECE_S, PIECE_T, PIECE_Z, NPIECES };

/* Rotation happens inside a square box; SRS uses 4x4 for I, 2x2 for O. */
static const int box_size[NPIECES] = { 4, 3, 3, 2, 3, 3, 3 };

/* Spawn cells as (row, col) inside the box, SRS orientation 0. */
static const int spawn_cells[NPIECES][4][2] = {
    { {1,0}, {1,1}, {1,2}, {1,3} },   /* I */
    { {0,0}, {1,0}, {1,1}, {1,2} },   /* J */
    { {0,2}, {1,0}, {1,1}, {1,2} },   /* L */
    { {0,0}, {0,1}, {1,0}, {1,1} },   /* O */
    { {0,1}, {0,2}, {1,0}, {1,1} },   /* S */
    { {0,1}, {1,0}, {1,1}, {1,2} },   /* T */
    { {0,0}, {0,1}, {1,1}, {1,2} },   /* Z */
};

/* shape[piece][rot] holds 4 (row, col) cells inside the piece box. */
static int shape[NPIECES][4][4][2];

typedef struct { float r, g, b; } Color;

static const Color piece_color[NPIECES] = {
    { 0.20f, 0.85f, 0.94f },   /* I cyan   */
    { 0.26f, 0.42f, 0.95f },   /* J blue   */
    { 0.98f, 0.56f, 0.16f },   /* L orange */
    { 0.98f, 0.84f, 0.20f },   /* O yellow */
    { 0.34f, 0.83f, 0.36f },   /* S green  */
    { 0.68f, 0.36f, 0.93f },   /* T purple */
    { 0.94f, 0.30f, 0.36f },   /* Z red    */
};

/*
 * SRS wall kicks. The tables use y-up, so the board applies -dy.
 * Index: [from_rotation][0 = clockwise, 1 = counter-clockwise][test][dx, dy].
 */
static const int kick_jlstz[4][2][5][2] = {
    { { {0,0},{-1,0},{-1, 1},{0,-2},{-1,-2} }, { {0,0},{ 1,0},{ 1, 1},{0,-2},{ 1,-2} } },
    { { {0,0},{ 1,0},{ 1,-1},{0, 2},{ 1, 2} }, { {0,0},{ 1,0},{ 1,-1},{0, 2},{ 1, 2} } },
    { { {0,0},{ 1,0},{ 1, 1},{0,-2},{ 1,-2} }, { {0,0},{-1,0},{-1, 1},{0,-2},{-1,-2} } },
    { { {0,0},{-1,0},{-1,-1},{0, 2},{-1, 2} }, { {0,0},{-1,0},{-1,-1},{0, 2},{-1, 2} } },
};

static const int kick_i[4][2][5][2] = {
    { { {0,0},{-2,0},{ 1,0},{-2,-1},{ 1, 2} }, { {0,0},{-1,0},{ 2,0},{-1, 2},{ 2,-1} } },
    { { {0,0},{-1,0},{ 2,0},{-1, 2},{ 2,-1} }, { {0,0},{ 2,0},{-1,0},{ 2, 1},{-1,-2} } },
    { { {0,0},{ 2,0},{-1,0},{ 2, 1},{-1,-2} }, { {0,0},{ 1,0},{-2,0},{ 1,-2},{-2, 1} } },
    { { {0,0},{ 1,0},{-2,0},{ 1,-2},{-2, 1} }, { {0,0},{-2,0},{ 1,0},{-2,-1},{ 1, 2} } },
};

/* -------------------------------------------------------------- game state */

enum { ST_PLAY, ST_CLEARING, ST_PAUSED, ST_OVER };

typedef struct {
    int type, rot, row, col;    /* row/col = board position of the box corner */
} Piece;

static int   board[ROWS][COLS];          /* 0 empty, else piece type + 1 */
static Piece cur;
static int   bag[NPIECES], bag_left;
static int   queue[4];                   /* upcoming piece types */
static int   hold = -1, hold_used;
static int   state = ST_PLAY;

static int   score, lines, level = 1;
static int   clear_rows[4], nclear;
static float clear_timer;
static float fall_timer, lock_timer;
static int   lock_resets;
static int   grounded;
static float shake;                      /* screen shake left, in seconds */

static int   key_left, key_right, key_down;
static float repeat_left, repeat_right, repeat_down;

#define DAS_DELAY   0.14f
#define DAS_RATE    0.035f
#define SOFT_RATE   0.030f
#define LOCK_DELAY  0.50f
#define MAX_RESETS  15
#define CLEAR_TIME  0.22f

/* ------------------------------------------------------------ piece tables */

static void build_shapes(void)
{
    for (int p = 0; p < NPIECES; p++) {
        int n = box_size[p];
        for (int i = 0; i < 4; i++) {
            shape[p][0][i][0] = spawn_cells[p][i][0];
            shape[p][0][i][1] = spawn_cells[p][i][1];
        }
        for (int r = 1; r < 4; r++)
            for (int i = 0; i < 4; i++) {
                int pr = shape[p][r - 1][i][0], pc = shape[p][r - 1][i][1];
                shape[p][r][i][0] = pc;             /* clockwise inside the box */
                shape[p][r][i][1] = n - 1 - pr;
            }
    }
}

static void refill_bag(void)
{
    for (int i = 0; i < NPIECES; i++) bag[i] = i;
    for (int i = NPIECES - 1; i > 0; i--) {
        int j = rand() % (i + 1), t = bag[i];
        bag[i] = bag[j]; bag[j] = t;
    }
    bag_left = NPIECES;
}

static int next_from_bag(void)
{
    if (bag_left == 0) refill_bag();
    return bag[--bag_left];
}

/* ------------------------------------------------------------ board checks */

static int collides(const Piece *p)
{
    for (int i = 0; i < 4; i++) {
        int r = p->row + shape[p->type][p->rot][i][0];
        int c = p->col + shape[p->type][p->rot][i][1];
        if (c < 0 || c >= COLS || r >= ROWS) return 1;
        if (r >= 0 && board[r][c]) return 1;
    }
    return 0;
}

static void spawn_piece(int type)
{
    cur.type = type;
    cur.rot  = 0;
    cur.col  = (type == PIECE_O) ? 4 : 3;
    cur.row  = (type == PIECE_I) ? 1 : 2;   /* whole piece inside the visible field */
    fall_timer = lock_timer = 0.0f;
    lock_resets = 0;
    grounded = 0;
    if (collides(&cur)) state = ST_OVER;
}

static void pull_next(void)
{
    int t = queue[0];
    queue[0] = queue[1]; queue[1] = queue[2]; queue[2] = queue[3];
    queue[3] = next_from_bag();
    spawn_piece(t);
    hold_used = 0;
}

static float fall_interval(void)
{
    /* Tetris guideline curve, in seconds per row. */
    float base = 0.8f - (level - 1) * 0.007f;
    if (base < 0.05f) base = 0.05f;
    float t = powf(base, (float)(level - 1));
    return t < 0.02f ? 0.02f : t;
}

static void reset_game(void)
{
    memset(board, 0, sizeof board);
    score = lines = 0;
    level = 1;
    hold = -1;
    hold_used = 0;
    nclear = 0;
    shake = 0.0f;
    state = ST_PLAY;
    refill_bag();
    for (int i = 0; i < 4; i++) queue[i] = next_from_bag();
    pull_next();
}

/* ------------------------------------------------------------------ actions */

static void touch_ground(void)
{
    Piece t = cur; t.row++;
    int now_grounded = collides(&t);
    if (now_grounded && !grounded) lock_timer = 0.0f;
    if (!now_grounded) lock_timer = 0.0f;
    grounded = now_grounded;
}

static int try_move(int drow, int dcol)
{
    Piece t = cur;
    t.row += drow; t.col += dcol;
    if (collides(&t)) return 0;
    cur = t;
    if (grounded && lock_resets < MAX_RESETS) { lock_timer = 0.0f; lock_resets++; }
    touch_ground();
    return 1;
}

static int try_rotate(int cw)
{
    if (cur.type == PIECE_O) return 0;
    int from = cur.rot;
    int to = (from + (cw ? 1 : 3)) % 4;
    const int (*table)[5][2] = cur.type == PIECE_I ? kick_i[from] : kick_jlstz[from];
    for (int k = 0; k < 5; k++) {
        Piece t = cur;
        t.rot = to;
        t.col += table[cw ? 0 : 1][k][0];
        t.row -= table[cw ? 0 : 1][k][1];
        if (!collides(&t)) {
            cur = t;
            if (grounded && lock_resets < MAX_RESETS) { lock_timer = 0.0f; lock_resets++; }
            touch_ground();
            return 1;
        }
    }
    return 0;
}

static int ghost_row(void)
{
    Piece t = cur;
    while (1) {
        Piece n = t; n.row++;
        if (collides(&n)) return t.row;
        t = n;
    }
}

static void lock_piece(void)
{
    for (int i = 0; i < 4; i++) {
        int r = cur.row + shape[cur.type][cur.rot][i][0];
        int c = cur.col + shape[cur.type][cur.rot][i][1];
        if (r >= 0 && r < ROWS && c >= 0 && c < COLS)
            board[r][c] = cur.type + 1;
    }

    nclear = 0;
    for (int r = 0; r < ROWS; r++) {
        int full = 1;
        for (int c = 0; c < COLS; c++) if (!board[r][c]) { full = 0; break; }
        if (full) clear_rows[nclear++] = r;
    }

    if (nclear > 0) {
        static const int pts[5] = { 0, 100, 300, 500, 800 };
        score += pts[nclear] * level;
        lines += nclear;
        level = 1 + lines / 10;
        clear_timer = 0.0f;
        shake = nclear == 4 ? 0.32f : 0.12f;
        state = ST_CLEARING;
    } else {
        pull_next();
    }
}

static void finish_clear(void)
{
    for (int k = 0; k < nclear; k++) {
        int r = clear_rows[k];
        for (int y = r; y > 0; y--)
            memcpy(board[y], board[y - 1], sizeof board[0]);
        memset(board[0], 0, sizeof board[0]);
    }
    nclear = 0;
    state = ST_PLAY;
    pull_next();
}

static void hard_drop(void)
{
    int target = ghost_row();
    score += 2 * (target - cur.row);
    cur.row = target;
    lock_piece();
}

static void do_hold(void)
{
    if (hold_used) return;
    int prev = hold;
    hold = cur.type;
    hold_used = 1;
    if (prev < 0) {
        int t = queue[0];
        queue[0] = queue[1]; queue[1] = queue[2]; queue[2] = queue[3];
        queue[3] = next_from_bag();
        spawn_piece(t);
    } else {
        spawn_piece(prev);
    }
}

/* ------------------------------------------------------------------ drawing */

static void quad(float x, float y, float w, float h)
{
    glBegin(GL_QUADS);
    glVertex2f(x, y); glVertex2f(x + w, y);
    glVertex2f(x + w, y + h); glVertex2f(x, y + h);
    glEnd();
}

static void draw_text(float x, float y, float size, const char *s)
{
    /* GLUT_STROKE_ROMAN glyphs are about 100 units tall. */
    float k = size / 100.0f;
    glPushMatrix();
    glTranslatef(x, y, 0.0f);
    glScalef(k, k, k);
    glLineWidth(size > 22.0f ? 2.4f : 1.6f);
    for (const char *p = s; *p; p++) glutStrokeCharacter(GLUT_STROKE_ROMAN, *p);
    glPopMatrix();
}

static float text_width(float size, const char *s)
{
    float w = 0.0f;
    for (const char *p = s; *p; p++) w += glutStrokeWidth(GLUT_STROKE_ROMAN, *p);
    return w * size / 100.0f;
}

/* One mineral block: flat face, lit top-left bevel, dark bottom-right bevel. */
static void draw_block(float x, float y, float s, Color c, float alpha)
{
    float b = s * 0.16f;

    glColor4f(c.r * 0.55f, c.g * 0.55f, c.b * 0.55f, alpha);
    quad(x, y, s, s);

    glColor4f(c.r, c.g, c.b, alpha);
    quad(x + b, y + b, s - 2 * b, s - 2 * b);

    glColor4f(c.r * 0.45f + 0.55f, c.g * 0.45f + 0.55f, c.b * 0.45f + 0.55f, alpha);
    glBegin(GL_TRIANGLES);
    glVertex2f(x, y + s); glVertex2f(x + s, y + s); glVertex2f(x + s - b, y + s - b);
    glVertex2f(x, y + s); glVertex2f(x + s - b, y + s - b); glVertex2f(x + b, y + s - b);
    glVertex2f(x, y + s); glVertex2f(x + b, y + s - b); glVertex2f(x + b, y + b);
    glVertex2f(x, y + s); glVertex2f(x + b, y + b); glVertex2f(x, y);
    glEnd();

    glColor4f(c.r * 0.30f, c.g * 0.30f, c.b * 0.30f, alpha);
    glBegin(GL_TRIANGLES);
    glVertex2f(x + s, y); glVertex2f(x, y); glVertex2f(x + b, y + b);
    glVertex2f(x + s, y); glVertex2f(x + b, y + b); glVertex2f(x + s - b, y + b);
    glVertex2f(x + s, y); glVertex2f(x + s - b, y + b); glVertex2f(x + s - b, y + s - b);
    glVertex2f(x + s, y); glVertex2f(x + s - b, y + s - b); glVertex2f(x + s, y + s);
    glEnd();
}

static void draw_ghost(float x, float y, float s, Color c)
{
    glColor4f(c.r, c.g, c.b, 0.16f);
    quad(x, y, s, s);
    glColor4f(c.r, c.g, c.b, 0.55f);
    glLineWidth(1.6f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x + 1, y + 1); glVertex2f(x + s - 1, y + 1);
    glVertex2f(x + s - 1, y + s - 1); glVertex2f(x + 1, y + s - 1);
    glEnd();
}

/* Board cell (row, col) -> GL position of its lower-left corner. */
static float cell_x(int col) { return FIELD_X + col * CELL; }
static float cell_y(int row) { return FIELD_TOP - (row - HIDDEN + 1) * CELL; }

static void draw_background(void)
{
    glBegin(GL_QUADS);
    glColor3f(0.055f, 0.062f, 0.098f); glVertex2f(0, 0); glVertex2f(CANVAS_W, 0);
    glColor3f(0.105f, 0.115f, 0.180f); glVertex2f(CANVAS_W, CANVAS_H); glVertex2f(0, CANVAS_H);
    glEnd();
}

static void draw_panel(float x, float y, float w, float h, const char *title)
{
    glColor4f(0.0f, 0.0f, 0.0f, 0.35f);
    quad(x, y, w, h);
    glColor4f(0.45f, 0.50f, 0.72f, 0.55f);
    glLineWidth(1.5f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x, y); glVertex2f(x + w, y); glVertex2f(x + w, y + h); glVertex2f(x, y + h);
    glEnd();
    if (title) {
        glColor4f(0.66f, 0.72f, 0.92f, 1.0f);
        draw_text(x + 10.0f, y + h + 10.0f, 17.0f, title);
    }
}

static void draw_field(void)
{
    glColor4f(0.02f, 0.03f, 0.06f, 0.85f);
    quad(FIELD_X, FIELD_TOP - FIELD_H, FIELD_W, FIELD_H);

    glColor4f(1.0f, 1.0f, 1.0f, 0.055f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (int c = 1; c < COLS; c++) {
        glVertex2f(cell_x(c), FIELD_TOP - FIELD_H);
        glVertex2f(cell_x(c), FIELD_TOP);
    }
    for (int r = 1; r < VIS_ROWS; r++) {
        glVertex2f(FIELD_X, FIELD_TOP - r * CELL);
        glVertex2f(FIELD_X + FIELD_W, FIELD_TOP - r * CELL);
    }
    glEnd();

    glColor4f(0.55f, 0.62f, 0.90f, 0.9f);
    glLineWidth(2.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(FIELD_X, FIELD_TOP - FIELD_H);
    glVertex2f(FIELD_X + FIELD_W, FIELD_TOP - FIELD_H);
    glVertex2f(FIELD_X + FIELD_W, FIELD_TOP);
    glVertex2f(FIELD_X, FIELD_TOP);
    glEnd();
}

static int is_clearing_row(int r)
{
    for (int k = 0; k < nclear; k++) if (clear_rows[k] == r) return 1;
    return 0;
}

static void draw_stack(void)
{
    float flash = 0.0f;
    if (state == ST_CLEARING) {
        float t = clear_timer / CLEAR_TIME;
        flash = 1.0f - t;
    }

    for (int r = HIDDEN; r < ROWS; r++) {
        int clearing = state == ST_CLEARING && is_clearing_row(r);
        for (int c = 0; c < COLS; c++) {
            if (!board[r][c]) continue;
            Color col = piece_color[board[r][c] - 1];
            float a = 1.0f;
            if (clearing) {
                col.r = col.r + (1.0f - col.r) * flash;
                col.g = col.g + (1.0f - col.g) * flash;
                col.b = col.b + (1.0f - col.b) * flash;
                a = 0.35f + 0.65f * flash;
            }
            draw_block(cell_x(c), cell_y(r), CELL, col, a);
        }
    }
}

static void draw_current(void)
{
    if (state == ST_OVER || state == ST_CLEARING) return;
    Color c = piece_color[cur.type];
    int g = ghost_row();

    for (int i = 0; i < 4; i++) {
        int r = g + shape[cur.type][cur.rot][i][0];
        int col = cur.col + shape[cur.type][cur.rot][i][1];
        if (r >= HIDDEN) draw_ghost(cell_x(col), cell_y(r), CELL, c);
    }
    for (int i = 0; i < 4; i++) {
        int r = cur.row + shape[cur.type][cur.rot][i][0];
        int col = cur.col + shape[cur.type][cur.rot][i][1];
        if (r >= HIDDEN) draw_block(cell_x(col), cell_y(r), CELL, c, 1.0f);
    }
}

/* Draw a piece centred inside a box, used by the hold and next previews. */
static void draw_preview(int type, float bx, float by, float bw, float bh, float s)
{
    if (type < 0) return;
    int minr = 9, maxr = -9, minc = 9, maxc = -9;
    for (int i = 0; i < 4; i++) {
        int r = shape[type][0][i][0], c = shape[type][0][i][1];
        if (r < minr) minr = r;
        if (r > maxr) maxr = r;
        if (c < minc) minc = c;
        if (c > maxc) maxc = c;
    }
    float w = (maxc - minc + 1) * s, h = (maxr - minr + 1) * s;
    float ox = bx + (bw - w) * 0.5f, oy = by + (bh - h) * 0.5f;
    for (int i = 0; i < 4; i++) {
        int r = shape[type][0][i][0] - minr, c = shape[type][0][i][1] - minc;
        draw_block(ox + c * s, oy + (maxr - minr - r) * s, s, piece_color[type], 1.0f);
    }
}

static void draw_hud(void)
{
    char buf[64];
    float lx = 30.0f, lw = 180.0f;

    glColor4f(0.85f, 0.88f, 1.0f, 1.0f);
    draw_text(FIELD_X + (FIELD_W - text_width(26.0f, "TETRIS")) * 0.5f,
              CANVAS_H - 42.0f, 26.0f, "TETRIS");

    draw_panel(lx, 560.0f, lw, 100.0f, "HOLD");
    draw_preview(hold, lx, 560.0f, lw, 100.0f, 24.0f);

    struct { const char *label; long value; float y; } stats[] = {
        { "SCORE", score, 430.0f },
        { "LINES", lines, 350.0f },
        { "LEVEL", level, 270.0f },
    };
    for (int i = 0; i < 3; i++) {
        draw_panel(lx, stats[i].y, lw, 52.0f, stats[i].label);
        snprintf(buf, sizeof buf, "%ld", stats[i].value);
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        draw_text(lx + lw - 12.0f - text_width(24.0f, buf), stats[i].y + 16.0f, 24.0f, buf);
    }

    static const char *help[] = {
        "MOVE     < >",
        "ROTATE   UP / Z",
        "SOFT     DOWN",
        "DROP     SPACE",
        "HOLD     C",
        "PAUSE    P",
        "RESET    R",
    };
    glColor4f(0.55f, 0.60f, 0.80f, 1.0f);
    for (int i = 0; i < 7; i++)
        draw_text(lx, 200.0f - i * 22.0f, 14.0f, help[i]);

    float rx = FIELD_X + FIELD_W + 30.0f, rw = 150.0f;
    draw_panel(rx, 560.0f, rw, 100.0f, "NEXT");
    draw_preview(queue[0], rx, 560.0f, rw, 100.0f, 24.0f);
    for (int i = 1; i < 4; i++) {
        float y = 470.0f - (i - 1) * 84.0f;
        draw_panel(rx + 18.0f, y, rw - 36.0f, 72.0f, NULL);
        draw_preview(queue[i], rx + 18.0f, y, rw - 36.0f, 72.0f, 17.0f);
    }
}

static void draw_overlay(const char *big, const char *small)
{
    glColor4f(0.02f, 0.02f, 0.05f, 0.72f);
    quad(FIELD_X, FIELD_TOP - FIELD_H, FIELD_W, FIELD_H);

    float cx = FIELD_X + FIELD_W * 0.5f, cy = FIELD_TOP - FIELD_H * 0.5f;
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    draw_text(cx - text_width(30.0f, big) * 0.5f, cy + 10.0f, 30.0f, big);
    glColor4f(0.70f, 0.76f, 0.95f, 1.0f);
    draw_text(cx - text_width(15.0f, small) * 0.5f, cy - 26.0f, 15.0f, small);
}

static void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();

    if (shake > 0.0f) {
        float k = shake * 18.0f;
        glTranslatef(k * ((rand() % 100) / 50.0f - 1.0f),
                     k * ((rand() % 100) / 50.0f - 1.0f), 0.0f);
    }

    draw_background();
    draw_field();
    draw_stack();
    draw_current();
    draw_hud();

    if (state == ST_PAUSED) draw_overlay("PAUSED", "PRESS P TO RESUME");
    if (state == ST_OVER)   draw_overlay("GAME OVER", "PRESS R TO PLAY AGAIN");

    glutSwapBuffers();
}

static void reshape(int w, int h)
{
    /* Letterbox the fixed canvas so the layout never distorts. */
    float sx = w / CANVAS_W, sy = h / CANVAS_H;
    float s = sx < sy ? sx : sy;
    int vw = (int)(CANVAS_W * s), vh = (int)(CANVAS_H * s);
    glViewport((w - vw) / 2, (h - vh) / 2, vw, vh);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, CANVAS_W, 0.0, CANVAS_H, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
}

/* -------------------------------------------------------------------- input */

static void key_down_cb(unsigned char k, int x, int y)
{
    (void)x; (void)y;
    if (k >= 'A' && k <= 'Z') k += 32;

    if (k == 27 || k == 'q') exit(0);
    if (k == 'r') { reset_game(); return; }
    if (k == 'p') {
        if (state == ST_PLAY) state = ST_PAUSED;
        else if (state == ST_PAUSED) state = ST_PLAY;
        return;
    }
    if (state != ST_PLAY) return;

    switch (k) {
    case ' ': hard_drop(); break;
    case 'c': do_hold(); break;
    case 'z': try_rotate(0); break;
    case 'x': try_rotate(1); break;
    }
}

static void special_down_cb(int k, int x, int y)
{
    (void)x; (void)y;
    if (state != ST_PLAY) return;
    switch (k) {
    case GLUT_KEY_LEFT:  key_left = 1;  repeat_left = -DAS_DELAY; try_move(0, -1); break;
    case GLUT_KEY_RIGHT: key_right = 1; repeat_right = -DAS_DELAY; try_move(0, 1); break;
    case GLUT_KEY_DOWN:  key_down = 1;  repeat_down = 0.0f; if (try_move(1, 0)) score++; break;
    case GLUT_KEY_UP:    try_rotate(1); break;
    }
}

static void special_up_cb(int k, int x, int y)
{
    (void)x; (void)y;
    switch (k) {
    case GLUT_KEY_LEFT:  key_left = 0; break;
    case GLUT_KEY_RIGHT: key_right = 0; break;
    case GLUT_KEY_DOWN:  key_down = 0; break;
    }
}

/* ------------------------------------------------------------------- update */

static void update(float dt)
{
    if (shake > 0.0f) { shake -= dt; if (shake < 0.0f) shake = 0.0f; }

    if (state == ST_CLEARING) {
        clear_timer += dt;
        if (clear_timer >= CLEAR_TIME) finish_clear();
        return;
    }
    if (state != ST_PLAY) return;

    if (key_left) {
        repeat_left += dt;
        while (repeat_left >= DAS_RATE) { repeat_left -= DAS_RATE; try_move(0, -1); }
    }
    if (key_right) {
        repeat_right += dt;
        while (repeat_right >= DAS_RATE) { repeat_right -= DAS_RATE; try_move(0, 1); }
    }
    if (key_down) {
        repeat_down += dt;
        while (repeat_down >= SOFT_RATE) {
            repeat_down -= SOFT_RATE;
            if (try_move(1, 0)) score++;
        }
    }

    touch_ground();
    if (grounded) {
        lock_timer += dt;
        if (lock_timer >= LOCK_DELAY) { lock_piece(); return; }
    } else {
        fall_timer += dt;
        float step = fall_interval();
        while (fall_timer >= step) {
            fall_timer -= step;
            if (!try_move(1, 0)) break;
        }
    }
}

static void tick(int unused)
{
    (void)unused;
    static int last = -1;
    int now = glutGet(GLUT_ELAPSED_TIME);
    if (last < 0) last = now;
    float dt = (now - last) / 1000.0f;
    last = now;
    if (dt > 0.10f) dt = 0.10f;    /* a stalled window must not fast-forward */

    update(dt);
    glutPostRedisplay();
    glutTimerFunc(16, tick, 0);
}

/* --------------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));
    build_shapes();
    reset_game();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);
    glutInitWindowSize((int)CANVAS_W, (int)CANVAS_H);
    glutCreateWindow("Tetris");

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    glutIgnoreKeyRepeat(1);
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(key_down_cb);
    glutSpecialFunc(special_down_cb);
    glutSpecialUpFunc(special_up_cb);
    glutTimerFunc(16, tick, 0);

    glutMainLoop();
    return 0;
}
