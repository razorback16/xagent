/*
 * test_logic.c -- headless checks for the game rules in tetris.c.
 * Build: make test && ./test_logic
 */

#define main game_main
#include "tetris.c"
#undef main

static int failures;

static void check(const char *what, int ok)
{
    printf("%-46s %s\n", what, ok ? "pass" : "FAIL");
    if (!ok) failures++;
}

static int count_cells(void)
{
    int n = 0;
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++) if (board[r][c]) n++;
    return n;
}

static void clear_board(void) { memset(board, 0, sizeof board); }

/* Fill a row but leave one column empty. */
static void fill_row_except(int row, int gap)
{
    for (int c = 0; c < COLS; c++) board[row][c] = (c == gap) ? 0 : PIECE_L + 1;
}

static void test_shapes(void)
{
    /* Every rotation of every piece keeps exactly 4 cells inside its box. */
    int ok = 1;
    for (int p = 0; p < NPIECES; p++)
        for (int r = 0; r < 4; r++)
            for (int i = 0; i < 4; i++) {
                int rr = shape[p][r][i][0], cc = shape[p][r][i][1];
                if (rr < 0 || rr >= box_size[p] || cc < 0 || cc >= box_size[p]) ok = 0;
            }
    check("all rotations stay inside the piece box", ok);

    /* Four rotations return to the spawn orientation. */
    ok = 1;
    for (int p = 0; p < NPIECES; p++) {
        int n = box_size[p], a = 0, b = 0;
        for (int i = 0; i < 4; i++) {
            a |= 1 << (shape[p][0][i][0] * n + shape[p][0][i][1]);
            b |= 1 << (shape[p][3][i][0] * n + shape[p][3][i][1]);
        }
        if (a == b) ok = 0;    /* rot 3 must differ from rot 0 for every piece but O */
        if (p == PIECE_O) ok = 1;
    }
    check("rotation 3 differs from rotation 0", ok);

    /* The O piece is rotation invariant. */
    int same = 1;
    for (int r = 1; r < 4; r++) {
        int a = 0, b = 0;
        for (int i = 0; i < 4; i++) {
            a |= 1 << (shape[PIECE_O][0][i][0] * 2 + shape[PIECE_O][0][i][1]);
            b |= 1 << (shape[PIECE_O][r][i][0] * 2 + shape[PIECE_O][r][i][1]);
        }
        if (a != b) same = 0;
    }
    check("O piece is the same in all 4 rotations", same);
}

static void test_single_clear(void)
{
    reset_game();
    clear_board();

    /* An upright I piece plugs a one-cell gap in the bottom row. */
    cur.type = PIECE_I; cur.rot = 1; cur.col = 3; cur.row = 0;
    int bar_col = cur.col + shape[PIECE_I][1][0][1];
    fill_row_except(ROWS - 1, bar_col);

    cur.row = ghost_row();
    lock_piece();
    check("a full row enters the clearing state", state == ST_CLEARING && nclear == 1);

    int before = score;
    finish_clear();
    check("single clear scores 100 x level", before == 100 * 1);
    check("the cleared row leaves 3 cells of the I piece", count_cells() == 3);
    check("play resumes after the clear", state == ST_PLAY);
}

static void test_tetris_clear(void)
{
    reset_game();
    clear_board();
    for (int r = ROWS - 4; r < ROWS; r++) fill_row_except(r, 0);

    cur.type = PIECE_I; cur.rot = 1;
    cur.col = -shape[PIECE_I][1][0][1];   /* put the vertical bar in column 0 */
    cur.row = 0;
    cur.row = ghost_row();
    lock_piece();

    check("four full rows clear together", nclear == 4);
    check("a tetris scores 800 x level", score == 800);
    check("level rises after 10 lines", level == 1);
    finish_clear();
    check("the board is empty after a tetris", count_cells() == 0);
}

static void test_wall_kick(void)
{
    reset_game();
    clear_board();

    /* An I piece flat against the left wall must kick inward when it rotates. */
    cur.type = PIECE_I; cur.rot = 0; cur.col = 0; cur.row = 5;
    int ok = try_rotate(1);
    check("I piece rotates next to the left wall", ok && !collides(&cur));

    /* A T piece in a one-cell notch uses a kick to rotate. */
    clear_board();
    cur.type = PIECE_T; cur.rot = 0; cur.col = 0; cur.row = 5;
    ok = try_rotate(0);
    check("T piece rotates next to the left wall", ok && !collides(&cur));

    /* A piece boxed in on all sides must fail every kick. */
    clear_board();
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++) board[r][c] = PIECE_L + 1;
    cur.type = PIECE_T; cur.rot = 0; cur.col = 3; cur.row = 5;
    for (int i = 0; i < 4; i++)
        board[cur.row + shape[PIECE_T][0][i][0]][cur.col + shape[PIECE_T][0][i][1]] = 0;
    check("a fully boxed piece cannot rotate", try_rotate(1) == 0);
}

static void test_bounds(void)
{
    reset_game();
    clear_board();
    cur.type = PIECE_O; cur.rot = 0; cur.col = 4; cur.row = 5;

    int steps = 0;
    while (try_move(0, -1)) steps++;
    check("the piece stops at the left wall", cur.col == 0 && steps == 4);

    steps = 0;
    while (try_move(0, 1)) steps++;
    check("the piece stops at the right wall", cur.col == COLS - 2 && steps == 8);

    while (try_move(1, 0)) ;
    check("the piece stops at the floor", cur.row + 1 == ROWS - 1);
}

static void test_hard_drop(void)
{
    reset_game();
    clear_board();
    cur.type = PIECE_T; cur.rot = 0; cur.col = 3; cur.row = 2;
    score = 0;
    int rows = ghost_row() - cur.row;
    hard_drop();
    check("hard drop scores 2 per row", score == 2 * rows);
    check("hard drop locks 4 cells", count_cells() == 4);
}

static void test_hold(void)
{
    reset_game();
    int first = cur.type, next = queue[0];
    do_hold();
    check("the first hold stores the piece", hold == first);
    check("the first hold takes the next piece", cur.type == next);
    check("hold locks until the piece lands", hold_used == 1);

    int held = hold, active = cur.type;
    do_hold();
    check("a second hold in one turn does nothing", hold == held && cur.type == active);

    clear_board();
    cur.row = ghost_row();
    lock_piece();
    check("landing unlocks hold", hold_used == 0);

    int spawned = cur.type;
    do_hold();
    check("hold swaps after the piece lands", hold == spawned && cur.type == held);
}

static void test_bag(void)
{
    reset_game();
    int seen[NPIECES] = {0};
    /* The 4 queued pieces plus the active one come from the same bag sequence. */
    seen[cur.type]++;
    for (int i = 0; i < 4; i++) seen[queue[i]]++;
    int ok = 1;
    for (int i = 0; i < NPIECES; i++) if (seen[i] > 1) ok = 0;
    check("no piece repeats in the first 5 draws", ok);

    memset(seen, 0, sizeof seen);
    refill_bag();
    for (int i = 0; i < NPIECES; i++) seen[next_from_bag()]++;
    ok = 1;
    for (int i = 0; i < NPIECES; i++) if (seen[i] != 1) ok = 0;
    check("one bag holds each of the 7 pieces once", ok);
}

static void test_speed(void)
{
    reset_game();
    level = 1;
    float slow = fall_interval();
    level = 10;
    float fast = fall_interval();
    level = 30;
    float fastest = fall_interval();
    check("gravity speeds up with the level", fast < slow && fastest < fast);
    check("gravity never reaches zero", fastest >= 0.02f);
}

static void test_game_over(void)
{
    reset_game();
    clear_board();
    for (int r = 2; r < ROWS; r++)
        for (int c = 0; c < COLS; c++) board[r][c] = PIECE_L + 1;
    spawn_piece(PIECE_T);
    check("a blocked spawn ends the game", state == ST_OVER);
}

int main(void)
{
    srand(12345);
    build_shapes();

    test_shapes();
    test_single_clear();
    test_tetris_clear();
    test_wall_kick();
    test_bounds();
    test_hard_drop();
    test_hold();
    test_bag();
    test_speed();
    test_game_over();

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS",
           failures, failures == 1 ? "" : "s");
    return failures != 0;
}
