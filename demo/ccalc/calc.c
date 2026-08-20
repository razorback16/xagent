/*
 * calc.c - a REPL calculator in C
 *
 * Build:  cc -O2 -Wall -Wextra -o calc calc.c -lm
 * Run:    ./calc
 *
 *   rad> 2 + 3 * 4
 *   = 14
 *
 * Features:
 *   - arithmetic with + - * / % ^ and parentheses
 *   - correct precedence, right-associative ^ (2^3^2 = 512)
 *   - functions:  sin cos tan  asin acos atan  sinh cosh tanh
 *                 sqrt cbrt ln log log10 log2 exp abs floor ceil round
 *                 pow(x,y) min(x,y) max(x,y) atan2(y,x) hypot(x,y)
 *   - constants:  pi  e  tau
 *   - angle modes: 'rad' (default) and 'deg'
 *   - REPL commands: help  clear  rad  deg  quit
 *
 * Implementation notes (speed):
 *   - evaluation is a single pass: the lexer reads input character by
 *     character with one token of lookahead, and implicit multiplication
 *     (2pi, 3(4+1), (2)5, 2sin(x)) is synthesized on the fly, so there is
 *     no token array to build, rescan, or memmove
 *   - function/constant lookup is a small hashed table, not an if-chain
 *   - the REPL prints its prompt and flushes only when stdout is a
 *     terminal, so redirected/batch output stays buffered
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

/* ---------------- tokens ---------------- */

enum tok {
    T_END = 0, T_NUM, T_IDENT,
    T_PLUS, T_MINUS, T_STAR, T_SLASH, T_PERCENT,
    T_CARET, T_LPAREN, T_RPAREN, T_COMMA
};

#define MAX_TOKENS 8192
#define MAX_PARSE_DEPTH 512

typedef struct {
    int     type;
    double  num;
    char    ident[48];
} Token;

static int deg_mode = 0;

static const char *tok_name(int t)
{
    switch (t) {
    case T_NUM:     return "number";
    case T_IDENT:   return "name";
    case T_PLUS:    return "'+'";
    case T_MINUS:   return "'-'";
    case T_STAR:    return "'*'";
    case T_SLASH:   return "'/'";
    case T_PERCENT: return "'%'";
    case T_CARET:   return "'^'";
    case T_LPAREN:  return "'('";
    case T_RPAREN:  return "')'";
    case T_COMMA:   return "','";
    default:        return "end of input";
    }
}

/* ---------------- functions & constants ---------------- */

struct fn {
    const char      *name;
    double          (*f1)(double);           /* 1-arg function   */
    double          (*f2)(double, double);   /* 2-arg function   */
    int             nargs;      /* -1: constant, 1 or 2: function */
    int             no_paren;   /* unary function usable without parens */
    int             angle;      /* -1: degree input, +1: degree output */
    double          val;        /* value, if constant */
};

static unsigned name_hash(const char *s)
{
    unsigned h = 5381;
    while (*s)
        h = ((h << 5) + h) + (unsigned char)*s++;
    return h;
}

/* common names first */
static const struct fn fns[] = {
    { "sin", sin, NULL, 1,  1, -1, 0.0 },
    { "cos", cos, NULL, 1,  1, -1, 0.0 },
    { "tan", tan, NULL, 1,  1, -1, 0.0 },
    { "sqrt", sqrt, NULL, 1,  1, 0, 0.0 },
    { "ln", log, NULL, 1,  1, 0, 0.0 },
    { "log", log10, NULL, 1,  1, 0, 0.0 },
    { "exp", exp, NULL, 1,  1, 0, 0.0 },
    { "abs", fabs, NULL, 1,  1, 0, 0.0 },
    { "pi", NULL, NULL, -1,  0, 0, M_PI },
    { "e", NULL, NULL, -1,  0, 0, M_E },
    { "tau", NULL, NULL, -1,  0, 0, 2.0 * M_PI },
    { "asin", asin, NULL, 1,  0, 1, 0.0 },
    { "acos", acos, NULL, 1,  0, 1, 0.0 },
    { "atan", atan, NULL, 1,  0, 1, 0.0 },
    { "atan2", NULL, atan2, 2,  0, 1, 0.0 },
    { "sinh", sinh, NULL, 1,  0, 0, 0.0 },
    { "cosh", cosh, NULL, 1,  0, 0, 0.0 },
    { "tanh", tanh, NULL, 1,  0, 0, 0.0 },
    { "cbrt", cbrt, NULL, 1,  0, 0, 0.0 },
    { "log10", log10, NULL, 1,  0, 0, 0.0 },
    { "log2", log2, NULL, 1,  0, 0, 0.0 },
    { "floor", floor, NULL, 1,  0, 0, 0.0 },
    { "ceil", ceil, NULL, 1,  0, 0, 0.0 },
    { "round", round, NULL, 1,  0, 0, 0.0 },
    { "pow", NULL, pow, 2,  0, 0, 0.0 },
    { "min", NULL, fmin, 2,  0, 0, 0.0 },
    { "max", NULL, fmax, 2,  0, 0, 0.0 },
    { "hypot", NULL, hypot, 2,  0, 0, 0.0 },
};
#define FN_COUNT ((int)(sizeof fns / sizeof fns[0]))

static unsigned fn_hash[FN_COUNT];
static int fn_hash_init = 0;

static const struct fn *find_fn(const char *name)
{
    if (!fn_hash_init) {
        for (int i = 0; i < FN_COUNT; i++)
            fn_hash[i] = name_hash(fns[i].name);
        fn_hash_init = 1;
    }
    unsigned h = name_hash(name);
    for (int i = 0; i < FN_COUNT; i++)
        if (fn_hash[i] == h && strcmp(fns[i].name, name) == 0)
            return &fns[i];
    return NULL;
}
/* ---------------- lexer: one token of lookahead, no token array ---------------- */

struct state {
    const char *s;
    size_t      i;
    int         ttype;     /* current token (lookahead) */
    double      tnum;
    char        tident[48];
    Token       pending;   /* operand held behind an implicit '*' */
    int         has_pending;
    int         raw_tokens;
    int         parse_depth;
    char        err[256];
};

static void set_err(struct state *st, const char *fmt, ...)
{
    va_list ap;

    if (st->err[0])
        return;
    va_start(ap, fmt);
    vsnprintf(st->err, sizeof st->err, fmt, ap);
    va_end(ap);
}

static void use_token(struct state *st, const Token *t)
{
    st->ttype = t->type;
    st->tnum = t->num;
    memcpy(st->tident, t->ident, sizeof st->tident);
}

static int needs_implicit_mul(int left, int right)
{
    int left_end = left == T_NUM || left == T_IDENT || left == T_RPAREN;
    int right_beg = right == T_NUM || right == T_IDENT || right == T_LPAREN;

    return left_end && right_beg && !(left == T_IDENT && right == T_LPAREN);
}

/*
 * Next token at st->s[st->i], synthesizing an implicit '*' where needed.
 * The implicit-multiplication rule is the same one the old lex() pass
 * applied afterwards:
 *     previous token in { NUMBER, IDENT, ')' }
 *     current token  in { NUMBER, IDENT, '(' }
 *     minus IDENT '(' which is a function call
 * When a product is needed, the raw token is retained in pending while '*'
 * is returned.  The following call returns that retained token without
 * rescanning or losing its value.
 */
static void next_tok(struct state *st)
{
    const char *s = st->s;
    size_t i = st->i;

    if (st->err[0]) {
        st->ttype = T_END;
        return;
    }
    if (st->has_pending) {
        use_token(st, &st->pending);
        st->has_pending = 0;
        return;
    }

    for (;;) {
        unsigned char c = (unsigned char)s[i];
        if (isspace(c)) { i++; continue; }
        if (c == '\0') { st->i = i; st->ttype = T_END; return; }

        Token next = { .type = T_END, .num = 0.0, .ident = { 0 } };
        if (isdigit(c) || (c == '.' && isdigit((unsigned char)s[i + 1]))) {
            char *end;
            double v = strtod(s + i, &end);
            if (end == s + i) {
                st->i = i;
                set_err(st, "malformed number near '%s'", s + i);
                st->ttype = T_END;
                return;
            }
            next.type = T_NUM;
            next.num = v;
            st->i = (size_t)(end - s);
        } else if (isalpha(c) || c == '_') {
            size_t n = 0;
            while (s[i] && (isalnum((unsigned char)s[i]) || s[i] == '_')) {
                if (n + 1 >= sizeof next.ident) {
                    st->i = i;
                    set_err(st, "name too long");
                    st->ttype = T_END;
                    return;
                }
                next.ident[n++] = s[i];
                i++;
            }
            next.ident[n] = '\0';
            next.type = T_IDENT;
            st->i = i;
        } else {
            switch (c) {
            case '+': next.type = T_PLUS;    break;
            case '-': next.type = T_MINUS;   break;
            case '*': next.type = T_STAR;    break;
            case '/': next.type = T_SLASH;   break;
            case '%': next.type = T_PERCENT; break;
            case '^': next.type = T_CARET;   break;
            case '(': next.type = T_LPAREN;  break;
            case ')': next.type = T_RPAREN;  break;
            case ',': next.type = T_COMMA;   break;
            default:
                st->i = i;
                set_err(st, "unexpected character '%c'", s[i]);
                st->ttype = T_END;
                return;
            }
            st->i = i + 1;
        }

        if (++st->raw_tokens > MAX_TOKENS) {
            set_err(st, "expression too long");
            st->ttype = T_END;
            return;
        }
        if (needs_implicit_mul(st->ttype, next.type)) {
            st->pending = next;
            st->has_pending = 1;
            st->ttype = T_STAR;
            st->tnum = 0.0;
            st->tident[0] = '\0';
        } else {
            use_token(st, &next);
        }
        return;
    }
}
/* ---------------- parser / evaluator ----------------
 *
 * expr    := term  (('+' | '-') term)*
 * term    := unary (('*' | '/' | '%') unary)*
 * unary   := ('-' | '+') unary | power
 * power   := primary ('^' unary)?          right-associative
 * primary := NUMBER | IDENT | IDENT '(' args ')' | '(' expr ')'
 */

static struct state st;

static int err_set(void) { return st.err[0] != '\0'; }

static double call_function(const char *name, double a, double b, int nargs)
{
    const struct fn *f = find_fn(name);
    if (!f) {
        set_err(&st, "unknown function '%s'", name);
        return 0.0;
    }
    if (f->nargs < 0) {
        set_err(&st, "'%s' is a constant, not a function", name);
        return 0.0;
    }
    if (f->nargs != nargs) {
        set_err(&st, "%s() expects %d argument%s", name, f->nargs,
                f->nargs == 1 ? "" : "s");
        return 0.0;
    }
    if (f->nargs == 1) {
        double x = a;
        if (deg_mode && f->angle < 0)
            x = x * M_PI / 180.0;
        double value = f->f1(x);
        if (deg_mode && f->angle > 0)
            value = value * 180.0 / M_PI;
        return value;
    }
    double value = f->f2(a, b);
    if (deg_mode && f->angle > 0)
        value = value * 180.0 / M_PI;
    return value;
}

static double parse_expr(void);
static double parse_unary(void);
static double parse_power(void);

static double parse_primary(void)
{
    int t = st.ttype;
    switch (t) {
    case T_NUM: {
        double value = st.tnum;
        next_tok(&st);
        return value;
    }

    case T_IDENT: {
        char name[48];
        strcpy(name, st.tident);
        next_tok(&st);

        if (st.ttype == T_LPAREN) {
            double a = 0.0, b = 0.0;
            int nargs = 0;

            next_tok(&st);
            if (st.ttype != T_RPAREN) {
                a = parse_expr();
                if (err_set()) return 0.0;
                nargs = 1;
            }
            if (st.ttype == T_COMMA && nargs == 1) {
                next_tok(&st);
                b = parse_expr();
                if (err_set()) return 0.0;
                nargs = 2;
            }
            if (st.ttype != T_RPAREN) {
                set_err(&st, "expected ')' after arguments of %s()", name);
                return 0.0;
            }
            next_tok(&st);
            return call_function(name, a, b, nargs);
        }

        const struct fn *f = find_fn(name);
        if (f) {
            if (f->nargs == -1)
                return f->val;
            if (f->no_paren) {
                /* An inserted '*' between "sin" and its bare argument is
                 * syntax, not multiplication at this point. */
                if (st.ttype == T_STAR)
                    next_tok(&st);
                double v = parse_unary();
                if (err_set()) return 0.0;
                return call_function(name, v, 0.0, 1);
            }
        }
        set_err(&st, "unknown name '%s' (constants: pi, e, tau)", name);
        return 0.0;
    }

    case T_LPAREN:
        next_tok(&st);
        {
            double v = parse_expr();
            if (err_set()) return 0.0;
            if (st.ttype != T_RPAREN) {
                set_err(&st, "expected ')' in subexpression");
                return 0.0;
            }
            next_tok(&st);
            return v;
        }

    case T_END:
        set_err(&st, "expression ends unexpectedly");
        return 0.0;

    default:
        set_err(&st, "unexpected %s", tok_name(t));
        return 0.0;
    }
}

static double parse_unary(void)
{
    double value;

    if (++st.parse_depth > MAX_PARSE_DEPTH) {
        --st.parse_depth;
        set_err(&st, "expression nesting is too deep");
        return 0.0;
    }
    if (st.ttype == T_MINUS) {
        next_tok(&st);
        value = err_set() ? 0.0 : -parse_unary();
    } else if (st.ttype == T_PLUS) {
        next_tok(&st);
        value = err_set() ? 0.0 : parse_unary();
    } else {
        value = parse_power();
    }
    --st.parse_depth;
    return value;
}

static double parse_power(void)
{
    double base = parse_primary();
    if (err_set()) return 0.0;
    if (st.ttype == T_CARET) {
        next_tok(&st);
        double expv = parse_unary();          /* right-associative, 2^-3 works */
        if (err_set()) return 0.0;
        return pow(base, expv);
    }
    return base;
}

static double parse_term(void)
{
    double v = parse_unary();
    if (err_set()) return 0.0;
    for (;;) {
        if (st.ttype == T_STAR) {
            next_tok(&st);
            double r = parse_unary();
            if (err_set()) return 0.0;
            v *= r;
        } else if (st.ttype == T_SLASH) {
            next_tok(&st);
            double r = parse_unary();
            if (err_set()) return 0.0;
            if (r == 0.0) {
                set_err(&st, "division by zero");
                return 0.0;
            }
            v /= r;
        } else if (st.ttype == T_PERCENT) {
            next_tok(&st);
            double r = parse_unary();
            if (err_set()) return 0.0;
            if (r == 0.0) {
                set_err(&st, "modulus by zero");
                return 0.0;
            }
            v = fmod(v, r);
        } else
            break;
    }
    return v;
}

static double parse_expr(void)
{
    double v = parse_term();
    if (err_set()) return 0.0;
    for (;;) {
        if (st.ttype == T_PLUS) {
            next_tok(&st);
            double r = parse_term();
            if (err_set()) return 0.0;
            v += r;
        } else if (st.ttype == T_MINUS) {
            next_tok(&st);
            double r = parse_term();
            if (err_set()) return 0.0;
            v -= r;
        } else
            break;
    }
    return v;
}

/* Parse and evaluate s.  Returns 0 on success, -1 on error (message in err). */
int eval(const char *s, double *out, char *out_err, size_t out_errsz)
{
    st.s = s;
    st.i = 0;
    st.ttype = T_END;   /* no previous token at the start */
    st.tnum = 0.0;
    st.tident[0] = '\0';
    st.has_pending = 0;
    st.raw_tokens = 0;
    st.parse_depth = 0;
    st.err[0] = '\0';
    next_tok(&st);
    if (err_set()) {
        snprintf(out_err, out_errsz, "%s", st.err);
        return -1;
    }
    if (st.ttype == T_END) {
        snprintf(out_err, out_errsz, "empty expression");
        return -1;
    }
    double result = parse_expr();
    if (err_set()) {
        snprintf(out_err, out_errsz, "%s", st.err);
        return -1;
    }
    if (st.ttype != T_END) {
        snprintf(out_err, out_errsz, "unexpected %s after expression",
                 tok_name(st.ttype));
        return -1;
    }
    if (isnan(result) || isinf(result)) {
        snprintf(out_err, out_errsz,
                 "result is %s", isnan(result) ? "not a number" : "infinite");
        return -1;
    }
    *out = result;
    return 0;
}
/* ---------------- help & REPL ---------------- */

static void print_help(void)
{
    printf(
"Operators    +  -  *  /  %%  ^   (and parentheses)\n"
"             implicit multiplication: 2pi, 3(4+1), 2sin(x)\n"
"             ^ is right-associative:  2^3^2 = 512\n"
"             %% is remainder (fmod), / and %% check for zero\n"
"Numbers      12  3.14  .5  1e-3\n"
"Functions    sin cos tan  asin acos atan  sinh cosh tanh\n"
"             sqrt cbrt ln log log10 log2 exp abs floor ceil round\n"
"             pow(x,y) min(x,y) max(x,y)  atan2(y,x) hypot(x,y)\n"
"             (sin cos tan sqrt ln log exp abs also work without parens:\n"
"              sin x)\n"
"Constants    pi  e  tau\n"
"Angle modes  'rad' (default) and 'deg' switch trig angles\n"
"Commands     help   clear   rad   deg   quit\n"
"");
}

static int command_is(const char *line, size_t n, const char *command)
{
    size_t command_len = strlen(command);

    if (n != command_len)
        return 0;
    for (size_t i = 0; i < n; i++)
        if (tolower((unsigned char)line[i]) != (unsigned char)command[i])
            return 0;
    return 1;
}

int main(void)
{
    int tty = isatty(fileno(stdout)) != 0;   /* interactive? */

    printf("calc - a small REPL calculator\n");
    printf("type 'help' for functions, 'quit' to exit\n\n");

    char *line = NULL;
    size_t capacity = 0;
    for (;;) {
        /*
         * The prompt is always printed, so output looks the same as before
         * in every mode.  The flush, though, only matters interactively:
         * with redirected stdout it is fully buffered, so skipping the
         * per-line fflush() removes ~2 syscalls from every line.
         */
        printf("%s> ", deg_mode ? "deg" : "rad");
        if (tty)
            fflush(stdout);

        ssize_t line_len = getline(&line, &capacity, stdin);
        if (line_len < 0)
            break;

        /* strip trailing newline */
        size_t n = (size_t)line_len;
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r'))
            line[--n] = '\0';

        /* skip blank lines */
        const char *ws = " \t\r\n";
        if (strspn(line, ws) == n)
            continue;

        /* REPL commands (checked case-insensitively, exact word) */
        if (command_is(line, n, "help")) {
            print_help();
            continue;
        }
        if (command_is(line, n, "clear")) {
            (void)!system("clear 2>/dev/null || cls 2>/dev/null");
            continue;
        }
        if (command_is(line, n, "rad")) {
            deg_mode = 0;
            printf("angles: radians\n");
            if (tty)
                fflush(stdout);
            continue;
        }
        if (command_is(line, n, "deg")) {
            deg_mode = 1;
            printf("angles: degrees\n");
            if (tty)
                fflush(stdout);
            continue;
        }
        if (command_is(line, n, "quit") || command_is(line, n, "exit"))
            break;

        /* otherwise: an expression */
        double v;
        char error[256];
        if (eval(line, &v, error, sizeof error)) {
            printf("error: %s\n", error);
        } else {
            /* print with as few digits as needed, fall back to 12 significant */
            char buf[64];
            snprintf(buf, sizeof buf, "%.12g", v);
            if (strlen(buf) > 17)
                snprintf(buf, sizeof buf, "%.15g", v);
            printf("= %s\n", buf);
        }
        if (tty)
            fflush(stdout);
    }
    free(line);
    printf("bye\n");
    return 0;
}
