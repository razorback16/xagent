# Tetris — C + OpenGL

A complete Tetris game written in C using the legacy OpenGL fixed-function
pipeline (with GLUT bitmap fonts for text), driven by an AppKit event loop.

## Build (macOS)

    cc -x objective-c -o tetris tetris.c \
       -framework AppKit -framework OpenGL -framework GLUT

## Run

    ./tetris

## Controls

| Key              | Action                          |
|------------------|---------------------------------|
| Left / Right     | Move piece                      |
| Down             | Soft drop (hold to fall faster) |
| X or A           | Rotate clockwise                |
| Z or right-click | Rotate counter-clockwise        |
| Left-click       | Rotate clockwise (drag)         |
| Space            | Hard drop                       |
| P                | Pause / resume                  |
| R                | Restart                         |

## Features

- 10x20 board, all 7 standard tetrominoes (SRS-style rotations with wall kicks)
- Ghost piece showing landing position
- NEXT-piece preview
- Line clearing with classic scoring (n² × 100), 10 lines = +1 level
- Gravity speeds up with level (0.8 s/row down to 0.07 s/row)
- Hard-drop scoring (1 point per row)
- Pause, game over, and restart
- 3D-beveled blocks via simple per-cell shading
- 120 Hz game tick via NSTimer

## Implementation notes

- The window is a standard `NSWindow` whose content view is `NSOpenGLView`,
  subclassed at runtime in C with `objc_allocateClassPair` + `class_addMethod`
  (no Objective-C class files needed) to hook `keyDown:`/`keyUp:` and
  `drawRect:`.
- The OpenGL context is **double-buffered**. On Apple silicon the Metal-backed
  "2.1" compatibility context does not reliably present double-buffered
  `glBegin`/`glVertex` geometry, so single buffering is used instead.
- A `TETRIS_CAPTURE=/path/frame.ppm` env var (optionally with
  `TETRIS_CAPTURE_AT=<tick>`) makes the game dump the framebuffer after N
  ticks and exit — used for headless verification.
- `NSOpenGLView`'s default `drawRect:` is invoked first (via the superclass
  IMP looked up with the ObjC runtime) to set the context current and bind the
  view, then the game renders and flushes.
