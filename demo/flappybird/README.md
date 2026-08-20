# Flappy Bird — C + OpenGL (freeglut)

A complete Flappy Bird game written in portable C using the classic GLUT/OpenGL
stack. All graphics are procedural (no image assets) and all sounds are
synthesized at startup as WAV data and fed to SDL_mixer (no audio assets).

## Features

- Classic flappy gameplay: pipes, gaps, scoring, game-over screen
- Procedurally drawn bird (flapping wing, rotation on dive), parallax clouds,
  scrolling ground, sun, dust/feather particles
- Freetype-rendered text (DejaVu Sans) in a single texture atlas
- SDL2_mixer sound effects: flap, score, hit, death, swoosh
- Best score persistence (`.flappy_best` in the working directory)
- 60 FPS frame cap (sane CPU use even without vsync)

## Build

Dependencies (Ubuntu/Debian):

    sudo apt install build-essential freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev \
                     libsdl2-dev libsdl2-mixer-dev libfreetype-dev

Then:

    make          # builds ./flappy

The Makefile uses pkg-config to find freetype2, sdl2 and SDL2_mixer.

## Run

    ./flappy

| Input                  | Action                          |
|------------------------|---------------------------------|
| SPACE / UP / W / click | flap (starts game from "Get Ready") |
| P                      | pause / resume                  |
| ESC or Q               | quit                            |

On the Game Over screen: SPACE or clicking "Play Again" restarts.

### Options

    ./flappy --noaudio         # force silent (dummy audio driver)
    ./flappy --font /path.ttf  # use a different TrueType font
    FLAPPY_DUMP=120 ./flappy   # debug: dump frame 120 to /tmp/flappy_dump.ppm and exit
    d                          # debug key: dump the current frame to /tmp/flappy_frame_1.ppm

## Tests

    make test

`test_flappy.c` runs the real game logic and rendering headless on a surfaceless
EGL context into an FBO (no display needed) and checks the state machine:
ready→play→death transitions, flapping physics, pipe spawning, scoring,
collision, banner timing and restart.

## Layout

- `flappy.c`      — the entire game (world constants, audio synthesis,
                    Freetype text atlas, procedural drawing, physics,
                    collision, state machine, GLUT callbacks)
- `test_flappy.c` — headless EGL smoke test (includes flappy.c)
- `Makefile`      — build + test targets
