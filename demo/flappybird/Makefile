# Flappy Bird - C + GLUT + SDL2_mixer + Freetype
CC      ?= gcc
CFLAGS  ?= -O2 -Wall
PKGS    := freetype2 sdl2 SDL2_mixer

# Homebrew keeps freeglut and Mesa outside the default search path, so ask
# pkg-config first. Debian ships no glut.pc, hence the plain -l fallback.
pc = $(shell pkg-config --exists $(1) && pkg-config --$(2) $(1) || echo $(3))

CFLAGS  += $(shell pkg-config --cflags $(PKGS))
LDLIBS  := $(shell pkg-config --libs $(PKGS)) -lm

# The game window. Homebrew freeglut is an X11 build and needs XQuartz, so on
# macOS link Apple GLUT instead. Apple deprecated it, hence the warning switch.
GLUT_CFLAGS := $(call pc,glut,cflags,) $(call pc,gl,cflags,)
GLUT_LIBS   := $(call pc,glut,libs,-lglut) $(call pc,gl,libs,-lGL)
ifeq ($(shell uname -s),Darwin)
GLUT_CFLAGS := -Wno-deprecated-declarations
GLUT_LIBS   := -framework GLUT -framework OpenGL
endif

# The headless test renders through Mesa on both platforms, never Apple GL.
EGL_CFLAGS := $(call pc,glut,cflags,) $(call pc,gl,cflags,) $(call pc,egl,cflags,)
EGL_LIBS   := $(call pc,glut,libs,-lglut) $(call pc,gl,libs,-lGL) $(call pc,egl,libs,-lEGL)

all: flappy

flappy: flappy.c
	$(CC) $(CFLAGS) $(GLUT_CFLAGS) -o $@ flappy.c $(GLUT_LIBS) $(LDLIBS)

run: flappy
	./flappy

clean:
	rm -f flappy test_flappy

test: test_flappy
	./test_flappy

test_flappy: test_flappy.c flappy.c
	$(CC) $(CFLAGS) $(EGL_CFLAGS) -I. -o $@ test_flappy.c $(EGL_LIBS) $(LDLIBS)

.PHONY: all run clean test
