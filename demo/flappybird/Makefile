# Flappy Bird - C + freeglut + SDL2_mixer + Freetype
CC      ?= gcc
CFLAGS  ?= -O2 -Wall
PKGS    := freetype2 sdl2 SDL2_mixer
CFLAGS  += $(shell pkg-config --cflags $(PKGS))
LDLIBS  := $(shell pkg-config --libs $(PKGS)) -lGL -lGLU -lglut -lm

all: flappy

flappy: flappy.c
	$(CC) $(CFLAGS) -o $@ flappy.c $(LDLIBS)

run: flappy
	./flappy

clean:
	rm -f flappy

test: test_flappy
	./test_flappy

test_flappy: test_flappy.c flappy.c
	$(CC) $(CFLAGS) -I. -o $@ test_flappy.c $(LDLIBS) $(shell pkg-config --libs egl)

.PHONY: all run clean test test_flappy
