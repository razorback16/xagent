CC      = cc
CFLAGS  = -O2 -std=c99 -Wall -Wextra
TARGET  = tetris

UNAME  := $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS = -framework OpenGL -framework GLUT
else
LDFLAGS = -lglut -lGL -lGLU -lm
endif

$(TARGET): tetris.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

test: test_logic.c tetris.c
	$(CC) $(CFLAGS) -o test_logic test_logic.c $(LDFLAGS)
	./test_logic

clean:
	rm -f $(TARGET) test_logic

.PHONY: run test clean
