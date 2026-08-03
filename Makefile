# Compiler
CC = gcc

# Compile flags: warnings, optimization, C99
CFLAGS = -Wall -Wextra -std=c99 -lm

# All source files
SRC = main.c tensor.c kv_cache.c quant.c gguf.c model.c

# Object files
OBJDIR = obj
OBJ = $(patsubst %.c,$(OBJDIR)/%.o,$(SRC))

# Output program name
TARGET = bin/chris_llama

.PHONY: all clean

# Build the final program
all: clean $(TARGET)

bin:
	mkdir -p bin

obj:
	mkdir -p obj

# Compile .c files into obj/*.o
$(OBJDIR)/%.o: %.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

# Link object files into executable
$(TARGET): $(OBJ) | bin
	$(CC) $(OBJ) -lm -o $@

# Clean build files
clean:
	rm -rf $(OBJDIR) bin