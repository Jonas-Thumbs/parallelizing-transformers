# Makefile outline taken from one used in an older class
   # I thought it looked nice, but we can simplify this if we want
   # so that it only contains the absolute neccessities

CC = gcc
CFLAGS = -O3 -std=c11 -Wall -Wextra

N = 1024
D = 64
BQ = 32
BK = 64

sequential_flash_attention: sequential_flash_attention.c
	$(CC) $(CFLAGS) sequential_flash_attention.c -o sequential_flash_attention -lm

run: sequential_flash_attention
	time ./sequential_flash_attention $(N) $(D) --bq $(BQ) --bk $(BK) --check

benchmark: sequential_flash_attention
	time ./sequential_flash_attention $(N) $(D) --bq $(BQ) --bk $(BK) --quiet

clean:
	rm -f sequential_flash_attention