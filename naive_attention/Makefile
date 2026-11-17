naive_attention: naive_attention.c
	gcc -O3 naive_attention.c -o naive_attention -lm


CC = gcc
CFLAGS = -O3 -std=c11 -Wall -Wextra

N = 2048#1024
D = 256#64

naive_attention: naive_attention.c
	$(CC) $(CFLAGS) naive_attention.c -o naive_attention -lm

run: naive_attention
	time ./naive_attention $(N) $(D) 

benchmark: naive_attention
	time ./naive_attention $(N) $(D) --quiet

clean:
	rm -f naive_attention