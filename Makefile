CC = gcc
CFLAGS = -Wall -Wconversion -Werror -Wextra -O3

LIBS = -lm
EXEC = cg_solver
TEST = test
OBJS = linalg.o mtxio.o
INCS = linalg.h mtxio.h

all: $(EXEC) $(TEST)

debug: CFLAGS += -DDEBUG -g
debug: all

profile: PROFILE += -pg
profile: $(EXEC)

$(EXEC): main.o $(OBJS)
	$(CC) -o $@ $^ $(LIBS) $(PROFILE)

$(TEST): unittests.o $(OBJS)
	$(CC) -o $@ $^ $(LIBS)

%.o: %.c $(INCS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean
clean:
	rm -f main.o unittests.o $(OBJS) $(EXEC) $(TEST) *.~ *.mtx *.dSYM *.out
