CC = g++ -g -Wall
PROJECT = new_output
SRC = new.cpp
LIBS = `pkg-config --cflags --libs opencv4` -lncurses
$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS)