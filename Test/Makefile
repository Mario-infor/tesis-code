CPPFLAGS = -g -Wall -DINTEL

CC = g++ 
PROJECT = new_output
SRC = new.cpp
LIBS = `pkg-config --cflags --libs opencv4` -lncurses
$(PROJECT) : $(SRC)
	$(CC) $(CPPFLAGS) $(SRC) -o $(PROJECT) $(LIBS)

clean:
	rm new_output
