CC = g++ -g -Wall 
PROJECT = new_output
SRC = new.cpp
LIBS = `pkg-config --cflags --libs opencv4` -lncurses

TARGETS = calibrate_camera_charuco calibrate_camera		\
	create_board_charuco create_board create_diamond	\
	create_marker detect_board_charuco detect_board		\
	detect_diamonds detect_markers detect_and_print_markers	\
	detector_params.yml detect_and_send_markers

$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS)