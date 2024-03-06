INCLUDE_JETSON =./include

CPPFLAGS = -g -Wall -std=c++11 -I $(BNO055_DIR)/include -I $(INCLUDE_JETSON)
CC = g++

BNO055_DIR = $(HOME)/Mario/BNO055-BBB_IMU-Driver

JETSON_CFLAGS = `pkg-config --cflags opencv4`


JETSON_LIBS =  `pkg-config --libs opencv4` -lncurses \
				$(BNO055_DIR)/objs/BNO055-BBB_driver.o -lpthread

LIB = libMario.a


all: build/jetson


build/jetson: objs/jetson.o lib/libMario.a
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -o $@ objs/jetson.o -L ./lib -lMario  $(JETSON_LIBS)

objs/jetson.o: src/jetson.cpp include/jetson.h include/readWriteData.h include/utils.h include/structsFile.h include/RingBuffer.h 
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -c -o $@ src/jetson.cpp  

objs/readWriteData.o: src/readWriteData.cpp include/readWriteData.h include/RingBuffer.h
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -c -o $@ src/readWriteData.cpp

objs/utils.o: src/utils.cpp include/utils.h include/structsFile.h
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -c -o $@ src/utils.cpp

lib/libMario.a: objs/readWriteData.o objs/utils.o 
	ar rcs $@ $^	
	ranlib $@


.PHONY: clean
clean:
	rm -f objs/*.o build/jetson build/EKF build/readShow