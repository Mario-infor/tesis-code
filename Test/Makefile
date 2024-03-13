INCLUDE_JETSON =./include
OBJS_DIR = objs
SRC_DIR = src
BUILD_DIR = build
LIB_DIR = lib

CPPFLAGS = -g -Wall -std=c++11 -I $(BNO055_DIR)/include -I $(INCLUDE_JETSON)
CC = g++

BNO055_DIR = $(HOME)/Mario/BNO055-BBB_IMU-Driver

JETSON_CFLAGS = `pkg-config --cflags opencv4`


JETSON_LIBS =  `pkg-config --libs opencv4` -lncurses \
				$(BNO055_DIR)/objs/BNO055-BBB_driver.o -lpthread

LIB = libMario.a

all: $(BUILD_DIR)/jetson.out $(BUILD_DIR)/readShow.out $(BUILD_DIR)/EKF.out

$(BUILD_DIR)/jetson.out: $(OBJS_DIR)/jetson.o $(LIB_DIR)/libMario.a
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -o $@ $(OBJS_DIR)/jetson.o -L ./lib -lMario $(JETSON_LIBS)

$(BUILD_DIR)/readShow.out: $(OBJS_DIR)/readShow.o
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -o $@ $(OBJS_DIR)/readShow.o $(JETSON_LIBS)

$(BUILD_DIR)/EKF.out: $(OBJS_DIR)/EKF.o $(LIB_DIR)/libMario.a
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -o $@ $(OBJS_DIR)/EKF.o -L ./lib -lMario $(JETSON_LIBS)

$(OBJS_DIR)/jetson.o: $(SRC_DIR)/jetson.cpp include/jetson.h include/cameraInfo.h include/readWriteData.h include/utils.h include/structsFile.h include/RingBuffer.h 
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -c -o $@ $(SRC_DIR)/jetson.cpp

$(OBJS_DIR)/readWriteData.o: $(SRC_DIR)/readWriteData.cpp include/readWriteData.h include/RingBuffer.h
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -c -o $@ $(SRC_DIR)/readWriteData.cpp

$(OBJS_DIR)/utils.o: $(SRC_DIR)/utils.cpp include/utils.h include/structsFile.h
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -c -o $@ $(SRC_DIR)/utils.cpp

$(OBJS_DIR)/interpolationUtils.o: $(SRC_DIR)/interpolationUtils.cpp include/interpolationUtils.h include/utils.h include/structsFile.h
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -c -o $@ $(SRC_DIR)/interpolationUtils.cpp

$(OBJS_DIR)/readShow.o: $(SRC_DIR)/readShow.cpp include/readShow.h include/cameraInfo.h
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -c -o $@ $(SRC_DIR)/readShow.cpp

$(OBJS_DIR)/EKF.o: $(SRC_DIR)/EKF.cpp include/EKF.h include/utils.h include/cameraInfo.h include/readWriteData.h
	$(CC) $(CPPFLAGS) $(JETSON_CFLAGS) -c -o $@ $(SRC_DIR)/EKF.cpp

lib/libMario.a: $(OBJS_DIR)/readWriteData.o $(OBJS_DIR)/utils.o $(OBJS_DIR)/interpolationUtils.o
	ar rcs $@ $^	
	ranlib $@

.PHONY: clean
clean:
	rm -f $(OBJS_DIR)/*.o $(LIB_DIR)/*.a $(BUILD_DIR)/jetson $(BUILD_DIR)/EKF $(BUILD_DIR)/readShow