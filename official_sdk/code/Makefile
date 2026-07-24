# GoldRush 2.0 C++ 选手 Makefile
# make        -> 编译出 player.so
# make clean  -> 清理产物

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -march=native -fPIC -Wall
TARGET   = player.so
SRC      = player.cpp

all: $(TARGET)

$(TARGET): $(SRC) game_api.h
	$(CXX) $(CXXFLAGS) -shared -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean