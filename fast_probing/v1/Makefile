# GoldRush 2.0 fast_probing v1 C++ strategy

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -march=native -fPIC -Wall
TARGET   = player.so
SRC      = player.cpp
SMOKE    = smoke_test

all: $(TARGET)

$(TARGET): $(SRC) ../../official_sdk/code/game_api.h
	$(CXX) $(CXXFLAGS) -shared -o $(TARGET) $(SRC)

smoke: $(TARGET) $(SMOKE)
	./$(SMOKE)

$(SMOKE): smoke_test.cpp ../../official_sdk/code/game_api.h
	$(CXX) -std=c++17 -O2 -Wall -o $(SMOKE) smoke_test.cpp -ldl

clean:
	rm -f $(TARGET) $(SMOKE)

.PHONY: all smoke clean
