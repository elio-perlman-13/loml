CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -I/opt/conda/include
TARGET   := wta_solver

$(TARGET): main.cpp heuristic.hpp solution.hpp wtv.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: clean
