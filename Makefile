NVCC ?= nvcc
NVCCFLAGS ?= -std=c++14

TARGET := edge_detection
SOURCE := src/edge_detection.cu
INPUT ?= input/test.pgm
OUTPUT ?= output/edges.pgm

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: build
	./$(TARGET) "$(INPUT)" "$(OUTPUT)"

clean:
	rm -f $(TARGET)
