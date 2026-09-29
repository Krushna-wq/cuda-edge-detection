# CUDA-Accelerated Image Edge Detection

This project applies Sobel edge detection to 8-bit binary PGM (`P5`) images with CUDA. It supports both one-image processing and sequential batch processing, making it suitable for a GPU programming capstone demonstration.

The implementation was tested on an NVIDIA Tesla T4 with 100 synthetic 256x256 PGM images.

## CUDA Sobel kernel

Each CUDA thread computes one output pixel. The kernel reads the surrounding 3x3 neighborhood, applies the horizontal and vertical Sobel operators, calculates the gradient magnitude, and clamps it to the 0-255 range. Boundary pixels are set to zero.

The kernel uses a fixed 16x16 CUDA block configuration. CUDA error checking is retained for runtime calls and launches, and CUDA events measure GPU kernel time. Batch processing is sequential; it does not use CPU multithreading.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit and `nvcc`
- GNU Make
- Bash (for `run.sh`)

## Build

From the project root:

```bash
make build
```

This produces the `edge_detection` executable.

## Single-image processing

Process one P5 PGM image:

```bash
./edge_detection <input.pgm> <output.pgm>
```

Example:

```bash
mkdir -p output
./edge_detection input/test.pgm output/edges.pgm
```

The original single-image command remains supported.

## Batch processing

Batch mode finds top-level `.pgm` files in the input directory, sorts the filenames, and processes the first `N` files. Each edge-detected image is written to the output directory with the same filename.

```bash
./edge_detection --input <input_directory> --output <output_directory> --count <N>
```

Example for 100 images:

```bash
mkdir -p output/batch
./edge_detection --input input --output output/batch --count 100
```

The program reports the number of images processed, image dimensions, 16x16 block size, total GPU kernel time, average GPU kernel time per image, and output directory.

## Using `run.sh`

The script builds the project and then runs the executable. With no arguments, it runs the default single-image example:

```bash
./run.sh
```

Pass the normal program arguments for a specific run:

```bash
./run.sh input/test.pgm output/edges.pgm
./run.sh --input input --output output/batch --count 100
```

## Google Colab / Tesla T4

1. In Colab, select **Runtime > Change runtime type > T4 GPU**.
2. Clone the repository and enter it:

   ```bash
   !git clone https://github.com/Krushna-wq/cuda-edge-detection.git
   %cd cuda-edge-detection
   ```

3. Confirm that CUDA sees the Tesla T4:

   ```bash
   !nvidia-smi
   ```

4. Build the CUDA executable:

   ```bash
   !make build
   ```

5. Place P5 PGM images in `input/`, create an output directory, and run batch mode:

   ```bash
   !mkdir -p output/batch
   !./edge_detection --input input --output output/batch --count 100
   ```

For an existing set of 100 synthetic 256x256 PGM images, the last command reproduces the tested batch-processing scenario on a Tesla T4.
