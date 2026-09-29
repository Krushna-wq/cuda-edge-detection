#include <cuda_runtime.h>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int BLOCK_WIDTH = 16;
constexpr int BLOCK_HEIGHT = 16;

void check_cuda(cudaError_t error, const char* operation) {
	if (error != cudaSuccess) {
		std::cerr << "CUDA error during " << operation << ": "
				  << cudaGetErrorString(error) << '\n';
		std::exit(EXIT_FAILURE);
	}
}

#define CUDA_CHECK(operation) check_cuda((operation), #operation)

bool read_pgm_token(std::ifstream& input, std::string& token) {
	token.clear();
	char character;

	while (input.get(character)) {
		if (std::isspace(static_cast<unsigned char>(character))) {
			continue;
		}
		if (character == '#') {
			input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			continue;
		}
		token.push_back(character);
		break;
	}

	if (token.empty()) {
		return false;
	}

	while (input.get(character)) {
		if (std::isspace(static_cast<unsigned char>(character))) {
			if (character == '\r' && input.peek() == '\n') {
				input.get();
			}
			return true;
		}
		token.push_back(character);
	}
	return true;
}

bool parse_positive_integer(const std::string& token, int& value) {
	try {
		std::size_t parsed = 0;
		const long parsed_value = std::stol(token, &parsed);
		if (parsed != token.size() || parsed_value <= 0 ||
			parsed_value > std::numeric_limits<int>::max()) {
			return false;
		}
		value = static_cast<int>(parsed_value);
		return true;
	} catch (...) {
		return false;
	}
}

bool load_pgm(const std::string& filename, int& width, int& height,
			  std::vector<unsigned char>& pixels) {
	std::ifstream input(filename, std::ios::binary);
	if (!input) {
		std::cerr << "Error: could not open input file '" << filename << "'.\n";
		return false;
	}

	std::string token;
	int max_value = 0;
	if (!read_pgm_token(input, token) || token != "P5" ||
		!read_pgm_token(input, token) || !parse_positive_integer(token, width) ||
		!read_pgm_token(input, token) || !parse_positive_integer(token, height) ||
		!read_pgm_token(input, token) || !parse_positive_integer(token, max_value) ||
		max_value > 255) {
		std::cerr << "Error: expected an 8-bit binary PGM (P5) image.\n";
		return false;
	}

	const std::size_t image_width = static_cast<std::size_t>(width);
	const std::size_t image_height = static_cast<std::size_t>(height);
	if (image_width > std::numeric_limits<std::size_t>::max() / image_height) {
		std::cerr << "Error: image dimensions are too large.\n";
		return false;
	}

	const std::size_t pixel_count = image_width * image_height;
	if (pixel_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
		pixel_count > static_cast<std::size_t>(
						  std::numeric_limits<std::streamsize>::max())) {
		std::cerr << "Error: image dimensions are too large.\n";
		return false;
	}
	pixels.resize(pixel_count);
	input.read(reinterpret_cast<char*>(pixels.data()),
			   static_cast<std::streamsize>(pixel_count));
	if (input.gcount() != static_cast<std::streamsize>(pixel_count)) {
		std::cerr << "Error: input PGM pixel data is incomplete.\n";
		return false;
	}

	if (max_value != 255) {
		for (unsigned char& pixel : pixels) {
			pixel = static_cast<unsigned char>(
				(static_cast<unsigned int>(pixel) * 255U) /
				static_cast<unsigned int>(max_value));
		}
	}
	return true;
}

__global__ void sobel_kernel(const unsigned char* input, unsigned char* output,
							 int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) {
		return;
	}

	const int index = y * width + x;
	if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
		output[index] = 0;
		return;
	}

	const int top_left = input[(y - 1) * width + (x - 1)];
	const int top = input[(y - 1) * width + x];
	const int top_right = input[(y - 1) * width + (x + 1)];
	const int left = input[y * width + (x - 1)];
	const int right = input[y * width + (x + 1)];
	const int bottom_left = input[(y + 1) * width + (x - 1)];
	const int bottom = input[(y + 1) * width + x];
	const int bottom_right = input[(y + 1) * width + (x + 1)];

	const int gradient_x = -top_left + top_right - 2 * left + 2 * right -
						   bottom_left + bottom_right;
	const int gradient_y = -top_left - 2 * top - top_right + bottom_left +
						   2 * bottom + bottom_right;
	const int magnitude = static_cast<int>(sqrtf(
		static_cast<float>(gradient_x * gradient_x + gradient_y * gradient_y)));
	output[index] = static_cast<unsigned char>(magnitude > 255 ? 255 : magnitude);
}

bool save_pgm(const std::string& filename, int width, int height,
			  const std::vector<unsigned char>& pixels) {
	std::ofstream output(filename, std::ios::binary);
	if (!output) {
		std::cerr << "Error: could not open output file '" << filename << "'.\n";
		return false;
	}

	output << "P5\n" << width << ' ' << height << "\n255\n";
	output.write(reinterpret_cast<const char*>(pixels.data()),
				 static_cast<std::streamsize>(pixels.size()));
	if (!output) {
		std::cerr << "Error: could not write output file '" << filename << "'.\n";
		return false;
	}
	return true;
}

}  // namespace

int main(int argc, char* argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <input.pgm> <output.pgm>\n";
		return EXIT_FAILURE;
	}

	int width = 0;
	int height = 0;
	std::vector<unsigned char> host_input;
	if (!load_pgm(argv[1], width, height, host_input)) {
		return EXIT_FAILURE;
	}
	std::vector<unsigned char> host_output(host_input.size());

	const std::size_t image_bytes = host_input.size() * sizeof(unsigned char);
	unsigned char* device_input = nullptr;
	unsigned char* device_output = nullptr;
	CUDA_CHECK(cudaMalloc(&device_input, image_bytes));
	CUDA_CHECK(cudaMalloc(&device_output, image_bytes));
	CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), image_bytes,
						  cudaMemcpyHostToDevice));

	const dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT);
	const dim3 grid_size((width - 1) / BLOCK_WIDTH + 1,
						 (height - 1) / BLOCK_HEIGHT + 1);
	cudaEvent_t start_event;
	cudaEvent_t stop_event;
	CUDA_CHECK(cudaEventCreate(&start_event));
	CUDA_CHECK(cudaEventCreate(&stop_event));
	CUDA_CHECK(cudaEventRecord(start_event));
	sobel_kernel<<<grid_size, block_size>>>(device_input, device_output, width, height);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop_event));
	CUDA_CHECK(cudaEventSynchronize(stop_event));

	float kernel_time_ms = 0.0F;
	CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event));
	CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, image_bytes,
						  cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaEventDestroy(start_event));
	CUDA_CHECK(cudaEventDestroy(stop_event));
	CUDA_CHECK(cudaFree(device_input));
	CUDA_CHECK(cudaFree(device_output));

	if (!save_pgm(argv[2], width, height, host_output)) {
		return EXIT_FAILURE;
	}

	std::cout << "Image dimensions: " << width << " x " << height << '\n'
			  << "Block size: " << BLOCK_WIDTH << ' x ' << BLOCK_HEIGHT << '\n'
			  << "GPU kernel execution time: " << kernel_time_ms << " ms\n"
			  << "Output file: " << argv[2] << '\n';
	return EXIT_SUCCESS;
}
