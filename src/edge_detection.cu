#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <dirent.h>
#endif

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
		if (std::isspace(static_cast<unsigned char>(character))) continue;
		if (character == '#') {
			input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			continue;
		}
		token.push_back(character);
		break;
	}
	if (token.empty()) return false;
	while (input.get(character)) {
		if (std::isspace(static_cast<unsigned char>(character))) {
			if (character == '\r' && input.peek() == '\n') input.get();
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
			parsed_value > std::numeric_limits<int>::max()) return false;
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
		pixel_count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
		std::cerr << "Error: image dimensions are too large.\n";
		return false;
	}
	pixels.resize(pixel_count);
	input.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixel_count));
	if (input.gcount() != static_cast<std::streamsize>(pixel_count)) {
		std::cerr << "Error: input PGM pixel data is incomplete.\n";
		return false;
	}
	if (max_value != 255) {
		for (unsigned char& pixel : pixels) {
			pixel = static_cast<unsigned char>((static_cast<unsigned int>(pixel) * 255U) /
									static_cast<unsigned int>(max_value));
		}
	}
	return true;
}

__global__ void sobel_kernel(const unsigned char* input, unsigned char* output,
						 int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
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

bool has_pgm_extension(const std::string& filename) {
	if (filename.size() < 4) return false;
	const std::size_t start = filename.size() - 4;
	return filename[start] == '.' &&
		std::tolower(static_cast<unsigned char>(filename[start + 1])) == 'p' &&
		std::tolower(static_cast<unsigned char>(filename[start + 2])) == 'g' &&
		std::tolower(static_cast<unsigned char>(filename[start + 3])) == 'm';
}

std::string join_path(const std::string& directory, const std::string& filename) {
	if (directory.empty() || directory.back() == '/' || directory.back() == '\\') {
		return directory + filename;
	}
	return directory + '/' + filename;
}

bool find_pgm_files(const std::string& input_directory, int count,
					std::vector<std::string>& filenames) {
	filenames.clear();
#ifdef _WIN32
	WIN32_FIND_DATAA find_data;
	const std::string pattern = join_path(input_directory, "*");
	HANDLE handle = FindFirstFileA(pattern.c_str(), &find_data);
	if (handle == INVALID_HANDLE_VALUE) {
		std::cerr << "Error: could not open input directory '" << input_directory << "'.\n";
		return false;
	}
	do {
		const std::string filename(find_data.cFileName);
		if ((find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 && has_pgm_extension(filename)) {
			filenames.push_back(filename);
		}
	} while (FindNextFileA(handle, &find_data));
	FindClose(handle);
#else
	DIR* directory = opendir(input_directory.c_str());
	if (directory == nullptr) {
		std::cerr << "Error: could not open input directory '" << input_directory
				  << "': " << std::strerror(errno) << ".\n";
		return false;
	}
	for (dirent* entry = readdir(directory); entry != nullptr; entry = readdir(directory)) {
		const std::string filename(entry->d_name);
		if (filename != "." && filename != ".." && has_pgm_extension(filename)) {
			filenames.push_back(filename);
		}
	}
	closedir(directory);
#endif
	std::sort(filenames.begin(), filenames.end());
	if (filenames.size() > static_cast<std::size_t>(count)) {
		filenames.resize(static_cast<std::size_t>(count));
	}
	if (filenames.empty()) {
		std::cerr << "Error: no PGM files found in input directory '"
				  << input_directory << "'.\n";
		return false;
	}
	return true;
}

bool process_image(const std::string& input_filename, const std::string& output_filename,
				   int& width, int& height, float& kernel_time_ms) {
	std::vector<unsigned char> host_input;
	if (!load_pgm(input_filename, width, height, host_input)) return false;
	std::vector<unsigned char> host_output(host_input.size());
	const std::size_t image_bytes = host_input.size() * sizeof(unsigned char);
	unsigned char* device_input = nullptr;
	unsigned char* device_output = nullptr;
	CUDA_CHECK(cudaMalloc(&device_input, image_bytes));
	CUDA_CHECK(cudaMalloc(&device_output, image_bytes));
	CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), image_bytes, cudaMemcpyHostToDevice));
	const dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT);
	const dim3 grid_size((width - 1) / BLOCK_WIDTH + 1, (height - 1) / BLOCK_HEIGHT + 1);
	cudaEvent_t start_event;
	cudaEvent_t stop_event;
	CUDA_CHECK(cudaEventCreate(&start_event));
	CUDA_CHECK(cudaEventCreate(&stop_event));
	CUDA_CHECK(cudaEventRecord(start_event));
	sobel_kernel<<<grid_size, block_size>>>(device_input, device_output, width, height);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop_event));
	CUDA_CHECK(cudaEventSynchronize(stop_event));
	CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event));
	CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, image_bytes, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaEventDestroy(start_event));
	CUDA_CHECK(cudaEventDestroy(stop_event));
	CUDA_CHECK(cudaFree(device_input));
	CUDA_CHECK(cudaFree(device_output));
	return save_pgm(output_filename, width, height, host_output);
}

void print_summary(int images_processed, int first_width, int first_height,
				   float total_kernel_time_ms, const std::string& output_directory) {
	std::cout << "Images processed: " << images_processed << '\n'
			  << "Image dimensions: " << first_width << " x " << first_height << '\n'
			  << "Block size: " << BLOCK_WIDTH << 'x' << BLOCK_HEIGHT << '\n'
			  << "Total GPU kernel time: " << total_kernel_time_ms << " ms\n"
			  << "Average GPU kernel time per image: "
			  << total_kernel_time_ms / static_cast<float>(images_processed) << " ms\n"
			  << "Output directory: " << output_directory << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
	if (argc == 3) {
		int width = 0;
		int height = 0;
		float kernel_time_ms = 0.0F;
		if (!process_image(argv[1], argv[2], width, height, kernel_time_ms)) return EXIT_FAILURE;
		print_summary(1, width, height, kernel_time_ms, argv[2]);
		return EXIT_SUCCESS;
	}
	if (argc != 7 || std::string(argv[1]) != "--input" ||
		std::string(argv[3]) != "--output" || std::string(argv[5]) != "--count") {
		std::cerr << "Usage: " << argv[0] << " <input.pgm> <output.pgm>\n"
				  << "   or: " << argv[0]
				  << " --input <input_directory> --output <output_directory> --count <N>\n";
		return EXIT_FAILURE;
	}
	int requested_count = 0;
	if (!parse_positive_integer(argv[6], requested_count)) {
		std::cerr << "Error: --count must be a positive integer.\n";
		return EXIT_FAILURE;
	}
	const std::string input_directory(argv[2]);
	const std::string output_directory(argv[4]);
	std::vector<std::string> filenames;
	if (!find_pgm_files(input_directory, requested_count, filenames)) return EXIT_FAILURE;
	float total_kernel_time_ms = 0.0F;
	int first_width = 0;
	int first_height = 0;
	for (std::size_t index = 0; index < filenames.size(); ++index) {
		int width = 0;
		int height = 0;
		float kernel_time_ms = 0.0F;
		if (!process_image(join_path(input_directory, filenames[index]),
						   join_path(output_directory, filenames[index]), width, height,
						   kernel_time_ms)) return EXIT_FAILURE;
		if (index == 0) {
			first_width = width;
			first_height = height;
		}
		total_kernel_time_ms += kernel_time_ms;
	}
	print_summary(static_cast<int>(filenames.size()), first_width, first_height,
				  total_kernel_time_ms, output_directory);
	return EXIT_SUCCESS;
}
