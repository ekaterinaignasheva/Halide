#include <chrono>
#include <cstdio>
#include <iostream>

#include "conv_layer.h"

#include "HalideBuffer.h"
#include "halide_benchmark.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    const int H = 64, W = 64;
    const int src_channels = 1024;
    const int dst_channels = 1024;

    const int T = 4, SD = src_channels / 4, DD = dst_channels / 4;
    const int window = 1;

    Buffer<float, 4> input(T, H, W, SD);
    Buffer<float, 6> filter(4, 4, SD, window, window, DD);
    Buffer<float, 2> bias(T, DD);
    input.fill(1);
    filter.fill(1);
    bias.fill(1);

    Buffer<float, 4> output(T, H, W, DD);

    for (int i = 0; i < 20; i++) {
      conv_layer(input, filter, bias, output);
    }
    // Manually-tuned version
    double min_t_manual = benchmark(10, 100, [&]() {
        conv_layer(input, filter, bias, output);
        output.device_sync();
    });
    printf("Latency: %gms\n", min_t_manual * 1e3);

    const int iterations = 10;
    const int iteration_size = 100;
    for (int i = 0; i < iterations; i++) {
      const auto start = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < iteration_size; j++) {
        conv_layer(input, filter, bias, output);
      }
      output.device_sync();
      const auto end = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> diff = end - start;
      const double execution_time_ms = diff.count() / static_cast<double>(iteration_size) * 1000.0;
      std::cout << execution_time_ms << " ms "<< std::endl;
    }
    return 0;
}
