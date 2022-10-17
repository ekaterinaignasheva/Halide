#include <chrono>
#include <cstdio>
#include <iostream>

#include "conv_layer.h"

#include "HalideBuffer.h"
#include "halide_benchmark.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    const int T = 4, SD = 32, DD = 16;
    const int H = 256, W = 256;
    const int window = 3;

    Buffer<float, 4> input(T, H, W, SD);
    Buffer<float, 6> filter(4, 4, SD, window, window, DD);
    Buffer<float, 2> bias(T, DD);
    input.fill(1);
    filter.fill(1);
    bias.fill(1);

    Buffer<float, 4> output(T, H, W, DD);

    for (int i =0; i < 20; i++)
    conv_layer(input, filter, bias, output);

    // Manually-tuned version
    double min_t_manual = benchmark(10, 100, [&]() {
        conv_layer(input, filter, bias, output);
        output.device_sync();
    });
    printf("Manually-tuned time: %gms\n", min_t_manual * 1e3);
    for (int i =0; i < 10; i++){
      std::cerr << "output " << i << ": " << output.begin()[i] << std::endl;
    }
    return 0;
}
