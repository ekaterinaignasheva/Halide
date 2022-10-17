#include "Halide.h"
#include <iostream>

namespace {

using namespace Halide;

class ConvolutionLayer : public Halide::Generator<ConvolutionLayer> {
public:
    Input<Buffer<float, 4>> input{"input"};
    // t_source, k_destination, source_depth, window_h, window_w, destination_depth
    Input<Buffer<float, 6>> weights{"weights"};
    Input<Buffer<float, 2>> bias{"bias"};
    Output<Buffer<float, 4>> output{"output"};

    void generate() {
        const int SD = 32, DD = 16;
        const int H = 256, W = 256;
        const int WG_X = 4, WG_Y = 4;
        const int window = 3;

        Var t("t"), h("h"), w("w"), d("d");
        Func conv("conv"), padded;

        RDom r(0, 4, 0, SD, -1, window-1);

        padded(t, h, w, d) = input(t, clamp(h, 0, H - 1),
                                clamp(w, 0, W - 1), d);
        conv(t, h, w, d) = bias(t, d);
        conv(t, h, w, d) += padded(r.x, h + r.z, w + r.z, r.y) *
                            weights(r.x, t, r.y, r.z + 1, r.z + 1, d);
        output(t, h, w, d) = max(0, conv(t, h, w, d));

        input.dim(0).set_bounds(0, 4).set_stride(1);
        input.dim(1).set_bounds(0, H).set_stride(4);
        input.dim(2).set_bounds(0, W).set_stride(4*H);
        input.dim(3).set_bounds(0, SD).set_stride(4*H*W);

        weights.dim(0).set_bounds(0, 4).set_stride(1); // t_source
        weights.dim(1).set_bounds(0, 4).set_stride(4); // k_destination
        weights.dim(2).set_bounds(0, SD).set_stride(16);// source_depth
        weights.dim(3).set_bounds(0, window).set_stride(16*SD);// window_h
        weights.dim(4).set_bounds(0, window).set_stride(16*SD*window);// window_w
        weights.dim(5).set_bounds(0, DD).set_stride(16*SD*window*window);// destination_depth

        bias.dim(0).set_bounds(0, 4).set_stride(1);
        bias.dim(1).set_bounds(0, DD).set_stride(4);

        output.dim(0).set_bounds(0, 4).set_stride(1);
        output.dim(1).set_bounds(0, H).set_stride(4);
        output.dim(2).set_bounds(0, W).set_stride(4*H);
        output.dim(3).set_bounds(0, DD).set_stride(4*H*W);

        // conv.update(0).atomic().vectorize(r.x);

        const int WG_Z = 2;
        Var ho, wo, d_o, hi, wi, di;
        output.vectorize(t)
        .gpu_tile(h, w, d, ho, wo, d_o, hi, wi, di, WG_Z, WG_Y, WG_X);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ConvolutionLayer, conv_layer)
