#include "Halide.h"
#include <iostream>

namespace {

using namespace Halide;

class ConvolutionLayer : public Halide::Generator<ConvolutionLayer> {
public:
    Input<Buffer<float, 4>> input{"input"};
    Input<Buffer<float, 6>> weights{"weights"};
    Input<Buffer<float, 2>> bias{"bias"};
    Output<Buffer<float, 4>> output{"output"};

    void generate() {
        const int H = 64, W = 64;
        const int src_channels = 1024;
        const int dst_channels = 1024;

        const int SD = src_channels / 4, DD = dst_channels / 4;
        const int window = 1;


        Var t("t"), h("h"), w("w"), d("d");
        Func conv("conv"), padded;

        RDom r(0, 4, 0, SD);

        padded(t, h, w, d) = input(t, clamp(h, 0, H - 1),
                                clamp(w, 0, W - 1), d);
        conv(t, h, w, d) = bias(t, d);
        conv(t, h, w, d) += padded(r.x, h, w, r.y) * weights(r.x, t, r.y, 0, 0, d);
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

        const int WG_X = 4, WG_Y = 4, WG_Z = 2;

        Var ho, hi, ho_i, ho_o;
        Var wo, wo_i, wo_o, wi;
        Var d_o, di;

        output.vectorize(t)
        .split(h, ho, hi, 2)
        .split(w, wo, wi, 4)
        .reorder(t, hi, wi, ho, wo)
        .unroll(hi)
        .unroll(wi)
        .gpu_tile(ho, wo, d, ho_o, wo_o, d_o, ho_i, wo_i, di, WG_Z, WG_Y, WG_X);

        conv.compute_at(output, ho_i);
        conv.vectorize(t)
            .unroll(h)
            .unroll(w);
        conv.update(0)
            .reorder(r.x, t, h, w, r.y, d)
            .atomic()
            .vectorize(r.x)
            .unroll(t)
            .unroll(w)
            .unroll(h);

    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ConvolutionLayer, conv_layer)
