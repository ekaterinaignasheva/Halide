#include "CodeGen_Posix.h"

namespace Halide {
namespace Internal {

using std::string;

#if defined(WITH_RISCV)

namespace {

/** A code generator that emits mips code from a given Halide stmt. */
class CodeGen_RISCV : public CodeGen_Posix {
public:
    /** Create a mips code generator. Processor features can be
     * enabled using the appropriate flags in the target struct. */
    CodeGen_RISCV(const Target &);

protected:
    using CodeGen_Posix::visit;

    string mcpu_target() const override;
    string mcpu_tune() const override;
    string mattrs() const override;
    string mabi() const override;
    bool use_soft_float_abi() const override;
    int native_vector_bits() const override;
    int target_vscale() const override;
};

CodeGen_RISCV::CodeGen_RISCV(const Target &t)
    : CodeGen_Posix(t) {
}

string CodeGen_RISCV::mcpu_target() const {
    return "";
}

string CodeGen_RISCV::mcpu_tune() const {
    return mcpu_target();
}

string CodeGen_RISCV::mattrs() const {
    // Note: the default march is "rv[32|64]imafdc",
    // which includes standard extensions:
    //   +m Integer Multiplication and Division,
    //   +a Atomic Instructions,
    //   +f Single-Precision Floating-Point,
    //   +d Double-Precision Floating-Point,
    //   +c Compressed Instructions,
    string arch_flags = "+m,+a,+f,+d,+c";

    if (target.has_feature(Target::RVV)) {
        arch_flags += ",+v";
    }
    return arch_flags;
}

string CodeGen_RISCV::mabi() const {
    string abi;
    if (target.bits == 32) {
        abi = "ilp32";
    } else {
        abi = "lp64";
    }
    if (!target.has_feature(Target::SoftFloatABI)) {
        abi += "d";
    }
    return abi;
}

bool CodeGen_RISCV::use_soft_float_abi() const {
    return target.has_feature(Target::SoftFloatABI);
}

int CodeGen_RISCV::native_vector_bits() const {
    if (target.vector_bits != 0 &&
        target.has_feature(Target::RVV)) {
        return target.vector_bits;
    }
    return 0;
}

int CodeGen_RISCV::target_vscale() const {
    if (target.vector_bits != 0 &&
        target.has_feature(Target::RVV)) {
        internal_assert((target.vector_bits % 64) == 0);
        return target.vector_bits / 64;
    }

    return 0;
}

}  // namespace

std::unique_ptr<CodeGen_Posix> new_CodeGen_RISCV(const Target &target) {
    return std::make_unique<CodeGen_RISCV>(target);
}

#else  // WITH_RISCV

std::unique_ptr<CodeGen_Posix> new_CodeGen_RISCV(const Target &target) {
    user_error << "RISCV not enabled for this build of Halide.\n";
    return nullptr;
}

#endif  // WITH_RISCV

}  // namespace Internal
}  // namespace Halide
