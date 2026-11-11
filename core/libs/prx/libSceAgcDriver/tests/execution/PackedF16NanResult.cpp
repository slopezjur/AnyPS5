#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Inputs = 4;
constexpr std::uint32_t Results = 8;
constexpr std::uint32_t Batches = 16;
alignas(256) std::array<std::uint32_t, Threads * Inputs> Input{};
alignas(256) std::array<std::uint32_t, Threads * Results> Output{};

alignas(256) constexpr std::array<std::uint32_t, 26> Code{
    0x34020084, 0x34060085, 0xe0381000, 0x80000401, 0xbf8c3f70,
    0xcc0f400a, 0x18020b04,
    0xcc10400b, 0x18020b04,
    0xcc0e400c, 0x1c1a0b04,
    0xcc0f4a0d, 0x30020b04,
    0xcc10c00e, 0x18020d04,
    0xcc0e330f, 0x8c1a0b04,
    0x7e200306, 0x78200b04,
    0xcc0ec011, 0x3c1a0f04,
    0xe0781000, 0x80010a03, 0xe0781010, 0x80010e03, 0xbf810000,
};

const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 6> Modes{{
    std::nullopt,
    ShaderRecompiler::ShaderFloatMode{0xc0u, true, true, true},
    ShaderRecompiler::ShaderFloatMode{0xc0u, false, true, false},
    ShaderRecompiler::ShaderFloatMode{0xc0u, false, false, true},
    ShaderRecompiler::ShaderFloatMode{0xc0u, false, false, false},
    ShaderRecompiler::ShaderFloatMode{0x80u, false, true, false},
}};

std::uint64_t Seed = 0x2545f4914f6cdd1dull;

std::uint32_t Next() {
    Seed += 0x9e3779b97f4a7c15ull;
    std::uint64_t z = Seed;
    z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
    return static_cast<std::uint32_t>(z ^ (z >> 31u));
}

std::uint32_t Half() {
    const std::uint32_t bits = Next();
    const std::uint32_t sign = (bits >> 31u) << 15u;
    const std::uint32_t mantissa = bits & 0x3ffu;
    switch ((bits >> 16u) % 8u) {
    case 0u: return sign;
    case 1u: return sign | std::max(mantissa, 1u);
    case 2u: return sign | 0x7c00u;
    case 3u: return sign | 0x7c00u | std::max(mantissa & 0x1ffu, 1u);
    case 4u: return sign | 0x7e00u | mantissa;
    case 5u: return sign | ((0x1du + (bits >> 20u) % 2u) << 10u) | mantissa;
    default: return sign | ((0x0du + (bits >> 20u) % 5u) << 10u) | mantissa;
    }
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::fprintf(stderr, "usage: agc_driver_packed_f16_nan_result_tests <out.bin>\n");
            return VulkanTestSkipped;
        }
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto target = device->Target();
        std::vector<std::uint32_t> withoutFloat16;
        std::ranges::copy_if(target.supportedCapabilities, std::back_inserter(withoutFloat16), [](std::uint32_t capability) { return capability != spv::CapabilityFloat16; });
        auto emulatedTarget = target;
        emulatedTarget.supportedCapabilities = withoutFloat16;

        const std::span<const std::uint32_t> code(Code);
        std::vector<std::uint32_t> userData(8, 0u);
        const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size() * 4u));
        const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
        std::copy(input.begin(), input.end(), userData.begin());
        std::copy(output.begin(), output.end(), userData.begin() + 4);
        const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
        const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
        const auto compile = [&](const ShaderRecompiler::SpirvTarget& shaderTarget, const std::optional<ShaderRecompiler::ShaderFloatMode>& mode) {
            ShaderRecompiler::RecompileRequest request{
                {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
                {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
                shaderTarget,
                {0, 0, 0, 128}
            };
            request.useCache = false;
            request.context.floatMode = mode;
            return ShaderRecompiler::Recompile(request);
        };
        std::ofstream file(argv[1], std::ios::binary);
        std::uint64_t written = 0u;
        std::uint64_t nanLanes = 0u;
        for (const auto& mode : Modes) {
            for (const ShaderRecompiler::SpirvTarget* shaderTarget : std::array<const ShaderRecompiler::SpirvTarget*, 2>{&target, &emulatedTarget}) {
                const auto shader = compile(*shaderTarget, mode);
                for (std::uint32_t batch = 0; batch < Batches; ++batch) {
                    for (auto& word : Input) word = Half() | (Half() << 16u);
                    Output.fill(0xdeadbeefu);
                    device->Dispatch(shader, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
                    device->WaitIdle();
                    for (const auto word : Output) {
                        Require(word != 0xdeadbeefu, "packed f16 NaN result: a result was not written");
                        for (const auto lane : {word & 0xffffu, word >> 16u}) nanLanes += (lane & 0x7fffu) > 0x7c00u;
                    }
                    file.write(reinterpret_cast<const char*>(Output.data()), static_cast<std::streamsize>(Output.size() * 4u));
                    written += Output.size();
                }
            }
        }
        Require(nanLanes != 0u, "packed f16 NaN result: no NaN result lane was produced");
        std::printf("packed f16 NaN results written (%llu words, %llu NaN lanes)\n", static_cast<unsigned long long>(written), static_cast<unsigned long long>(nanLanes));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
