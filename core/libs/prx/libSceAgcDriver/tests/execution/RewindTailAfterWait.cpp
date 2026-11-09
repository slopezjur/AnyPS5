#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

alignas(256) std::array<std::uint32_t, 12> ReaderCode{0xf4040100u, 0xfa000000u, 0xf4080201u, 0xfa000000u, 0xbf8cc07fu, 0xf4000182u, 0xfa000000u, 0xbf8cc07fu, 0x7e000206u, 0xe0700000u, 0x80020000u, 0xbf810000u};

constexpr std::uint32_t StaleValue = 0x57a1e000u;
constexpr std::uint32_t FreshValue = 0xf7e54000u;

struct Slot {
    alignas(256) std::array<std::uint32_t, 4> stale{};
    alignas(256) std::array<std::uint32_t, 4> fresh{};
    alignas(256) std::array<std::uint32_t, 2> staleTable{};
    alignas(256) std::array<std::uint32_t, 2> freshTable{};
    alignas(256) std::array<std::uint32_t, 4> output{};
    alignas(256) std::array<std::uint32_t, 4> outputDescriptor{};
    alignas(256) volatile std::uint32_t label = 0;
    alignas(256) volatile std::uint32_t done = 0;
};

std::array<Slot, 3> Slots;

std::uint32_t Low(const volatile void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const volatile void* pointer) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)) >> 32u);
}

std::array<std::uint32_t, 4> Descriptor(const void* pointer, std::uint32_t bytes) {
    return {Low(pointer), High(pointer) & 0xffffu, bytes, 0x31016facu};
}

void SetShaderRegister(std::vector<std::uint32_t>& words, std::uint32_t offset, std::uint32_t value) {
    words.insert(words.end(), {0xc0017600u, offset, value});
}

std::size_t Dispatch(std::vector<std::uint32_t>& words, const void* code, std::initializer_list<std::uint32_t> userData) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(code));
    SetShaderRegister(words, 0x20c, static_cast<std::uint32_t>(address >> 8u));
    SetShaderRegister(words, 0x20d, static_cast<std::uint32_t>(address >> 40u) & 0xffu);
    for (const std::uint32_t offset : {0x207u, 0x208u, 0x209u}) SetShaderRegister(words, offset, 1);
    SetShaderRegister(words, 0x212, 0);
    SetShaderRegister(words, 0x213, static_cast<std::uint32_t>(userData.size()) << 1u);
    const auto first = words.size() + 2u;
    std::uint32_t offset = 0x240;
    for (const auto value : userData) SetShaderRegister(words, offset++, value);
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x8041u});
    return first;
}

void WriteData(std::vector<std::uint32_t>& words, volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0033700u, 0x00100200u, Low(target), High(target), value});
}

void WaitEqual(std::vector<std::uint32_t>& words, volatile std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0053c00u, 0x13u, Low(target), High(target), value, 0xffffffffu, 0x19u});
}

void Prepare(Slot& slot) {
    slot.stale[0] = StaleValue;
    slot.fresh[0] = FreshValue;
    slot.staleTable = {Low(slot.stale.data()), High(slot.stale.data())};
    slot.freshTable = {Low(slot.fresh.data()), High(slot.fresh.data())};
    slot.outputDescriptor = Descriptor(slot.output.data(), sizeof(slot.output));
    slot.output = {};
    slot.label = 0;
    slot.done = 0;
}

void WaitDone(const Slot& slot, const std::string& what) {
    const auto start = std::chrono::steady_clock::now();
    while (slot.done == 0) {
        Require(std::chrono::steady_clock::now() - start < std::chrono::seconds(10), what + ": the submission behind the wait did not finish");
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    AgcDriverWaitIdle_nid_postfix();
}

void TailRewrittenBeforeTheLabel(Slot& slot, bool earlyLabel) {
    const std::string what = earlyLabel ? "a wait that holds at submit" : "a wait that lands after the tail is rewritten";
    Prepare(slot);
    if (earlyLabel) slot.label = 1;
    std::vector<std::uint32_t> words;
    WaitEqual(words, &slot.label, 1);
    const auto table = Dispatch(words, ReaderCode.data(), {Low(slot.staleTable.data()), High(slot.staleTable.data()), Low(slot.outputDescriptor.data()), High(slot.outputDescriptor.data())});
    WriteData(words, &slot.done, 1);
    if (earlyLabel) {
        words[table] = Low(slot.freshTable.data());
        words[table + 3u] = High(slot.freshTable.data());
    }
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    Require(sceAgcDriverSubmitAcb(0x20, &packet) == 0, what + ": the compute submission failed");
    if (!earlyLabel) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        Require(slot.done == 0, what + ": the dispatch behind the wait ran before its label was written");
        std::atomic_ref<std::uint32_t>(words[table]).store(Low(slot.freshTable.data()), std::memory_order_release);
        std::atomic_ref<std::uint32_t>(words[table + 3u]).store(High(slot.freshTable.data()), std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        slot.label = 1;
    }
    WaitDone(slot, what);
    Require(slot.output[0] == FreshValue, what + ": the dispatch read " + (slot.output[0] == StaleValue ? std::string("the table its tail named at submit") : "neither table (" + std::to_string(slot.output[0]) + ")") + ", not the table the tail named when its wait landed");
}

}

int main() {
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        device.reset();
        TailRewrittenBeforeTheLabel(Slots[0], true);
        TailRewrittenBeforeTheLabel(Slots[1], false);
        TailRewrittenBeforeTheLabel(Slots[2], false);
        std::cout << "rewind tail after wait tests passed\n";
        AgcDriverShutdown_nid_postfix();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (const std::exception& shutdownError) {
            std::cerr << shutdownError.what() << '\n';
        }
        return 1;
    }
}
