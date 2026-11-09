#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>

namespace AgcDriver::DriverDetail {

std::atomic<std::uint64_t>& DeferredWaitAddress() {
    static std::atomic<std::uint64_t> address{0};
    return address;
}

void Driver::dumpPackets(std::span<const std::uint32_t> commands, const std::uint32_t* guest) {
    std::map<std::string, std::pair<std::size_t, std::string>> summary;
    std::vector<std::pair<std::size_t, std::uint32_t>> walk;
    for (std::size_t cursor = 0; cursor < commands.size();) {
        const auto header = commands[cursor];
        if (Pm4::FillerPacket(header)) { ++cursor; continue; }
        if ((header & 0xc0000000u) != 0xc0000000u) break;
        const auto count = Pm4::PacketWords(header);
        if (count > commands.size() - cursor) break;
        walk.emplace_back(cursor, header);
        auto& entry = summary[Pm4::Name(header)];
        if (entry.first++ == 0) {
            try {
                Pm4::Validate(commands.subspan(cursor, count), 0);
            } catch (const std::exception& error) {
                entry.second = error.what();
            }
        }
        cursor += count;
    }
    std::fprintf(stderr, "[gpu] rejected submission of %zu dwords at guest %p:\n", commands.size(), static_cast<const void*>(guest));
    for (const auto& [name, entry] : summary) std::fprintf(stderr, "[gpu]   %-28s x%-5zu %s\n", name.c_str(), entry.first, entry.second.c_str());

    for (std::size_t i = walk.size() > 24 ? walk.size() - 24 : 0; i < walk.size(); ++i) {
        const auto [offset, header] = walk[i];
        std::fprintf(stderr, "[gpu]   at DWORD %-6zu header 0x%08x %-24s %zu dwords:", offset, header, Pm4::Name(header).c_str(), Pm4::PacketWords(header));
        for (std::size_t j = offset + 1; j < std::min(commands.size(), offset + std::min<std::size_t>(Pm4::PacketWords(header), 12)); ++j) std::fprintf(stderr, " %08x", commands[j]);
        std::fprintf(stderr, "\n");
    }
    static const bool dumpRejected = std::getenv("APS5_DUMP_REJECTED") != nullptr;
    if (!dumpRejected) return;
    static int dumps = 0;
    char fileName[64];
    std::snprintf(fileName, sizeof(fileName), "rejected_submission_%d.bin", dumps++);
    if (FILE* file = std::fopen(fileName, "wb")) {
        std::fwrite(commands.data(), sizeof(std::uint32_t), commands.size(), file);
        std::fclose(file);
        std::fprintf(stderr, "[gpu] rejected submission written to %s\n", fileName);
    }
}

void Driver::validate(const Submission& submission, const std::uint32_t* guest) {
    const std::span<const std::uint32_t> commands = submission.commands;
    const auto queue = submission.queue;
    std::vector<std::size_t> guarded;
    for (std::size_t cursor = 0; cursor < commands.size();) {
        std::erase_if(guarded, [cursor](std::size_t end) { return end <= cursor; });
        const auto header = commands[cursor];
        if (Pm4::FillerPacket(header)) { ++cursor; continue; }
        if ((header & 0xc0000000u) != 0xc0000000u) {
            char what[96];
            std::snprintf(what, sizeof(what), "unsupported PM4 packet type: header 0x%08x at DWORD %zu of %zu", header, cursor, commands.size());

            dumpPackets(commands, guest);
            std::fprintf(stderr, "[gpu] dwords %zu..%zu:", cursor >= 8 ? cursor - 8 : 0, std::min(commands.size(), cursor + 8));
            for (std::size_t i = cursor >= 8 ? cursor - 8 : 0; i < std::min(commands.size(), cursor + 8); ++i) std::fprintf(stderr, " %08x", commands[i]);
            std::fprintf(stderr, "\n");
            if (guest != nullptr) {
                std::size_t firstWait = commands.size();
                for (std::size_t at = 0; at < cursor;) {
                    const auto word = commands[at];
                    if (Pm4::FillerPacket(word)) { ++at; continue; }
                    if (((word >> 8u) & 0xffu) == 0x3cu && firstWait == commands.size()) firstWait = at;
                    at += Pm4::PacketWords(word);
                }
                const auto waitAddress = DeferredWaitAddress().load(std::memory_order_acquire);
                const auto waitWord = [&]() -> std::uint64_t {
                    if (waitAddress == 0) return 0;
                    return std::atomic_ref<const std::uint32_t>(*reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(waitAddress))).load(std::memory_order_acquire);
                };
                const auto start = std::chrono::steady_clock::now();
                std::size_t lastChanged = ~std::size_t{0};
                std::size_t lastValid = ~std::size_t{0};
                std::uint64_t lastWord = ~std::uint64_t{0};
                int reports = 0;
                bool becameValid = false;
                while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(10000) && reports < 40) {
                    std::size_t changed = 0;
                    std::size_t firstChanged = commands.size();
                    for (std::size_t i = 0; i < commands.size(); ++i) {
                        const auto now = std::atomic_ref<const std::uint32_t>(guest[i]).load(std::memory_order_acquire);
                        if (now != commands[i]) {
                            ++changed;
                            firstChanged = std::min(firstChanged, i);
                        }
                    }
                    std::size_t valid = 0;
                    for (std::size_t at = 0; at < commands.size();) {
                        const auto word = std::atomic_ref<const std::uint32_t>(guest[at]).load(std::memory_order_acquire);
                        if (Pm4::FillerPacket(word)) { ++at; continue; }
                        if ((word & 0xc0000000u) != 0xc0000000u) break;
                        const auto words = Pm4::PacketWords(word);
                        if (words == 0 || words > commands.size() - at) break;
                        at += words;
                        valid = at;
                    }
                    const auto word = waitWord();
                    if (changed != lastChanged || valid != lastValid || word != lastWord) {
                        const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                        std::fprintf(stderr, "[torn] queue 0x%x t=%.1f ms: %zu of %zu dwords changed (first at %zu), packets parse to dword %zu, wait word 0x%llx at 0x%llx; first WAIT_REG_MEM at dword %zu\n", queue, elapsed, changed, commands.size(), firstChanged, valid, static_cast<unsigned long long>(word), static_cast<unsigned long long>(waitAddress), firstWait);
                        lastChanged = changed;
                        lastValid = valid;
                        lastWord = word;
                        ++reports;
                    }
                    if (valid == commands.size()) {
                        becameValid = true;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(500));
                }
                std::fprintf(stderr, "[torn] queue 0x%x: %s within 3 s\n", queue, becameValid ? "the guest buffer became a valid packet stream" : "the guest buffer never became a valid packet stream");
            }
            throw std::runtime_error(std::string("AGC driver: ") + what);
        }
        const auto count = Pm4::PacketWords(header);
        require(count <= commands.size() - cursor, "truncated PM4 packet");
        try {
            Pm4::Validate(commands.subspan(cursor, count), queue);
            if (header == FlipPacketHeader && !guarded.empty()) throw std::runtime_error("a flip inside a conditional execution range (conditional flip reservation) is not implemented");
            if (((header >> 8u) & 0xffu) == 0x22u) {
                const auto end = submission.conditionalEnds.find(cursor);
                if (end == submission.conditionalEnds.end()) throw std::runtime_error("conditional execution range is not made of whole packets of its command buffer");
                guarded.push_back(end->second);
            }
        } catch (const std::exception& error) {
            dumpPackets(commands, guest);
            throw std::runtime_error("AGC driver: " + Pm4::Name(header) + " at DWORD " + std::to_string(cursor) + ": " + error.what());
        }
        cursor += count;
    }
}

}
