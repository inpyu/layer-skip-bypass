#ifndef NN_TOPOLOGY_H
#define NN_TOPOLOGY_H

#include "nn-core.hpp"
#include <stdexcept>

typedef struct {
    NnUint globalRank;
    NnUint ppRank;
    NnUint spRank;
    NnUint tpRank;
    NnUint tpGroupStart;
    NnUint tpGroupEnd;
    NnUint spGroupStart; // 같은 PP stage + SP group 내 첫 globalRank
    NnUint spGroupEnd;   // 같은 PP stage + SP group 내 마지막 globalRank (exclusive)
} NnNodePlacement;

// 3D topology: PP × SP × TP
// globalRank = ppRank * (spSize * tpSize) + spRank * tpSize + tpRank
class NnParallelTopology {
public:
    NnUint nNodes;
    NnUint ppSize;
    NnUint spSize; // Sequence Parallelism (default 1 = disabled)
    NnUint tpSize;

    NnParallelTopology(NnUint nNodes, NnUint ppSize, NnUint spSize = 1)
        : nNodes(nNodes), ppSize(ppSize), spSize(spSize), tpSize(0) {
        if (nNodes < 1)
            throw std::runtime_error("nNodes must be >= 1");
        if (ppSize < 1)
            throw std::runtime_error("ppSize must be >= 1");
        if (spSize < 1)
            throw std::runtime_error("spSize must be >= 1");
        if (nNodes % (ppSize * spSize) != 0)
            throw std::runtime_error("nNodes must be divisible by ppSize * spSize");
        this->tpSize = nNodes / (ppSize * spSize);
    }

    NnNodePlacement getPlacement(NnUint globalRank) const {
        if (globalRank >= nNodes)
            throw std::runtime_error("global rank out of range");
        const NnUint ppRank  = globalRank / (spSize * tpSize);
        const NnUint rem     = globalRank % (spSize * tpSize);
        const NnUint spRank  = rem / tpSize;
        const NnUint tpRank  = rem % tpSize;
        const NnUint tpGroupStart = ppRank * spSize * tpSize + spRank * tpSize;
        const NnUint tpGroupEnd   = tpGroupStart + tpSize;
        const NnUint spGroupStart = ppRank * spSize * tpSize;
        const NnUint spGroupEnd   = spGroupStart + spSize * tpSize;
        return NnNodePlacement{globalRank, ppRank, spRank, tpRank,
                               tpGroupStart, tpGroupEnd, spGroupStart, spGroupEnd};
    }

    // spSize=1 호환: (ppRank, tpRank)로 globalRank 계산
    NnUint getGlobalRank(NnUint ppRank, NnUint tpRank) const {
        return getGlobalRank(ppRank, 0, tpRank);
    }

    NnUint getGlobalRank(NnUint ppRank, NnUint spRank, NnUint tpRank) const {
        if (ppRank >= ppSize) throw std::runtime_error("pp rank out of range");
        if (spRank >= spSize) throw std::runtime_error("sp rank out of range");
        if (tpRank >= tpSize) throw std::runtime_error("tp rank out of range");
        return ppRank * (spSize * tpSize) + spRank * tpSize + tpRank;
    }

    // Lane = one full PP chain with fixed (spRank, tpRank)
    NnUint getLaneCount() const {
        return spSize * tpSize;
    }

    NnUint getLaneId(const NnNodePlacement &placement) const {
        return placement.spRank * tpSize + placement.tpRank;
    }
};

static inline NnParallelTopology createPPxTPTopology(NnUint nNodes, NnUint ppSize) {
    return NnParallelTopology(nNodes, ppSize, 1);
}

static inline NnParallelTopology createPPxSPxTPTopology(NnUint nNodes, NnUint ppSize, NnUint spSize) {
    return NnParallelTopology(nNodes, ppSize, spSize);
}

#endif
