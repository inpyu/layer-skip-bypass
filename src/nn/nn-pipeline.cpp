#include "nn-pipeline.hpp"
#include "nn-quants.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

static inline unsigned long long pipelineNowUs() {
    return (unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static NnFloatType selectWireDtype(NnFloatType requested, NnSize srcBytes) {
    if (requested == F_Q40 || requested == F_Q80) {
        if (srcBytes % sizeof(float) != 0)
            return F_16;
        NnSize n = srcBytes / sizeof(float);
        if (n % Q80_BLOCK_SIZE != 0)
            return F_16;
    }
    return requested;
}

static void encodeFromF32(const NnByte *src, NnSize srcBytes, NnFloatType dtype, std::vector<NnByte> &out) {
    const float *srcF32 = reinterpret_cast<const float *>(src);
    if (dtype == F_32) {
        out.assign(src, src + srcBytes);
        return;
    }

    if (srcBytes % sizeof(float) != 0)
        throw std::runtime_error("Invalid source bytes for activation encoding");
    const NnSize n = srcBytes / sizeof(float);

    if (dtype == F_16) {
        out.resize(n * sizeof(NnFp16));
        NnFp16 *dst = reinterpret_cast<NnFp16 *>(out.data());
        for (NnSize i = 0; i < n; i++)
            dst[i] = CONVERT_F32_TO_F16(srcF32[i]);
        return;
    }

    if (dtype == F_Q80) {
        out.resize(getBytes(F_Q80, n));
        quantizeF32toQ80(srcF32, reinterpret_cast<NnBlockQ80 *>(out.data()), n, 1, 0);
        return;
    }

    if (dtype == F_Q40) {
        out.resize(getBytes(F_Q40, n));
        quantizeF32toQ40(srcF32, reinterpret_cast<NnBlockQ40 *>(out.data()), n, 1, 0);
        return;
    }

    throw std::runtime_error("Unsupported activation transport dtype");
}

static bool decodeToF32(
    const NnByte *wire,
    NnSize wireBytes,
    NnFloatType dtype,
    NnByte *dst,
    NnSize dstBytes,
    NnSize *decodedElemsOut = nullptr
) {
    if (dtype == F_32) {
        if (wireBytes > dstBytes)
            return false;
        std::memcpy(dst, wire, wireBytes);
        if (decodedElemsOut != nullptr)
            *decodedElemsOut = wireBytes / sizeof(float);
        return true;
    }

    float *dstF32 = reinterpret_cast<float *>(dst);
    if (dstBytes % sizeof(float) != 0)
        return false;
    const NnSize maxElems = dstBytes / sizeof(float);

    if (dtype == F_16) {
        if (wireBytes % sizeof(NnFp16) != 0)
            return false;
        NnSize n = wireBytes / sizeof(NnFp16);
        if (n > maxElems)
            return false;
        const NnFp16 *src = reinterpret_cast<const NnFp16 *>(wire);
        for (NnSize i = 0; i < n; i++)
            dstF32[i] = CONVERT_F16_TO_F32(src[i]);
        if (decodedElemsOut != nullptr)
            *decodedElemsOut = n;
        return true;
    }

    if (dtype == F_Q80) {
        if (wireBytes % sizeof(NnBlockQ80) != 0)
            return false;
        NnSize nBlocks = wireBytes / sizeof(NnBlockQ80);
        NnSize n = nBlocks * Q80_BLOCK_SIZE;
        if (n > maxElems)
            return false;
        dequantizeQ80toF32(reinterpret_cast<const NnBlockQ80 *>(wire), dstF32, n, 1, 0);
        if (decodedElemsOut != nullptr)
            *decodedElemsOut = n;
        return true;
    }

    if (dtype == F_Q40) {
        if (wireBytes % sizeof(NnBlockQ40) != 0)
            return false;
        NnSize nBlocks = wireBytes / sizeof(NnBlockQ40);
        NnSize n = nBlocks * Q40_BLOCK_SIZE;
        if (n > maxElems)
            return false;
        dequantizeQ40toF32(reinterpret_cast<const NnBlockQ40 *>(wire), dstF32, n, 1, 0);
        if (decodedElemsOut != nullptr)
            *decodedElemsOut = n;
        return true;
    }

    return false;
}

NnPipelineCommunicator::NnPipelineCommunicator(
    NnNetwork *network,
    const NnParallelTopology *topology,
    NnUint myNodeIndex,
    bool deltaEnabled,
    NnSize deltaMinBytes,
    NnSize chunkBytes
) {
    this->network = network;
    this->topology = topology;
    this->myNodeIndex = myNodeIndex;
    
    NnNodePlacement placement = topology->getPlacement(myNodeIndex);
    this->myPpRank = placement.ppRank;
    this->mySpRank = placement.spRank;
    this->myTpRank = placement.tpRank;
    this->deltaEnabled = deltaEnabled;
    this->deltaMinBytes = deltaMinBytes;
    this->chunkBytes = chunkBytes;
    if (network != nullptr) {
        this->prevSendBySocket.resize(network->nSockets);
        this->prevRecvBySocket.resize(network->nSockets);
    }
    std::memset(&lastSendStats, 0, sizeof(lastSendStats));
    std::memset(&lastRecvStats, 0, sizeof(lastRecvStats));
}

NnPipelineCommunicator::~NnPipelineCommunicator() {
    // Nothing to cleanup - network is owned externally
}

NnUint NnPipelineCommunicator::getPipelineSocketIndex(NnUint targetPpRank, NnUint targetTpRank) const {
    NnUint targetGlobalRank = topology->getGlobalRank(targetPpRank, mySpRank, targetTpRank);

    if (targetGlobalRank == myNodeIndex) {
        throw std::runtime_error("Pipeline socket target cannot be current node");
    }

    if (myNodeIndex == 0) {
        if (targetGlobalRank == 0)
            throw std::runtime_error("Invalid root pipeline socket target");
        return targetGlobalRank - 1;
    }

    if (targetGlobalRank == 0)
        return 0;

    if (targetGlobalRank < myNodeIndex)
        return targetGlobalRank;

    return targetGlobalRank - 1;
}

NnUint NnPipelineCommunicator::calculateChecksum(const NnByte *data, NnSize bytes, NnFloatType dtype) const {
    // Simple checksum: sum of first few values
    NnUint checksum = 0;
    
    if (dtype == F_32 && bytes >= sizeof(float) * 4) {
        const float *floats = reinterpret_cast<const float*>(data);
        for (int i = 0; i < 4; i++) {
            // Convert float to int for checksum
            std::memcpy(&checksum, &floats[i], sizeof(NnUint));
            checksum ^= (i + 1); // XOR with position
        }
    } else {
        // For other types or small data, just use first bytes
        NnSize checksumBytes = bytes < sizeof(NnUint) ? bytes : sizeof(NnUint);
        std::memcpy(&checksum, data, checksumBytes);
    }
    
    return checksum;
}

bool NnPipelineCommunicator::sendActivation(
    NnUint targetPpRank,
    NnUint seqPosition,
    NnUint batchIndex,
    NnUint sliceId,
    const NnByte *data,
    NnSize bytes,
    NnFloatType dtype,
    NnUint extraFlags
) {
    if (targetPpRank >= topology->ppSize) {
        throw std::runtime_error("Invalid target PP rank for pipeline communication");
    }
    
    // Slice-preserving: send to same TP rank in target stage
    // Prepare wire payload (activation transport may be quantized)
    NnFloatType wireDtype = selectWireDtype(dtype, bytes);
    const NnUint targetSocketIndex = getPipelineSocketIndex(targetPpRank, myTpRank);
    const bool canDelta = deltaEnabled &&
        bytes >= deltaMinBytes &&
        (bytes % sizeof(float) == 0) &&
        (targetSocketIndex < prevSendBySocket.size());
    bool useDelta = false;
    std::vector<NnByte> &wirePayload = sendWireScratch;
    const float *srcF32 = reinterpret_cast<const float *>(data);
    const NnSize nElems = bytes / sizeof(float);
    if (bytes == 0) {
        wirePayload.clear();
    } else if (canDelta && prevSendBySocket[targetSocketIndex].size() == nElems) {
        deltaScratch.resize(nElems);
        const float *prev = prevSendBySocket[targetSocketIndex].data();
        for (NnSize i = 0; i < nElems; i++)
            deltaScratch[i] = srcF32[i] - prev[i];
        encodeFromF32(reinterpret_cast<const NnByte *>(deltaScratch.data()), bytes, wireDtype, wirePayload);
        useDelta = true;
    } else {
        encodeFromF32(data, bytes, wireDtype, wirePayload);
    }

    try {
        const unsigned long long tWire0 = pipelineNowUs();
        const NnSize totalWireBytes = wirePayload.size();
        const NnSize writeChunkBytes = (chunkBytes > 0 && totalWireBytes > chunkBytes) ? chunkBytes : 0;
        NnPipelineActivationHeader header;
        header.seqPosition = seqPosition;
        header.batchIndex = batchIndex;
        header.sliceId = sliceId;
        header.dtype = wireDtype;
        header.payloadBytes = totalWireBytes;
        header.totalPayloadBytes = totalWireBytes;
        header.streamChunkBytes = writeChunkBytes;
        header.flags = 0u;
        if (writeChunkBytes > 0)
            header.flags |= PIPELINE_FLAG_CHUNKED;
        if (useDelta)
            header.flags |= PIPELINE_FLAG_DELTA;
        header.flags |= extraFlags;
        header.checksum = calculateChecksum(wirePayload.data(), totalWireBytes, wireDtype);

        // Send one metadata header, then stream payload in chunks (if enabled).
        network->write(targetSocketIndex, &header, sizeof(header));
        if (writeChunkBytes == 0) {
            network->write(targetSocketIndex, wirePayload.data(), totalWireBytes);
        } else {
            NnSize sent = 0;
            while (sent < totalWireBytes) {
                const NnSize chunk = std::min(writeChunkBytes, totalWireBytes - sent);
                network->write(targetSocketIndex, wirePayload.data() + sent, chunk);
                sent += chunk;
            }
        }
        const unsigned long long tWire1 = pipelineNowUs();
        NnSize txBytes = sizeof(NnPipelineActivationHeader) + totalWireBytes;
        network->addTaggedTraffic(false, txBytes, 0);
        // ACK 제거: wave pipelining에서 logits send ↔ activation send 교차 블로킹 deadlock 방지
        // TCP가 전달을 보장하며 checksum으로 데이터 무결성 검증
        if (targetSocketIndex < prevSendBySocket.size() && bytes > 0)
            prevSendBySocket[targetSocketIndex].assign(srcF32, srcF32 + nElems);
        lastSendStats.wireUs = (NnUint)(tWire1 - tWire0);
        lastSendStats.decodeUs = 0;
        lastSendStats.payloadBytes = bytes;
        lastSendStats.wireBytes = totalWireBytes;
        lastSendStats.streamChunkBytes = writeChunkBytes;
        lastSendStats.flags = header.flags;

        return true;
    } catch (const std::exception &e) {
        printf("🚨 Pipeline send error (target PP=%u, socket=%u): %s\n", 
               targetPpRank, targetSocketIndex, e.what());
        return false;
    }
}

bool NnPipelineCommunicator::recvActivation(
    NnUint sourcePpRank,
    NnPipelineActivationHeader *header,
    NnByte *buffer,
    NnSize bufferSize,
    unsigned long headerMaxAttempts
) {
    if (sourcePpRank >= topology->ppSize) {
        throw std::runtime_error("Invalid source PP rank for pipeline communication");
    }
    
    // Slice-preserving: receive from same TP rank in source stage
    NnUint sourceSocketIndex = getPipelineSocketIndex(sourcePpRank, myTpRank);
    
    try {
        const unsigned long long tWire0 = pipelineNowUs();
        // Receive single metadata header.
        if (headerMaxAttempts == 0) {
            network->read(sourceSocketIndex, header, sizeof(NnPipelineActivationHeader));
        } else if (!network->tryReadWithMaxAttempts(
            sourceSocketIndex,
            header,
            sizeof(NnPipelineActivationHeader),
            headerMaxAttempts
        )) {
            return false;
        }
        const NnSize totalWireBytes = header->payloadBytes;
        if (header->totalPayloadBytes != 0 && header->totalPayloadBytes != totalWireBytes) {
            printf("🚨 Pipeline recv error: invalid payload sizing (payload=%zu total=%zu)\n",
                totalWireBytes, header->totalPayloadBytes);
            return false;
        }
        recvWireScratch.resize(totalWireBytes);
        std::vector<NnByte> &wirePayload = recvWireScratch;
        const NnSize streamChunkBytes = ((header->flags & PIPELINE_FLAG_CHUNKED) != 0u) ? header->streamChunkBytes : 0;
        if ((header->flags & PIPELINE_FLAG_CHUNKED) == 0u || streamChunkBytes == 0) {
            network->read(sourceSocketIndex, wirePayload.data(), totalWireBytes);
        } else {
            NnSize received = 0;
            while (received < totalWireBytes) {
                const NnSize chunk = std::min(streamChunkBytes, totalWireBytes - received);
                network->read(sourceSocketIndex, wirePayload.data() + received, chunk);
                received += chunk;
            }
        }
        const unsigned long long tWire1 = pipelineNowUs();
        NnSize rxBytes = sizeof(NnPipelineActivationHeader) + totalWireBytes;
        network->addTaggedTraffic(false, 0, rxBytes);

        // Header-only marker payload.
        if (totalWireBytes == 0) {
            const unsigned long long tDecode1 = pipelineNowUs();
            lastRecvStats.wireUs = (NnUint)(tWire1 - tWire0);
            lastRecvStats.decodeUs = (NnUint)(tDecode1 - tWire1);
            lastRecvStats.payloadBytes = 0;
            lastRecvStats.wireBytes = 0;
            lastRecvStats.streamChunkBytes = 0;
            lastRecvStats.flags = header->flags;
            return true;
        }

        // Verify checksum for full payload.
        NnUint receivedChecksum = calculateChecksum(wirePayload.data(), totalWireBytes, header->dtype);
        if (receivedChecksum != header->checksum) {
            printf("⚠️  Pipeline recv warning: checksum mismatch (expected=%u, got=%u)\n",
                   header->checksum, receivedChecksum);
        }

        const unsigned long long tDecode0 = pipelineNowUs();
        if ((header->flags & PIPELINE_FLAG_DELTA) != 0u) {
            if (!(bufferSize % sizeof(float) == 0)) {
                printf("🚨 Pipeline recv error: delta requires float destination buffer\n");
                return false;
            }
            deltaScratch.resize(bufferSize / sizeof(float));
            NnSize decodedElems = 0;
            if (!decodeToF32(
                wirePayload.data(),
                totalWireBytes,
                header->dtype,
                reinterpret_cast<NnByte *>(deltaScratch.data()),
                bufferSize,
                &decodedElems
            )) {
                printf("🚨 Pipeline recv error: delta decode failed (dtype=%d payload=%zu dst=%zu)\n",
                    (int)header->dtype, totalWireBytes, bufferSize);
                return false;
            }
            if (sourceSocketIndex >= prevRecvBySocket.size() || prevRecvBySocket[sourceSocketIndex].size() != decodedElems) {
                printf("🚨 Pipeline recv error: delta base activation missing (socket=%u elems=%zu)\n",
                    sourceSocketIndex, decodedElems);
                return false;
            }
            float *dst = reinterpret_cast<float *>(buffer);
            const float *prev = prevRecvBySocket[sourceSocketIndex].data();
            for (NnSize i = 0; i < decodedElems; i++)
                dst[i] = prev[i] + deltaScratch[i];
            prevRecvBySocket[sourceSocketIndex].assign(dst, dst + decodedElems);
            lastRecvStats.payloadBytes = decodedElems * sizeof(float);
        } else {
            NnSize decodedElems = 0;
            if (!decodeToF32(
                wirePayload.data(),
                totalWireBytes,
                header->dtype,
                buffer,
                bufferSize,
                &decodedElems
            )) {
                printf("🚨 Pipeline recv error: decode failed (dtype=%d payload=%zu dst=%zu)\n",
                    (int)header->dtype, totalWireBytes, bufferSize);
                return false;
            }
            if (sourceSocketIndex < prevRecvBySocket.size()) {
                const float *src = reinterpret_cast<const float *>(buffer);
                prevRecvBySocket[sourceSocketIndex].assign(src, src + decodedElems);
            }
            lastRecvStats.payloadBytes = decodedElems * sizeof(float);
        }
        const unsigned long long tDecode1 = pipelineNowUs();
        lastRecvStats.wireUs = (NnUint)(tWire1 - tWire0);
        lastRecvStats.decodeUs = (NnUint)(tDecode1 - tDecode0);
        lastRecvStats.wireBytes = totalWireBytes;
        lastRecvStats.streamChunkBytes = streamChunkBytes;
        lastRecvStats.flags = header->flags;
        // ACK 제거: sendActivation과 대칭으로 ACK 왕복 없앰

        return true;
    } catch (const std::exception &e) {
        // Polled recv path intentionally returns false on "no data yet" style probes.
        if (headerMaxAttempts == 0) {
            printf("🚨 Pipeline recv error (source PP=%u, socket=%u): %s\n",
                   sourcePpRank, sourceSocketIndex, e.what());
        }
        return false;
    }
}

bool NnPipelineCommunicator::shouldSendActivations() const {
    // Send if not in the last stage
    return myPpRank < topology->ppSize - 1;
}

bool NnPipelineCommunicator::shouldRecvActivations() const {
    // Receive if not in the first stage
    return myPpRank > 0;
}

NnUint NnPipelineCommunicator::getTargetPpRank() const {
    if (!shouldSendActivations()) {
        throw std::runtime_error("Cannot get target PP rank: already in last stage");
    }
    return myPpRank + 1;
}

NnUint NnPipelineCommunicator::getSourcePpRank() const {
    if (!shouldRecvActivations()) {
        throw std::runtime_error("Cannot get source PP rank: already in first stage");
    }
    return myPpRank - 1;
}

void NnPipelineCommunicator::setTransportOptions(bool deltaEnabled, NnSize deltaMinBytes, NnSize chunkBytes) {
    this->deltaEnabled = deltaEnabled;
    this->deltaMinBytes = deltaMinBytes;
    this->chunkBytes = chunkBytes;
}
