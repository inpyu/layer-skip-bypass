#ifndef NN_PIPELINE_H
#define NN_PIPELINE_H

#include "nn-core.hpp"
#include "nn-network.hpp"
#include "nn-topology.hpp"
#include <vector>

// Pipeline activation transfer packet header
// Used for sending activations between pipeline stages
typedef struct {
    NnUint seqPosition;      // Sequence position of this activation
    NnUint batchIndex;       // Batch index (for batched inference)
    NnUint sliceId;          // TP slice ID (which TP rank this belongs to)
    NnFloatType dtype;       // Data type of the payload
    NnSize payloadBytes;     // Total payload bytes
    NnSize totalPayloadBytes;// Reserved (same as payloadBytes)
    NnSize streamChunkBytes; // Stream chunk size for payload transfer (0 = single write/read)
    NnUint flags;            // bit0: chunked stream, bit1: delta payload
    NnUint checksum;         // Simple checksum for data integrity (sum of first 4 floats)
} NnPipelineActivationHeader;

typedef struct {
    NnUint wireUs;           // socket read/write time
    NnUint decodeUs;         // decode/reconstruct time (recv only)
    NnSize payloadBytes;     // logical payload bytes (F32 domain)
    NnSize wireBytes;        // transferred payload bytes on wire
    NnSize streamChunkBytes; // configured stream chunk bytes (0 = disabled)
    NnUint flags;            // PIPELINE_FLAG_*
} NnPipelineTransferStats;

// Activation header flags.
static const NnUint PIPELINE_FLAG_CHUNKED = 1u << 0;
static const NnUint PIPELINE_FLAG_DELTA = 1u << 1;
// Header-only marker from decision stage to skipped stage.
static const NnUint PIPELINE_FLAG_STAGE_SKIP_BYPASS = 1u << 2;

// Pipeline communicator for stage-to-stage activation transfer
class NnPipelineCommunicator {
private:
    NnNetwork *network;
    const NnParallelTopology *topology;
    NnUint myNodeIndex;
    NnUint myPpRank;
    NnUint mySpRank;
    NnUint myTpRank;
    bool deltaEnabled;
    NnSize deltaMinBytes;
    NnSize chunkBytes;
    std::vector<std::vector<float>> prevSendBySocket;
    std::vector<std::vector<float>> prevRecvBySocket;
    // Reused scratch buffers to avoid per-token heap churn on decode hot path.
    std::vector<NnByte> sendWireScratch;
    std::vector<NnByte> recvWireScratch;
    std::vector<float> deltaScratch;
    NnPipelineTransferStats lastSendStats;
    NnPipelineTransferStats lastRecvStats;
    
    // Timeout configuration (in milliseconds)
    static const NnUint DEFAULT_SEND_TIMEOUT_MS = 5000;
    static const NnUint DEFAULT_RECV_TIMEOUT_MS = 10000;
    
    // Calculate socket index for pipeline communication
    // Returns the socket index to communicate with the corresponding node in next/prev stage
    NnUint getPipelineSocketIndex(NnUint targetPpRank, NnUint targetTpRank) const;
    
    // Simple checksum for data integrity
    NnUint calculateChecksum(const NnByte *data, NnSize bytes, NnFloatType dtype) const;

public:
    NnPipelineCommunicator(
        NnNetwork *network,
        const NnParallelTopology *topology,
        NnUint myNodeIndex,
        bool deltaEnabled = false,
        NnSize deltaMinBytes = 0,
        NnSize chunkBytes = 0
    );
    ~NnPipelineCommunicator();
    
    // Send activation to the next pipeline stage
    // - targetPpRank: Pipeline rank of the target stage
    // - seqPosition: Sequence position
    // - batchIndex: Batch index
    // - sliceId: TP slice ID (usually same as myTpRank for slice-preserving)
    // - data: Activation data
    // - bytes: Size of data
    // - dtype: Data type
    // Returns: true on success, false on timeout/error
    bool sendActivation(
        NnUint targetPpRank,
        NnUint seqPosition,
        NnUint batchIndex,
        NnUint sliceId,
        const NnByte *data,
        NnSize bytes,
        NnFloatType dtype,
        NnUint extraFlags = 0u
    );
    
    // Receive activation from the previous pipeline stage
    // - sourcePpRank: Pipeline rank of the source stage
    // - header: Output parameter for received header
    // - buffer: Buffer to receive activation data (must be pre-allocated)
    // - bufferSize: Size of the buffer
    // Returns: true on success, false on timeout/error
    bool recvActivation(
        NnUint sourcePpRank,
        NnPipelineActivationHeader *header,
        NnByte *buffer,
        NnSize bufferSize,
        unsigned long headerMaxAttempts = 0
    );
    
    // Check if this node should send activations (not in last stage)
    bool shouldSendActivations() const;
    
    // Check if this node should receive activations (not in first stage)
    bool shouldRecvActivations() const;
    
    // Get the target PP rank for sending (myPpRank + 1)
    NnUint getTargetPpRank() const;
    
    // Get the source PP rank for receiving (myPpRank - 1)
    NnUint getSourcePpRank() const;

    // Runtime transport tuning (keeps root/worker protocol in sync per request)
    void setTransportOptions(bool deltaEnabled, NnSize deltaMinBytes, NnSize chunkBytes);
    const NnPipelineTransferStats& getLastSendStats() const { return lastSendStats; }
    const NnPipelineTransferStats& getLastRecvStats() const { return lastRecvStats; }
};

#endif
