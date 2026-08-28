/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * Ring-exchange kernel -- the transport proof for device-payload channels.
 *
 * Deliberately minimal. It computes nothing: every rank publishes its own payload into its
 * slice of the symmetric window, and then reads its *neighbour's* slice back out. If the
 * output a rank produces is the input a different rank supplied, the bytes crossed the device
 * channel and nothing else can explain it.
 *
 *   Phase 1 (publish):  input -> my "payload" buffer, inside my window
 *   Phase 2 (barrier):  notify every peer, wait for every peer
 *   Phase 3 (collect):  peer (rank-1)'s "payload" buffer -> my output
 *
 * With two ranks that is a swap: rank 0 ends up holding rank 1's input and vice versa.
 *
 * `input` / `output` are per-rank host tensors passed through TaskArgs; the runtime does the
 * H2D / D2H. `payload` and `signal` are two named buffers carved out of the communication
 * window by the serving layer -- cross-rank addressable, and the reason this kernel needs no
 * host involvement once it starts.
 *
 * The signal buffer holds one int32 slot per rank: peer r bumps my slot[r] once its publish is
 * visible, and I wait on slot[r] before reading r's payload.
 *
 * args layout (see ring_exchange_orch.cpp):
 *   tensor(0) = input    (host-backed, framework-supplied device addr)
 *   tensor(1) = output   (host-backed, framework-supplied device addr)
 *   tensor(2) = payload  (window buffer, cross-rank addressable)
 *   tensor(3) = signal   (window buffer, cross-rank addressable)
 *   scalar(0) = nranks
 *   scalar(1) = CommContext device pointer
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "platform_comm/comm_context.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// The element count is a property of this kernel, not of the channel: tile extents are
// compile-time. The serving layer still chooses the buffer names, byte sizes, window size and
// rank set; it just has to size "payload" to at least this many floats.
static constexpr size_t RING_COUNT = 256;
static constexpr int kMaxSupportedRanks = 16;

/** Translate a pointer inside my own window into the same offset in peer `pe`'s window. */
template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *input_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *output_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *payload_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *signal_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    int nranks = static_cast<int>(args[4]);
    __gm__ CommContext *commCtx = reinterpret_cast<__gm__ CommContext *>(args[5]);

    __gm__ float *input = reinterpret_cast<__gm__ float *>(input_tensor->buffer.addr) + input_tensor->start_offset;
    __gm__ float *output = reinterpret_cast<__gm__ float *>(output_tensor->buffer.addr) + output_tensor->start_offset;
    __gm__ float *payload =
        reinterpret_cast<__gm__ float *>(payload_tensor->buffer.addr) + payload_tensor->start_offset;
    __gm__ int32_t *signal_base =
        reinterpret_cast<__gm__ int32_t *>(signal_tensor->buffer.addr) + signal_tensor->start_offset;

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, RING_COUNT, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 0 || nranks > kMaxSupportedRanks) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, RING_COUNT);
    StrideDyn stride(RING_COUNT, RING_COUNT, RING_COUNT, RING_COUNT, 1);

    TileData stageTile(1, RING_COUNT);
    TileData recvTile(1, RING_COUNT);
    TASSIGN(stageTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    Global inputG(input, shape, stride);
    Global payloadG(payload, shape, stride);
    Global outputG(output, shape, stride);

    // ------------------------------------------------------------------
    // Phase 1: publish -- copy my input into my "payload" slice of the
    // window, where peers can reach it.
    // ------------------------------------------------------------------
    TLOAD(stageTile, inputG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(payloadG, stageTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 2: device barrier -- announce that my publish is visible, then
    // wait until every peer has announced theirs. Reading a peer's payload
    // before this point would race its Phase 1.
    // ------------------------------------------------------------------
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        __gm__ int32_t *remote_signal = CommRemotePtr(commCtx, signal_base + my_rank, peer);
        pto::comm::Signal sig(remote_signal);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        pto::comm::Signal sig(signal_base + peer);
        pto::comm::TWAIT(sig, (int32_t)1, pto::comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 3: collect -- read my neighbour's payload across the channel and
    // stage it into my output. This is the only remote read, and it is the
    // whole proof.
    // ------------------------------------------------------------------
    int src_rank = (my_rank + nranks - 1) % nranks;
    __gm__ float *remote_payload = CommRemotePtr(commCtx, payload, src_rank);
    Global remoteG(remote_payload, shape, stride);
    TLOAD(recvTile, remoteG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    TSTORE(outputG, recvTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    pipe_barrier(PIPE_ALL);
}
