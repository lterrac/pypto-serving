#include <serving/bindings/bindings.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <serving/bindings/repr.hpp>
#include <serving/config/types.hpp>

namespace py = pybind11;

// In the types' own namespace so the repr templates find them by ADL.
namespace serving::config
{

std::string show(const KVCacheSpec &v)
{
  return bindings::Repr("KVCacheSpec").field("block_size", v.blockSize).field("page_size_bytes", v.pageSizeBytes).field("compress_ratio", v.compressRatio).str();
}
std::string show(const KVCacheGroupSpec &v)
{
  return bindings::Repr("KVCacheGroupSpec")
    .field("name", v.name)
    .field("layer_indices", v.layerIndices)
    .field("spec", v.spec)
    .field("max_blocks_per_seq", v.maxBlocksPerSeq)
    .field("num_blocks", v.numBlocks)
    .field("num_partitions", v.numPartitions)
    .field("sliding_window", v.slidingWindow)
    .field("is_eagle_group", v.isEagleGroup)
    .str();
}
std::string show(const SamplingParams &v)
{
  return bindings::Repr("SamplingParams").field("temperature", v.temperature).field("top_p", v.topP).field("top_k", v.topK).field("seed", v.seed).str();
}
std::string show(const KvAllocation &v)
{
  return bindings::Repr("KvAllocation")
    .field("request_id", v.requestId)
    .field("model_id", v.modelId)
    .field("page_ids", v.pageIds)
    .field("tokens_capacity", v.tokensCapacity)
    .field("tokens_used", v.tokensUsed)
    .str();
}

} // namespace serving::config

namespace serving::bindings
{

namespace
{

using namespace serving::config;

/// `ring_*` is `int | list[int] | None` in Python and a vector here: None is
/// empty, an int is a one-element list.
std::vector<int> ringFromPython(const py::handle &value)
{
  if (value.is_none()) { return {}; }
  if (py::isinstance<py::int_>(value)) { return {value.cast<int>()}; }
  return value.cast<std::vector<int>>();
}

py::object ringToPython(const std::vector<int> &value)
{
  if (value.empty()) { return py::none(); }
  return py::cast(value);
}

const char *statusName(RequestStateStatus status)
{
  switch (status)
  {
  case RequestStateStatus::Waiting: return "waiting";
  case RequestStateStatus::Prefill: return "prefill";
  case RequestStateStatus::Decode: return "decode";
  case RequestStateStatus::Finished: return "finished";
  case RequestStateStatus::Aborted: return "aborted";
  case RequestStateStatus::Error: return "error";
  }
  return "waiting";
}

RequestStateStatus statusFromName(const std::string &name)
{
  for (const auto status :
       {RequestStateStatus::Waiting, RequestStateStatus::Prefill, RequestStateStatus::Decode, RequestStateStatus::Finished, RequestStateStatus::Aborted, RequestStateStatus::Error})
  {
    if (name == statusName(status)) { return status; }
  }
  throw std::invalid_argument("invalid request status '" + name + "' (expected waiting, prefill, decode, finished, aborted or error)");
}

} // namespace

void bindConfigTypes(py::module_ &m)
{
  // Frozen dataclasses are def_readonly; the others def_readwrite.

  py::class_<GenerateConfig>(m, "GenerateConfig", "User-facing options that control text generation.")
    .def(py::init(
           [](int maxNewTokens, double temperature, double topP, std::optional<int> topK, std::optional<uint64_t> seed, std::vector<std::string> stop, bool stream, bool ignoreEos) {
             GenerateConfig c;
             c.maxNewTokens = maxNewTokens;
             c.temperature  = temperature;
             c.topP         = topP;
             c.topK         = topK;
             c.seed         = seed;
             c.stop         = std::move(stop);
             c.stream       = stream;
             c.ignoreEos    = ignoreEos;
             return c;
           }),
         py::arg("max_new_tokens") = 256,
         py::arg("temperature")    = 0.0,
         py::arg("top_p")          = 1.0,
         py::arg("top_k")          = py::none(),
         py::arg("seed")           = py::none(),
         py::arg("stop")           = std::vector<std::string>{},
         py::arg("stream")         = false,
         py::arg("ignore_eos")     = false)
    .def_readonly("max_new_tokens", &GenerateConfig::maxNewTokens)
    .def_readonly("temperature", &GenerateConfig::temperature)
    .def_readonly("top_p", &GenerateConfig::topP)
    .def_readonly("top_k", &GenerateConfig::topK)
    .def_readonly("seed", &GenerateConfig::seed)
    .def_readonly("stop", &GenerateConfig::stop)
    .def_readonly("stream", &GenerateConfig::stream)
    .def_readonly("ignore_eos", &GenerateConfig::ignoreEos)
    .def(py::self == py::self)
    .def("__repr__", [](const GenerateConfig &c) {
      return Repr("GenerateConfig")
        .field("max_new_tokens", c.maxNewTokens)
        .field("temperature", c.temperature)
        .field("top_p", c.topP)
        .field("top_k", c.topK)
        .field("seed", c.seed)
        .field("stop", c.stop)
        .field("stream", c.stream)
        .field("ignore_eos", c.ignoreEos)
        .str();
    });

  py::class_<ModelConfig>(m, "ModelConfig", "Static description of one loaded model, as read from its config.json.")
    .def(py::init([](std::string        modelId,
                     std::string        architecture,
                     int                vocabSize,
                     int                hiddenSize,
                     int                intermediateSize,
                     int                numHiddenLayers,
                     int                numAttentionHeads,
                     int                numKeyValueHeads,
                     int                headDim,
                     int                maxPositionEmbeddings,
                     double             rmsNormEps,
                     double             ropeTheta,
                     std::optional<int> bosTokenId,
                     std::optional<int> eosTokenId,
                     std::optional<int> padTokenId,
                     std::string        torchDtype) {
           ModelConfig c;
           c.modelId               = std::move(modelId);
           c.architecture          = std::move(architecture);
           c.vocabSize             = vocabSize;
           c.hiddenSize            = hiddenSize;
           c.intermediateSize      = intermediateSize;
           c.numHiddenLayers       = numHiddenLayers;
           c.numAttentionHeads     = numAttentionHeads;
           c.numKeyValueHeads      = numKeyValueHeads;
           c.headDim               = headDim;
           c.maxPositionEmbeddings = maxPositionEmbeddings;
           c.rmsNormEps            = rmsNormEps;
           c.ropeTheta             = ropeTheta;
           c.bosTokenId            = bosTokenId;
           c.eosTokenId            = eosTokenId;
           c.padTokenId            = padTokenId;
           c.torchDtype            = std::move(torchDtype);
           return c;
         }),
         py::arg("model_id"),
         py::arg("architecture"),
         py::arg("vocab_size"),
         py::arg("hidden_size"),
         py::arg("intermediate_size"),
         py::arg("num_hidden_layers"),
         py::arg("num_attention_heads"),
         py::arg("num_key_value_heads"),
         py::arg("head_dim"),
         py::arg("max_position_embeddings"),
         py::arg("rms_norm_eps"),
         py::arg("rope_theta"),
         py::arg("bos_token_id"),
         py::arg("eos_token_id"),
         py::arg("pad_token_id"),
         py::arg("torch_dtype"))
    .def_readonly("model_id", &ModelConfig::modelId)
    .def_readonly("architecture", &ModelConfig::architecture)
    .def_readonly("vocab_size", &ModelConfig::vocabSize)
    .def_readonly("hidden_size", &ModelConfig::hiddenSize)
    .def_readonly("intermediate_size", &ModelConfig::intermediateSize)
    .def_readonly("num_hidden_layers", &ModelConfig::numHiddenLayers)
    .def_readonly("num_attention_heads", &ModelConfig::numAttentionHeads)
    .def_readonly("num_key_value_heads", &ModelConfig::numKeyValueHeads)
    .def_readonly("head_dim", &ModelConfig::headDim)
    .def_readonly("max_position_embeddings", &ModelConfig::maxPositionEmbeddings)
    .def_readonly("rms_norm_eps", &ModelConfig::rmsNormEps)
    .def_readonly("rope_theta", &ModelConfig::ropeTheta)
    .def_readonly("bos_token_id", &ModelConfig::bosTokenId)
    .def_readonly("eos_token_id", &ModelConfig::eosTokenId)
    .def_readonly("pad_token_id", &ModelConfig::padTokenId)
    .def_readonly("torch_dtype", &ModelConfig::torchDtype)
    .def(py::self == py::self)
    .def("__repr__", [](const ModelConfig &c) {
      return Repr("ModelConfig")
        .field("model_id", c.modelId)
        .field("architecture", c.architecture)
        .field("vocab_size", c.vocabSize)
        .field("hidden_size", c.hiddenSize)
        .field("num_hidden_layers", c.numHiddenLayers)
        .field("num_attention_heads", c.numAttentionHeads)
        .field("num_key_value_heads", c.numKeyValueHeads)
        .field("head_dim", c.headDim)
        .field("max_position_embeddings", c.maxPositionEmbeddings)
        .field("torch_dtype", c.torchDtype)
        .str();
    });

  py::class_<KVCacheSpec>(m, "KVCacheSpec", "Source-token and physical-storage layout of one cache block.")
    .def(py::init([](int blockSize, int pageSizeBytes, int compressRatio) {
           KVCacheSpec s;
           s.blockSize     = blockSize;
           s.pageSizeBytes = pageSizeBytes;
           s.compressRatio = compressRatio;
           s.validate();
           return s;
         }),
         py::arg("block_size"),
         py::arg("page_size_bytes"),
         py::arg("compress_ratio") = 1)
    .def_readonly("block_size", &KVCacheSpec::blockSize)
    .def_readonly("page_size_bytes", &KVCacheSpec::pageSizeBytes)
    .def_readonly("compress_ratio", &KVCacheSpec::compressRatio)
    .def_property_readonly("token_capacity", &KVCacheSpec::tokenCapacity)
    .def_property_readonly("storage_block_size", &KVCacheSpec::storageBlockSize)
    .def(py::self == py::self)
    .def("__repr__", [](const KVCacheSpec &s) { return show(s); });

  py::class_<KVCacheGroupSpec>(m, "KVCacheGroupSpec", "One independently allocated model-specific cache family.")
    .def(py::init([](std::string        name,
                     std::vector<int>   layerIndices,
                     KVCacheSpec        spec,
                     int                maxBlocksPerSeq,
                     std::optional<int> numBlocks,
                     int                numPartitions,
                     std::optional<int> slidingWindow,
                     bool               isEagleGroup) {
           KVCacheGroupSpec g;
           g.name            = std::move(name);
           g.layerIndices    = std::move(layerIndices);
           g.spec            = spec;
           g.maxBlocksPerSeq = maxBlocksPerSeq;
           g.numBlocks       = numBlocks;
           g.numPartitions   = numPartitions;
           g.slidingWindow   = slidingWindow;
           g.isEagleGroup    = isEagleGroup;
           g.validate();
           return g;
         }),
         py::arg("name"),
         py::arg("layer_indices"),
         py::arg("spec"),
         py::arg("max_blocks_per_seq"),
         py::arg("num_blocks")     = py::none(),
         py::arg("num_partitions") = 1,
         py::arg("sliding_window") = py::none(),
         py::arg("is_eagle_group") = false)
    .def_readonly("name", &KVCacheGroupSpec::name)
    .def_readonly("layer_indices", &KVCacheGroupSpec::layerIndices)
    .def_readonly("spec", &KVCacheGroupSpec::spec)
    .def_readonly("max_blocks_per_seq", &KVCacheGroupSpec::maxBlocksPerSeq)
    .def_readonly("num_blocks", &KVCacheGroupSpec::numBlocks)
    .def_readonly("num_partitions", &KVCacheGroupSpec::numPartitions)
    .def_readonly("sliding_window", &KVCacheGroupSpec::slidingWindow)
    .def_readonly("is_eagle_group", &KVCacheGroupSpec::isEagleGroup)
    .def(py::self == py::self)
    .def("__repr__", [](const KVCacheGroupSpec &g) { return show(g); });

  py::class_<RuntimeConfig>(m, "RuntimeConfig", "Runtime limits and device placement for one loaded model.")
    .def(py::init([](int                           pageSize,
                     int                           maxBatchSize,
                     int                           maxSeqLen,
                     std::string                   device,
                     std::string                   kvDtype,
                     std::string                   weightDtype,
                     std::optional<int>            totalKvPages,
                     double                        npuMemoryUtilization,
                     int                           maxNumBatchedTokens,
                     std::optional<int>            maxPrefillTokensPerRequest,
                     std::vector<int>              prefillChunkSizeChoices,
                     bool                          supportsChunkedPrefillWithSpeculation,
                     bool                          requiresHomogeneousPrefillDecode,
                     int                           maxNewTokens,
                     int                           numSpeculativeTokens,
                     std::vector<KVCacheGroupSpec> kvCacheGroups,
                     const py::handle             &ringDepPool,
                     const py::handle             &ringTaskWindow,
                     const py::handle             &ringHeap) {
           RuntimeConfig r;
           r.pageSize                              = pageSize;
           r.maxBatchSize                          = maxBatchSize;
           r.maxSeqLen                             = maxSeqLen;
           r.device                                = std::move(device);
           r.kvDtype                               = std::move(kvDtype);
           r.weightDtype                           = std::move(weightDtype);
           r.totalKvPages                          = totalKvPages;
           r.npuMemoryUtilization                  = npuMemoryUtilization;
           r.maxNumBatchedTokens                   = maxNumBatchedTokens;
           r.maxPrefillTokensPerRequest            = maxPrefillTokensPerRequest;
           r.prefillChunkSizeChoices               = std::move(prefillChunkSizeChoices);
           r.supportsChunkedPrefillWithSpeculation = supportsChunkedPrefillWithSpeculation;
           r.requiresHomogeneousPrefillDecode      = requiresHomogeneousPrefillDecode;
           r.maxNewTokens                          = maxNewTokens;
           r.numSpeculativeTokens                  = numSpeculativeTokens;
           r.kvCacheGroups                         = std::move(kvCacheGroups);
           r.ringDepPool                           = ringFromPython(ringDepPool);
           r.ringTaskWindow                        = ringFromPython(ringTaskWindow);
           r.ringHeap                              = ringFromPython(ringHeap);
           return r;
         }),
         py::arg("page_size")                                 = 64,
         py::arg("max_batch_size")                            = 1,
         py::arg("max_seq_len")                               = 4096,
         py::arg("device")                                    = "cpu",
         py::arg("kv_dtype")                                  = "bfloat16",
         py::arg("weight_dtype")                              = "bfloat16",
         py::arg("total_kv_pages")                            = py::none(),
         py::arg("npu_memory_utilization")                    = 0.90,
         py::arg("max_num_batched_tokens")                    = 4096,
         py::arg("max_prefill_tokens_per_request")            = py::none(),
         py::arg("prefill_chunk_size_choices")                = std::vector<int>{},
         py::arg("supports_chunked_prefill_with_speculation") = true,
         py::arg("requires_homogeneous_prefill_decode")       = false,
         py::arg("max_new_tokens")                            = 256,
         py::arg("num_speculative_tokens")                    = 0,
         py::arg("kv_cache_groups")                           = std::vector<KVCacheGroupSpec>{},
         py::arg("ring_dep_pool")                             = py::none(),
         py::arg("ring_task_window")                          = py::none(),
         py::arg("ring_heap")                                 = py::none())
    .def_readonly("page_size", &RuntimeConfig::pageSize)
    .def_readonly("max_batch_size", &RuntimeConfig::maxBatchSize)
    .def_readonly("max_seq_len", &RuntimeConfig::maxSeqLen)
    .def_readonly("device", &RuntimeConfig::device)
    .def_readonly("kv_dtype", &RuntimeConfig::kvDtype)
    .def_readonly("weight_dtype", &RuntimeConfig::weightDtype)
    .def_readonly("total_kv_pages", &RuntimeConfig::totalKvPages)
    .def_readonly("npu_memory_utilization", &RuntimeConfig::npuMemoryUtilization)
    .def_readonly("max_num_batched_tokens", &RuntimeConfig::maxNumBatchedTokens)
    .def_readonly("max_prefill_tokens_per_request", &RuntimeConfig::maxPrefillTokensPerRequest)
    .def_readonly("prefill_chunk_size_choices", &RuntimeConfig::prefillChunkSizeChoices)
    .def_readonly("supports_chunked_prefill_with_speculation", &RuntimeConfig::supportsChunkedPrefillWithSpeculation)
    .def_readonly("requires_homogeneous_prefill_decode", &RuntimeConfig::requiresHomogeneousPrefillDecode)
    .def_readonly("max_new_tokens", &RuntimeConfig::maxNewTokens)
    .def_readonly("num_speculative_tokens", &RuntimeConfig::numSpeculativeTokens)
    .def_readonly("kv_cache_groups", &RuntimeConfig::kvCacheGroups)
    .def_property_readonly("ring_dep_pool", [](const RuntimeConfig &r) { return ringToPython(r.ringDepPool); })
    .def_property_readonly("ring_task_window", [](const RuntimeConfig &r) { return ringToPython(r.ringTaskWindow); })
    .def_property_readonly("ring_heap", [](const RuntimeConfig &r) { return ringToPython(r.ringHeap); })
    .def(py::self == py::self)
    .def("__repr__", [](const RuntimeConfig &r) {
      return Repr("RuntimeConfig")
        .field("page_size", r.pageSize)
        .field("max_batch_size", r.maxBatchSize)
        .field("max_seq_len", r.maxSeqLen)
        .field("device", r.device)
        .field("kv_dtype", r.kvDtype)
        .field("weight_dtype", r.weightDtype)
        .field("total_kv_pages", r.totalKvPages)
        .field("npu_memory_utilization", r.npuMemoryUtilization)
        .field("max_num_batched_tokens", r.maxNumBatchedTokens)
        .field("max_new_tokens", r.maxNewTokens)
        .field("num_speculative_tokens", r.numSpeculativeTokens)
        .field("kv_cache_groups", r.kvCacheGroups)
        .str();
    });

  py::class_<LayerSpec>(m, "LayerSpec", "Per-layer geometry handed to the model layer.")
    .def(py::init([](int layerIdx, int hiddenSize, int intermediateSize, int numAttentionHeads, int numKeyValueHeads, int headDim) {
           LayerSpec l;
           l.layerIdx          = layerIdx;
           l.hiddenSize        = hiddenSize;
           l.intermediateSize  = intermediateSize;
           l.numAttentionHeads = numAttentionHeads;
           l.numKeyValueHeads  = numKeyValueHeads;
           l.headDim           = headDim;
           return l;
         }),
         py::arg("layer_idx"),
         py::arg("hidden_size"),
         py::arg("intermediate_size"),
         py::arg("num_attention_heads"),
         py::arg("num_key_value_heads"),
         py::arg("head_dim"))
    .def_readonly("layer_idx", &LayerSpec::layerIdx)
    .def_readonly("hidden_size", &LayerSpec::hiddenSize)
    .def_readonly("intermediate_size", &LayerSpec::intermediateSize)
    .def_readonly("num_attention_heads", &LayerSpec::numAttentionHeads)
    .def_readonly("num_key_value_heads", &LayerSpec::numKeyValueHeads)
    .def_readonly("head_dim", &LayerSpec::headDim)
    .def(py::self == py::self)
    .def("__repr__", [](const LayerSpec &l) {
      return Repr("LayerSpec")
        .field("layer_idx", l.layerIdx)
        .field("hidden_size", l.hiddenSize)
        .field("intermediate_size", l.intermediateSize)
        .field("num_attention_heads", l.numAttentionHeads)
        .field("num_key_value_heads", l.numKeyValueHeads)
        .field("head_dim", l.headDim)
        .str();
    });

  py::class_<SamplingParams>(m, "SamplingParams", "Per-request sampling knobs resolved from a GenerateConfig.")
    .def(py::init([](double temperature, double topP, std::optional<int> topK, std::optional<uint64_t> seed) {
           SamplingParams p;
           p.temperature = temperature;
           p.topP        = topP;
           p.topK        = topK;
           p.seed        = seed;
           return p;
         }),
         py::arg("temperature"),
         py::arg("top_p"),
         py::arg("top_k") = py::none(),
         py::arg("seed")  = py::none())
    .def_readwrite("temperature", &SamplingParams::temperature)
    .def_readwrite("top_p", &SamplingParams::topP)
    .def_readwrite("top_k", &SamplingParams::topK)
    .def_readwrite("seed", &SamplingParams::seed)
    .def(py::self == py::self)
    .def("__repr__", [](const SamplingParams &p) { return show(p); });

  py::class_<KvAllocation>(m, "KvAllocation", "Paged KV-cache allocation assigned to one request.")
    .def(py::init([](std::string requestId, std::string modelId, std::vector<int> pageIds, int tokensCapacity, int tokensUsed) {
           KvAllocation a;
           a.requestId      = std::move(requestId);
           a.modelId        = std::move(modelId);
           a.pageIds        = std::move(pageIds);
           a.tokensCapacity = tokensCapacity;
           a.tokensUsed     = tokensUsed;
           return a;
         }),
         py::arg("request_id"),
         py::arg("model_id"),
         py::arg("page_ids"),
         py::arg("tokens_capacity"),
         py::arg("tokens_used") = 0)
    .def_readwrite("request_id", &KvAllocation::requestId)
    .def_readwrite("model_id", &KvAllocation::modelId)
    .def_readwrite("page_ids", &KvAllocation::pageIds)
    .def_readwrite("tokens_capacity", &KvAllocation::tokensCapacity)
    .def_readwrite("tokens_used", &KvAllocation::tokensUsed)
    .def(py::self == py::self)
    .def("__repr__", [](const KvAllocation &a) { return show(a); });

  py::class_<RequestState>(m, "RequestState", "Mutable per-request state carried alongside a batch.")
    .def(py::init([](std::string                   requestId,
                     std::string                   modelId,
                     std::string                   prompt,
                     std::vector<int>              promptTokenIds,
                     std::vector<int>              generatedTokenIds,
                     std::optional<SamplingParams> samplingParams,
                     const std::string            &status,
                     int                           maxNewTokens,
                     std::vector<std::string>      stopStrings,
                     std::optional<int>            eosTokenId,
                     int                           seqLen,
                     int                           numPromptTokens,
                     std::optional<KvAllocation>   kvAllocation,
                     std::string                   outputText) {
           RequestState s;
           s.requestId         = std::move(requestId);
           s.modelId           = std::move(modelId);
           s.prompt            = std::move(prompt);
           s.promptTokenIds    = std::move(promptTokenIds);
           s.generatedTokenIds = std::move(generatedTokenIds);
           s.samplingParams    = std::move(samplingParams);
           s.status            = statusFromName(status);
           s.maxNewTokens      = maxNewTokens;
           s.stopStrings       = std::move(stopStrings);
           s.eosTokenId        = eosTokenId;
           s.seqLen            = seqLen;
           s.numPromptTokens   = numPromptTokens;
           s.kvAllocation      = std::move(kvAllocation);
           s.outputText        = std::move(outputText);
           return s;
         }),
         py::arg("request_id"),
         py::arg("model_id"),
         py::arg("prompt"),
         py::arg("prompt_token_ids"),
         py::arg("generated_token_ids") = std::vector<int>{},
         py::arg("sampling_params")     = py::none(),
         py::arg("status")              = "waiting",
         py::arg("max_new_tokens")      = 0,
         py::arg("stop_strings")        = std::vector<std::string>{},
         py::arg("eos_token_id")        = py::none(),
         py::arg("seq_len")             = 0,
         py::arg("num_prompt_tokens")   = 0,
         py::arg("kv_allocation")       = py::none(),
         py::arg("output_text")         = "")
    .def_readwrite("request_id", &RequestState::requestId)
    .def_readwrite("model_id", &RequestState::modelId)
    .def_readwrite("prompt", &RequestState::prompt)
    .def_readwrite("prompt_token_ids", &RequestState::promptTokenIds)
    .def_readwrite("generated_token_ids", &RequestState::generatedTokenIds)
    .def_readwrite("sampling_params", &RequestState::samplingParams)
    // A string on the Python side (a Literal of six names); the enum stays C++.
    .def_property(
      "status", [](const RequestState &s) { return std::string(statusName(s.status)); }, [](RequestState &s, const std::string &name) { s.status = statusFromName(name); })
    .def_readwrite("max_new_tokens", &RequestState::maxNewTokens)
    .def_readwrite("stop_strings", &RequestState::stopStrings)
    .def_readwrite("eos_token_id", &RequestState::eosTokenId)
    .def_readwrite("seq_len", &RequestState::seqLen)
    .def_readwrite("num_prompt_tokens", &RequestState::numPromptTokens)
    .def_readwrite("kv_allocation", &RequestState::kvAllocation)
    .def_readwrite("output_text", &RequestState::outputText)
    .def(py::self == py::self)
    .def("__repr__", [](const RequestState &s) {
      return Repr("RequestState")
        .field("request_id", s.requestId)
        .field("model_id", s.modelId)
        .field("status", std::string(statusName(s.status)))
        .field("seq_len", s.seqLen)
        .field("num_prompt_tokens", s.numPromptTokens)
        .field("generated_token_ids", s.generatedTokenIds)
        .str();
    });

  py::class_<GenerateResult>(m, "GenerateResult", "Final text, generated IDs, and stop reason for one request.")
    .def(py::init([](std::string text, std::vector<int> tokenIds, std::string finishReason) {
           GenerateResult r;
           r.text         = std::move(text);
           r.tokenIds     = std::move(tokenIds);
           r.finishReason = std::move(finishReason);
           return r;
         }),
         py::arg("text"),
         py::arg("token_ids"),
         py::arg("finish_reason"))
    .def_readwrite("text", &GenerateResult::text)
    .def_readwrite("token_ids", &GenerateResult::tokenIds)
    .def_readwrite("finish_reason", &GenerateResult::finishReason)
    .def(py::self == py::self)
    .def("__repr__",
         [](const GenerateResult &r) { return Repr("GenerateResult").field("text", r.text).field("token_ids", r.tokenIds).field("finish_reason", r.finishReason).str(); });

  m.attr("PREFILL_CHUNK_SIZE_CHOICES") = py::cast(std::vector<int>(PREFILL_CHUNK_SIZE_CHOICES.begin(), PREFILL_CHUNK_SIZE_CHOICES.end()));
}

} // namespace serving::bindings
