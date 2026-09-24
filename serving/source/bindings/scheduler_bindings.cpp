#include <serving/bindings/bindings.hpp>

#include <optional>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include <serving/sched/scheduler.hpp>

namespace py = pybind11;

namespace serving::bindings
{

void bindScheduler(py::module_ &m)
{
  py::class_<sched::SchedulerConfig>(m, "SchedulerConfig")
    .def(py::init([](int                maxNumRunningReqs,
                     int                maxNumScheduledTokens,
                     int                longPrefillTokenThreshold,
                     std::optional<int> maxPrefillTokensPerRequest,
                     std::vector<int>   prefillChunkSizeChoices,
                     int                maxSeqLen,
                     bool               enablePrefixCache,
                     bool               enableChunkPrefill,
                     int                numSpeculativeTokens,
                     bool               supportsChunkedPrefillWithSpeculation,
                     bool               requiresHomogeneousPrefillDecode) {
           sched::SchedulerConfig c;
           c.maxNumRunningReqs                     = maxNumRunningReqs;
           c.maxNumScheduledTokens                 = maxNumScheduledTokens;
           c.longPrefillTokenThreshold             = longPrefillTokenThreshold;
           c.maxPrefillTokensPerRequest            = maxPrefillTokensPerRequest;
           c.prefillChunkSizeChoices               = std::move(prefillChunkSizeChoices);
           c.maxSeqLen                             = maxSeqLen;
           c.enablePrefixCache                     = enablePrefixCache;
           c.enableChunkPrefill                    = enableChunkPrefill;
           c.numSpeculativeTokens                  = numSpeculativeTokens;
           c.supportsChunkedPrefillWithSpeculation = supportsChunkedPrefillWithSpeculation;
           c.requiresHomogeneousPrefillDecode      = requiresHomogeneousPrefillDecode;
           return c;
         }),
         py::arg("max_num_running_reqs")                      = 32,
         py::arg("max_num_scheduled_tokens")                  = 4096,
         py::arg("long_prefill_token_threshold")              = 2048,
         py::arg("max_prefill_tokens_per_request")            = py::none(),
         py::arg("prefill_chunk_size_choices")                = std::vector<int>{},
         py::arg("max_seq_len")                               = 4096,
         py::arg("enable_prefix_cache")                       = true,
         py::arg("enable_chunk_prefill")                      = true,
         py::arg("num_speculative_tokens")                    = 0,
         py::arg("supports_chunked_prefill_with_speculation") = true,
         py::arg("requires_homogeneous_prefill_decode")       = false)
    .def_readwrite("max_num_running_reqs", &sched::SchedulerConfig::maxNumRunningReqs)
    .def_readwrite("max_num_scheduled_tokens", &sched::SchedulerConfig::maxNumScheduledTokens)
    .def_readwrite("long_prefill_token_threshold", &sched::SchedulerConfig::longPrefillTokenThreshold)
    .def_readwrite("max_prefill_tokens_per_request", &sched::SchedulerConfig::maxPrefillTokensPerRequest)
    .def_readwrite("prefill_chunk_size_choices", &sched::SchedulerConfig::prefillChunkSizeChoices)
    .def_readwrite("max_seq_len", &sched::SchedulerConfig::maxSeqLen)
    .def_readwrite("enable_prefix_cache", &sched::SchedulerConfig::enablePrefixCache)
    .def_readwrite("enable_chunk_prefill", &sched::SchedulerConfig::enableChunkPrefill)
    .def_readwrite("num_speculative_tokens", &sched::SchedulerConfig::numSpeculativeTokens)
    .def_readwrite("supports_chunked_prefill_with_speculation", &sched::SchedulerConfig::supportsChunkedPrefillWithSpeculation)
    .def_readwrite("requires_homogeneous_prefill_decode", &sched::SchedulerConfig::requiresHomogeneousPrefillDecode);
}

} // namespace serving::bindings
