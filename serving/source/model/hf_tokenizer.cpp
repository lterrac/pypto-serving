#include <serving/model/hf_tokenizer.hpp>

#include <serving/model/model_files.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <tokenizers_cpp.h>

namespace serving::model
{

namespace
{} // namespace

HfTokenizer::HfTokenizer(std::unique_ptr<tokenizers::Tokenizer> tokenizer, std::optional<int> bos, std::optional<int> eos, std::unordered_set<int> specialIds)
  : _tokenizer(std::move(tokenizer)),
    _bos(bos),
    _eos(eos),
    _specialIds(std::move(specialIds))
{}

HfTokenizer::~HfTokenizer() = default;

std::unique_ptr<HfTokenizer> HfTokenizer::fromModelDir(const std::string &modelDir)
{
  const auto tokenizerJson = readFile(modelDir + "/tokenizer.json");
  if (!tokenizerJson.has_value()) { throw std::runtime_error("cannot read tokenizer.json in " + modelDir); }

  auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(*tokenizerJson);
  if (tokenizer == nullptr) { throw std::runtime_error("failed to parse tokenizer.json in " + modelDir); }

  // `added_tokens` is the authoritative id <-> content table for special tokens;
  // the Python loader rebuilds its specials from the same place.
  std::unordered_set<int>    specialIds;
  std::map<std::string, int> contentToId;
  try
  {
    const auto parsed = nlohmann::json::parse(*tokenizerJson);
    if (parsed.contains("added_tokens"))
    {
      for (const auto &entry : parsed.at("added_tokens"))
      {
        if (!entry.contains("id") || !entry.contains("content")) { continue; }
        const int         id      = entry.at("id").get<int>();
        const std::string content = entry.at("content").get<std::string>();
        contentToId[content]      = id;
        if (entry.value("special", false)) { specialIds.insert(id); }
      }
    }
  }
  catch (const std::exception &)
  {
    // Without the table there are no special ids to skip; encoding and decoding
    // still work, so this is not fatal.
  }

  std::optional<int> bos;
  std::optional<int> eos;
  try
  {
    // Absent or malformed leaves bos/eos unset, which is what the catch below
    // is for: a tokenizer without them still works.
    const auto configText = readFile(modelDir + "/tokenizer_config.json");
    const auto config     = nlohmann::json::parse(configText.value_or(""));
    const auto bosContent = specialTokenContent(config, "bos_token");
    const auto eosContent = specialTokenContent(config, "eos_token");
    if (const auto it = contentToId.find(bosContent); !bosContent.empty() && it != contentToId.end()) { bos = it->second; }
    if (const auto it = contentToId.find(eosContent); !eosContent.empty() && it != contentToId.end()) { eos = it->second; }
  }
  catch (const std::exception &)
  {
    // A missing tokenizer_config.json costs the bos/eos ids, never the tokenizer.
  }

  return std::make_unique<HfTokenizer>(std::move(tokenizer), bos, eos, std::move(specialIds));
}

std::vector<int> HfTokenizer::encode(const std::string &text) const { return _tokenizer->Encode(text); }

std::string HfTokenizer::decode(const std::vector<int> &tokenIds, bool skipSpecialTokens) const
{
  // tokenizers-cpp's public `Decode(ids)` hard-codes skip_special_tokens=FALSE
  // (huggingface_tokenizer.cc: `Decode(ids) final { return Decode(ids, false); }`),
  // and the two-argument form is not on the installed interface. The Python
  // engine decodes with skip_special_tokens=True, so left alone every streamed
  // chunk would carry <|im_end|> and friends. Dropping the ids before decoding
  // gives the same result, and `all_special_ids` is the Python adapter's own
  // answer to the same question.
  if (!skipSpecialTokens || _specialIds.empty()) { return _tokenizer->Decode(tokenIds); }

  std::vector<int> filtered;
  filtered.reserve(tokenIds.size());
  for (const int id : tokenIds)
  {
    if (_specialIds.find(id) == _specialIds.end()) { filtered.push_back(id); }
  }
  return _tokenizer->Decode(filtered);
}

size_t HfTokenizer::vocabSize() const { return _tokenizer->GetVocabSize(); }

bool HfTokenizer::isSpecial(int tokenId) const { return _specialIds.find(tokenId) != _specialIds.end(); }

} // namespace serving::model
