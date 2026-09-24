#include <serving/coordinator/rule_message.hpp>

#include <cstring>

namespace serving::coordinator
{

namespace
{

/// Little-endian on the wire, written byte by byte so the frame does not depend
/// on the host's endianness or on struct padding.
void putU64(std::vector<uint8_t> &out, uint64_t value)
{
  for (int i = 0; i < 8; ++i) { out.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFF)); }
}

void putString(std::vector<uint8_t> &out, const std::string &value)
{
  putU64(out, value.size());
  out.insert(out.end(), value.begin(), value.end());
}

/// Every read is bounds-checked against the frame it was given: a truncated or
/// hostile frame must be rejected, never read past.
bool takeU64(const uint8_t *data, size_t size, size_t &offset, uint64_t &value)
{
  if (offset + 8 > size) { return false; }
  value = 0;
  for (int i = 0; i < 8; ++i) { value |= static_cast<uint64_t>(data[offset + static_cast<size_t>(i)]) << (i * 8); }
  offset += 8;
  return true;
}

bool takeString(const uint8_t *data, size_t size, size_t &offset, std::string &value)
{
  uint64_t length = 0;
  if (!takeU64(data, size, offset, length)) { return false; }
  // Checked before the addition so a length near UINT64_MAX cannot wrap the sum
  // and slip past the bound.
  if (length > size - offset) { return false; }
  value.assign(reinterpret_cast<const char *>(data + offset), static_cast<size_t>(length));
  offset += static_cast<size_t>(length);
  return true;
}

} // namespace

std::vector<uint8_t> encodeRulePublish(const RulePublish &rule)
{
  std::vector<uint8_t> out;
  putU64(out, rule.generation);
  putString(out, rule.sessionId);
  putString(out, rule.nextHopName);
  putU64(out, rule.nextHopInstance);
  return out;
}

std::optional<RulePublish> decodeRulePublish(const uint8_t *data, size_t size)
{
  if (data == nullptr) { return std::nullopt; }
  RulePublish rule;
  size_t      offset = 0;
  if (!takeU64(data, size, offset, rule.generation)) { return std::nullopt; }
  if (!takeString(data, size, offset, rule.sessionId)) { return std::nullopt; }
  if (!takeString(data, size, offset, rule.nextHopName)) { return std::nullopt; }
  if (!takeU64(data, size, offset, rule.nextHopInstance)) { return std::nullopt; }
  // Trailing bytes mean the frame is not what this version expects.
  if (offset != size) { return std::nullopt; }
  return rule;
}

std::vector<uint8_t> encodeRuleRevoke(const RuleRevoke &revoke)
{
  std::vector<uint8_t> out;
  putU64(out, revoke.generation);
  putString(out, revoke.sessionId);
  return out;
}

std::optional<RuleRevoke> decodeRuleRevoke(const uint8_t *data, size_t size)
{
  if (data == nullptr) { return std::nullopt; }
  RuleRevoke revoke;
  size_t     offset = 0;
  if (!takeU64(data, size, offset, revoke.generation)) { return std::nullopt; }
  if (!takeString(data, size, offset, revoke.sessionId)) { return std::nullopt; }
  if (offset != size) { return std::nullopt; }
  return revoke;
}

} // namespace serving::coordinator
