#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include <serving/model/chat_template.hpp>

namespace
{

/**
 * Golden fixture captured from the Python stack. Regenerate with
 * `serving/tests/fixtures/README.md`'s recipe when a model's template changes.
 */
nlohmann::ordered_json loadFixture()
{
  const std::string path = std::string(SERVING_TEST_FIXTURE_DIR) + "/qwen3_chat_template.json";
  std::ifstream     in(path);
  EXPECT_TRUE(in.good()) << "missing fixture: " << path;
  nlohmann::ordered_json fixture;
  in >> fixture;
  return fixture;
}

} // namespace

TEST(Qwen3ChatTemplateTest, RendersByteIdenticallyToTransformers)
{
  const auto fixture = loadFixture();

  serving::model::ChatTemplate chatTemplate(fixture.at("chat_template").get<std::string>(), "", "<|im_end|>");

  for (const auto &testCase : fixture.at("cases"))
  {
    const auto name   = testCase.at("name").get<std::string>();
    const auto kwargs = testCase.at("kwargs");
    const bool addGen = kwargs.contains("add_generation_prompt") ? kwargs.at("add_generation_prompt").get<bool>() : true;

    nlohmann::ordered_json extraContext = nlohmann::ordered_json::object();
    for (const auto &entry : kwargs.items())
    {
      if (entry.key() != "add_generation_prompt") { extraContext[entry.key()] = entry.value(); }
    }

    const auto rendered = chatTemplate.apply(testCase.at("messages"), addGen, extraContext);
    EXPECT_EQ(rendered, testCase.at("expected").get<std::string>()) << "case: " << name;
  }
}

TEST(Qwen3ChatTemplateTest, FixtureCoversTheCasesWeCareAbout)
{
  const auto fixture = loadFixture();
  EXPECT_EQ(fixture.at("chat_template").get<std::string>().size(), 4168u);
  EXPECT_GE(fixture.at("cases").size(), 6u);
}
