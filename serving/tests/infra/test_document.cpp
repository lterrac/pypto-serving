#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include <serving/infra/document.hpp>

using serving::infra::Channel;
using serving::infra::DocumentError;
using serving::infra::loadDocument;
using serving::infra::Node;
using serving::infra::Phase;
using serving::infra::readDocument;
using serving::infra::Replica;
using serving::infra::saveDocument;
using serving::infra::StateMap;
using serving::infra::writeDocument;

namespace
{

/// A deployment with everything a document can carry.
StateMap populated()
{
  StateMap map;
  map.addNode(Node{"hg-atlas-01", "192.168.150.11", {0, 1, 2, 3, 4, 5, 6, 7}});
  map.addNode(Node{"hg-atlas-02", "192.168.150.12", {0, 1}});

  Replica a;
  a.name         = "qwen-0";
  a.partition    = 0;
  a.node         = "hg-atlas-01";
  a.devices      = {4};
  a.port         = 8000;
  a.instance     = 10;
  a.phase        = Phase::Active;
  a.desiredPhase = Phase::Active;
  map.addReplica(a);

  Replica b;
  b.name         = "qwen-1";
  b.partition    = 1;
  b.node         = "hg-atlas-02";
  b.devices      = {0};
  b.port         = 8001;
  b.scheme       = "https";
  b.phase        = Phase::Draining;
  b.desiredPhase = Phase::Dead;
  map.addReplica(b);

  map.addChannel(Channel{"qwen-0", "qwen-1"});
  return map;
}

/// A scratch path that cleans itself up.
class TempFile
{
  public:

  TempFile()
    : _path(std::filesystem::temp_directory_path() / ("serving-doc-" + std::to_string(::getpid()) + "-" + std::to_string(++_seq) + ".json"))
  {}

  ~TempFile()
  {
    std::error_code ec;
    std::filesystem::remove(_path, ec);
    std::filesystem::remove(std::filesystem::path(_path).replace_extension(".json.tmp"), ec);
  }

  [[nodiscard]] const std::filesystem::path &path() const { return _path; }

  private:

  std::filesystem::path _path;
  static inline int     _seq = 0;
};

} // namespace

TEST(DocumentTest, RoundTripsEverythingTheMapHolds)
{
  const StateMap original = populated();
  const StateMap reloaded = readDocument(writeDocument(original));

  EXPECT_EQ(reloaded.nodes(), original.nodes());
  EXPECT_EQ(reloaded.replicas(), original.replicas());
  EXPECT_EQ(reloaded.channels(), original.channels());
}

TEST(DocumentTest, KeepsTheVersionRatherThanCountingTheLoad)
{
  const StateMap original = populated();
  const StateMap reloaded = readDocument(writeDocument(original));

  // Building a map replica by replica bumps the version once per edit, so a
  // naive load would report a version counting the load and every reader that
  // cached one would think the deployment changed.
  EXPECT_EQ(reloaded.version(), original.version());
}

TEST(DocumentTest, WritesTheSameBytesForTheSameMap) { EXPECT_EQ(writeDocument(populated()), writeDocument(populated())); }

TEST(DocumentTest, CarriesBothPhasesSoSpecAndStatusStayTogether)
{
  const StateMap reloaded = readDocument(writeDocument(populated()));
  const Replica *draining = reloaded.replica("qwen-1");

  ASSERT_NE(draining, nullptr);
  EXPECT_EQ(draining->phase, Phase::Draining);
  EXPECT_EQ(draining->desiredPhase, Phase::Dead);
  // Which is exactly the gap reconcile() exists to report.
  ASSERT_EQ(reconcile(reloaded).size(), 1U);
  EXPECT_EQ(reconcile(reloaded)[0].replica, "qwen-1");
}

TEST(DocumentTest, AnEmptyDocumentIsAnEmptyDeployment)
{
  const StateMap map = readDocument("{}");
  EXPECT_TRUE(map.nodes().empty());
  EXPECT_TRUE(map.replicas().empty());
}

TEST(DocumentTest, DefaultsAReplicaToPendingAndWanted)
{
  const StateMap map = readDocument(R"({"replicas": [{"name": "r"}]})");
  const Replica *r   = map.replica("r");

  ASSERT_NE(r, nullptr);
  // Not observed yet, but in the document because it is wanted.
  EXPECT_EQ(r->phase, Phase::Pending);
  EXPECT_EQ(r->desiredPhase, Phase::Active);
  EXPECT_EQ(r->scheme, "http");
}

TEST(DocumentTest, RejectsWhatItCannotRead)
{
  EXPECT_THROW((void)readDocument("not json"), DocumentError);
  EXPECT_THROW((void)readDocument("[]"), DocumentError);
  // A silently half-read document is how a deployment routes somewhere nobody
  // intended, so an unknown key is an error rather than a shrug.
  EXPECT_THROW((void)readDocument(R"({"replicaz": []})"), DocumentError);
  EXPECT_THROW((void)readDocument(R"({"replicas": [{"name": "r", "colour": "red"}]})"), DocumentError);
  EXPECT_THROW((void)readDocument(R"({"replicas": [{}]})"), DocumentError);
  EXPECT_THROW((void)readDocument(R"({"replicas": [{"name": "r", "phase": "zombie"}]})"), DocumentError);
  EXPECT_THROW((void)readDocument(R"({"nodes": [{"name": "n", "devices": "all"}]})"), DocumentError);
  EXPECT_THROW((void)readDocument(R"({"format": 99})"), DocumentError);
}

TEST(DocumentTest, ErrorsNameWhereTheyHappened)
{
  try
  {
    (void)readDocument(R"({"replicas": [{"name": "a"}, {"name": "b", "oops": 1}]})", "deployment.json");
    FAIL() << "expected a DocumentError";
  }
  catch (const DocumentError &e)
  {
    const std::string message = e.what();
    EXPECT_NE(message.find("deployment.json"), std::string::npos) << message;
    EXPECT_NE(message.find("replica 1"), std::string::npos) << message;
    EXPECT_NE(message.find("oops"), std::string::npos) << message;
  }
}

TEST(DocumentTest, SurvivesAFileRoundTrip)
{
  const TempFile file;
  const StateMap original = populated();

  saveDocument(original, file.path());
  ASSERT_TRUE(std::filesystem::exists(file.path()));

  const StateMap reloaded = loadDocument(file.path());
  EXPECT_EQ(reloaded.replicas(), original.replicas());
  EXPECT_EQ(reloaded.version(), original.version());
}

TEST(DocumentTest, LeavesNoTemporaryBehind)
{
  const TempFile file;
  saveDocument(populated(), file.path());

  // A reader polling the document must never catch it half-written, which is
  // why it is renamed into place; the temporary must not survive either.
  const auto temporary = std::filesystem::path(file.path()).replace_extension(".json.tmp");
  EXPECT_FALSE(std::filesystem::exists(temporary));
}

TEST(DocumentTest, OverwritesAnExistingDocumentWhole)
{
  const TempFile file;
  saveDocument(populated(), file.path());

  StateMap smaller;
  smaller.addNode(Node{"solo", "127.0.0.1", {0}});
  saveDocument(smaller, file.path());

  const StateMap reloaded = loadDocument(file.path());
  EXPECT_TRUE(reloaded.replicas().empty());
  EXPECT_EQ(reloaded.nodes().size(), 1U);
}

TEST(DocumentTest, AMissingFileSaysSoRatherThanReturningAnEmptyDeployment) { EXPECT_THROW((void)loadDocument("/no/such/deployment.json"), DocumentError); }
