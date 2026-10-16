/**
 * Starting a replica on another node.
 *
 * The reach command is a template, so these drive the remote path with a local
 * shell standing in for ssh: everything the remote path does differently --
 * composing the command, recording the group on the far side, signalling it
 * there before tearing the reach down -- runs for real. What a local shell
 * cannot stand in for is ssh itself, and nothing here claims it does.
 */

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include <csignal>
#include <unistd.h>

#include <gtest/gtest.h>

#include <serving/infra/remote_starter.hpp>

using serving::infra::Node;
using serving::infra::Phase;
using serving::infra::RemoteLaunch;
using serving::infra::RemoteStarter;
using serving::infra::Replica;
using serving::infra::StateMap;

namespace
{

StateMap twoNodes()
{
  StateMap map;
  map.addNode(Node{"here", "127.0.0.1", {0, 1}});
  map.addNode(Node{"there", "10.9.9.9", {0, 1}});
  return map;
}

Replica on(const std::string &node, const std::string &name = "qwen-0", int device = 0, int port = 8000)
{
  Replica replica;
  replica.name         = name;
  replica.node         = node;
  replica.devices      = {device};
  replica.port         = port;
  replica.phase        = Phase::Pending;
  replica.desiredPhase = Phase::Active;
  return replica;
}

/// A reach that runs the command here instead of on another machine.
RemoteLaunch locally(const std::filesystem::path &directory)
{
  RemoteLaunch launch;
  launch.reach          = "sh -c {command}";
  launch.stateDirectory = directory.string();
  launch.graceSeconds   = 1;
  return launch;
}

std::filesystem::path scratch()
{
  const auto directory = std::filesystem::temp_directory_path() / ("remote-starter-" + std::to_string(::getpid()));
  std::filesystem::create_directories(directory);
  return directory;
}

} // namespace

TEST(RemoteStarterTest, AReplicaOnThisNodeIsStartedHereWithNoReachAtAll)
{
  const StateMap      map = twoNodes();
  const RemoteStarter starter(map, "here", "run {name} on {device}");
  EXPECT_TRUE(starter.isLocal(on("here")));
  EXPECT_EQ(starter.commandFor(on("here")), "run qwen-0 on 0");
}

TEST(RemoteStarterTest, AReplicaElsewhereIsReachedAtItsNodesAddressNotItsName)
{
  const StateMap      map = twoNodes();
  const RemoteStarter starter(map, "here", "run {name} on {device}");
  const auto          command = starter.commandFor(on("there"));

  EXPECT_FALSE(starter.isLocal(on("there")));
  // The address, because a node name is not one -- the same fault the health
  // probe had.
  EXPECT_NE(command.find("10.9.9.9"), std::string::npos) << command;
  EXPECT_EQ(command.find(" there "), std::string::npos) << command;
  EXPECT_NE(command.find("ssh"), std::string::npos) << command;
  // Attached, not detached: the reach has to last as long as the replica or
  // its liveness means nothing and the supervisor retries over the top of a
  // replica that is merely still loading.
  EXPECT_EQ(command.find("setsid"), std::string::npos) << command;
  EXPECT_NE(command.find("serving-qwen-0.pgid"), std::string::npos) << command;
  EXPECT_NE(command.find("exec run qwen-0 on 0"), std::string::npos) << command;
}

TEST(RemoteStarterTest, EveryReplicaIsRemoteWhenThisProcessRunsNone)
{
  const StateMap      map = twoNodes();
  const RemoteStarter starter(map, "", "run {name}");
  EXPECT_FALSE(starter.isLocal(on("here")));
  EXPECT_NE(starter.commandFor(on("here")).find("127.0.0.1"), std::string::npos);
}

TEST(RemoteStarterTest, AnArgumentWithASpaceSurvivesTheHop)
{
  const StateMap      map = twoNodes();
  const RemoteStarter starter(map, "here", "serve --generate-config '{\"max_new_tokens\": 5}' --name {name}");
  const auto          command = starter.commandFor(on("there"));
  // Quoted rather than split: the far side must receive one argument, and an
  // embedded quote must not end the quoting early.
  EXPECT_NE(command.find("max_new_tokens"), std::string::npos) << command;
  EXPECT_NE(command.find("'\\''"), std::string::npos) << command << "\nembedded quotes must be broken out";
}

TEST(RemoteStarterTest, RecordsTheReplicaOnTheFarSideAndStoppingReachesIt)
{
  const auto directory = scratch();
  const auto marker    = directory / "alive";

  StateMap      map = twoNodes();
  RemoteStarter starter(map, "here", "sh -c 'while true; do touch " + marker.string() + "; sleep 0.2; done'", locally(directory));

  const Replica replica = on("there");
  ASSERT_TRUE(starter.start(replica));

  const auto pidFile  = directory / "serving-qwen-0.pgid";
  bool       recorded = false;
  for (int waited = 0; waited < 50 && !recorded; ++waited)
  {
    recorded = std::filesystem::exists(pidFile);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  ASSERT_TRUE(recorded) << "the far side has to record what it started, or nothing can stop it";
  EXPECT_TRUE(starter.running("qwen-0"));

  int recordedPid = 0;
  {
    std::ifstream file(pidFile);
    file >> recordedPid;
  }
  ASSERT_GT(recordedPid, 0);
  // The replica itself, not a wrapper: `$$` is written before `exec`, so the
  // shell that recorded it is the process that is now serving.
  EXPECT_EQ(::kill(recordedPid, 0), 0) << "the recorded process must exist";

  starter.stop(replica);

  // Gone on the far side, not merely unreferenced here.
  bool reaped = false;
  for (int waited = 0; waited < 60 && !reaped; ++waited)
  {
    reaped = ::kill(recordedPid, 0) != 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  EXPECT_TRUE(reaped) << "stopping a remote replica must reach the replica, not just the reach command";
  EXPECT_FALSE(starter.running("qwen-0"));

  std::filesystem::remove_all(directory);
}

TEST(RemoteStarterTest, RefusesASecondStartSoAReplicaCannotBeLostTrackOf)
{
  const auto    directory = scratch();
  StateMap      map       = twoNodes();
  RemoteStarter starter(map, "here", "sleep 30", locally(directory));

  const Replica replica = on("there");
  ASSERT_TRUE(starter.start(replica));
  EXPECT_FALSE(starter.start(replica)) << "a second start would lose the first and leave it holding a card";

  starter.stop(replica);
  std::filesystem::remove_all(directory);
}

TEST(RemoteStarterTest, TheReachOutlivesItsOwnLaunchSoLivenessStillMeansSomething)
{
  // Found on two cards. The remote command was wrapped in `setsid`, which
  // returns as soon as it has forked, so the reach exited at once. The
  // supervisor read that as a failed launch, retried, and started a second
  // replica on the same card while the first was still loading -- then leaked
  // both, because only the newer one was still referenced.
  //
  // A replica takes minutes to load. The reach has to still be there.
  const auto    directory = scratch();
  StateMap      map       = twoNodes();
  RemoteStarter starter(map, "here", "sleep 30", locally(directory));

  const Replica replica = on("there");
  ASSERT_TRUE(starter.start(replica));

  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  EXPECT_TRUE(starter.running("qwen-0")) << "the reach exited while the replica was still running, so the supervisor will start another";
  EXPECT_FALSE(starter.start(replica)) << "and would have been allowed to start a second one";

  starter.stop(replica);
  std::filesystem::remove_all(directory);
}
