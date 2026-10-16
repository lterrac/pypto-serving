# The control plane

`scaling-pypto-serving.pptx` describes a control plane that runs alongside
serving and never inside it: it starts processes, offers the fabric for
inter-node channels, and holds what the infrastructure looks like. It takes no
decisions — it never schedules a request, manages KV, or decides anything about
the model — and it stays off the per-token path.

Four mechanisms, and what exists of each.

| | | what exists |
| --- | --- | --- |
| **M1** | partitioning: describe the app as partitions on separate nodes, joined by channels | a path of replicas, and forwarding rules reaching them over HTTP or channels; nothing **executes** a split model |
| **M2** | elastic capacity: wire new nodes in without a restart | local and CloudR capacity, a launcher that starts and supervises replicas, warm spares, a scaling policy the operator triggers |
| **M3** | failure recovery: deal with dead or unreachable nodes | detection in ~4 s, restart with backoff, in-flight requests re-sent to a sibling |
| **M4** | infrastructure state map: one versioned map of what runs where | the map, the on-disk document, and the deployer's RPC to hand out copies |

M4 comes first because the deck makes it the precondition for the others: a
scaling policy, a cache-aware router and a recovery procedure all read it.

## The map

`serving/infra/state.hpp`. Plain data with no HiCR in it, the way
`coordinator/module.hpp` is: a test, the router and a single-node deployment can
all hold one without a control plane underneath, and the platform can host and
distribute it later without the type changing.

A **node** owns devices. Serving has treated a device as a bare integer with no
way to say which host it belongs to, which is why a multi-node deployment could
not be described at all. A device here is that integer plus its node, which is
what makes *does this partition fit that node* answerable.

A **replica** names its node, the devices it holds, its partition and endpoint,
and two phases: the one it is observed in and the one the deployment wants.

| phase | holds its devices | |
| --- | --- | --- |
| `pending` | no | wanted, not started |
| `active` | yes | |
| `draining` | yes | still serving what it has |
| `dead` | no | lost rather than drained; its capacity can be rebuilt on that card |

Every edit moves `version()`, so a reader that cached a decision can tell the
ground shifted without diffing the document.

### Validation

`validate()` returns one message per problem, empty when the deployment is
coherent: a replica on a node that is not in the map, a replica claiming a card
its node does not own, two live replicas on the same card, duplicate names or
endpoints, a pipeline with a hole, a channel to nowhere. This is slide 6's
question answered before anything starts rather than after it fails.

### Declarative

Per slide 12, serving writes what it wants and something else closes the gap.
`reconcile()` reports that gap as `start` / `drain` / `stop` per replica.
**Nothing executes those actions** — that is the launcher, and there is no
launcher. A dead replica whose desired phase is still `active` reconciles to
`start`, not to a restart in place: the deployment asks for the capacity back
and lets the launcher decide where it lands.

## Where a replica's identity comes from

A node is named by configuration, not discovered: inside a container the
hostname is the container's id. Its cards are discovered, by reading the
`davinci<N>` character devices the driver creates (`infra/discovery.hpp`);
`davinci_manager` and the other control nodes are skipped, and a machine with
none is a dev box rather than an error.

The map records the **physical** card. The task broker pins the card it granted
through `ASCEND_RT_VISIBLE_DEVICES`, so inside the job the device is always
logical 0; `TASK_PHYS_DEVICE` carries the real number. Recording the logical one
would put every replica on the node on card 0.

`Runtime.placement()` returns this process as a map entry. Verified on
hg-atlas-01 with Qwen3-14B: eight cards discovered, the replica recorded on
card 4 while running on logical 0, `validate()` coherent, a second replica on
card 4 reported as a collision, golden tokens unchanged, and the card freed when
the replica is marked dead.

## Routing reads it

The coordinator holds a map (`stateMap()`) and `service()` brings routing to it
whenever `version()` has moved. A quiet tick costs one integer comparison, which
is what the version is for.

| phase in the map | routing |
| --- | --- |
| `active` | routable |
| `draining` | kept, not routable, finishing what it has |
| `pending` | kept, not routable: it cannot serve yet |
| `dead` | taken out through the loss path, and its paths revoked |

`dead` is deliberately not a flag. It runs the same `onReplicaLost` a health
signal runs, so a replica holding a stale forwarding rule hears that the session
moved — M4 driving the recovery machinery M3 already had.

Edits are applied on the service tick, never on the writing thread: a health
signal marking a replica dead must not run routing policy where it lands, which
is the reason `reportReplicaLost` queues.

`initialize()` seeds the map from the configuration, so routing and the map
agree before anything edits either; a configured replica is `active`, since the
file is a statement that it should be serving.

A replica in a partition routing does not know about is reported in
`SyncReport::unplaceable` rather than dropped. Growing the pipeline is
repartitioning, which is M1.

## The document

`serving/infra/document.hpp`. JSON, and the thing processes agree through
rather than by calling each other: one holds both phases, so spec and status
stay in one place the way a Kubernetes object does.

```json
{ "format": 1, "version": 12,
  "nodes":    [ { "name": "hg-atlas-01", "host": "192.168.150.11", "devices": [0,1,2,3,4,5,6,7] } ],
  "replicas": [ { "name": "qwen-0", "partition": 0, "node": "hg-atlas-01", "devices": [4],
                  "port": 8000, "scheme": "http", "instance": 0,
                  "phase": "active", "desired_phase": "active" } ],
  "channels": [] }
```

Parsing is strict, like the coordinator's configuration file: an unknown key is
an error and every message names where it happened, because a document that is
silently half-read is how a deployment routes somewhere nobody intended.

Two details that are easy to get wrong and are pinned by tests:

- **The version is restored, not recounted.** Building a map replica by replica
  bumps the version once per edit, so a naive load reports a version counting
  the load, and every reader that cached one concludes the deployment changed.
- **Saving is a write to a sibling temporary and a rename.** A reader polling
  the document must never catch it half-written; rename within a directory is
  atomic, a partial write is not.

A coordinator adopts one by assigning it over `stateMap()`; the next
`service()` sees the version moved and brings routing to it. No extra API: the
document and the coordinator compose through the map they already share.

### One copy, on the deployer

A pipeline has a coordinator per partition, possibly on different machines, and
each has to know the whole deployment to plan a path across it. They do not each
keep a map and they do not replicate: the **deployer** — the node that started
the application, known to everyone — owns the one copy and answers an RPC with
it (`infra/document_rpc.hpp`, `serving.deployment.document`). One writer, one
truth, nothing to reconcile.

The handler serialises on demand rather than at registration, because the
document moves and a snapshot taken when the server was built would be stale by
the time anyone asked. A coordinator that cannot read it throws rather than
carrying on with an empty map, which would route nowhere and say so only much
later.

It goes over HiCR's RPC engine — the control plane's own fabric, so it stays off
the per-token path. Built with the `cloudr` feature, which is what brings HiCR
in; `infra/document.hpp` itself stays HiCR-free, and a single-node deployment
reads the file off disk.

`examples/document_rpc` runs it: a deployer holding a three-partition
deployment across two machines, and two coordinators that start with nothing,
ask, and end up able to plan over all three partitions.

Verified on hg-atlas-01 with Qwen3-14B, two processes and one file: the replica
writes where it landed (card 4, discovered from the broker's grant), generates
its golden tokens, then marks itself draining in the document; a second process
that never met it reads the file, finds the deployment coherent, and sees card 4
still held — until the phase goes to dead and it is free again.

## M2: elastic capacity

### Where capacity comes from

`infra/capacity.hpp`. The deck names two sources — spare workers held idle, and
a provider asked for nodes beyond the local pool — and both answer the same
question, so both are a `CapacityProvider` and a policy cannot tell them apart.

- `LocalCapacity` places on the free cards the map already knows about. That is
  the whole of elastic capacity for one machine. It tracks what it handed out
  but the caller has not yet placed, because the map does not know about a grant
  until then and two acquires in a row would otherwise get the same card.
Running out returns nothing rather than failing — a policy that asked too
eagerly should back off.

Warm capacity is not a third provider: it is a replica already loaded and
waiting, which is `Phase::Reserved` below.

### Who decides

A person. `infra/deployment_server.hpp` runs on the deployer — the node that
owns the document, and so the only one with the whole picture — and exposes
what an operator needs to decide and the lever to act:

| | |
| --- | --- |
| `GET /deployment` | what runs where, each replica's load, and anything `validate()` objects to |
| `GET /deployment/document` | the document itself |
| `GET /deployment/advice` | what the policy *would* do |
| `POST /deployment/scale` | `{"partition": N, "direction": "up"\|"down"}` |

**Advice is reported, never acted on.** Looking at the policy's opinion does
not enact it; `decideScaling` still exists and still computes, but nothing
moves until someone asks.

A replica that does not answer its load endpoint reports `null`, not zero. Zero
reads as idle and would invite an operator to drain a replica that is merely
unreachable.

Each replica publishes its own load at `GET /control/load` — ready, pending
tokens, preemptions. Not `/metrics`: upstream serves Prometheus there, and this
is three numbers for a control plane rather than an observability surface.

The operator API is HTTP because the other end is a person or a tool. Inter-node
control traffic goes over channels instead, where there is neither.

### When to scale

`infra/scaling.hpp`, and this one is serving's: the deck is explicit that the
control plane executes and decides nothing, so what counts as overloaded is
serving's to say. `decideScaling` reads and changes nothing; `applyScaling` acts,
as a separate call, so a deployment can watch the policy's opinion before
letting it run.

| signal | effect |
| --- | --- |
| mean outstanding above `scale_up_above` | add a replica |
| preemptions above `preemptions_above` | add a replica, whatever the queue says |
| mean outstanding below `scale_down_below` | drain the least loaded |

Preemptions are checked first and separately because they are the harder signal:
a preempted request has already thrown work away, which a queue depth cannot
tell you. A partition can be quiet enough to argue for scaling *down* while
thrashing its KV pool.

`validate()` refuses a configuration that would flap (`scale_up_above` not above
`scale_down_below`) or scale a partition to nothing.

Load is passed in, not read off the map. The map is a durable document whose
version readers use as a change signal; folding in a figure that moves every
step would churn that version and make it worthless. What runs where is durable,
how busy it is is not.

Two counting rules carry the weight:

- **Pending counts towards the partition.** A replica that is loading carries
  nothing, so it does not lift the load average — but it is on its way up, and
  without counting it the policy orders the same capacity on every tick until
  the first one finishes.
- **Draining does not.** It is leaving, so it neither holds the partition above
  its floor nor argues against replacing it.

A new replica is added `pending`, never `active`: it cannot serve for the
minutes its model takes to load, and routing already refuses a pending replica.
Whatever starts it marks it active when it is up.

### Starting them

`infra/launcher.hpp`. `reconcile()` says what the deployment is missing; the
supervisor acts on it. Split because deciding is cheap and idempotent while
starting a process is neither, and because the brief asks for the retry to
behave the way Kubernetes does.

A started replica is not a serving one. The supervisor leaves it `pending` —
which routing already refuses to send to — and promotes it to `active` only
when its own `/health` says it is ready. That is the brief's *route requests to
the replica only when it's ready*, enforced rather than hoped for.

**Readiness is three states, not two.** `Loading`, `Ready`, `Failed`. "Not
serving" covers two cases that look identical over HTTP — a refused connection
— but that a supervisor must treat differently: one is worth waiting minutes
for, the other will never resolve. Without the distinction a replica whose
process has already exited waits out the full ready timeout before anyone
retries, which is the difference between a slow deployment and a stuck one.
`LiveAndReadyProbe` supplies it by pairing the health check with whether the
process is still alive.

**Backoff** is `base * 2^(n-1)`, capped, with the first attempt immediate — a
first try is not a retry. After `max_attempts` the replica is marked `dead`
rather than left `pending`, because dead releases its devices and the capacity
can be rebuilt somewhere that might work.

`ProcessStarter` runs a command template per replica (`{name}`, `{node}`,
`{device}`, `{devices}`, `{port}`) through `posix_spawn` — not fork and exec,
because a supervisor has threads. It refuses to start a replica twice over its
own pid, and reaps what it kills.


M2's loop runs end to end on one node: the supervisor starts a real Qwen
replica on the granted card, holds it `pending` for the 85 s its model takes to
load, promotes it when `/health` turns 200, serves a request, and takes it down
when the deployment stops wanting it — card released, process reaped. The
failure path is exercised too: a replica that cannot start is retried on the
growing interval and marked `dead` after the third attempt.

Note the device numbers differ by design. The map records the **physical** card
the node owns; the launch command passes **logical 0**, because the broker pins
the granted card through `ASCEND_RT_VISIBLE_DEVICES`. `{device}` substitutes the
physical one, which is right for a deployment that is not behind a broker and
wrong behind one — a template under `task-submit` hardcodes `--device 0`.

`infra/cloudr_capacity.hpp` is the other source: CloudR emulates instances on
one machine, so the control plane can be shown taking delivery of a node it did
not have. A cloud grant is a **node**, not a card on an existing one — that is
the whole of "wire new nodes in without a system restart" — so `Grant` carries
the machine and `applyScaling` registers it before the replica that names it.

Behind `-Dcloudr=`; the default build links no MPI. CloudR does not pull HiCR in
when it is itself a subproject, so `serving/meson.build` supplies it from
`extern/CloudR/extern/HiCR`, the same commit `platform/` pins.

Run on both machines (`mpirun -np 4 ./cloudr-capacity`): a full local node
scales onto two emulated ones carrying real HiCR instance ids, the map stays
coherent, and running the pool out is declined rather than failed. **CloudR
needs OpenMPI** — under the images' stock MPICH it hangs with no output.

### Warm capacity

The deck's first source: *spare worker processes start at launch and sit idle;
activating one is a single message*.

The warm spare is `Phase::Reserved`: up, loaded, healthy, and deliberately not
routed to. It holds its devices because it really is holding them, the monitor
watches it because a spare that died while waiting is worse than none, and the
scaling policy counts it because ordering capacity on top of one is waste.
Activating it is one edit to its desired phase; the supervisor finds the process
already answering and changes the phase without launching anything.

Measured on hg-atlas-01 with Qwen3-14B on two cards:

| | |
| --- | ---: |
| cold start → serving | **84.0 s** |
| spare's own load, paid at launch | 84.0 s |
| **activation → serving** | **one probe, below measurement** |

So the saving is the whole model load, moved from the moment of demand to the
moment of launch. Which is the point, and which a pool of placements could
never have delivered.

The shortcut that makes it cheap — the supervisor adopting a replica whose
probe says it is already serving — trusts the probe. Behind `LiveAndReadyProbe`
that is safe, because a replica with no process reports `Failed`; behind a bare
health check a stale server at the same address would be adopted. There is a
test naming that.

## M3: failure recovery

`infra/monitor.hpp` polls the replicas that are supposed to be serving and
writes `dead` in the map when one stops answering. That is all it does —
dropping it from routing, revoking the paths through it, releasing its cards
and starting a replacement are consequences of the map saying `dead`, which is
what makes one map the source of truth rather than three components each with
an opinion.

The deck's complaint about the Python stack is the bar: a dead worker surfaced
only after a twenty-minute timeout, while the server kept accepting requests,
returned truncated answers marked as normal completions, and `/health` stayed
green. Measured on hg-atlas-01 with a replica killed from outside: **noticed in
4 s, rebuilt and serving again 88 s later**.

Failures must be consecutive. One refused connection is as likely to be a blip
as a death, and declaring a healthy replica dead costs a model load, so a
replica that answers again has its count forgotten. Only `active` replicas are
watched: a `pending` one is the supervisor's business and is expected not to
answer yet, a `draining` one is leaving under its own steam.

`dead`, not `draining`: draining is an orderly exit that keeps its devices, and
a replica that has gone is not exiting. Dead releases the cards, which is what
lets "a spare replica rebuilds the lost capacity" mean anything.

### Re-sending an in-flight request

The deck's last M3 item: *re-send its in-flight requests to another replica of
the same partition*. `router/proxy.cpp` retries a request whose replica turned
out to be unreachable, on another replica, and the session's pin moves with it.

**The retry only ever happens while nothing has reached the client.** Once the
upstream status is known the relay has begun, and a second attempt would put two
answers on one connection. That is not a heuristic to tune — it is the whole
correctness argument, and it is why the retry sits exactly where the transport
failure is detected before any header is written.

A replica that dies *during* its answer cannot be retried. The transfer is
broken rather than terminated cleanly, because a short answer framed as a whole
one is the exact fault the deck names in the Python stack; it is counted
(`truncated()`) and logged, because an operator who is never told is the other
half of that fault.

Whether the *client* notices is the client's business: httplib and curl report
the broken transfer, a urllib line-iterator silently stops. What the router
guarantees is that it does not frame a truncated answer as a complete one.

Verified on hg-atlas-01 against two real Qwen replicas behind the real router.
Killing one before it answers: four of four requests re-sent and served by the
survivor. Killing one mid-stream: the upstream came back `ok=0 error=4
status=200` — headers received, read failed — and the relay aborted.

The failed replica leaves rotation before the retry, so the next request does
not pay the connect timeout again: `select` then skips it naturally and the
retry is a recovery rather than a routing policy. `--max-attempts 1` turns it
off; the default is 3.

The status codes stay meaningful: 503 only when there was never a routable
replica, 502 once we have actually tried one and the upstreams failed.

### A replica is a process group

`ProcessStarter` runs its template through a shell, and a shell that spawns
rather than execs leaves the replica one indirection away: the recorded pid is
the wrapper's. Signalling the wrapper alone can leave the replica running and
holding its card — exactly the failure recovery exists to undo.

This was visible in the first recovery run on hardware:

```
129548 /bin/sh -c bash .../run-replica.sh qwen-0 1 8110   ← the recorded pid
129550 python3 -m pypto_serving.cpp ... --port 8110        ← the replica
142152 python3 -m pypto_serving.cpp ...                    ← a simpler chip child
```

Killing only the replica left the wrapper alive, so liveness still said "up" and
the rebuild was refused. Replicas are now spawned into their own process group
and signalled as a group, with SIGTERM, a grace period, then SIGKILL.

## M1: partitioning

A model split across nodes runs as a chain of partitions, each with a
coordinator and replicas, and a request follows a path of one replica per
partition. Routing across that chain, and both ways of deciding it, were built
early (see **Routing**). What was missing was the other end of Option B.

### The replica's half of Option B

`replica/rule_table.hpp`. The coordinator plans a session once and publishes a
forwarding rule to each replica on the path; the replicas forward directly,
which is what keeps the coordinator off the per-token path. Until now it
published into nothing — the transport had five fakes and no consumer.

**Generations are what make it safe.** Channels are per replica and nothing
orders them against each other, so a message can arrive after a newer one. A
rule is applied only if its generation is at least the one held, so a late
message cannot send a session back to a replica the coordinator has already
moved it off. The same rule applies to revokes: a revoke older than the rule it
finds is ignored, so a revoke racing the replan that replaced it cannot strand
the session.

That is also why a generation's high half is the coordinator's epoch. A
restarted coordinator counting from 1 again would otherwise issue rules that
every surviving replica reads as stale.

The table is bounded and evicts least-recently-used. A replica serving a long
tail of conversations would otherwise hold a rule per session for ever, and a
session whose rule is gone is a cache miss the coordinator can replan — which
is not true of an arbitrary eviction.

The table is wired into the replica. A serving process started with
`--forwarding` holds one and gains three control routes — `POST /control/rules`,
`POST /control/rules/revoke`, `GET /control/rules?session=` — and a request
whose session has a rule is **forwarded to the next hop instead of served
here**, with no round trip to the coordinator. Without a table the routes are
not registered at all and the replica behaves exactly as before.

A rule carries an HTTP endpoint alongside the instance id, because the binary
wire format addresses the next hop by instance — which is what a control plane
routes on, and what an HTTP forward cannot use. A hop with no endpoint is
served locally rather than guessed at; a hop with one that refuses is a 502,
because an answer from the wrong stage of a pipeline is worse than a failure.

Measured on hg-atlas-01, two Qwen replicas, one per partition:

```
before any rule          answered-by=p0   forwarded-by=None
publish p0 -> p1         outcome=applied
the same request again   answered-by=p1   forwarded-by=p0
a generation-0 rule      outcome=stale, still points at p1
revoke                   answered-by=p0   forwarded-by=None
next hop refusing        HTTP 502 "next hop gone is unreachable"
```

What this does **not** do is split the model: both replicas hold all of Qwen,
so the forward moves a whole request rather than a half-computed activation.
The control plane's half is real; the pipeline behind it is not.

`tests/replica/test_pipeline.cpp` runs the coordinator's publisher into real
tables through the real encode/decode, then walks the pipeline from the tables
alone: planned once, three hops, and the last holds nothing because there is
nothing after it. Losing the middle replica empties the tables through the
revoke, and the next turn follows a new path at a higher generation.

### The same rules over channels

HTTP is the wrong transport for this. The deck's deployment already has
channels between the replicas; a second transport to configure, secure and keep
alive buys nothing, and under MPI there is no HTTP to reach for.

`replica/channel_rules.hpp` is the consumer, and `replica/rule_module.hpp` runs
it as a platform module beside `CoordinatorModule`. Every service tick drains
the inbound rule channel into the same `RuleTable` the HTTP routes write, so
nothing downstream changes: the forward is still a table lookup. The two forms
of the consumer exist because the platform offers two ways to read a channel —
`ChannelRuleConsumer::service()` drains one itself, and
`rulePublishHandler` / `ruleRevokeHandler` fit `modules/subscription.hpp`,
which binds a type to a channel and a handler.

Both are header-only, compiled only where the platform headers are, so
`libserving` still links no HiCR.

A frame is popped whatever happens to it. One the replica cannot decode, or
carrying a type the two ends do not agree on, is counted as `rejected` rather
than dropped silently and must not block the ones behind it — a replica quietly
ignoring its routing rules is the hardest kind of fault to find.

**The table now locks itself.** With rules arriving on the module's service
thread while handler threads read the table to route requests, an external
mutex in the HTTP server covered only one of the two writers; and `lookup`
writes, because it reorders the LRU list. `RuleTableTest.SurvivesAChannelWritingWhileHandlersRead`
is the guard: under ThreadSanitizer it reports a data race in the hash table —
and then segfaults — with the lock in `lookup` removed, and passes clean with it.

What is **not** proven is a live channel. Standing an `Input`/`Output` pair up
takes the platform engine exchanging memory slots between two instances, which
is a deployment, not a test. What is proven is everything either side does with
the bytes: `examples/channel_transport` runs the publisher's frame through the
replica's handlers, including a late generation and an undecodable frame, and
`examples/coordinator` drives the whole coordinator through those handlers into
one table per replica:

```
what each replica now holds for chat-1
  p0a      -> p1b       (applied=2 revoked=1 stale=0 unknown=0 rejected=0)
  p1a      -> (nothing)  (applied=1 revoked=1 stale=0 unknown=0 rejected=0)
  p1b      -> p2a       (applied=1 revoked=0 stale=0 unknown=0 rejected=0)
```

`p1a` died mid-conversation; the revoke emptied it, and `p0a` now forwards to
`p1b` at the higher generation.

### Routing to real replicas

Slide 3's complaint was that nothing identified a conversation, so the rotation
sent a follow-up to a replica that did not hold its cache and the whole thing
was recomputed. Measured on hg-atlas-01, two Qwen replicas behind the real
router, a conversation that grows by ~220 tokens a turn:

| | turn 1..5 median | replicas used |
| --- | ---: | --- |
| one session id kept | **0.29 s** | `r0` only |
| a new session id each turn | **0.43 s** | `r0` and `r1` |

**1.49x slower without a session id**, and the gap widens as the conversation
grows — by turn 5 it is 0.38 s against 0.46 s on a 1342-token prompt. Eight
distinct conversations split 4/4 across the two replicas, so affinity does not
cost balance.

The measurement only works because the conversation *grows*. A first attempt
sent a fixed prompt every turn, which warmed both replicas after turn 0 and
showed 1.00x — the control group had heated the very replica it was supposed to
be cold on.

### What partitioning still does not do

Nothing **executes** a partitioned model. The path is a path of HTTP replicas;
there are no inter-partition data channels, and the deck's "dependencies that
become inter-node become a channel towards the next partition" is about tensors
between kernels, which is pypto's business and not serving's. This layer can
say where the next partition is; it cannot yet hand it a half-computed
activation.

## Running it: the deployer

Everything above was built and tested on its own, and nothing put the pieces
together. A deployment was assembled by hand in a shell script every time one
was needed — the "coordinator" in the Option B measurement above was a Python
script POSTing to `/control/rules`, and the `Coordinator` class had never run
outside a test.

`pypto-serving-deployer-cpp` is the assembly. One process on the node that
started the application — the deployer of **One copy, on the deployer** — which:

- reads the document and writes it back as the deployment changes
- launches the replicas it asks for and supervises them to serving
- watches their health and marks the dead ones
- runs the coordinator, which plans a session's path and, under Option B,
  publishes the forwarding rules that let replicas skip it afterwards
- serves the ingress requests arrive on, and the operator API

```
pypto-serving-deployer-cpp --config deployment-config.json [--dry-run]
```

The configuration names a document and says how to run it; `--dry-run` prints
what it would start, including each launch command, and exits — the one thing
worse than a deployment that will not start is one that starts the wrong thing.

Two pieces had to exist for this to work at all:

**`router/http_rule_transport.hpp`** publishes rules over HTTP. The channel
transport was the only `RuleTransport` there had ever been, so a coordinator
could not publish to a deployment of plain HTTP replicas — which is every
deployment that exists today. It runs on the routing thread, once per session
rather than per token, with short timeouts and no dialling of a replica the
deployment has given up on.

**`--forwarding` on `pypto-serving-cpp`.** The flag existed only on
`python -m pypto_serving.cpp`; the native binary built its HTTP server with no
rule table and had no way to ask for one, so a native replica could not take
part in Option B at all.

`RouterServer` takes an optional coordinator. With one it is the front of a
partitioned deployment — the request enters at the hop the coordinator planned,
pinned for the first attempt only, because a retry means that hop did not answer
and the registry knows which sibling serves the same stage. Without one it is
the flat affinity router it has always been. `GET /routing?session=` reports the
plan without sending a request through it, and answers 503 rather than a short
hop list when a partition has nothing routable: an empty path reads like a
healthy deployment with one stage.

`tests/deployer/test_deployment.cpp` drives the whole thing without a device.
The replicas are real `HttpServer`s over a scripted executor, so the rules
travel real HTTP into real rule tables: the coordinator plans, `p0` is told
where `chat-1` goes next, `p0` forwards there itself, and the answer comes back
from `p1`. Killing `p1` marks it dead, stops the transport dialling it, and
turns the plan into an honest 503.

Two faults came out of running it under ThreadSanitizer, in code that predates
it:

- `router/proxy.cpp` raced on the attempt's `status` and `transportFailed`.
  The upstream thread keeps writing them after `send()` has returned — a
  transfer that dies mid-body sets `transportFailed` long after the caller read
  it to decide whether to retry — and a third thread, the chunked provider,
  reads it again. Both are now atomic.
- The remaining report, a "double lock" in `RouterServer`'s health poller, is
  the toolchain: 23 lines of a plain `condition_variable::wait_for` with a
  predicate reproduce it on this gcc, and a genuine double lock of a
  `std::mutex` would deadlock rather than pass. Suppressed, with that stated in
  `tests/tsan.supp`.

The suite is clean under ThreadSanitizer: 342 tests, zero warnings.

### On a card

hg-atlas-01, Qwen3-14B on card 4 through `task-submit`. The deployer is given a
document and nothing else:

```
=== what it would start ===
  qwen-0  partition 0 on hg-atlas-01  devices 4  port 8160
      bash /mounted_home/.cpp-port/run-replica.sh qwen-0 4 8160

[deployer] starting qwen-0
[deployer] qwen-0 is serving          replica serving after 85s

the coordinator's own plan      hops=[qwen-0 @ http://127.0.0.1:8160]  generation 7696131676996370433
a request through the ingress   0.26s  served by qwen-0: ' Paris. What is the capital of the'
the operator's view             qwen-0 active, load {pending_tokens: 0, preemptions: 0, outstanding: 0}
```

Nothing in that was posted by hand. The deployer launched the replica,
supervised it to serving, planned the route through its own `Coordinator`, and
answered on the ingress — which is the whole of what was missing.

`rules_published` is 0 because there is one partition, and the last hop of a
path gets no rule: there is nothing after it. **Option B's forwarding is still
unproven on a device** — it needs two cards, and one was the budget here. What
the measurement does prove is that the coordinator runs in a binary, which it
never had before.

Two runs were burnt on a stale artifact first. `/control/load` answered 404 and
the operator's view reported `load: null`, which looked like the design's
"a silent replica is not idle" rule firing — it was a build that named its
targets. The replica is launched through Python and loads the
`_pypto_serving` extension, which `ninja <two target names>` leaves untouched.
`DeploymentTest.ServesTheOperatorApiFromTheSameMap` now asserts the load is not
null, so the same stale build fails locally instead of on a card.

### Scaling and recovery, on four cards

The deployer driving its own components, with nothing holding the map by hand.
hg-atlas-01, Qwen3-14B, four cards through `task-submit`.

**Scaling.** The document asks for one replica; the operator asks for three
more:

```
[   0.0s] starting qwen-0
[  93.1s] qwen-0 is serving
[  93.1s] three POSTs to /deployment/scale, back to back
              -> p0-auto-0, p0-auto-1, p0-auto-2   (applied, versions 2..4)
[ 192.2s] p0-auto-0 active card [4] port 8161
          p0-auto-1 active card [5] port 8162
          p0-auto-2 active card [6] port 8163
          qwen-0    active card [3] port 8160
[ 193.3s] 8 conversations landed on {qwen-0: 3, p0-auto-0: 3, p0-auto-1: 2}
[ 193.3s] scale down -> qwen-0 draining, still holding its card
```

Three replicas loaded in **99s together**, because the supervisor starts every
pending replica on one tick. The earlier harness scaled one at a time and took
348s for the same four.

**Recovery.** Three replicas, one killed outright:

```
[ 117.5s] killed qwen-1 (both its processes)
[ 123.9s] qwen-1 marked dead after 6.3s; requests during the outage:
              qwen-0: 3, qwen-2: 1, failed: 0
[ 211.0s] qwen-1 active again, rebuilt 93s after it died, and serving
```

Against slide 3's complaint -- twenty minutes, truncated answers reported as
complete, `/health` green -- this is **6.3s to notice, no failed request, and
93s to rebuild**, with routing going round the hole in between.

Two faults came out of these runs, both of which made the feature useless and
neither of which the stub tests could see:

- **A scaled-up replica had no port.** `applyScaling` built a `CapacityRequest`
  without one, so the grant came back on port 0: nothing to launch, nothing to
  probe, nothing to route to. A provider hands out machines and cards; the port
  is a property of the deployment, so it now comes from the map -- one past the
  highest in use, which keeps endpoints unique as `validate()` requires.
- **The ingress ran its own health poller** and added replicas unready, so
  between the map saying active and that poller's next cycle every request got
  503 with three loaded replicas sitting idle. Routability now comes from the
  map, like everything else; the deployer's monitor is already probing these
  replicas, and a second opinion could only disagree with it.

Both are pinned by tests that fail without the fix.

### Option B, on two cards, planned by the coordinator

The earlier Option B measurement published its rules from a Python script. The
coordinator class had never done it. Through the deployer, on two cards, one
replica per partition:

```
[  96.1s] coordinator plans: p0 (partition 0) -> p1 (partition 1)   generation 7697229578011410433
[  96.3s] a request for chat-1 enters at p0
          answered by p1   x-forwarded-by=p0   ' Paris. What is the capital'
          p0 holds: next_hop p1 @ 127.0.0.1:8161, instance 20, same generation
          p1 holds: nothing -- it is the last hop
[  96.7s] plans=2  rules_published=2  invalidated=0
```

The request entered the pipeline at the hop the coordinator chose, and the
replica forwarded it on **without asking anyone** -- which is the whole of
slides 9 and 10. One plan and one rule per session, not per request.

That run also found a silent zero: a rule published over HTTP carried
`next_hop_instance: 0` while the document said 20. `ReplicaSpec` has no
instance, so the transport had nothing to send and the replica's handler
defaulted the field. Nothing over HTTP reads it -- a forward there goes by URL
-- but it is what a channel deployment routes on, and a real value replaced by
a default is the kind of fault that surfaces much later. The transport now
takes the name-to-instance map the deployer builds from the document.

### Replicas on another node

`ProcessStarter` spawns a command here, which is why the map's nodes had never
meant anything to the launcher: every replica was started locally whatever node
it claimed. `infra/remote_starter.hpp` acts on the claim -- replicas on this
process's own node are started directly, everything else is reached through a
command template. The default is ssh; a cluster with its own launcher
substitutes it without touching the code. `--dry-run` prints what a node would
actually receive:

```
there  partition 0 on node-b  devices 3  port 8161   (reached)
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new 192.168.150.12 \
      'sh -c '\''echo $$ > /tmp/serving-there.pgid; exec bash run-replica.sh there 3 8161'\'''
```

A remote replica needs more bookkeeping than a local one. The local pid belongs
to the reach, not the replica, so the far side records what it started and
stopping signals it there before the reach is torn down -- the same thing
`ProcessStarter` does with a process group, one machine further away.

Measured on two cards, with the reach standing in for ssh so the remote
bookkeeping runs for real against real Qwen replicas:

```
[   0.0s] starting here / starting there
[  90.1s] here active, there active       one started locally, one reached
[  90.1s] the far side recorded its own group, alive
[  90.9s] both serve through the one ingress: {here: 3, there: 3}
          shutting down -> group file removed by the remote stop
```

**Four faults came out of those runs**, every one of which leaked a Qwen
replica onto a card off the broker's lock, and three of which no test here
could have found:

- **The deployer never stopped anything.** `run()` cleared the flag `stop()`
  checks, so `stop()` returned having done nothing.
- **SIGTERM killed it outright.** `awaitShutdownSignal` uses `sigwait`, which
  needs the signals blocked first, and the deployer never called
  `blockShutdownSignals()` -- the other two binaries do. So the shutdown path
  above was never reached at all, which hid the first fault behind it.
- **The reach was detached.** Wrapping the remote command in `setsid` returns
  as soon as it has forked, so the reach exited at once, the supervisor read a
  replica that was still loading as a failed launch, retried, and started a
  second one on the same card.
- **The supervisor adopted a stranger.** The "already serving" shortcut
  trusted the probe alone, so a leaked replica still listening on the port had
  the next deployment promote a replica it had never started and could not
  stop. Only a reserved spare takes that path now.

What is **not** proven is ssh itself. `hg-atlas-02` has NPUs and a broker but
no serving tree, no model and no container of ours, and the container here has
no key and cannot resolve it. The composition is proven and the bookkeeping is
proven; the hop is not.

## Not yet

| | |
| --- | --- |
| nothing executes a partitioned model | the path is a path of HTTP replicas; splitting the model is the model layer's problem |
| the deployer RPC is not wired into the coordinator | `Coordinator::initialize()` still builds its map from a file; the deployer now owns the document, which is the precondition for handing copies out |
