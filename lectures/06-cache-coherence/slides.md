# Cache Coherence
## Keeping Multiple Caches Consistent in Parallel Systems
### GWU ECE 6125: Parallel Computer Architecture

Note: Today we dive into one of the most critical problems in parallel architecture: ensuring that when multiple processors each have their own cache, they all agree on what the "current" value of shared data is. This problem has driven decades of hardware innovation — from the simple MSI protocol to the complex SCD directories in today's 192-core server chips.

---

## Lecture Roadmap

| Part | Topic |
|------|-------|
| 1 | **The Coherence Problem** — what goes wrong with caches? |
| 2 | **Snooping Protocols** — MSI → MESI → MOESI → MESIF |
| 3 | **Directory-Based Coherence** — scaling beyond the bus |
| 4 | **Memory Consistency Models** — SC (Sequential Consistency), TSO (Total Store Order), relaxed ordering |
| 5 | **False Sharing** — the hidden performance killer |
| 6 | **Modern CPU Implementations** — AMD Zen 5, Intel, Apple |
| 7 | **Heterogeneous Coherence** — CPU + GPU |
| 8 | **Emerging Topics** — CXL, chiplets |

Note: We go from fundamentals to state-of-the-art. By the end, you'll understand the design decisions behind every modern processor's cache subsystem — and why each one made different choices.

---

## Part 1: The Coherence Problem

---

## The Stale Data Problem

Consider two processors sharing a variable X in memory:

| Time | CPU 0 | CPU 1 | Memory[X] |
|------|-------|-------|-----------|
| t₀ | — | — | **0** |
| t₁ | Read X → **0** | — | 0 |
| t₂ | — | Read X → **0** | 0 |
| t₃ | Write X = **1** | — | stale |
| t₄ | — | Read X → **0** ← WRONG! | 1 |

CPU 1's cache holds a **stale copy**. Without coherence, parallel programs produce incorrect results — silently.

> **Think of it like Google Docs:** if two people have the same document open and one edits it, the other's tab can briefly show stale content. Cache coherence is the hardware equivalent of Google's sync mechanism — except it must resolve in nanoseconds, not seconds.

![Two CPUs with shared variable and stale read scenario](images/coherence-problem.svg)

Note: This isn't a theoretical edge case. Before hardware coherence protocols, programmers had to manually flush caches before reading shared data. Operating system kernels, databases, and any code that shared memory across cores would fail. The hardware must handle this automatically and efficiently — programmers cannot be trusted to flush caches correctly in every case.

---

## Why This is Hard: The Speed-Correctness Tension

- Main memory is **~100× slower** than the processor
- **Private caches** keep hot data nearby → low latency, high performance
- But private copies can **diverge** when one processor writes

> "Cache coherence is the price we pay for making parallel systems fast with private caches."

**The fundamental dilemma:**
- No private caches → correct but slow (every access waits for memory)
- Private caches + no coherence → fast but wrong
- Private caches + coherence protocol → fast AND correct

Note: If every processor shared a single, monolithic cache, there'd be no coherence problem — but also terrible performance and scalability. Private caches at each core are the right answer for performance. Coherence protocols are what make private caches safe.

---

## Two Requirements for Coherence

A memory system is **coherent** if it satisfies:

**1. Write Propagation**
> A write by any processor must eventually become visible to all other processors.

**2. Write Serialization**
> All processors must observe **all** writes to the same address in the **same order**.

**Example — why serialization is essential:**

P1 writes X=1, then P2 writes X=2 (nearly simultaneously)

- **Without serialization:** P3 sees X=1 → X=2 (reads 2), P4 sees X=2 → X=1 (reads 1) — inconsistent!
- **With serialization:** one agreed-upon global order for all writes to X.

Note: Write propagation alone isn't enough. Even if every write eventually reaches every cache, if different caches see writes in different orders, programs can behave incorrectly. A lock-free queue relies on both propagation AND serialization. Serialization is what lets us reason about "who wrote last."

---

## Write-Invalidate vs. Write-Update

When a processor writes to a shared line, what should happen to other copies?

| Strategy | Action on Write | Analogy |
|----------|----------------|---------|
| **Write-Invalidate** | Send "your copy is stale, discard it" to all sharers | Google Docs marks other tabs as out of date — they reload on next access |
| **Write-Update** | Send the new value to all sharers immediately | Google Docs pushes every keystroke live to all open tabs |

**Modern processors almost universally use write-invalidate. Why?**

![Write-invalidate vs write-update side-by-side comparison](images/write-invalidate-vs-update.svg)

Note: Write-update seems more helpful — sharers immediately have the new value. But it consumes bus bandwidth on EVERY write, even for data that other caches won't read for a while. Write-invalidate pays a one-time cost to force a future cache miss, and that miss only occurs if someone actually reads the data. If a value is written 10 times before anyone reads it, write-update sends 10 updates; write-invalidate sends 1 invalidation.

---

## The Write-Invalidate Win Condition

**Write-invalidate is optimal when:**
- A location is written **≥2 times** before the next read — after the first write, all other copies are already invalidated, so subsequent writes cost zero messages
- This is the common case in real programs (loop accumulators, local counters, flag updates)

**Write-update wins when:**
- A consumer reads immediately after every single producer write (tight producer-consumer pipelines)
- Very few sharers (broadcast overhead is low)

**The quantitative argument:**
- Write-invalidate: 1 invalidation on the first write, then **0 messages** for every subsequent write (others already invalid)
- Write-update: **1 broadcast per write**, every time, regardless of whether anyone will read

> Think of Google Docs: write-update is like syncing every single keystroke to all open tabs — wasteful if the reader only opens the doc once an hour.

> Intel, AMD, ARM, RISC-V, and IBM all chose write-invalidate for general-purpose coherence.

Note: The Dragon protocol (Xerox PARC, 1982) and Firefly protocol (DEC, 1987) were serious write-update implementations. Both were eventually abandoned for the same reason — bus bandwidth could not keep up. Write-update survives only in specialized GPU contexts where access patterns are known to be tightly coupled producer-consumer.

---

## Coherence vs. Consistency: Two Separate Guarantees

These are two distinct properties of a shared memory system — and confusing them is the most common mistake in parallel programming.

| Property | Question it answers | Scope |
|----------|---------------------|-------|
| **Cache Coherence** | Do all CPUs agree on the *value* of address X? | Single address |
| **Memory Consistency** | In what *order* do CPUs observe writes to *different* addresses? | Multiple addresses |

**Analogy:** Think of a shared Google Doc.
- **Coherence** guarantees: "Everyone's copy eventually shows the same text" — no stale reads forever.
- **Consistency** guarantees: "If I first edited paragraph 1, then paragraph 2, you see them update in that order" — not paragraph 2 first.

> Coherence is about **accuracy** per address. Consistency is about **ordering** across addresses.

A system can be perfectly coherent yet still give surprising results when writes to different addresses appear out of order.

Note: Both properties are required for a correct shared-memory system. Coherence is handled entirely in hardware (the protocols we study in this lecture). Consistency requires hardware support AND explicit programmer annotations — memory fences, acquire/release semantics. The next slide shows why with a concrete example.

---

## Why the Distinction Matters — The Flag/Data Trap

A classic pattern that looks correct but can silently fail:

```c
// Initially: data = 0, flag = 0
// CPU 0:          // CPU 1:
data = 42;         while (flag == 0);  // wait for signal
flag = 1;          print(data);        // what prints here?
```

**What coherence guarantees:**
- CPU 1 *will* eventually see `flag = 1` ✓
- CPU 1 *will* eventually see `data = 42` ✓

**What coherence does NOT guarantee:**
- That CPU 1 sees `data = 42` *before* or *at the same time as* `flag = 1`

On some processors, `flag = 1` can become visible to CPU 1 *before* `data = 42` does.
`print(data)` prints **0** — even though the hardware is fully coherent.

> The ordering of writes across *two different addresses* is a **consistency** question, not a coherence question.
> **Fix:** a memory fence between CPU 0's writes, and an acquire load on CPU 1's spin.

Memory consistency models (SC — Sequential Consistency, TSO — Total Store Order, ARM weak ordering) and how fences restore ordering are covered later in this lecture.

Note: Students often blame "cache bugs" when they see this failure. The hardware is doing exactly what it's designed to do — coherence is satisfied. The missing piece is a memory ordering guarantee, which requires explicit programmer annotations on weakly-ordered architectures. On x86 (TSO), this pattern happens to work without fences — which is why the bug is often discovered only when porting to ARM or RISC-V.

---

## Fences vs. Locks — What's the Difference?

The natural reaction to the flag/data trap is: *"why not just use a lock?"* They are related but solve different problems.

| | **Lock** | **Fence** |
|---|---|---|
| **Purpose** | Mutual exclusion — only one thread in the critical section | Ordering — my writes appear to others in the right sequence |
| **Blocks execution?** | Yes — other threads wait outside | No — both threads keep running |
| **Overhead** | High (contention, OS involvement on slow path) | Medium (pipeline stall, store buffer drain) |
| **Prevents data races?** | Yes | No — races are still possible |
| **Contains fences?** | Yes — internally | It *is* the primitive |

**A lock secretly wraps fences inside it:**
- `lock()` → implicit **acquire fence**: "I will see all writes committed before I enter"
- `unlock()` → implicit **release fence**: "all my writes are visible before I signal I'm done"

So if you use locks correctly, **you never need to think about fences.** The lock library handles ordering for you.

**Then when do you use fences directly?**

When you want the *ordering guarantee* of a lock but **without the mutual exclusion cost.** In the flag/data pattern there is only one writer and one reader — they never conflict. A lock would unnecessarily serialize them. A fence gives you the ordering at lower cost:

```c
// CPU 0 — writer:          // CPU 1 — reader:
data = 42;                  while (flag == 0);
// release fence            // acquire fence
flag = 1;                   print(data);  // guaranteed to see 42
```

> **Rule of thumb:** Use a lock when two threads must not run concurrently. Use a fence (or atomics) when they can run concurrently but need to agree on the order of what they see.

Note: In C++11 and later, you rarely write raw fences. Instead you use std::atomic with memory_order_acquire and memory_order_release — which compile down to the correct fence instructions per architecture. The compiler and hardware together handle the rest. Raw fences (std::atomic_thread_fence) exist for advanced cases like seqlocks or lock-free data structures.

---

## Sequential Consistency: The Intuitive Model

**SC is the memory model you naturally assume when writing parallel code.**

Two rules, both must hold simultaneously:

1. **Each CPU's operations happen in the order it issued them** — no CPU skips ahead or reorders its own instructions
2. **All CPUs agree on one global order** — every CPU sees the same interleaving of everyone's operations

Rule 1 is easy to understand. Rule 2 is the hard one.

**What "one global order" means:**

Think of two decks of cards — CPU 0's operations and CPU 1's operations. Each deck must keep its internal order. But the two decks can be shuffled together in any way. SC says: **every CPU must see the same shuffle result.** Not just their own cards in order — the same complete sequence of all cards from all CPUs.

> If CPU 0 sees: write X, write Y, read Z — then CPU 1 must also see those operations in exactly that position in the global sequence. No CPU gets a different view.

This is what makes SC powerful — and what makes it expensive to implement in hardware.

Note: SC is the model most programmers implicitly assume. When you write parallel code and reason about it on paper, you're almost certainly assuming SC. The surprising fact is that almost no modern hardware actually provides SC by default — because enforcing it would require stalling the pipeline every time a write is issued, waiting for it to become visible globally before moving on.

---

## Sequential Consistency: The (0, 0) Litmus Test

The classic test for whether a system is SC:

```c
// Initially: X = 0, Y = 0
// CPU 0:    // CPU 1:
X = 1;       Y = 1;
print(Y);    print(X);
```

| Outcome (Y, X) | Under SC? | Why |
|----------------|-----------|-----|
| (1, 1) | ✓ | Both writes committed before either read |
| (0, 1) | ✓ | CPU 0 ran entirely before CPU 1's write reached it |
| (1, 0) | ✓ | CPU 1 ran entirely before CPU 0's write reached it |
| **(0, 0)** | **✗ Never** | No valid interleaving of the two programs can produce this |

**(0, 0) is impossible under SC** — if CPU 0 sees Y=0, then Y=1 hasn't happened yet, which means CPU 0's entire sequence runs before CPU 1's write, which means CPU 1 must see X=1. You cannot have both reads return 0 in any single consistent ordering.

> Real hardware **can** produce (0, 0) — because stores sit in a buffer and are not immediately visible. This is covered in Part 4.

Note: The (0,0) outcome is the canonical fingerprint of a non-SC system. If you ever observe it, the hardware is definitely not SC. This test (called a "litmus test") is the standard tool for probing memory model behavior — researchers run millions of these on real hardware to map out exactly what orderings a given CPU permits.

---

## So How Do We Keep Caches in Sync?

We now know the problem: multiple CPUs each have a private cache, and any of them can write to the same memory address — leaving other caches holding a stale copy.

**Hardware solves this with a coherence protocol.** Two fundamental approaches exist:

| Approach | Core idea | Works best at |
|---|---|---|
| **Snooping** | Every cache watches every memory transaction on a shared bus. If a write affects your copy, you act on it. | Small systems — few cores |
| **Directory** | A central directory tracks which caches hold each block. Only the relevant caches are notified on a write. | Large systems — many cores |

Both approaches enforce the same guarantee: **no cache holds a stale value indefinitely.**

They differ in *how* they communicate that guarantee — broadcast vs. targeted messaging.

**We start with snooping** — it is simpler, and understanding it deeply makes directory protocols much easier to follow.

Note: The snooping vs. directory split is one of the most important architectural decisions in parallel system design. Snooping dominated from the 1980s through the early 2000s when core counts were low (2–8 cores). As core counts grew beyond ~16, the broadcast overhead of snooping became a bottleneck and directory protocols took over. Modern chips often use a hybrid — snooping within a cluster of cores, directory between clusters.

---

## Part 2: Snooping Protocols

### Bus-Based Coherence — Simplicity Through Broadcast

> Every cache watches every transaction. Coherence emerges from universal visibility.

Note: Snooping protocols exploit the broadcast property of a shared bus. If every memory transaction is visible to every cache controller simultaneously, each controller can independently determine whether it needs to act — invalidate, update, or supply data. No central coordinator required. The bus does the work.

---

## How Snooping Works

**Hardware setup:**
- All caches connect to a **shared bus**
- Every bus transaction is seen by **all caches simultaneously**
- Each cache controller "snoops" (monitors) all transactions

**When CPU 0 writes to address X:**
1. CPU 0 puts `BusRdX(address=X)` on the bus
2. All other caches see this transaction immediately
3. Any cache with X in its cache **invalidates** that line
4. CPU 0 receives data and ownership

![Bus topology with coherence broadcast arrows](images/snooping-protocol.svg)

Note: The bus provides two critical properties: (1) broadcast — every controller sees every transaction, and (2) total order — the bus arbiter ensures all caches see transactions in exactly the same order. Property 2 provides write serialization for free. The bus arbiter IS the serialization point. This is elegant, but the bus becomes the bottleneck as core count grows.

---

## MSI: The Foundational Protocol

Every cache line is always in exactly one of three states:

| State | What it means | This cache can read? | This cache can write? | Other caches can hold a copy? |
|-------|--------------|----------------------|-----------------------|-------------------------------|
| **M**odified | This cache has the only copy, and it has been written — memory is stale | ✓ | ✓ | No |
| **S**hared | This cache has a clean copy — identical to what is in memory | ✓ | ✗ | Yes |
| **I**nvalid | This cache has no usable copy — must fetch before use | ✗ | ✗ | Unknown |

**Google Docs analogy:**
- **M** = You have the doc open and have typed unsaved changes. Your version is the latest. No one else has it.
- **S** = Multiple people have the doc open in view-only mode. Everyone sees the same saved version.
- **I** = Your tab is closed. You have nothing.

![MSI 3-state FSM with all labeled transitions](images/msi-state-diagram.svg)

**MSI in the real world:**

| System | Years | Notes |
|---|---|---|
| SGI Challenge (MIPS R4400) | 1993–1997 | Early commercial multiprocessor; pure MSI over a shared bus |
| Sun UltraSPARC II systems | 1997–2001 | Used MSI-based snooping before moving to MESI |
| Early IBM POWER systems | 1990s | MSI with bus-based coherence on 2–8 socket servers |
| Academic research platforms | 1980s–present | MSI is the standard teaching and verification baseline |

No modern high-volume processor ships with pure MSI today — they all use MESI or beyond. But MSI remains the **reference protocol**: every coherence proof, textbook, and formal verification tool starts here.

Note: MSI is the minimal coherent protocol. Every more complex protocol (MESI, MOESI, MESIF) is just MSI with extra states added to avoid unnecessary bus traffic. Understand MSI and the rest follow naturally.

---

## MSI: What Triggers a State Change?

Two types of events cause a cache line to change state:

**1. This CPU wants to read or write the line:**

| Current State | This CPU reads | This CPU writes |
|---------------|----------------|-----------------|
| **M** | Serve from cache — no bus needed | Serve from cache — no bus needed |
| **S** | Serve from cache — no bus needed | Broadcast **BusUpgr**: "everyone else invalidate your copy" |
| **I** | Broadcast **BusRd**: "someone give me a copy" | Broadcast **BusRdX**: "give me a copy AND everyone else invalidate" |

**2. This cache sees another CPU's request on the bus:**

| Current State | Another CPU reads (BusRd) | Another CPU writes (BusRdX or BusUpgr) |
|---------------|---------------------------|----------------------------------------|
| **M** | Give them the data, go to **S** | Give them the data, go to **I** |
| **S** | Stay **S** — they can share | Go to **I** — my copy is now stale |
| **I** | Do nothing | Do nothing |

Note: The S→M transition sends BusUpgr even if this cache is the ONLY reader. That wasted broadcast is the key inefficiency MSI has — and exactly what the E (Exclusive) state in MESI eliminates.

**"If I'm in M and forced to go to S or I — are my changes lost?"**

No — and this is critical to understand. A cache in M state is the sole owner of the most recent copy of that data. The hardware will NEVER silently discard it. Before any M→S or M→I transition completes, the cache must first supply the data:

- **M → S** (another CPU reads): your cache writes the data back to memory (or supplies it directly to the requester via a cache-to-cache transfer). Memory is now up to date. Both you and the requester hold clean S copies. Your changes are preserved.

- **M → I** (another CPU writes): your cache again supplies the data — either writing back to memory or handing it directly to the requester. Memory and the requester both get the latest value. Your copy is then invalidated. Your changes are preserved — just no longer in your cache.

The phrase "supply data" in the transition table is not optional or cosmetic. It is the writeback step that makes the protocol correct. Without it, the requester would get stale data from memory and the write your cache did would be lost forever.

---

## MSI: Example Trace

| Step | Event | CPU 0 | CPU 1 | Memory | Bus transaction |
|------|-------|-------|-------|--------|-----------------|
| Start | — | **I** | **I** | X = 0 | — |
| 1 | CPU 0 reads X | **S** | I | X = 0 | BusRd → memory supplies X = 0 |
| 2 | CPU 1 reads X | S | **S** | X = 0 | BusRd → memory supplies X = 0 |
| 3 | CPU 0 writes X = 1 | **M** | **I** | X = 0 ⚠️ stale | BusUpgr → CPU 1 must invalidate |
| 4 | CPU 1 reads X | **S** | **S** | X = 1 | BusRd → **CPU 0** supplies X = 1 |

**Two things to notice:**

- **Step 3:** CPU 0 was already the only writer, yet it still had to broadcast BusUpgr to invalidate CPU 1. The bus was used even though CPU 0 knew it wanted to write all along. This is the wasted transaction MESI eliminates with the E state.

- **Step 4:** Memory did not supply the data — CPU 0's cache did. This is a **cache-to-cache transfer**. CPU 0 held the only up-to-date copy (memory was stale at X=0), so it intervenes on the bus and hands X=1 directly to CPU 1.

Note: Cache-to-cache transfers are faster than going to memory and are essential for correctness — if memory had supplied X=0 in step 4, CPU 1 would have read a stale value. The M-state cache intercepts the BusRd and overrides memory.

---

## The MSI Tax on Private Writes

In Step 3 of the trace, CPU 0 was **already the only cache holding X** — yet it still had to broadcast a BusUpgr and wait for acks before writing.

| What happened | Why it's wasteful |
|---|---|
| CPU 0 read X → S (shared) | Makes sense — maybe others will read too |
| CPU 0 wants to write X | Forces BusUpgr onto the bus |
| CPU 1 must invalidate and ack | But CPU 1 was never going to write! |
| CPU 0 finally writes X | After burning a round-trip bus transaction |

**The insight:** if the hardware had known at read time that no other cache had X, it could have given CPU 0 an **exclusive** copy — and the write would have been silent, no bus message at all.

> ~60-70% of cache lines are private (stack frames, local variables, thread-local data). MSI charges a bus transaction for every write to every one of them.

Note: This isn't a corner case — it's the common case. Most data in a program is private. The MSI protocol was designed for correctness first, and it achieves that, but it pays a heavy tax on the most frequent operation. MESI was designed specifically to eliminate this tax by adding a fourth state that says "I have the only copy."

---

## MESI: Adding the Exclusive State

**The Exclusive (E) state:** "I have the only copy, and it's clean (matches memory)."

**How E is granted:** On a BusRd, if **no other cache** signals it has the line, the miss is granted as E instead of S. (Requires one extra "shared" wire on the bus.)

**The payoff — silent E→M transition:**

| | MSI (before) | MESI (after) |
|---|---|---|
| CPU 0 reads X (only copy) | → **S** | → **E** |
| CPU 0 writes X | BusUpgr → wait for acks → **M** | Silent → **M** (no bus message) |
| Bus transactions | **2** (BusRd + BusUpgr) | **1** (BusRd only) |

![MESI 4-state FSM highlighting E state benefit](images/mesi-state-diagram.svg)

Note: The E state is valuable because most data is private — local variables, private data structures, stack frames. Studies of SPLASH-2 and PARSEC benchmarks show 60-70% of cache lines are touched by only one core. Without E, every write to private data burns a bus transaction. With E, private-data writes are entirely local.

---

## MESI: Real-World Impact

**Typical workload breakdown:**

| Data Type | Protocol State | Coherence Overhead |
|-----------|---------------|-------------------|
| Local variables, stack | E → M | **Zero** (silent transition) |
| Read-only shared data | S | Zero (read-only) |
| Producer writes, consumer reads | M/S exchange | 1 transaction per write-then-read |
| Write-shared "hot" data | M ping-pong | 1 transaction per write |

**Why MESI replaced MSI everywhere:**
- ~60-70% of lines are private → E eliminates their upgrade traffic
- Remaining ~30-40% benefit equally from MSI and MESI

**Real-world systems using MESI:**

| System | Years | Notes |
|--------|-------|-------|
| Intel i486 | 1989–1995 | First commercial MESI implementation — Intel invented the E state |
| Intel Pentium / Pentium Pro | 1993–1999 | MESI at L1; Pentium Pro added L2 bus coherence |
| Intel Core / Xeon (all generations) | 2006–present | MESI at L1/L2 between cores; MESIF added at socket level |
| ARM Cortex-A (A9, A15, A53, A72…) | 2007–present | MESI at L1/L2 in all multi-core Cortex-A configurations |
| IBM Cell Broadband Engine | 2006–2012 | Used by PS3; MESI between SPE local stores and main memory |
| RISC-V (SiFive, Rocket Chip) | 2016–present | Reference implementations default to MESI |

> MESI is the **baseline** for almost every coherence protocol in use today. AMD extended it to MOESI; Intel extended it to MESIF. But the M, E, S, I states are in every one of them.

Note: The E state is a pure win with a tiny hardware cost (one shared-line wire per bus, plus one extra state bit per cache line). The improvement in real workloads is typically 15-30% fewer bus transactions. This is why every serious coherence protocol since 1985 includes an E state.

---

## MESI's Remaining Weakness: The Dirty-Data Round-Trip

MESI solved private writes. It did not solve **shared dirty data**.

Consider what happens when CPU 0 has X in M state and CPU 1 wants to read it:

| Step | MESI |
|------|------|
| 1 | CPU 0 must **write back** X to memory (memory was stale) |
| 2 | CPU 1 fetches X from memory |
| Result | **2 memory transactions** — one to write stale memory, one to read it back |

The second transaction is reading data that was just written a moment ago. Memory is acting as an unnecessary middleman.

> In a producer–consumer pattern where CPU 0 repeatedly writes X and CPU 1 reads it, every single read burns two memory transactions. On a 4-socket server, each memory transaction can cost 200–400 ns. This adds up fast.

**The insight:** if CPU 0 can hand X directly to CPU 1 and simply *stay responsible* for writing it back later, both memory transactions are eliminated.

Note: This is the classic "dirty sharing" problem. It's most painful in producer-consumer workloads, streaming data pipelines, and any pattern where one core writes data that another must immediately read. The Owner state in MOESI is the direct answer.

---

## MOESI: Eliminating the Memory Writeback

**Problem with MESI:** When a Modified line is requested by another cache:
1. Owner must write back dirty data to memory
2. Requester fetches from memory
→ 2 slow memory transactions, one of which writes data that's about to be read

**MOESI solution — the O (Owner) state:**

> "I have a modified copy, and I'm sharing it. I'm responsible for keeping it coherent. Memory does NOT need to be updated yet."

| | MESI | MOESI |
|---|---|---|
| CPU 0 state before | M (X=5, memory stale) | M (X=5, memory stale) |
| CPU 1 reads X | CPU 0 writes back → memory → CPU 1 fetches | CPU 0 → **O**, supplies X=5 directly to CPU 1 |
| Memory transactions | **2** (writeback + fetch) | **0** |
| Memory state after | X=5 (updated) | X=old (still stale — owner responsible) |

![MOESI 5-state FSM highlighting O state benefit](images/moesi-state-diagram.svg)

Note: The Owner state is AMD's signature innovation. The owner is the cache responsible for maintaining the coherent view — it supplies data to any requester and writes back to memory when it finally evicts the line. The trade-off: the owner must track that it's the authoritative source, and the directory/other caches must know to ask the owner, not memory.

---

## MOESI in Practice

**Owner state transitions:**
- M → O: Another cache requests the line (owner stays, sharer added)
- O → M: All other copies invalidated (write request from another core)
- O → I: Line is evicted → must write back to memory now

**Benefit quantification (from AMD's performance data):**
- In workloads with heavy shared modified data, MOESI reduces memory traffic by 20-40%
- Critical for large server systems where memory bandwidth is the bottleneck

**AMD uses MOESI in:** Zen 1 through Zen 5, EPYC, Threadripper. All AMD processor families.

Note: The Owner state is especially valuable on multi-socket systems where going to memory means crossing the inter-socket interconnect (200-400 ns). With MOESI, another cache in the same socket can supply the data at L3 speed (30-50 ns). The 4-8× latency reduction is often the difference between a scalable workload and a memory-bound one.

---

## MESIF: Intel's Forward State

**Problem without F:** When many caches have X in Shared state, who supplies data to a new requester?
- Option A: All sharers could respond → thundering herd, multiple conflicting replies
- Option B: Memory responds → no thundering herd, but memory is slow

**MESIF solution — the F (Forward) state:**
- Exactly one S-state cache is designated **Forward** (the most recent reader)
- On a new BusRd, the F-cache **supplies data directly** (2-hop: requester ↔ F-cache)
- The F-cache transitions: F → S; the new requester becomes F

```
Caches: CPU 0 in F, CPU 1 in S, CPU 2 in S
CPU 3 reads X:
  CPU 0 (F) → CPU 3: supplies X directly   [memory not accessed!]
  CPU 0: F → S
  CPU 3: I → F
```

![MESIF 6-state FSM with F (Forward) state](images/mesif-state-diagram.svg)

Note: MESIF is used in Intel's QPI and UPI interconnects for multi-socket Xeon systems. The F state eliminates memory from the critical path for read sharing in multi-socket configurations. Benchmarks show MESIF reduces average read latency by 10-20% in multi-socket workloads vs. MESI, by avoiding the ~200 ns round-trip to the home node's memory controller.

---

## Protocol Comparison: The Full Picture

| Protocol | States | Key Addition | Eliminates |
|----------|--------|-------------|------------|
| MSI | 3 | — (baseline) | — |
| MESI | 4 | Exclusive (E) | BusUpgr for private data |
| MOESI | 5 | Owner (O) | Memory writeback on sharing |
| MESIF | 6 | Forward (F) | Memory on read-sharing response |
| MOESIF | 6 | Owner + Forward | Both writebacks and memory reads |

**Design philosophy:** Each state addition trades hardware complexity (more state bits, more protocol logic, more verification effort) for eliminating a class of redundant transactions.

Note: You'll sometimes see MOSI (early AMD), MESIF (Intel), MOESI (AMD), and MOESIF (theoretical / some research processors). The trend is always: add states to eliminate waste. But more states means more complex verification hardware and more coverage needed in simulation. MESIF and MOESI represent pragmatic stopping points where the cost-benefit ratio is favorable.

---

## Snooping: The Scalability Wall

**The bus is snooping's strength and its fatal weakness:**
- **Strength:** Total order → write serialization for free; simple protocol
- **Weakness:** One bus = one serialization bottleneck for ALL coherence traffic

**Bus bandwidth math:**
```
Bus: 1600 MHz × 64-bit wide = 12.8 GB/s total bandwidth
Each core generates ~1-2 GB/s coherence traffic
→ 8-12 cores saturate the bus
Coherence traffic is additive with data traffic
→ Practical snooping limit: 16-32 cores
```

**Modern core counts:** AMD EPYC has 96-192 cores. Intel Xeon has 60 cores. ARM Neoverse has 128 cores.

**Conclusion:** Every server-class chip needs directory-based coherence for inter-cluster communication.

Note: The bus bandwidth wall was well understood by the late 1980s. Stanford DASH (1992) demonstrated that directory protocols could scale to hundreds of processors. SGI Origin (1996) used directory coherence at commercial scale. Today, snooping may still be used within a small cluster of cores (e.g., 8 cores sharing an L3 slice), but directory protocols handle all cross-cluster communication.

---

## Part 3: Directory-Based Coherence

### Scaling Coherence Beyond the Bus

> Instead of broadcasting to everyone, track exactly who has what — and message only them.

Note: Directory protocols replace the broadcast medium with targeted point-to-point messages. A directory entry for each memory block tracks exactly which caches hold copies. When coherence action is needed, the directory sends messages to only the affected caches — not to everyone. This is the key scalability insight.

---

## Directory Structure: Tracking Sharers

**A directory entry for each memory block:**

```
┌──────────────────────────────────────────────────────────────────┐
│  State (2 bits)  │  Owner/Sharers  (N bits for N-core system)   │
│  U / S / M       │  Bitmap or pointer to cache(s) with copies   │
└──────────────────────────────────────────────────────────────────┘
```

**Directory state field:**
- **Uncached (U):** No cache has this block
- **Shared (S):** ≥1 caches have clean copies; sharer bitmap indicates which ones
- **Modified (M):** Exactly one cache has it (dirty); owner pointer indicates which one

**Sharer tracking approaches:**

| Approach | Storage | Max Practical Sharers | Use Case |
|----------|---------|----------------------|----------|
| Full-map bitmap | N bits/block | All N caches | ≤ 64 cores |
| Limited pointer | k × log₂N bits | k sharers exactly | 64–256 cores |
| Sparse directory | Variable (hash/list) | Unlimited | Large systems |

![Directory entry format and pointer-to-sharers structure](images/directory-structure.svg)

Note: Storage overhead is the core scalability problem with directories. For N=64 cores with 64-byte cache lines, full-map adds 64 bits = 8 bytes per line — 12.5% overhead. Acceptable. For N=1024 cores, full-map adds 1024 bits = 128 bytes per 64-byte line — 200% overhead. Unacceptable. This is why limited-pointer and sparse directories exist.

---

## Directory Protocol: Read Miss

**Case 1 — No cache has the block (Uncached):**
```
1. CPU 2 → Directory:   ReadReq(block B)
2. Directory checks:    State = Uncached
3. Directory → Memory:  Fetch(B)
4. Memory → CPU 2:      Data(B)
5. Directory updates:   State = Shared, Sharers = {CPU 2}
```

**Case 2 — Block is Modified (CPU 5 is Owner):**
```
1. CPU 2 → Directory:   ReadReq(B)
2. Directory checks:    State = Modified, Owner = CPU 5
3. Directory → CPU 5:   Intervention: supply B to CPU 2, go to Shared
4. CPU 5 → CPU 2:       Data(B)   [cache-to-cache, bypasses memory]
5. CPU 5 → Directory:   AckOwner (I've transferred ownership)
6. Directory updates:   State = Shared, Sharers = {CPU 2, CPU 5}
```

Note: Case 2 demonstrates the directory's key advantage: it KNOWS to go to CPU 5, not to memory. In snooping, a read miss would broadcast to all cores — most of which have no stake in block B. With directory, only CPU 5 is messaged. At 64+ cores, this is the difference between O(1) and O(N) message complexity.

---

## Directory Protocol: Write Miss

**Block B is currently Shared by CPUs 1, 4, 7:**

```
1. CPU 3 → Directory:     WriteReq(B)
2. Directory checks:      State = Shared, Sharers = {1, 4, 7}
3. Directory → CPU 1:     Invalidate(B)
   Directory → CPU 4:     Invalidate(B)
   Directory → CPU 7:     Invalidate(B)
4. CPU 1, 4, 7 → Directory: AckInval (I've invalidated my copy)
5. Directory → CPU 3:     WriteAck + Data(B)
6. Directory updates:     State = Modified, Owner = CPU 3
```

**Critical:** CPU 3 must wait for ALL AckInval messages before proceeding. This enforces write serialization — no cache can read stale data once all Acks are received.

![Step-by-step message flow for directory read/write/eviction](images/directory-protocol-trace.svg)

Note: The write must wait for ALL invalidation acknowledgments. Why? If CPU 3 wrote X=5 and CPU 1 hadn't yet invalidated, CPU 1 might return old data X=3 to a future reader — violating coherence. The directory acts as a serialization point: it processes one request per block at a time, naturally providing the same total order the bus provided in snooping.

---

## Directory Protocol: Eviction (Writeback)

**Dirty eviction (CPU 3 is Owner, State = Modified):**
```
1. CPU 3 → Directory:   WritebackReq(B, data)
2. Directory → Memory:  Update(B, data)
3. Directory updates:   State = Uncached
4. Directory → CPU 3:   WritebackAck
```

**Clean eviction (CPU 1, one of several Shared copies):**
```
1. CPU 1 → Directory:   EvictShare(B)
2. Directory updates:   Remove CPU 1 from Sharers bitmap
3. If Sharers = ∅:      State = Uncached
```

**Why acknowledge writebacks?** Prevents races: if CPU 3 evicts B, then CPU 4 reads B before the writeback reaches memory, the directory must not grant CPU 4 the old value. WritebackAck serializes this.

Note: Evictions are one of the harder parts of directory protocol implementation. The directory must handle races between simultaneous requests and evictions for the same block. Real implementations use "transient states" (IMAD, IMAD-WB, etc.) to track these race conditions. Full protocol verification requires model checking — there are O(N) transient states even for simple 3-state protocols.

---

## SCD: Scalable Coherence Directory

**Problem:** For 1024 cores, full-map directory = 128 bytes overhead per 64-byte block (200%). Impractical.

**SCD's key insight (Sanchez & Kozyrakis, 2012):** The actual distribution of sharing is **bimodal** — most blocks are either **private** (1 sharer) or **broadcast** (many sharers). The "2–16 sharers" region is rare.

**Variable-size encoding:**

| Encoding | Bits Used | Represents |
|----------|-----------|------------|
| Single owner pointer | log₂N | Exactly 1 sharer (private) |
| Pair of pointers | 2 × log₂N | Exactly 2 sharers (producer-consumer) |
| Coarse-grained group vector | N/G bits | Many sharers (grouped by G cores) |
| Broadcast flag | 1 bit | All caches are sharers |

**Result at 1024 cores:** Only ~5% storage overhead vs. 200% for full-map.

![Variable-size sharer set encodings: 1, 2, many](images/scd-sharer-encoding.svg)

Note: SCD was demonstrated at 1024 cores with 5% storage overhead and less than 2% increase in coherence latency vs. full-map. AMD uses similar ideas in their Infinity Fabric directory. Intel's scalable socket architecture uses compressed sharer lists. The bimodal sharing distribution is universal: most data is either truly private or truly broadcast-shared. Applications that have large numbers of sharers in the 3-32 range are the edge case SCD handles with coarse vectors.

---

## Scalability Comparison

| Protocol | Max Practical Cores | Message Pattern | Directory Storage |
|----------|--------------------|-----------------|--------------------|
| Bus snooping | 8–32 | O(N) broadcast | None |
| Snooping (ring) | 32–64 | O(N) traversal | None |
| Directory (full-map) | 64–128 | O(1) targeted | O(N bits/block) |
| Directory (limited ptr) | 128–512 | O(k) targeted | O(k × log N) |
| SCD variable | 512–1024+ | O(1) targeted | ~5% |
| CXL fabric | Multi-socket, multi-device | O(1) targeted | Distributed |

Note: This is why every server CPU uses directory protocols between clusters. Modern AMD EPYC (96-192 cores) and Intel Xeon (60 cores) always use directory-based coherence at the inter-chiplet or inter-socket level. Snooping may be used within a small cluster of 8-16 cores sharing an L3 slice, but not across clusters. Understanding this hierarchy is key to understanding modern CPU performance.

---

## Part 4: Memory Consistency Models

### How Relaxed Can We Get Without Breaking Programs?

> Coherence ensures everyone agrees on the order of writes to a single address. Consistency defines what ordering guarantees exist across DIFFERENT addresses.

Note: This is where things get subtle. Coherence is a hardware property — the hardware either guarantees it or not. Consistency is a contract between hardware and software. The programming language (C11, Java, CUDA) maps to hardware instructions, and the hardware's consistency model determines what the programmer must add (fences, atomics) to get correct behavior.

---

## Sequential Consistency: What Programmers Want

**Lamport's SC:**
1. Operations of each processor appear in program order
2. The complete execution looks like some interleaving of all processors' program orders

**Why SC is desirable — Dekker's mutex works without fences:**
```c
// Initially: flag0=0, flag1=0
// CPU 0:                         // CPU 1:
flag0 = 1;                        flag1 = 1;
if (flag1 == 0) enter_CS();       if (flag0 == 0) enter_CS();
// Under SC: at most one enters. ✓
```

**Under relaxed models (ARM, without fences):** Both could read 0 → both enter. Mutual exclusion fails! ✗

Note: This is the canonical example of why consistency models matter. The algorithm is logically correct. With SC, it's practically correct. With a relaxed model, it fails — and the failure is silent. No crash, no error — just two threads both "safely" in the critical section at once. Memory model bugs are notorious for being hard to reproduce and hard to debug.

---

## Why SC is Too Strict for Performance

**SC forbids hardware optimizations that all modern CPUs use:**

| Optimization | Violates SC? | Typical Performance Gain |
|--------------|-------------|--------------------------|
| Store buffer (write queue) | Yes — loads can bypass stores | 10–30% IPC |
| Out-of-order load execution | Yes — loads may not wait for prior stores | 20–40% IPC |
| Non-blocking caches | Yes — multiple outstanding misses | 15–25% IPC |
| Speculative loads | Yes — load before branch resolved | 10–20% IPC |

**The store buffer is the key culprit:**
```
CPU 0: store X=1  (goes into store buffer, not yet visible)
CPU 1: load X → 0 (sees old value from cache, not CPU 0's store buffer)
→ This violates SC. But the store buffer gives 10-30% speedup.
```

Note: Every modern high-performance processor has a store buffer. Every modern high-performance processor is NOT sequentially consistent by default. The question is: how far do we relax, and what does the programmer need to add back to get correct behavior?

---

## TSO: Total Store Order (x86's Model)

**TSO is SC with one relaxation:** Stores can be delayed in a per-processor FIFO write buffer before becoming globally visible.

**What TSO permits that SC forbids:**
```
// Initially: X=0, Y=0

CPU 0:              CPU 1:
store X=1           store Y=1
(buffer)            (buffer)
load Y → 0          load X → 0
```
Both see 0 for the other's write — impossible under SC, allowed under TSO.

**What TSO still guarantees:**
- Stores become globally visible in order (total store order)
- A load sees all prior stores from the **same** processor (via store-to-load forwarding)
- `MFENCE` drains the store buffer completely

Note: x86-TSO was formalized by Owens, Sarkar, and Sewell in 2009 — surprising that x86's memory model wasn't formally specified until then! TSO is one of the strongest relaxed models in practice, which is why x86 code is often more portable across architectures. The store buffer is the minimal relaxation needed for good performance.

---

## Relaxed Models: ARM and RISC-V

**ARM's memory model (Weakly Ordered):**
- Loads and stores can be reordered in almost any way
- Only restrictions: data dependencies, explicit barriers, and acquire/release atomics

**RISC-V's memory model (RVWMO):**
- Similar to ARM: aggressively relaxed
- `FENCE r,w` instructions provide ordering guarantees

**Why such aggressive relaxation?**
- ARM targets everything from IoT devices to supercomputers
- Weak ordering allows maximum hardware optimization at every power/performance point
- The compiler and programmer are responsible for inserting barriers

**Memory barrier instructions:**

| Architecture | Full Barrier | Store Barrier | Load Barrier |
|-------------|--------------|---------------|--------------|
| x86 | `MFENCE` | `SFENCE` | `LFENCE` |
| ARM | `DMB ISH` | `DMB ISHST` | `DMB ISHLD` |
| RISC-V | `FENCE rw,rw` | `FENCE w,w` | `FENCE r,r` |

![Memory consistency models: SC vs TSO vs WO operation ordering](images/memory-consistency-models.svg)

Note: RISC-V's RVWMO is actually more carefully specified than ARM's model — RISC-V provides a formal axiomatic model in the ISA specification. Both allow significant reordering. In practice, the C11/C++11 memory model provides the best abstraction: `memory_order_acquire`, `memory_order_release`, and `memory_order_seq_cst` map to the minimum necessary barriers on each architecture.

---

## Coherence vs. Consistency: A Concrete Example

**This code is correct under SC but fails on ARM without fences:**
```c
// Initially: X=0, Y=0

// CPU 0:           // CPU 1:
X = 1;              while (Y == 0);   // spin until Y is set
Y = 1;              print(X);         // may print 0 on ARM!
```

**Coherence perspective:** All writes to X propagate ✓, all writes to Y propagate ✓. Coherent!

**Consistency perspective:** CPU 0's `X=1` store may still be in the write buffer when CPU 1 reads X, even though CPU 1 already saw `Y=1`.

**Fix:** `memory_order_release` on `Y=1`, `memory_order_acquire` on the spin.

Note: Coherence says "all writes to X will eventually be seen by everyone in the same order." But it says nothing about WHEN they're seen relative to writes to Y. Consistency fills that gap. This is why both concepts are necessary: coherence for single-variable correctness, consistency for multi-variable ordering. The Coherence hardware guarantees are preserved — the consistency violation is about cross-variable ordering.

---

## Part 5: False Sharing

### The Hidden Performance Killer

> Two processors, two different variables, one cache line — catastrophic performance, silently correct results.

Note: False sharing is insidious because the code is correct — it produces the right answers. But it can run 10-100× slower than expected, and there's no error to alert you. It's one of the most common parallel programming pitfalls discovered during performance optimization.

---

## What is False Sharing?

**Cache coherence operates at cache-line granularity (64 bytes on all modern x86/ARM).**

If two processors access **different variables** that happen to share a **64-byte cache line**, the coherence protocol treats the whole line as shared — even though no data is actually shared.

```c
struct {
    long counter_A;   // Thread 0 writes this   ─┐ same 64-byte
    long counter_B;   // Thread 1 writes this   ─┘ cache line!
} data;
```

**What the hardware sees:**
1. Thread 0 writes `counter_A` → cache line enters M state on CPU 0
2. Thread 1 writes `counter_B` → sends WriteReq for the same cache line
3. CPU 0's line is invalidated; CPU 1 gets the line in M state
4. Thread 0 writes `counter_A` again → sends WriteReq again
5. **The cache line ping-pongs between CPUs indefinitely**

![False sharing: same cache line, different variables, ping-pong](images/false-sharing.svg)

Note: In the worst case, false sharing reduces performance to *below* sequential speed — you have the overhead of constant cache-line invalidation on top of the serial computation. I've seen production systems show 50× slowdown from a single false-sharing struct. It's the most common cause of "parallel code that's slower than single-threaded code."

---

## False Sharing: Quantified Performance Impact

**Benchmark: Two threads increment adjacent counters, 1 billion iterations each**

| Configuration | Time | vs. Sequential |
|---------------|------|----------------|
| Single thread (no parallelism) | 2.1 s | 1.0× |
| Two threads — **false sharing** | 9.8 s | **0.2× (5× SLOWER)** |
| Two threads — padded to separate lines | 1.1 s | 1.9× |
| Two threads — completely separate arrays | 1.0 s | 2.1× |

**The false-sharing case is 4.7× slower than single-threaded.**

The constant cache-line invalidation traffic completely dominates, consuming all available memory bandwidth.

Note: The 5× slowdown relative to sequential is striking. You're not just failing to parallelize — you're actively making it worse. Every time Thread 0 writes, Thread 1's copy is invalidated. Thread 1 must fetch the line, write, and then Thread 0's copy is invalidated. The two CPUs are serializing each other's memory access through the coherence protocol.

---

## Detecting False Sharing

**Linux `perf c2c` — designed specifically for this:**
```bash
# Record cache-to-cache (c2c) events
perf c2c record ./program

# Report: shows lines with high HITM counts
perf c2c report --call-graph

# Key metric: HITM = Hit In The other processor's Modified cache line
# High HITM count → false sharing
```

**What to look for:**
- High `HITM` event count (accesses that hit a Modified line in another core)
- High cache miss rate but most misses are cache-to-cache (not main memory)
- Two specific cache line addresses bouncing between specific CPUs

**Intel VTune:** Shows "Memory Access" issues, highlights false sharing automatically in the GUI.

Note: `perf c2c` has been available since Linux 4.10 and was specifically designed to detect false sharing. It tracks HITM events — accesses that find a Modified line in another core's cache. Normal cache misses go to memory (slow). HITM misses go to another core's cache (also slow, and bandwidth-intensive). A cluster of HITM events on the same cache line address is the fingerprint of false sharing.

---

## Fixing False Sharing: Padding and Alignment

```c
// ❌ BEFORE: False sharing (counter_A and counter_B share a cache line)
struct {
    long counter_A;
    long counter_B;
} data;

// ✓ AFTER: Each counter on its own 64-byte cache line
#define CACHE_LINE 64

struct {
    alignas(CACHE_LINE) long counter_A;
    char _pad_A[CACHE_LINE - sizeof(long)];  // 56 bytes padding
    alignas(CACHE_LINE) long counter_B;
    char _pad_B[CACHE_LINE - sizeof(long)];
} data;

// ✓ BEST (C++17): Portable, self-documenting
#include <new>  // for hardware_destructive_interference_size
struct alignas(std::hardware_destructive_interference_size) PerThreadCounter {
    long value;
};
PerThreadCounter counters[NUM_THREADS];
```

**Result:** Each counter resides on its own cache line. Thread 0's writes never invalidate Thread 1's copy.

Note: `hardware_destructive_interference_size` is the C++17 portable way — it's defined per-platform to equal the actual cache line size. Don't hardcode 64: ARM platforms can have 64 or 128 byte cache lines. In production HPC and systems code, you'll see padding patterns everywhere in concurrent data structures: lock-free queues, thread-local storage, NUMA-aware allocators.

---

## Part 6: Modern CPU Cache Coherence

### How AMD, Intel, and Apple Actually Do It

Note: Real systems combine multiple protocols, hierarchical directories, and hardware-specific optimizations tailored to their die topology. The "correct" protocol depends on your target workload, die area budget, and interconnect topology. Let's see the three leading approaches.

---

## AMD Zen 5: Probe Filtering at Scale

**Architecture:** Up to 16 cores per CCD (Core Complex Die) connected via Infinity Fabric

**Key coherence design choices:**
- **MOESI protocol** within each CCD and across CCDs
- **L3 cache as probe filter:** Before broadcasting a coherence probe across the Infinity Fabric, the L3 checks whether it holds the line. If not → the line is private to its CCD → no need to probe other CCDs
  - Reduces inter-CCD coherence traffic by ~30-50%
- **124 outstanding L1 misses per core** (up from 44 in Zen 4) — allows aggressive latency hiding while waiting for coherence responses
- **Infinity Fabric directory:** Acts as the home node for cross-CCD coherence, with one home region per address range

Note: The probe filter is the key scalability feature. Without it, every L3 miss would broadcast to all 12 CCDs in a Genoa chip to check for dirty copies — saturating the Infinity Fabric with coherence probes. With the probe filter, only misses to lines actually in SOME L3 cache generate probes. AMD claims this reduces Infinity Fabric coherence traffic by 30-50% for typical server workloads (database, HPC, virtualization).

---

## Intel Raptor Lake: MESIF and Adaptive Snoop Modes

**Architecture:** P-cores + E-cores sharing L3 cache ring; Xeon uses mesh interconnect

**Key coherence design choices:**
- **MESIF protocol** — Forward state enables direct cache-to-cache transfers on the ring, eliminating memory from the critical path for read sharing
- **Two snoop modes:**
  - *Home Snoop:* Request goes to home LLC slice → home checks all sharers → lower bandwidth, higher latency
  - *Source Snoop:* Request broadcasts first → lower latency if nearby cache has it → higher bandwidth
- **Dynamic mode selection:** Hardware switches modes based on system load in real time
- **Snoop filter in LLC:** Tracks which cores have copies of which lines; avoids unnecessary probes

Note: The dual snoop mode is Intel's response to the latency vs. bandwidth tradeoff. Under low load, Source Snoop provides minimum latency (direct cache-to-cache). Under high load, Home Snoop reduces bandwidth. The hardware tracks which mode is more efficient. Intel's Xeon Scalable Family has been using variants of this since Skylake-SP (2017) and refined it through each generation.

---

## Apple M-Series: ARM ACE and Zero-Copy GPU

**Architecture:** CPU + GPU on same die, unified LPDDR memory, ARM AMBA ACE coherence fabric

**Key coherence design choices:**
- **ARM ACE protocol (AXI Coherency Extensions):** CPU and GPU participate in the same coherence protocol — the GPU has full read/write coherence with all CPU caches
- **Asymmetric inclusiveness:**
  - CPU L2: *Exclusive* (L1 evictions go to L2, not duplicated)
  - GPU L2: *Inclusive* of GPU L1 (simplifies CPU→GPU probes; CPU only needs to probe GPU L2)
- **Zero-copy data transfer:** GPU reads/writes CPU memory directly. No `memcpy` to GPU buffer.
- **Fabric-level coherence:** All agents (CPU, GPU, Neural Engine, DMA) connect to the interconnect with ACE ports

**Impact:** On Metal, a CPU-written texture can be read by the GPU with zero additional overhead. On discrete GPUs, this transfer typically dominates frame setup time.

Note: Apple's unified memory architecture is only possible because they designed CPU and GPU coherence domains together from scratch. x86+discrete GPU architectures must use PCI-E or NVLink for GPU coherence — both add latency and bandwidth overhead. Apple's M-series shows what's possible when the entire stack is co-designed.

---

## Modern CPU Coherence: Design Comparison

| Architecture | Protocol | Directory Location | Notable Feature |
|-------------|----------|--------------------|-----------------|
| AMD Zen 5 | MOESI | Infinity Fabric | L3 probe filter, 124 MSHRs |
| Intel Raptor Lake | MESIF | LLC slices | Dual snoop modes |
| Apple M4 | ARM ACE | Fabric-integrated | CPU+GPU unified, zero-copy |
| IBM POWER10 | MESI+ | NUMA directory | Memory Clustering Domains |
| Ampere Altra | MESI | ARM CMN-700 mesh | 128-core coherent mesh |

![AMD/Intel/Apple side-by-side protocol choices](images/cpu-coherence-comparison.svg)

Note: There is no single best protocol — each company optimized for their target workload and die topology. AMD's MOESI eliminates writebacks, important for server workloads with many L3 slices. Intel's MESIF optimizes cache-to-cache latency for latency-sensitive applications in multi-socket Xeon systems. Apple's ACE prioritizes CPU-GPU zero-copy for the mobile/laptop workload where copy overhead dominates. The "right" choice is workload-dependent.

---

## Part 7: Heterogeneous Coherence — CPU + GPU

### When Bandwidth Optimization Meets Latency Optimization

> CPUs optimize for latency (single-thread response time). GPUs optimize for throughput (aggregate bandwidth). Making them coherent without destroying both is a hard engineering problem.

Note: This is one of the hottest areas of current architecture research. The GPU's memory subsystem is built for bandwidth — hundreds to thousands of GB/s, with thousands of concurrent threads to hide latency. CPU caches optimize for single-thread latency — a few nanoseconds. Making them coherent without bottlenecking either side requires new protocol designs.

---

## The CPU+GPU Coherence Problem

**CPU memory characteristics:**
- Cache hierarchy: L1/L2/L3 private caches per core
- Coherence at **64-byte line granularity**
- Optimized for **latency** (single-digit ns for L1 hit)

**GPU memory characteristics:**
- HBM (H100: 3.35 TB/s bandwidth)
- Optimized for **throughput** (thousands of threads in flight)
- Coherence typically at **128-byte or page granularity** (or none)

**The mismatch:**
- CPU-style fine-grained coherence on GPU → too many coherence messages saturate the NVLink/PCIe interconnect
- GPU-style coarse coherence on CPU → CPU latency skyrockets (must flush pages, not lines)

![CPU+GPU shared memory with coherence traffic challenge](images/cpu-gpu-coherence.svg)

Note: The "two separate memory pools" model (pre-2020 GPU computing) was the industry's pragmatic answer: just don't make them coherent. CPU copies data to GPU memory, GPU computes, GPU copies results back. Zero coherence overhead, but massive data movement latency. With NVLink and CXL, we're now building systems where this copy overhead is the primary bottleneck — hence the push for hardware coherence.

---

## AMD's Approach: Selective Caching

**Key insight:** Not all GPU data needs to be coherent with the CPU.

**Selective caching (AMD CDNA architecture and APUs):**
- GPU memory pages are tagged as **coherent** or **non-coherent** in the page table
- Coherent pages: GPU uses the standard CPU coherence protocol (slower but correct for shared data)
- Non-coherent pages: GPU uses its own high-bandwidth memory subsystem (no coherence overhead for GPU-private data)

```cuda
// Non-coherent: GPU-private, maximum bandwidth
float* gpu_buf = allocate_device(size);        // non-coherent HBM

// Coherent: shared with CPU, explicit protocol
float* shared  = allocate_coherent(size);      // participates in CPU coherence
```

**Result:** ~3× bandwidth improvement vs. fully-coherent GPU for typical GPU-only compute workloads, with correct coherence for the subset that needs it.

Note: AMD's ROCm runtime and HIP handle this automatically for common patterns (host-device data transfers). The programmer doesn't need to manually tag pages in most cases. But performance-critical HPC code often does explicit management to maximize GPU bandwidth on the non-coherent allocations.

---

## Region Directories: Coarse-Grained GPU Coherence

**Problem:** Even for coherent GPU data, 64-byte line coherence is too fine for GPU access patterns (GPU threads access 128+ byte aligned, bulk sequential regions).

**Region directory approach:**
- Track coherence at **page granularity (4KB)** rather than line granularity
- If the CPU has no dirty lines in a region, the GPU can bypass coherence entirely for that region
- Only pages with recent CPU writes need invalidation before GPU access

**Message reduction math:**
```
Naive fine-grained: 64 invalidations per 4KB page (64 lines × 64B)
Region directory:    1 page-level invalidation
→ 64× reduction in coherence messages for bulk GPU accesses
```

**Trade-off:** Some false sharing at page granularity — a page may be partially dirty, requiring more invalidation than strictly necessary.

Note: Region directories are used in AMD's NUMA GPU systems and in Apple's M-series for CPU-GPU data sharing. The coarser granularity means some waste (a page might be 50% dirty, forcing full invalidation) but the bandwidth savings for GPU workloads usually outweigh the false-sharing cost, because GPU access patterns are typically large and regular.

---

## NVIDIA HMG: Hierarchical Multi-GPU Coherence

**Problem:** Multi-GPU nodes (DGX H100: 8 × H100 GPUs) need inter-GPU coherence. Software-managed coherence requires explicit `cudaMemcpy` between GPUs — high programmer burden and latency.

**NVIDIA HMG (Hierarchical Memory & Coherence, SC '22):**
- **Two-level directory hierarchy:**
  - Per-GPU local directory: tracks lines in that GPU's L2 cache
  - Global inter-GPU directory: tracks cross-GPU sharing over NVLink
- **Scope-based coherence:** GPU threads declare the coherence scope of their accesses (warp, CTA, device, system)

**Measured results (SC '22 paper):**
- 26% reduction in coherence traffic vs. software invalidation
- Bandwidth scales with GPU count up to 8 GPUs
- Enables GPU threads to directly read data cached in another GPU's L2

Note: HMG is significant because it brings CPU-style hardware coherence to multi-GPU systems. Previously, GPU-to-GPU data sharing required explicit memory copies — the programmer had to know the data location and issue the copy. With HMG, a CUDA thread can access data in another GPU's cache with hardware-maintained coherence, enabling new programming models like unified GPU cluster memory.

---

## Part 8: Emerging Topics

### CXL, Chiplets, and the Next Frontier

---

## CXL: Compute Express Link

**What is CXL?** An open standard (v1.0: 2019, v3.0: 2022) built on PCIe physical layer that adds three coherent protocols:

| Sub-protocol | Direction | What It Enables |
|-------------|-----------|----------------|
| `CXL.io` | Host ↔ Device | Standard PCIe I/O |
| `CXL.cache` | Device → Host | Accelerator caches host memory (coherently) |
| `CXL.mem` | Host → Device | Host accesses device-attached memory (coherently) |

**Why CXL matters:**
- Accelerators (FPGAs, AI chips, SmartNICs) join the CPU's coherence domain
- Memory pooling: multiple CPUs share a "pool" of CXL-attached DRAM
- CXL 3.0 switches: multi-host, multi-device coherence fabrics

**Practical example:** A 2TB CXL memory expander appears as regular RAM to Linux — cached by CPU caches, with full coherence — just higher latency (~100 ns additional).

![CXL device and chiplet topology with coherence domains](images/cxl-chiplet-coherence.svg)

Note: CXL is likely the most important architectural development of the 2020s. It breaks the assumption that coherence exists within one chip or socket. Intel Sapphire Rapids (2023) introduced CXL 1.1 support. AMD Genoa (2022) added CXL 1.1. CXL 3.0's fabric feature allows building disaggregated systems where memory is a service, accessed coherently over PCIe by any compute device on the network.

---

## CXL Coherence: Granularity Modes

**CXL.cache provides two coherence modes:**

| Mode | Granularity | Use Case |
|------|-------------|----------|
| **Fully coherent** | 64-byte cache lines | CPU ↔ AI accelerator with fine-grained sharing |
| **Memory-mapped coherent** | 4KB pages | CPU ↔ memory expander (mostly read-heavy) |

**CXL.mem host-side caching:**
- CPU can cache lines from device memory (home node = device)
- Device memory controller runs a directory protocol for cached lines
- Uncached regions are accessed like MMIO (no coherence overhead)

**Multi-host CXL (CXL 3.0):**
- Up to 16 hosts share a CXL fabric
- Each host can cache regions of pooled memory
- Hardware coherence across all hosts — without any software involvement

Note: CXL 3.0's multi-host coherence is the key differentiator. Earlier versions allowed one host to coherently access one device. CXL 3.0 allows many hosts to coherently share memory — essentially NUMA-across-boxes, enabled by hardware coherence. This enables rack-scale memory pooling: 10 servers share 10TB of CXL DRAM, each server caching the regions it accesses most.

---

## Chiplet Coherence: Inter-Die Bandwidth Contention

**Modern chiplet systems:**
- AMD EPYC Genoa: 12 CCDs + 1 I/O die → up to 96 cores
- Intel Ponte Vecchio (datacenter GPU): 47 chiplets on one package
- Apple M2 Ultra: Two M2 Max dies connected via die-to-die interconnect

**The chiplet coherence challenge:** Die-to-die interconnects have **finite bandwidth shared between coherence traffic and data traffic.**

```
AMD EPYC Genoa — xGMI inter-CCD bandwidth: ~800 GB/s total
  → Data traffic (computation results):        ~600 GB/s
  → Coherence control messages:                ~200 GB/s
If coherence traffic > budget → data bandwidth is stolen → performance collapse
```

**AMD's solution:** L3 probe filter reduces coherence traffic by filtering out probes for privately-cached lines — preserving data bandwidth on the xGMI fabric.

Note: This is a hard real engineering constraint in chiplet design. Every coherence message sent across the die-to-die interconnect consumes bandwidth that could carry data. The probe filter, SCD-style directory compression, and careful thread/data placement are all tools for keeping coherence traffic under budget. Intel's Ponte Vecchio had to carefully partition on-package mesh bandwidth to avoid coherence traffic starving compute traffic.

---

## Why Software Must Become Topology-Aware

**In chiplet and CXL systems, performance depends on topology in new ways:**

Software must consider:
1. **Core-to-core distance:** Same CCD (fast coherence) vs. different CCDs (Infinity Fabric hop, ~200ns extra)
2. **Thread-to-data affinity:** Is shared data in a cache near both threads?
3. **Coherence domain boundaries:** Where do protocols change? (snooping within CCD, directory across CCD)
4. **CXL latency:** Is accessed memory local DRAM (~80ns) or CXL-attached (~180ns)?

**Practical tools:**
```bash
# Show NUMA topology
numactl --hardware
lstopo --of png topology.png   # hwloc: visual topology map

# Pin threads to cores near their data
numactl --cpunodebind=0 --membind=0 ./program

# Profile for cross-NUMA access
perf stat -e dtlb_load_misses.miss_causes_a_walk \
          -e offcore_requests.all_data_rd ./program
```

Note: Operating systems are increasingly topology-aware (Linux's NUMA scheduler, Windows Core Parking). But application-level optimization still matters enormously for HPC and database workloads. Libraries like MPI, OpenMP, and Intel TBB provide topology-aware task placement APIs. Understanding the hardware's coherence topology is now a first-class performance engineering skill.

---

## Part 9: Summary

---

## Cache Coherence: The Big Picture

**Protocol evolution — each step eliminates one class of waste:**

```
MSI (3 states)
  └→ MESI (4): +Exclusive → eliminates BusUpgr for private data
       └→ MOESI (5): +Owner → eliminates memory writeback on sharing
       └→ MESIF (6): +Forward → eliminates memory on read-sharing response
            └→ MOESIF (6): +Owner+Forward → eliminates both
```

**Scaling evolution — each step reduces message complexity:**

```
Bus snooping (≤32 cores, O(N) broadcast)
  └→ Directory full-map (≤128 cores, O(1) targeted)
       └→ Limited pointer (≤512 cores, O(k) targeted)
            └→ SCD variable (≤1024+ cores, ~5% overhead)
                 └→ CXL fabric (cross-socket, cross-device)
```

**The universal design principle:**
> **No free lunch.** Every coherence protocol trades hardware complexity (extra states, protocol logic, verification effort) for eliminating a class of redundant work (bus upgrades, writebacks, broadcasts, memory reads). The best protocol minimizes total cost for your specific workload, die topology, and target core count.

Note: Every major concept in this lecture was motivated by a specific bottleneck in an earlier design. MSI→MESI to eliminate private-data upgrade traffic. Snooping→directory to eliminate broadcast at scale. Full-map→SCD to eliminate directory storage overhead. CXL to eliminate copy overhead for accelerators. The pattern is universal: identify the dominant cost, add mechanism to eliminate it, accept the added complexity. This is how computer architecture evolves.
