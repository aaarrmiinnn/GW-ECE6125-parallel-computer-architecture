# Parallel Programming II
## Communication, Scaling, and Patterns
### GWU ECE 6125: Parallel Computer Architecture

---

## Lecture Roadmap

| Part | Topic | Key Question |
|------|-------|-------------|
| 1 | **Communication Costs** | Why is moving data (not computing) the real bottleneck? |
| 2 | **Communication Patterns** | Broadcast, scatter, gather, all-to-all, stencil: when to use which? |
| 3 | **Scaling Laws in Depth** | When does doubling processors double performance? |
| 4 | **Parallel Patterns** | What reusable templates solve most parallel problems? |
| 5 | **Programming Models** | MPI, OpenMP, CUDA, PGAS, Spark: how do we pick? |
| 6 | **Case Study** | End-to-end: parallel matrix multiply with real numbers |
| 7 | **Modern Context** | Heterogeneous, cloud, energy, fault tolerance |
| 8 | **Pitfalls & Wrap-Up** | What goes wrong, and how to avoid it |

Note: Lecture 7 gave you the four-step framework (decompose → assign → orchestrate → map) and Amdahl's Law. This lecture goes deeper into what makes real parallel programs fast or slow. The answer is almost always communication, not computation. We'll assume you already know the basics of decomposition, synchronization, and SPMD from Lecture 7.

---

## Part 1: Communication Costs

### Why moving data dominates everything

Note: Before we can pick the right programming model or pattern, we need to understand how expensive communication actually is. The numbers are shocking, and they shape every design decision in parallel computing.

---

## The Central Truth of Parallel Computing

> **Modern hardware computes far faster than it moves data. Performance is almost always limited by communication, not computation.**

A single CPU core in 2026 can execute ~10 billion floating-point operations per second. But reading one value from DRAM takes ~100 nanoseconds, during which the core could have done **1,000 floating-point operations**.

**Analogy:** Imagine a chef who can chop vegetables at superhuman speed, but the pantry is a 10-minute walk away. The chef's peak speed is irrelevant because the walking time decides how many meals get made.

> **Think about it:** What strategies can we use to hide or reduce the cost of moving data? We've already seen the hardware's answers (caches, coherence protocols, fast interconnects). Today's question: what can the *programmer* do?

Note: This fact explains almost everything about modern parallel architecture. Why do we have caches? To avoid the walk. Why are GPUs fast for ML? Because matrix multiply does O(n³) work on O(n²) data, so the ratio of compute to communication is high. Why is all-reduce the hot topic in distributed training? Because it's the step that limits how many GPUs you can usefully throw at a model. The question at the end bridges prior lectures (hardware solutions) to today's content (software solutions): overlapping communication with computation, choosing the right collective, raising arithmetic intensity, and picking the right programming model.

---

## Latency, Bandwidth, and Arithmetic Intensity

Three numbers tell you whether your program is <span class="accent">communication-bound</span> or <span class="accent">compute-bound</span>:

| Metric | Measures | Bottleneck Sign | Fix |
|---|---|---|---|
| **Latency** ($\alpha$) | Fixed startup cost per message | Many small messages | Batch messages together |
| **Bandwidth** ($\beta$) | Max transfer rate (GB/s) | Moving large volumes | Send less data, compress |
| **Arith. Intensity** | FLOPs / bytes moved | Low ratio = <span class="accent">IO-bound</span> | Restructure algorithm, block data |

$$T_{\text{message}} = \alpha + \frac{n}{\beta}$$

> **The diagnostic:** Compute your kernel's arithmetic intensity. If it's low, you're <span class="accent">IO-bound</span>, so move less data or do more work per byte. If it's high, you're <span class="accent">compute-bound</span>, which is the good case.

Note: This is the single most important slide in the lecture. Every performance question in parallel computing reduces to: am I moving too much data (bandwidth-bound), sending too many small messages (latency-bound), or actually limited by compute (rare, and the good case)? Arithmetic intensity is the number that answers this instantly. We'll see it again in the roofline model two slides from now.

---

## Message Time: Latency vs. Bandwidth Regimes

![Message time vs message size: latency vs bandwidth regimes](images/message_time_curve.svg)

- **Left of crossover (~few KB):** Sending 1 byte costs almost the same as sending 1000 bytes because latency dominates
- **Right of crossover:** Doubling the message doubles the time because bandwidth dominates
- **Practical implication:** Batching many small messages into one large message can be a massive win

Note: This curve explains why MPI programmers batch communications and why GPU programmers fuse kernels. If you're on the left side of the curve, reducing message count matters more than reducing message size. If you're on the right side, reducing how much data you send is what matters.

---

## A Table You Should Memorize

Approximate costs on a modern system (2026):

| Operation | Latency | Bandwidth |
|---|---|---|
| L1 cache access | ~1 ns | ~1 TB/s |
| L2 cache access | ~4 ns | ~600 GB/s |
| L3 cache access | ~10 ns | ~400 GB/s |
| Local DRAM access | ~100 ns | ~100 GB/s |
| Remote DRAM (NUMA) | ~200 ns | ~50 GB/s |
| NVLink (GPU↔GPU, same node) | ~1 μs | ~900 GB/s |
| InfiniBand (node↔node) | ~1–2 μs | ~50 GB/s |
| Ethernet (cloud, cross-rack) | ~50 μs | ~25 GB/s |
| Cross-region network (WAN) | ~50 ms | varies |

Note: Every step down this table is roughly 10× slower than the one above it. Good parallel algorithms structure themselves around this hierarchy: do as much work as possible at the top of the table, and move data across the expensive boundaries only when absolutely necessary. This is why GPU programmers obsess over shared memory and coalesced access: they are fighting this table at every level.

---

## The Roofline Model

The roofline answers one question: **is my program limited by computation or by data movement?**

![Roofline model: arithmetic intensity vs achievable performance](images/roofline_model.svg)

- **Diagonal slope** (left) = memory bandwidth ceiling, where performance rises with arithmetic intensity
- **Flat roof** (right) = peak FLOP/s ceiling, the processor's maximum
- **Ridge point** = where you transition from <span class="accent">IO-bound</span> to <span class="accent">compute-bound</span>

> If your kernel is on the slope, a faster processor won't help. Move less data or restructure the algorithm.

*Kernels on the plot: **SpMV** = sparse matrix-vector multiply (~2 FLOPs/byte, IO-bound). **Stencil** = neighbor grid update (~5-10). **GEMM** = dense matrix multiply (~50+, compute-bound).*

Note: The roofline model was popularized by Sam Williams at Berkeley around 2009 and is now the standard way performance engineers reason about kernels. NVIDIA Nsight, Intel Advisor, and AMD uProf all generate roofline plots automatically. If your point is well below the roof, there's optimization headroom; if it's on the roof, you need algorithmic or architectural changes.

---

## The Cost of Synchronization

Communication isn't just data movement. Waiting counts too.

| Event | Typical Cost |
|---|---|
| Lock/unlock (uncontended) | ~20 ns |
| Lock (contended) | ~1 μs and up |
| Barrier across 16 threads (node) | ~1 μs |
| Barrier across 1000 MPI ranks | ~100 μs–1 ms |
| GPU kernel launch | ~5–10 μs |

> **Rule of thumb:** If you synchronize more often than you compute, no algorithm will scale.

Note: This is why kernel fusion is such a big deal on GPUs: each kernel launch costs 10 μs, so if you have 100 tiny kernels you spend a millisecond just launching them. Fusing ten kernels into one saves 90 μs. The same idea applies to MPI: batching messages and overlapping communication with computation is the bread and butter of HPC optimization.

---

## Overlapping Communication with Computation

The single most effective optimization: **don't wait for messages**.

```c
// BAD: wait for receive, then compute (CPU idle during transfer)
MPI_Recv(buffer, ...);
compute(buffer);

// GOOD: start receive, compute on local data, then wait
MPI_Irecv(buffer, ..., &request);
compute_local_data();          // overlaps with network transfer
MPI_Wait(&request, &status);
compute(buffer);
```

- `MPI_Irecv` is **non-blocking**: it returns immediately and the NIC fills the buffer in the background
- Hide as much communication latency as you can behind useful work
- The best case: communication takes zero observable time

Note: Non-blocking communication is the standard practice in production HPC codes. The technique applies everywhere: async I/O in web servers, CUDA streams on GPUs, prefetching in CPU caches. All the same principle: start the slow thing early, do other work in the meantime, and ideally never wait.

---

## Recap: Synchronization and Ordering Mechanisms

A quick reference of every mechanism that controls **when** and **in what order** parallel work happens:

| Mechanism | What It Does | Scope |
|---|---|---|
| **Mutex / Lock** | Only one thread enters the critical section at a time | Shared memory (threads) |
| **Atomic operation** | Read-modify-write in one hardware instruction (CAS, fetch-add) | Shared memory (threads) |
| **Barrier** | All threads/ranks must arrive before any proceed | Threads or MPI ranks |
| **Semaphore** | Allow up to N concurrent accesses | Threads |
| **Memory fence** | Force all prior loads/stores to complete before proceeding | Single thread; controls what *other* threads see |
| **Acquire / Release** | Fence variants: acquire = "see all writes before the lock"; release = "flush my writes before unlocking" | Lock/unlock boundaries |
| **Volatile / _Atomic** | Compiler: don't optimize away or reorder this access | Single variable |
| **MPI_Barrier** | Global synchronization across all ranks | Distributed (MPI) |
| **MPI_Wait / Waitall** | Block until non-blocking operation completes | Distributed (MPI) |
| **cudaDeviceSynchronize** | Host waits for all GPU kernels to finish | GPU stream |
| **cudaStreamSynchronize** | Host waits for one specific GPU stream | GPU stream |

> **Fences vs. locks:** A lock protects a *section* of code. A fence controls *memory visibility*, ensuring other cores see your writes in the right order. You often need both: the lock for mutual exclusion, the fence (built into the lock) for ordering.

Note: Students often confuse fences with locks. A fence doesn't block other threads; it tells the hardware "make sure my writes are visible before I continue." Locks use fences internally (acquire fence on lock, release fence on unlock), but you can also use standalone fences for lock-free algorithms. The C11/C++11 memory model formalizes this with memory_order_acquire, memory_order_release, and memory_order_seq_cst. We covered the hardware side (store buffers, invalidation queues) in Lecture 6 on cache coherence.

---

## Part 2: Communication Patterns

### The vocabulary of parallel data exchange

Note: Nearly every parallel algorithm uses one or more of these patterns. Learning to recognize them lets you reach for the right collective call instead of hand-coding point-to-point messages.

---

## Communication Patterns Are Universal

Every time parallel tasks need to share data, the programmer chooses a **pattern**. The wrong choice can be 100× slower than the right one.

These patterns are **not MPI-specific**. They appear in every framework:

| Framework | How You Access Them |
|---|---|
| **MPI** | `MPI_Bcast`, `MPI_Scatter`, `MPI_Allreduce`, ... |
| **NCCL** (GPU) | `ncclBroadcast`, `ncclAllReduce`, ... |
| **OpenMP** | `#pragma omp for reduction(+:sum)`, implicit barriers |
| **Spark / Dask** | `reduceByKey`, `groupBy`, `broadcast` variables |
| **CUDA** | Warp shuffles, shared memory, cooperative groups |

> The API names change; the ideas don't.

Note: This is a key insight students miss: they think collectives are an MPI thing. In reality, every parallel framework implements the same small set of patterns because the underlying math of data movement is the same regardless of the programming model.

---

## Collectives: One-to-Many and Many-to-One

![Broadcast, scatter, gather patterns](images/collectives_1.svg)

| Pattern | Meaning | Typical Use |
|---|---|---|
| **Broadcast** | One process sends the *same* data to all | Distributing model weights, config |
| **Scatter** | One process sends *different* pieces to each | Handing out work chunks |
| **Gather** | Every process sends its piece to one collector | Assembling partial results |

Note: These three cover the most common data distribution needs. Broadcast is by far the most frequent because any time every worker needs the same configuration, weights, or parameters.

---

## Why Collectives Beat Point-to-Point

Why not just write a loop of sends?

```c
// NAIVE: root sends to each process one at a time -- O(P) steps
for (int i = 1; i < num_procs; i++)
    MPI_Send(data, size, MPI_INT, i, tag, comm);

// BETTER: one call, library uses a tree internally -- O(log P) steps
MPI_Bcast(data, size, MPI_INT, root, comm);
```

**The difference is a tree vs. a chain:**

- **Naive loop:** Root sends P-1 messages, one after another. At P=1024, that's 1024 steps.
- **Tree broadcast:** Root sends to 2 nodes, they each send to 2 more, and so on. At P=1024, that's only 10 steps (log₂ 1024).

> **Rule:** Never hand-roll collectives. The library is faster because it uses tree algorithms, pipelining, and sometimes even hardware offload (InfiniBand switches can do reductions *inside the network*).

Note: This is not a minor optimization. It's the difference between O(P) and O(log P). At scale this matters enormously. Every major library (MPI, NCCL, Gloo) has spent years tuning these implementations. Hand-rolling a broadcast loop is one of the most common performance mistakes in parallel code.

---

## Collectives: All-to-All and Reductions

![Reduce, all-reduce, all-to-all patterns](images/collectives_2.svg)

| Pattern | Meaning | Typical Use |
|---|---|---|
| **Reduce** | Combine values from all → one result on one process | Sum, max, min |
| **All-reduce** | Combine, and the result ends up on *every* process | Averaging gradients in ML training |
| **All-to-all** | Every process sends a different chunk to every other | FFT, matrix transpose |

> **Key distinction:** Reduce sends the result to *one* process. All-reduce sends it to *all*. All-to-all is the most expensive because every process talks to every other.

Note: When do you pick which? Reduce when only the root needs the answer (e.g., printing a final result). All-reduce when every process needs the answer to continue (e.g., every GPU needs the averaged gradients for the next training step). All-to-all when the data needs to be completely rearranged (e.g., transposing a distributed matrix). All-reduce is by far the most common in practice.

---

## Deep Dive: How All-Reduce Actually Works

All-reduce is the most performance-critical collective in modern computing. Every step of training a neural network across N GPUs ends with:

**"Average all the gradients, and give every GPU the result."**

The naive approach (gather everything to one node, sum, broadcast back) creates a bottleneck at that single node. The solution: **ring all-reduce**.

**How ring all-reduce works (P = 4 GPUs):**

1. Arrange GPUs in a logical ring: GPU0 → GPU1 → GPU2 → GPU3 → GPU0
2. **Reduce-scatter phase** (P-1 steps): each GPU sends a chunk to its neighbor and accumulates. After this, each GPU holds 1/P of the final sum.
3. **All-gather phase** (P-1 steps): each GPU passes its completed chunk around the ring until everyone has the full result.

**Why it's optimal:** Every GPU sends and receives at full bandwidth simultaneously.

$$\text{Data per GPU} = 2 \cdot \frac{P-1}{P} \cdot N$$

where $P$ = number of GPUs and $N$ = total message size in bytes. As $P$ grows, $\frac{P-1}{P} \approx 1$, so each GPU moves ~$2N$ bytes regardless of how many GPUs there are.

> This is why NVIDIA built NCCL, why Meta obsesses over network topology, and why a bad switch can slow down LLM training by 2×.

Note: Ring all-reduce was popularized by Baidu researchers around 2017 and adopted by Horovod, then NCCL. Modern implementations also use tree all-reduce for latency-sensitive small messages (trees have O(log P) latency vs. ring's O(P) latency, but ring wins on bandwidth for large messages). In practice, NCCL picks the best algorithm automatically based on message size and topology.

---

## Stencils and Neighbor Exchange

Many scientific codes compute each point from its **neighbors**:

```c
new[i][j] = 0.25 * (old[i-1][j] + old[i+1][j]
                  + old[i][j-1] + old[i][j+1]);
```

When we split the grid across processes, **edge values** live on the neighbor:

![Stencil halo exchange across process boundaries](images/stencil_halo.svg)

- Each process owns a **block** of the grid
- It keeps a **halo** (ghost region) with copies of the neighbors' edges
- Before each step, processes exchange halos with their neighbors

> **Scalability is good**: communication is O(boundary), computation is O(area). Doubling the grid per processor halves the communication-to-computation ratio.

Note: This is the pattern behind weather simulation, computational fluid dynamics, seismic imaging, and finite-element analysis. The halo exchange is usually the bottleneck, and it's where non-blocking sends really shine. Start the exchange, compute the *interior* of the block (which doesn't need neighbor data), then wait and compute the boundary.

---

## Picking the Right Pattern

A decision table for common situations:

| You Need To… | Use |
|---|---|
| Send the same data to everyone | Broadcast |
| Split work among workers | Scatter |
| Assemble results | Gather |
| Compute a global statistic | Reduce / All-reduce |
| Rearrange data globally (e.g., FFT, transpose) | All-to-all |
| Share edges in a grid algorithm | Neighbor / halo exchange |
| Merge streaming updates | Reduction tree |

> **Wrong pattern = wasted bandwidth.** An all-to-all when broadcast would do is 100× more expensive.

Note: The biggest performance wins in real HPC codes come from *replacing* communication patterns, not tuning them. Someone writes a naive version that does O(P²) point-to-point messages, you notice it's really an all-reduce, and swap in MPI_Allreduce for a 100× speedup, no other changes. Pattern recognition is the single highest-leverage skill in performance engineering.

---

## Part 3: Scaling Laws in Depth

### When doubling processors actually doubles performance

Note: Lecture 7 introduced Amdahl's Law. Here we go deeper: what do "strong" and "weak" scaling really mean, when does each apply, and why is weak scaling the only thing that matters for modern ML?

---

## Strong Scaling vs. Weak Scaling

The two questions you can ask about parallel performance:

| Scaling | What's Fixed | What Changes | Question Answered |
|---|---|---|---|
| **Strong** | Problem size | Processor count | *Can I finish the same work faster?* |
| **Weak** | Work per processor | Problem size and processors grow together | *Can I solve a bigger problem in the same time?* |

![Strong vs weak scaling](images/strong_vs_weak_scaling.svg)

> **Strong scaling hits a wall.** As you add processors to a fixed-size problem, the per-processor work shrinks until communication dominates. Speedup plateaus, often well before the theoretical maximum.

> **Weak scaling is more forgiving.** Communication usually grows slower than computation when the problem grows, so efficiency holds up much better.

Note: Here's the real-world punch line: modern ML training is designed around weak scaling. You don't train GPT-4 on a fixed dataset "faster" by adding more GPUs; you train it on a *larger* model or larger batch. Amdahl's Law tells you strong scaling is hopeless past a few hundred GPUs. Gustafson's Law tells you weak scaling can reach tens of thousands, and this is what actually happens in every large training cluster.

---

## Amdahl vs. Gustafson: The Same Equation, Different Assumptions

Where $s$ = serial fraction (portion that can't be parallelized) and $P$ = number of processors.

**Amdahl (fixed problem size):**

$$\text{Speedup}_{\text{Amdahl}} = \frac{1}{s + \frac{1-s}{P}}$$

- As $P \to \infty$, speedup $\to 1/s$
- If $s = 5\%$, speedup is capped at 20×, no matter how many processors

**Gustafson (fixed time, scale the problem):**

$$\text{Speedup}_{\text{Gustafson}} = s + (1-s) \cdot P$$

- As $P \to \infty$, speedup grows linearly
- Adding processors lets you solve a proportionally larger problem in the same time

> **They don't contradict each other.** Amdahl asks "how fast can I finish *this* problem?" Gustafson asks "how big a problem can I solve in *this* time?"

Note: When people say "Amdahl was wrong," they usually mean "we shouldn't optimize for fixed problem sizes." That's fair for HPC and ML, where the goal is often to solve problems that were previously impossible. But Amdahl is *still* the right answer if your problem size is genuinely fixed, for example, a real-time simulation that must finish in 16 ms per frame. Know which question you're asking.

---

## Why Efficiency Drops as P Grows

Parallel efficiency = speedup ÷ P. Four forces push it below 1.0:

| Force | Why It Kills Efficiency |
|---|---|
| **Serial fraction** | Non-parallelizable code sets a hard ceiling (Amdahl) |
| **Communication overhead** | More processors → more messages and more synchronization |
| **Load imbalance** | The slowest processor sets the pace, and idle time wastes resources |
| **Contention** | Shared resources (memory, network, locks) saturate |

> **Quick diagnostic:** Run at P=2, 4, 8, 16, 32. Plot efficiency. The shape tells you the cause:

![Efficiency curve shapes and what they diagnose](images/efficiency_curves.svg)

Note: This is the most practical slide in Part 3. Before reaching for a profiler, just plotting efficiency at a few values of P tells you where to look. Different causes need different fixes: communication overhead needs batching or overlap, load imbalance needs dynamic scheduling, serial fraction needs algorithmic redesign, contention needs fewer shared resources.

---

## Part 4: Parallel Patterns

### Reusable templates for parallel computation

Note: Just as object-oriented programming has design patterns (Factory, Observer, Strategy), parallel programming has a small set of templates that cover most problems. Learning these lets you recognize "oh, this is a map-reduce" and reach for the right library instead of reinventing the wheel.

---

## Map, Reduce, and Map-Reduce

**Map**: apply the same independent operation to every element.

```python
# Python / multiprocessing
result = pool.map(square, [1, 2, 3, 4, 5])
# → [1, 4, 9, 16, 25]
```

**Reduce**: combine all elements using an associative operation (+, max, ...).

```python
total = sum(result)
```

**Map-Reduce**: map first, then reduce. This is trivially parallel because both steps have no dependencies between elements.

> **Why it scales:** Map is embarrassingly parallel. Reduce is O(log P) with a tree. Together they scale to thousands of machines.

Note: Google's original MapReduce paper (2004) launched the big-data era. The insight: express computation as map + reduce, and the framework handles distribution, fault tolerance, and scheduling automatically.

---

## Map-Reduce Is Everywhere

More problems are map-reduce than you'd think:

| Problem | Map | Reduce |
|---|---|---|
| Word counting | Count words in each chunk | Sum counts |
| <span class="accent">ML training</span> | Each GPU computes gradients | All-reduce to average |
| Image rendering | Each core renders a tile | Stitch tiles |
| Log analysis | Each node filters its logs | Merge results |

Note: The hard part is recognizing that a problem *is* map-reduce in disguise. Once you see it, frameworks like Spark, Dask, Ray, and BigQuery give you parallelism for free.

---

## Fork-Join and Task Parallelism

**Fork-join**: split work into subtasks, run them in parallel, wait at a join point.

```c
#pragma omp parallel       // FORK: spawn a team of threads
{
    #pragma omp for        // split loop iterations across threads
    for (int i = 0; i < N; i++) {
        a[i] = heavy_work(i);
    }
}                          // JOIN: all threads wait here, then one continues
```

- Works naturally for **recursive divide-and-conquer**: quicksort, tree traversals
- Each branch can spawn more branches, and load balances via <span class="accent">work stealing</span> (idle processors steal from busy ones)

> **How is this different from map-reduce?** Map-reduce is <span class="accent">flat</span>: one map phase, one reduce phase. Fork-join is <span class="accent">recursive</span>: subtasks spawn more subtasks of unpredictable size. Use map-reduce when every piece is the same shape; use fork-join when the work is irregular.

*Cilk (MIT, 1994) pioneered work stealing for fork-join. Its ideas live on in Intel TBB, OpenMP tasks, and Java's ForkJoinPool.*

Note: Quicksort is the canonical fork-join example: each partition creates two subproblems of unpredictable size. Static assignment would leave processors idle; work stealing handles this gracefully because idle processors grab work dynamically.

---

## Pipeline Parallelism

Different stages run in parallel on different data items, like an assembly line.

![Pipeline parallelism: stages overlap on different inputs](images/pipeline.svg)

<span class="accent">Throughput = 1 / (slowest stage)</span>. The programmer designs the stages and assigns each to a processor.

Note: CPU instruction pipelines use this idea in hardware, but here we're talking about the software version the programmer explicitly builds. Pipeline parallelism is also the key technique for LLM training, where models too big for one GPU split layers across GPUs, with micro-batches flowing through (GPipe, PipeDream).

---

## Pipeline Example: CUDA Streams

Overlap data transfer and computation by putting them in different streams:

```c
// Without pipeline: transfer ALL data, then compute ALL
cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);
kernel<<<blocks, threads>>>(d_data, N);

// With pipeline: split into chunks, overlap transfer and compute
for (int i = 0; i < NUM_CHUNKS; i++) {
    // Stream i: copy chunk i while stream i-1 computes
    cudaMemcpyAsync(d_chunk[i], h_chunk[i], chunk_size,
                    cudaMemcpyHostToDevice, stream[i]);
    kernel<<<blocks, threads, 0, stream[i]>>>(d_chunk[i], chunk_size);
}
```

- **Without pipeline:** GPU sits idle during transfer, then CPU sits idle during compute
- **With pipeline:** transfer of chunk N+1 overlaps with compute on chunk N

Note: This is one of the most common CUDA optimizations. The same idea applies with thread pools: one thread decodes, another processes, another writes, each working on a different item. Unix pipes (`cat file | grep pattern | sort`) are the simplest version: three processes in a pipeline, the OS handles the handoffs.

---

## Stencil / Structured Grid Pattern

Each cell updates from its neighbors (we saw the halo exchange in Part 2).

<div class="cols">
<div class="left">

```c
// 2D heat equation: average of 4 neighbors
new_u[i][j] = 0.25 * (u[i-1][j] + u[i+1][j]
                     + u[i][j-1] + u[i][j+1]);
```

- **Used in:** weather, seismic imaging, fluid dynamics, image processing
- **Scales well:** border exchange is O(√N), interior compute is O(N). More grid = less relative communication.

</div>
<div class="right">

![Stencil halo exchange](images/stencil_halo.svg)

</div>
</div>

Note: Any physics on a grid is likely a stencil. Stencils are the reason supercomputers exist: climate models, nuclear simulations, structural analysis. The pattern is so important that specialized DSLs (Halide, Exo) exist just to optimize stencil kernels.

---

## Task Graphs and Dataflow

A **DAG** (Directed Acyclic Graph) of tasks:

- **Directed:** edges point from producer to consumer
- **Acyclic:** no circular dependencies, so the runtime can always find a ready task

<div class="cols">
<div class="left">

- Each node is a task; each edge is a data dependency
- The runtime runs any task whose inputs are ready
- <span class="accent">Critical path</span> = longest path through the DAG = minimum wall-clock time
- TensorFlow, PyTorch, Dask, and Ray all build task graphs under the hood

</div>
<div class="right">

![Task DAG with dependencies](images/task_graph.svg)

</div>
</div>

Note: This is how deep learning frameworks get parallelism without asking you to manage threads. When you write `y = model(x)` in PyTorch, it silently builds a DAG of tensor operations that can be fused, reordered, and dispatched to CPU or GPU. CUDA graphs go further: capture a DAG once and replay it thousands of times with zero kernel-launch overhead.

---

## Part 5: Programming Models Compared

### How do you actually write these programs?

Note: The patterns we just covered are abstract. In practice, you pick a programming model (a concrete library and runtime) that supports the memory model and parallelism style of your hardware. Choosing correctly can save months of work.

---

## The Five Major Models

| Model | Memory Style | Parallelism | Hardware |
|---|---|---|---|
| **OpenMP** | Shared | Threads + SIMD | Multi-core CPU (1 node) |
| **MPI** | Distributed | Message passing | Clusters of any size |
| **CUDA** / **HIP** | GPU-local + host | Massive SIMT | NVIDIA / AMD GPUs |
| **PGAS** (Chapel, UPC) | Partitioned global | Global address space | Clusters w/ fast interconnect |
| **Spark / Dask** | Distributed | Map-reduce | Big-data clusters (cloud) |

*HIP = AMD's portable GPU API. HIP code compiles for both AMD and NVIDIA GPUs.*

> <span class="accent">Real systems combine these ("MPI+X"):</span> MPI between nodes + OpenMP within a node + CUDA on the GPU. Each layer matches a level of the hardware hierarchy.

Note: The MPI+X pattern has been the dominant HPC approach for a decade. The hierarchy mirrors hardware: MPI crosses node boundaries, OpenMP crosses socket boundaries, CUDA crosses the host/device boundary.

---

## OpenMP: Shared-Memory, Minimal Friction

```c
// dot product: each thread gets a chunk of iterations
// reduction(+:sum) safely combines per-thread partial sums
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; i++) {
    sum += a[i] * b[i];   // no locks needed, reduction handles it
}
```

- Incremental: add pragmas one loop at a time, no restructuring needed
- No explicit data movement since it's all shared memory
- Limited to a single node (~100 cores, ~1 TB memory)

> **When to pick OpenMP:** your problem fits on one machine and you want parallelism with minimal code changes.

*Used in: MATLAB's parallel internals, LLVM/GCC compiler parallelization, most scientific simulation codes (LAMMPS, GROMACS, OpenFOAM).*

Note: OpenMP started in 1997 and is still going strong. The 2021 5.2 spec even added GPU offloading, making it a credible alternative to CUDA for portable code. It's the easiest parallel programming model on the planet: if you can understand a for loop, you can parallelize one with OpenMP.

---

## MPI: Distributed-Memory, Explicit Control

```c
// Distributed dot product: same program runs on every process
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // "who am I?" (0..P-1)
MPI_Comm_size(MPI_COMM_WORLD, &size);  // "how many of us?"

double local_sum = 0;
for (int i = rank; i < N; i += size) { // each process takes every P-th element
    local_sum += a[i] * b[i];
}

double global_sum;
// sum all local_sums and deliver result to every process
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE,
              MPI_SUM, MPI_COMM_WORLD);
```

- Every process runs the same program (SPMD) with a different `rank`
- All data movement is explicit: you see every byte that crosses the network
- Scales to the world's largest systems (~10 million ranks)

> **When to pick MPI:** your problem doesn't fit on one machine, or you need full control over communication.

*Used in: every top-500 supercomputer, weather forecasting (WRF, ECMWF), molecular dynamics (LAMMPS, NAMD), Llama/GPT distributed training via NCCL (which uses MPI-like collectives).*

Note: MPI is harder than OpenMP because it forces you to think about data ownership and movement. But that explicitness is why it scales further: there's no hidden magic, so nothing degrades unexpectedly.

---

## CUDA: Massive SIMT on GPUs

```c
// __global__ = this function runs on the GPU, called from the CPU
__global__ void saxpy(int N, float a, float *x, float *y) {
    // each thread computes its own index from block/thread IDs
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = a * x[i] + y[i];  // y = a*x + y
}

// Host (CPU) launches thousands of threads in one call
int threads_per_block = 256;                              // threads per block
int num_blocks = (N + threads_per_block - 1) / threads_per_block;  // enough blocks to cover N
saxpy<<<num_blocks, threads_per_block>>>(N, 2.0f, d_x, d_y);      // <<<>>> = GPU launch syntax
```

- Threads run in **warps** of 32 executing the same instruction (SIMT)
- You manage the memory hierarchy explicitly: global, shared, registers
- The dominant model for ML, graphics, and scientific simulation

> **When to pick CUDA:** high arithmetic intensity and regular data access.

*Used in: PyTorch and TensorFlow (all tensor ops are CUDA kernels), Blender/game engines, molecular dynamics (AMBER, GROMACS), every LLM training run.*

Note: We'll cover GPU architecture in depth in Lecture 9. For now, the key mental model: a GPU is a massively parallel processor with deep memory hierarchy. CUDA exposes all of it for aggressive optimization. Frameworks like PyTorch hide this behind autograd, but under the hood it's CUDA kernels all the way down.

---

## PGAS: Partitioned Global Address Space

The middle ground between shared and distributed memory.

```chapel
// Chapel: array is split across processors automatically
var A: [1..N] int dmapped Block(boundingBox = {1..N});
forall i in 1..N do
    A[i] = compute(i);   // runs on whichever processor owns A[i]
```

- **One address space, many owners:** you write `A[i]` as if the array is local. The runtime figures out which processor owns element `i` and fetches it if needed.
- **Hidden network messages:** if `A[i]` lives on another node, the compiler inserts a network transfer (`MPI_Put`/`MPI_Get`) for you. No explicit send/recv.
- **The tradeoff:** easy to write, but easy to accidentally trigger slow remote accesses without realizing it.

**Languages:** Chapel (Cray/HPE), UPC, UPC++, Fortran coarrays

> **When to pick PGAS:** you want shared-memory simplicity at distributed-memory scale.

Note: PGAS is a beautiful idea that never quite took over because the abstraction hides where data lives. It's easy to write code with hidden remote accesses that kill performance. Chapel has seen a renaissance with HPE/Cray pushing it for exascale systems.

---

## Spark and Dask: Big Data in the Cloud

```python
# PySpark: count words across terabytes of logs
counts = (spark.read.text("s3://logs/")     # read from cloud storage
          .rdd                               # treat as distributed dataset
          .flatMap(lambda line: line.value.split())  # split each line into words
          .map(lambda word: (word, 1))       # emit (word, 1) pairs
          .reduceByKey(lambda a, b: a + b))  # sum counts per word
```

- **You never manage nodes:** data sits in distributed storage (S3, HDFS). The framework partitions it, schedules tasks, and moves data between nodes.
- **Automatic fault tolerance:** if a node dies mid-job, the framework re-runs that partition from the last checkpoint. In MPI, a single failed rank kills the whole job.
- **Functional style:** you chain transformations (map, filter, reduce). The framework decides what runs where.

> **When to pick Spark/Dask:** data-heavy problems on commodity cloud hardware where failures are expected.

*Used in: Netflix recommendations, Uber surge pricing, genomics pipelines, most enterprise ETL/analytics, Databricks platform.*

*Connection to Part 4: Map-reduce is the **pattern**. Spark/Dask are **implementations** of that pattern with fault tolerance, scheduling, and cloud storage built in. Google's original MapReduce (2004) was the first; Spark replaced it by keeping data in memory instead of writing to disk between steps.*

Note: Spark/Dask/Ray exist because MPI and OpenMP weren't designed for cheap cloud hardware where individual machines fail constantly. These frameworks trade a small performance overhead for automatic fault tolerance and ease of use.

---

## When to Use Which Model

A pragmatic decision table:

| Situation | Model |
|---|---|
| One machine, multi-core, existing C/C++/Fortran | **OpenMP** |
| Cluster, scientific code, need to scale big | **MPI** (often with OpenMP inside) |
| Dense compute, regular data access | **CUDA / HIP** |
| Cluster, global data view, productivity matters | **PGAS (Chapel)** |
| Large data, cloud, fault tolerance, SQL-ish | **Spark / Dask** |
| Portable code across CPUs, GPUs, and accelerators | **SYCL / Kokkos / oneAPI** |

*SYCL is a C++ standard for writing one kernel that compiles to NVIDIA, AMD, and Intel hardware. Kokkos (Sandia National Lab) and oneAPI (Intel) solve the same problem with different APIs. All three aim to avoid CUDA vendor lock-in.*

> **Reality check:** Most production systems combine two or three of these. Pick the simplest one that works, add others only when forced.

Note: A common trap is over-engineering: someone reaches for MPI + CUDA + OpenMP on day one when OpenMP alone would have worked. Start simple, measure, and add complexity only when the simpler tool hits a wall. Your future maintenance self will thank you.

---

## Part 6: Case Study: Parallel Matrix Multiply

### From naive to near-optimal in four steps

Note: Matrix multiply is the canonical parallel algorithm and the computational core of deep learning. Walking through how we parallelize it touches every concept in this lecture: decomposition, communication patterns, arithmetic intensity, scaling.

---

## The Problem

Compute `C = A × B`, where A, B, C are n × n matrices.

- Work: O(n³) multiply-add operations
- Data: O(n²) elements per matrix
- **Arithmetic intensity:** ~ n / 3, which grows with n, which is why matrix multiply is *great* for parallel hardware

Sequential cost: 2n³ floating-point operations. On a single modern CPU at 50 GFLOP/s, a 4096×4096 multiply takes about 2.7 seconds. We want to make it go faster using P processors.

Note: The fact that matrix multiply has high and growing arithmetic intensity is *why* GPUs are so fast at ML workloads. A GPU's peak FLOP rate is only achievable on kernels with high intensity; matrix multiply hits that target easily, which is why tensor cores exist and why ML training uses them nearly 100% of the time.

---

## Approach 1: 1D Row Decomposition

![1D row decomposition of matrix multiply](images/matmul_1d.svg)

- Split A by rows: each of P processes owns n/P rows
- Every process needs the **entire matrix B** to compute its block of C
- Broadcast B to everyone, then compute locally

**Cost analysis:**

- Compute per process: 2n³ / P
- Communication: broadcast of n² elements → O(n² log P) with tree algorithm
- Memory per process: O(n²), a full copy of B

> **Problem:** Memory footprint is *constant* in P because every process holds an n² matrix. Can't scale to matrices that don't fit on one machine.

Note: The 1D approach is easy to code and fine for small clusters with small matrices. It fails exactly when you need parallelism most: big matrices on big clusters. The per-process memory doesn't shrink as you add processors, so you run out of RAM long before you run out of parallelism.

---

## Approach 2: 2D Block Decomposition

![2D block decomposition of matrix multiply](images/matmul_2d.svg)

- Arrange P processes in a √P × √P grid
- Each process owns an (n/√P) × (n/√P) block of A, B, and C
- To compute its block of C, a process needs the corresponding **row of A blocks** and **column of B blocks**

**Cost analysis:**

- Compute: 2n³ / P
- Communication: O(n² / √P) per process
- Memory: O(n² / P) per process, which **scales with P**

> **Win:** Communication grows only as √P, not P. Memory shrinks linearly. This is why 2D decomposition is the standard.

Note: The square-root scaling of communication volume is the central result of 2D decomposition. It's not obvious the first time you see it, but it's the reason dense linear algebra on supercomputers uses 2D grids universally. ScaLAPACK, PLASMA, and every modern dense LA library works this way.

---

## Approach 3: Cannon's Algorithm and SUMMA

Two famous refinements of 2D decomposition:

**Cannon's Algorithm (1969)**

- Each step: shift A left by one block, shift B up by one block, multiply-accumulate
- After √P steps, every block of C is complete
- Only neighbor communication, which maps perfectly to a 2D torus network

**SUMMA (Scalable Universal Matrix Multiplication, 1995)**

- At step k: broadcast the k-th column of A blocks across rows, broadcast the k-th row of B blocks down columns, multiply-accumulate
- Uses collective broadcasts (which are efficient) instead of careful shifts
- Easier to implement and works for non-square process grids

> **Both achieve the same O(n² / √P) communication.** SUMMA is what you'll find in production libraries (ScaLAPACK's `PDGEMM`) because it's simpler and handles irregular sizes.

Note: Cannon's algorithm is a historical artifact that every HPC student learns because it beautifully illustrates neighbor-only communication. SUMMA, which came 26 years later, is what everyone actually uses because collective broadcasts are well-optimized in every MPI implementation. A good example of how the "simpler algorithm that uses a better primitive" often beats the clever one.

---

## Real Numbers: Strong Scaling of GEMM

Approximate strong-scaling efficiency for a fixed 32768×32768 matrix multiply on a typical HPC cluster:

| Processors | Time | Speedup | Efficiency |
|---|---|---|---|
| 1 | 2400 s | 1.0× | 100% |
| 16 | 160 s | 15.0× | 94% |
| 256 | 11 s | 218× | 85% |
| 4096 | 0.9 s | 2667× | 65% |
| 16384 | 0.35 s | 6857× | 42% |

> **Efficiency drops** as communication dominates: at 16k processors, each one has only a 256×256 block to compute, and communication overhead catches up with compute.

Note: These numbers are representative. Actual efficiencies depend heavily on network quality, with InfiniBand clusters doing much better than Ethernet clusters of the same size. On modern GPU systems, the equivalent scaling for mixed-precision GEMM is what enables training foundation models: a single H100 does 1 petaFLOP/s, and 1024 of them in a well-tuned cluster deliver close to 500 petaFLOP/s on large matrix multiplies.

---

## Part 7: Modern Context

### What's actually happening in the real world

Note: The patterns we covered are timeless, but the hardware and workloads driving them have shifted dramatically in the last few years. This section grounds the theory in 2026 reality.

---

## Heterogeneous Parallelism

Modern systems are not uniform. They combine multiple kinds of compute:

| Unit | Strength | Weakness |
|---|---|---|
| **CPU** | Flexible control flow, large caches | Modest peak throughput |
| **GPU** | Massive throughput, high intensity kernels | Weak at branches, kernel-launch overhead |
| **TPU / NPU** | Matmul-optimized, power-efficient | Inflexible, fixed-function |
| **FPGA** | Custom datapaths, deterministic latency | Hard to program, long build times |

> **Challenge:** Writing code that uses the right unit for each piece of work, and moves data between them efficiently, is the central problem of modern performance engineering.

Note: Frameworks like SYCL, oneAPI, Kokkos, and Raja try to unify these with a single source-level abstraction. They mostly work, but "zero cost abstraction" is a lie. There's always a gap between hand-tuned CUDA and portable code. The gap is narrowing, though, and for many applications portable is good enough.

---

## CXL and the Blurring of Memory Boundaries

**Compute Express Link (CXL)** is an emerging interconnect (CXL 3.0 shipping in 2026) that lets CPUs, GPUs, and accelerators share **coherent memory pools**.

- A pool of DRAM on the network can be mapped into a server's address space
- Multiple servers can access the same pool
- The traditional wall between shared-memory and distributed-memory erodes

> **Implication:** You may soon write PGAS-style code on commodity hardware with hardware coherence. The programming model shifts back toward shared memory, at least for medium-scale systems.

Note: CXL isn't just theoretical. Intel Sapphire Rapids, AMD Genoa, and every major cloud provider are actively deploying it. Meta's OCP-style servers use CXL memory expansion to disaggregate memory from compute. The long-term vision is a data center where any server can access any memory with coherent semantics, eliminating the need for most explicit data movement.

---

## Cloud and Serverless Parallelism

Parallel computing has left the supercomputer and moved to the cloud:

| Approach | What It Looks Like | Typical Use |
|---|---|---|
| **Dedicated cluster** | EC2 instances with SLURM / K8s | HPC, training |
| **Managed Spark** | Databricks, EMR, Dataproc | ETL, analytics |
| **Serverless** | AWS Lambda, Cloud Run fanout | Embarrassingly parallel batch jobs |
| **Managed training** | SageMaker, Vertex AI | ML at scale |

> **Serverless insight:** If your problem is embarrassingly parallel (like image processing or Monte Carlo), you can spawn 10,000 Lambda functions for 30 seconds each and pay only for what you use. No cluster to manage.

Note: Serverless parallelism flips the economics of HPC on its head. Traditional HPC amortizes a $10M cluster over five years; serverless charges per millisecond. For bursty workloads (rendering a movie frame, processing satellite imagery, running a one-off Monte Carlo) serverless often wins dramatically on cost. For sustained workloads like LLM training, dedicated clusters still win.

---

## Fault Tolerance at Scale

At 1000 nodes, something fails every day. At 10,000 nodes, something fails every hour.

| Strategy | How | Used By |
|---|---|---|
| **Checkpoint / restart** | Periodically save state; restart from last checkpoint | HPC, LLM training |
| **Recompute from lineage** | Remember *how* to reconstruct a partition | Spark, Dask |
| **Replication** | Keep copies on multiple nodes | HDFS, Kafka |
| **Redundant computation** | Run the same task twice, vote | Mission-critical systems |

> **LLM training example:** Meta's Llama training recorded failures every few hours on ~16k GPUs. Restart from checkpoints cost millions of GPU-hours over a training run.

Note: Fault tolerance is usually absent from intro parallel programming courses, but it's unavoidable at real scale. Every large ML training job has a dedicated reliability team whose full-time job is detecting, isolating, and working around hardware failures. The checkpoint-restart loop is so central that frameworks like PyTorch Lightning and DeepSpeed include it as a first-class feature.

---

## Energy Efficiency

Power is a first-class constraint, not an afterthought.

- A supercomputer like Frontier (Oak Ridge) consumes ~20 MW, enough for 20,000 homes
- Training GPT-class models uses **gigawatt-hours**, comparable to the annual consumption of thousands of homes
- Data center electricity is projected to be **~10% of global electricity** by 2030

> **Design implication:** Performance per watt matters as much as raw performance. This is why TPUs, tensor cores, and FP8 training exist: lower precision means less energy per operation.

Note: The new axis of hardware competition is performance per watt, not peak performance. NVIDIA's H100 is faster than its predecessor but also more efficient per operation. Google's TPU v5 trades flexibility for efficiency on a narrow set of ML operations. Energy constraints are also pushing accelerators closer to the source of power. Microsoft recently signed deals for nuclear reactors to power AI training campuses.

---

## Part 8: Pitfalls and Wrap-Up

### The mistakes that actually kill performance

Note: Knowing the theory isn't enough. Real parallel programs fail in specific, repeated ways. Recognizing these anti-patterns saves enormous amounts of debugging time.

---

## The Top Parallel Programming Pitfalls

| Pitfall | What It Looks Like | The Fix |
|---|---|---|
| **Premature parallelization** | Parallel version is *slower* than serial for small inputs | Only parallelize when profiling says you must |
| **False sharing** | Threads modify different variables on the same cache line | Pad data, use per-thread accumulators |
| **Load imbalance** | One processor finishes last, everyone else waits | Dynamic scheduling, work stealing |
| **Too-fine granularity** | Overhead of synchronizing exceeds work done | Chunk tasks to amortize overhead |
| **Over-synchronization** | Locks or barriers at every step | Only synchronize on real dependencies |
| **Hidden serial sections** | Library call that secretly takes a global lock | Profile; check library docs for thread safety |

Note: False sharing is especially insidious because the code *looks* correct (each thread writes a distinct variable) but performance crashes because those variables happen to share a 64-byte cache line. The fix is often just adding padding or aligning data structures to cache-line boundaries. Lecture 6 covered this from the hardware side; here we care about recognizing it in your own code.

---

## Deadlocks, Livelocks, and Race Conditions

A quick glossary of correctness failures (Lecture 7 covered these in detail):

| Bug | Symptom | Classic Cause |
|---|---|---|
| **Race condition** | Wrong answer, non-deterministic | Missing synchronization |
| **Deadlock** | Program hangs, no progress | Circular lock dependency |
| **Livelock** | Program runs but makes no progress | Retry loops colliding |
| **Starvation** | One thread never gets to run | Unfair scheduling / locking |

> **The golden rules:**
> 1. Acquire locks in a **global order** to prevent deadlock
> 2. Minimize the size of critical sections
> 3. Prefer atomics and lock-free data structures when possible
> 4. Use tools (ThreadSanitizer, Helgrind, Intel Inspector) to find races automatically

Note: Every large parallel code has *some* race condition that hasn't been caught yet. Modern tools find most of them automatically by instrumenting memory accesses at runtime. The cost is a 5-10× slowdown during testing, but it's worth it because a race that hits production once a week is much more expensive to chase than a slow test run.

---

## Key Takeaways from Parallel Programming II

1. **Communication, not computation, is the bottleneck.** Know the latency/bandwidth table cold.
2. **Use collective patterns.** Broadcast, scatter, gather, reduce, all-reduce, stencil. Picking the right one beats tuning the wrong one.
3. **Amdahl and Gustafson answer different questions.** Fixed problem vs. fixed time. ML training uses Gustafson.
4. **Learn the parallel patterns.** Map-reduce, fork-join, pipeline, stencil, task graphs. They cover most problems.
5. **Pick the simplest programming model that works.** OpenMP → MPI → CUDA → PGAS, in order of complexity.
6. **Heterogeneous and cloud are the future.** CXL, GPUs, serverless. The old shared vs. distributed wall is crumbling.
7. **Profile before optimizing.** The efficiency curve tells you whether you're fighting Amdahl, communication, or imbalance.

> **Next lecture: GPU Architecture.** We'll go deep on the SIMT execution model, memory hierarchy, and why GPUs dominate modern AI workloads.

Note: These seven points are what I'd want you to remember five years from now, after the specific library names and syntax have changed. Communication costs, scaling laws, and pattern recognition are durable skills; APIs come and go. The four-step framework from Lecture 7 plus these seven ideas give you the mental model to approach any new parallel system and quickly figure out what matters.

---

## Further Reading

- **Culler, Singh, Gupta.** *Parallel Computer Architecture: A Hardware/Software Approach.* Still the canonical textbook.
- **McCool, Robison, Reinders.** *Structured Parallel Programming.* Best modern treatment of parallel patterns.
- **Williams, Waterman, Patterson.** *Roofline: An Insightful Visual Performance Model.* The paper that launched the roofline model.
- **Horace He.** *"Making Deep Learning Go Brrrr From First Principles."* The best modern essay on arithmetic intensity and GPU performance.
- **MPI Forum.** The MPI-4.1 standard, free online.
- **NVIDIA NCCL documentation.** How all-reduce actually works at scale.
- **Apache Spark / Dask / Ray docs.** Hands-on experience with data-parallel frameworks in the cloud.

Note: The single best way to learn this material is to write parallel code and profile it. Pick a small problem (matrix multiply, N-body, image filter) and implement it in OpenMP, then MPI, then CUDA. Measure. Discover why your first version is slow. That hands-on loop is worth more than any number of lectures.
