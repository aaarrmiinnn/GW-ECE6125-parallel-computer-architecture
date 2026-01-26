# Parallel Computer Architecture
## Lecture 2: Intuition for Parallelism
### GWU ECE 6125 | Armin Mehrabian | Spring 2026

---

## Parallelism Intuitions

![Parallelism Intuitions](images/image2.png)

---

## A Program's Perspective of Memory

A computer's memory is structured as an array of bytes.

| Address | Value |
|---------|-------|
| 0x0 | 10 |
| 0x1 | 0 |
| 0x2 | 1 |
| ... | ... |
| 0x1F | 255 |

- Each byte is assigned a unique **address**, which represents its position in the memory array
- **Examples:**
  - The byte at address **0x2** contains the value **1**
  - The byte at address **0x1F (31)** contains the value **255**

---

## What Are Caches?

A **cache** is a small, high-speed hardware storage layer that holds frequently accessed data to reduce memory access latency.

**Key Properties of Caches:**

| Property | Description |
|----------|-------------|
| **Transparent to Software** | Does not alter program behavior, only impacts performance |
| **On-Chip Storage** | Stores subset of data from main memory. Faster than DRAM (1-5 cycles vs 100+ cycles) |
| **Cache Line Granularity** | Data stored in fixed-size blocks (cache lines), e.g., 64 bytes |

**Two ways caches help:** Spatial and Temporal locality

![Cache Concept](images/image1.png)

---

## Caches and Data Locality

![Caches and Data Locality](images/image4.png)

---

## Caches and Data Locality (Continued)

![Data Locality Detailed](images/image18.png)

---

## Cache Mapping Policies

How does a processor decide what data to keep in cache?

**1. Direct-Mapped Cache**
- Each memory block maps to **exactly one location** in the cache

**2. Set-Associative Cache**
- Each memory block maps to a **set** of cache locations (e.g., 4-way associative)

**3. Fully Associative Cache**
- Data can be placed **anywhere** in the cache (rarely used for large caches)

---

## Direct-Mapped Cache

**Rule:** Each memory address maps to **exactly one cache line**

**Mapping Formula:** `Cache Line = (Address / Block Size) % Total Lines`

| Pros | Cons |
|------|------|
| **Simple hardware:** Easy to implement | **High conflict misses:** Multiple addresses compete for same line |
| **Fast access time:** No decision-making overhead | **Inflexible replacement:** Forced replacement on collision |
| **Low power consumption:** Minimal circuitry | **Poor utilization:** Frequently accessed blocks may evict each other |

![Direct-Mapped Cache](images/image5.png)

---

## Set-Associative Cache

**Rule:** Each memory address maps to a **set** (e.g., 2 lines per set)

**Mapping Formula:** `Set = (Address / Block Size) % Total Sets`

| Pros | Cons |
|------|------|
| **Reduced conflict misses:** Multiple slots per set | **Increased complexity:** Additional logic for sets |
| **Balanced performance:** Better hit rate than direct-mapped | **Slower than direct-mapped:** Slightly longer access time |
| **Flexible replacement:** LRU can be applied within a set | **Higher power usage:** More circuitry for tag comparison |

![Set-Associative Cache](images/image6.png)

---

## Fully-Associative Cache

**Rule:** Any memory address can map to **any cache line**

| Pros | Cons |
|------|------|
| **Minimal conflict misses:** Any block can occupy any line | **High complexity:** Requires searching all entries |
| **Optimal replacement:** Global LRU maximizes utilization | **Impractical for large sizes:** Hardware cost grows exponentially |
| **Flexibility:** Ideal for small caches (e.g., TLB) | **Power-intensive:** Parallel tag comparison consumes energy |

![Fully-Associative Cache](images/image7.png)

---

## Example Program

```c
int main() {
    int result = 1;
    for (int i = 0; i < 10; i++) {
        result = result * 2;
    }
    printf("%d\n", result);
    return 0;
}
```

**Compiled to assembly:**
```
ld   r0, addr[r1]
mul  r1, r0, r0
mul  r1, r1, r0
...
st   addr[r2], r0
```

---

## Program Execution Visualization

![Program Execution](images/image17.png)

---

## Instruction Pipeline

![Instruction Pipeline](images/image10.png)

---

## Pipeline Stages

![Pipeline Stages](images/image20.png)

---

## Pipeline Hazards

![Pipeline Hazards](images/image13.png)

---

## First Attempt at Parallelism: ILP

**Instruction-Level Parallelism (ILP):** Executing multiple instructions simultaneously within a single program thread.

**Key Techniques:**

- **Superscalar Execution:** Multiple instructions decoded/executed per clock cycle
- **Pipelining:** Overlap stages of instruction processing (Fetch, Decode, Execute, Writeback)
- **Out-of-Order Execution:** Dynamically reorder instructions to avoid stalls

```
1. a = x + y  ──┐
2. b = z * 2    │ Independent → Can run in parallel
3. c = a + b  ──┘ (Depends on 1 and 2)
4. d = m - n  ──► Independent of 1-3
```

---

## Example Program with ILP

![ILP Example](images/image30.png)

---

## ILP: One Processor, Multiple Execution Units

![ILP Multiple Execution Units](images/image16.png)

---

## Example Program: Computing Kinetic Energy

**Inputs (x):** A single 1D array containing both masses and velocities for the 2D grid:
- First half contains **mass values** (`x[index]`)
- Second half contains **velocity values** (`x[index + M]`)
- Organized in row-major format for grid with dimensions N × M

**Outputs (y):** A 1D array where each element represents the **total kinetic energy** for a row in the grid.

![Kinetic Energy Program](images/image9.png)

---

## Example Program: Code

![Program Code](images/image46.png)

---

## Example Program: Compiled Instructions

![Compiled Instructions](images/image8.png)

---

## Example Program: Execution

![Program Execution](images/image17.png)

---

## Example Program: ILP Analysis

![ILP Analysis](images/image16.png)

**Dependency!** Some instructions must wait for others to complete.

---

## ILP vs Higher-Level Parallelism

![ILP vs Higher Level](images/image9.png)

**Is there any parallelism left here?**

---

## ILP Diminishing Returns

The majority of chip transistors are utilized to enhance the speed of executing a single instruction stream.

**Problems:**
- **Diminishing Returns:** More transistors for ILP yield smaller performance gains
- **Higher Costs:** Complex ILP logic increases design complexity, power usage, and manufacturing costs
- **Underutilized Potential:** Many workloads lack sufficient parallelism to fully benefit
- **Better Alternatives:** Transistors could be better spent on more cores or specialized accelerators
- **Memory Bottlenecks:** Larger caches and smarter predictors can't fully mitigate memory delays
- **ILP Saturation:** Techniques like out-of-order execution face hard limits

![ILP Diminishing Returns](images/image31.png)

---

## Any Other Ideas for Parallelism?

Beyond ILP, we can exploit parallelism at higher levels...

---

## Many-Core vs Single-Core

**Old Approach: Enhancing Single-Core Performance**
- More transistors → Faster execution of one instruction stream
- Techniques: Out-of-order execution, speculative operations

**New Approach: Multi-Core Processing**
- More transistors → More cores, not just a faster single core
- Enables parallel execution, better efficiency, and scalability
- **Simpler cores:** Each core may be slower at running a single instruction stream (e.g., 25% slower)
- But there are now two cores: 2 × 0.75 = **1.5 (potential for speedup!)**

![Multi-Core vs Single-Core](images/image21.png)

---

## But... No Parallelism in the Code

This C program compiles into a single instruction stream that executes on one thread and one processor core.

If our simple processor cores are 25% slower than the original single complicated one, our program now runs **25% slower** than before.

![Single Thread Problem](images/image9.png)

---

## Spawning Multiple Threads

We can create two threads, which generates two instruction streams.

**"We"** the **programmer**, or sometimes the **compiler**, has the knowledge of which parts of the code are **parallel**.

![Spawning Threads](images/image12.png)

---

## 4 Cores → 4 Instruction Streams

![4 Cores](images/image14.png)

---

## 16 Cores → 16 Instruction Streams

![16 Cores](images/image15.png)

---

## Spawning Multiple Threads

**Parallelism is across iterations of the loop:**
- The iterations of the loop carry out the **exact same sequence of instructions** (defined by the loop body)
- But on **different input data** given by X

![Thread Spawning](images/image12.png)

---

## Data-Level Parallelism (DLP) with SIMD

**Key Concept - Parallelism Across Loop Iterations:**
- Executes the **same instructions** for each loop iteration
- Operates on **different input data** (e.g., array `x[i]`)

**SIMD Architecture:**
- **Fetch/Decode Unit:** Fetches and decodes instructions once for all iterations
- **ALUs (Arithmetic Logic Units):** Execute the same operation on multiple data points in parallel
- **Execution Context:** Stores input data and states for all ALUs

![SIMD Architecture](images/image47.png)

---

## Single-Core vs. Data-Parallel

**Sequential vs Vector Program**

![Sequential vs Vector](images/image26.png)

---

## Single-Core vs. Data-Parallel (AVX)

**Vector program using AVX intrinsics:**

- **AVX Parallelism:** Processes 8 floating-point values simultaneously using 256-bit registers
- **Loop Vectorization:** Replaces inner loop with SIMD instructions to compute kinetic energy for multiple particles in parallel

![AVX Vector Program](images/image44.png)

---

## Intel Multi-Core: Alder Lake (2022)

![Intel Alder Lake Die](images/image29.jpg)

*Source: https://x.com/aschilling/status/1453391035577495553*

---

## Intel Alder Lake Architecture Detail

![Intel Alder Lake Detail](images/image33.jpg)

*Source: https://x.com/aschilling/status/1453391035577495553*

---

## NVIDIA V100 Architecture

![NVIDIA V100](images/image43.png)

*Source: NVIDIA Volta Architecture Whitepaper*

---

## Apple A18 Pro

![Apple A18 Pro](images/image23.jpg)

---

## 16 SIMD Cores: 128 Elements in Parallel

![16 SIMD Cores](images/image49.png)

---

## Modern Multi-Core: SIMD Instruction Sets

**SIMD Instruction Sets:**
- **Intel AVX2:** 256-bit operations (8×32-bit floats or 4×64-bit doubles)
- **Intel AVX-512:** 512-bit operations (16×32-bit floats or 8×64-bit doubles)
- **ARM NEON:** 128-bit operations (4×32-bit floats or 2×64-bit doubles)

**Compiler-Generated SIMD:**
- **Programmer-Controlled:** Explicit SIMD via intrinsics (e.g., `_mm256_mul_ps`)
- **Auto-Vectorization:** Compilers analyze loops for dependencies and generate SIMD instructions automatically

---

## Implicit vs Explicit SIMD

| Implicit SIMD | Explicit SIMD |
|---------------|---------------|
| Compiler generates scalar instructions; hardware executes SIMD | Programmer explicitly requests SIMD using intrinsics |
| Managed by **hardware** at runtime | Managed by **programmer** at compile time |
| Used in **GPUs** (NVIDIA, AMD) with wide SIMD lanes | Used in **CPUs** with AVX, AVX-512, ARM NEON |
| Performance depends on hardware managing **divergence** | Programmer optimizes loop unrolling and vector operations |
| Easier for programmers but can lose efficiency | **Higher efficiency** if optimized correctly |

---

## Three Forms of Parallelism

**1. Superscalar Processing:**
- Executes multiple instructions from the same instruction stream in parallel within a single core
- Hardware dynamically identifies and exploits ILP during execution

**2. SIMD (Single Instruction, Multiple Data):**
- Uses multiple ALUs under control of a single instruction
- Well-suited for data-parallel workloads
- Vectorization by compiler (explicit) or hardware (implicit)

**3. Multi-Core Processing:**
- Multiple processor cores execute different instruction streams concurrently
- Enables thread-level parallelism via threading APIs

**YOU CAN COMBINE ALL THREE FOR MORE PARALLELISM**

---

## Hiding Stalls With Multi-Threading

![Hiding Stalls 1](images/image27.png)

---

## Multi-Threading: Thread Switching

![Thread Switching](images/image22.png)

---

## Multi-Threading: Overlapping Execution

![Overlapping Execution](images/image28.png)

---

## Multi-Threading: Full Utilization

![Full Utilization](images/image24.png)

---

## Multi-Threading: Maximum Throughput

![Maximum Throughput](images/image32.png)

---

## Expanding the Execution Context

![Execution Context](images/image25.png)

---

## Large Context vs. Many Small Contexts

![Large vs Small Context](images/image38.png)

---

## Utilizing One Core: One Thread

![One Core One Thread](images/image40.png)

---

## Utilizing One Core: Two Threads

![One Core Two Threads](images/image36.png)

---

## Utilizing One Core: Five Threads

![One Core Five Threads](images/image34.png)

---

## Utilizing One Core: More Than 5 Threads

![One Core Many Threads](images/image37.png)

---

## Utilization of Compute-Heavy Computation

![Compute Heavy 1](images/image35.png)

---

## Utilization of Compute-Heavy Computation (Continued)

![Compute Heavy 2](images/image45.png)

---

## Multithreading and Stalls

A processor equipped with multiple hardware threads can **reduce idle time (stalls)** by executing instructions from other threads while one thread is waiting for a high-latency operation.

**Important:** Multithreading does **not decrease the latency** of the operation itself; it ensures that the processor's resources remain **utilized** by switching to other ready-to-run threads.

**Problem Without Multithreading:**
- In single-threaded processors, when a thread encounters a high-latency operation, the processor sits idle, wasting resources

**How Multithreading Solves This:**
- By maintaining multiple active threads, the processor can switch to another thread and continue executing
- This improves **processor utilization** and reduces the impact of stalls

---

## Compute vs. Memory Ratio

A multi-threaded processor reduces the impact of memory latency by performing arithmetic operations from other active threads while waiting for memory access.

**Key Insight:** Programs with a **higher ratio of compute to memory** require fewer threads to effectively hide memory stalls.

| Program Type | Thread Requirement |
|--------------|-------------------|
| Memory-bound | More threads needed to hide latency |
| Compute-heavy | Fewer threads achieve high utilization |

---

## Hardware-Supported Multi-Threading

**1. Core Manages Multiple Threads**
- Same ALU resources, but multi-threading helps utilize them efficiently
- Processor decides which thread to run each clock cycle

**2. Interleaved Multithreading (Temporal)**
- At every clock cycle, core switches between threads
- Example: Rotating thread scheduling

**3. Simultaneous Multithreading (SMT)**
- At each clock cycle, core executes instructions from **multiple threads simultaneously**
- Example: **Intel Hyper-Threading** enables 2 threads per core to share resources

---

## Intel Skylake Architecture

![Intel Skylake](images/image42.png)

---

## NVIDIA Tesla V100 Architecture

![NVIDIA Tesla V100](images/image41.png)

---

## References

- "Introduction to Parallel Computing" by Ananth Grama, Anshul Gupta, George Karypis, and Vipin Kumar
- "Computer Architecture: A Quantitative Approach" by John L. Hennessy and David A. Patterson
- Stanford CS149: Parallel Computing
- Highly recommended: [Stanford CS149 YouTube Playlist](https://www.youtube.com/watch?v=V1tINV2-9p4&list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp)
