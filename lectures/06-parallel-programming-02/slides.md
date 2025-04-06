# Parallel Programming: Concepts and Strategies
## GWU ECE 6125: Parallel Computer Architecture

---

## Shared vs. Distributed Memory Architectures
![Shared vs Distributed Memory](images/shared_vs_dist_mem.svg)

---

## Shared vs. Distributed Memory Architectures

### Shared Memory
- **Single address space** shared by all processors  
- Processors **directly read/write** shared variables  
- **Synchronization** through locks, semaphores, or atomic operations  

#### Example (Chapel)
```chapel
// Shared array across all threads
var A: [1..10] int;

// Parallel update (shared memory)
forall i in A.domain do
  A[i] = i * 2;
```
- `A.domain` refers to the index set of array `A`  
- This example runs in parallel with **shared access** to `A`

- **Advantages**  
  - Easier to program (no explicit communication)  
  - Fast communication via shared memory  

- **Challenges**  
  - Scalability limited by memory bandwidth  
  - Risk of contention and race conditions  

---

## Shared vs. Distributed Memory Architectures (Cont.)

### Distributed Memory
- Each processor has its **own private memory**  
- Communication happens via **explicit message passing** (e.g., MPI)  
- Synchronization through **message coordination**

#### Example (MPI-style pseudocode)
```c
// Each process computes a local result
local_A = compute_local_chunk();

// Send local results to process 0
MPI_Gather(local_A, global_A, root=0);
```

- **Advantages**  
  - Scales well across many nodes  
  - Clear data ownership  

- **Challenges**  
  - Requires explicit communication  
  - Higher latency and programming complexity  

---

## Hybrid Memory Architectures

### Hybrid Systems
- Combine **shared memory** (within nodes) and **distributed memory** (across nodes)  
- Typical in modern clusters and supercomputers  

#### Example: MPI + OpenMP  
- MPI distributes work across nodes  
- OpenMP runs threads on shared-memory within a node  

- **Advantages**  
  - Good **performance** and **scalability**  
  - Matches hierarchical hardware architectures  

- **Challenges**  
  - More complex programming  
  - Careful tuning needed for optimal performance  

---



## Orchestration in SPMD and Data Parallel Systems

- **SPMD orchestration**:
  - Global synchronization points (barriers)
  - Collective operations (gather, scatter, reduce)
  - Local computation phases

- **Example: MPI barrier synchronization**:
  ```
  // All processes compute locally
  local_result = compute_local_chunk();
  
  // Synchronize before communication
  MPI_Barrier(comm);
  
  // Exchange data
  MPI_Allgather(...);
  ``` 

---

## 6. Communication

### Communication in Parallel Programs

#### Purpose
- Exchange data between parallel tasks  
- Enable coordination and data sharing  

#### Types of Communication
- **Point-to-Point**  
  - Direct exchange between two tasks  
  - Examples: `MPI_Send`, `MPI_Recv`  
  - Can be **blocking** (synchronous) or **non-blocking** (asynchronous)  
  - Common in **nearest-neighbor** communication  

- **Collective**  
  - Involves a group of tasks working together  
  - Examples: broadcast, scatter, gather, reduce, all-to-all  
  - Often more **efficient** than multiple point-to-point operations  
  - Optimized with **tree-based algorithms** in libraries like MPI  

---

## 6. Communication (Cont.)

#### Types of Communication (Cont.)
- **One-Sided Communication**  
  - Direct remote memory access (RMA) without explicit participation by the target  
  - Examples: `MPI_Put`, `MPI_Get`, PGAS models (like Chapel)  
  - Reduces synchronization overhead  
  - Requires hardware/software support  

#### Performance Considerations
- **Latency**  
  - Time to initiate a communication  
  - Dominates small messages  
- **Bandwidth**  
  - Data transfer rate (bytes per second)  
  - Dominates large messages  
- **Contention**  
  - Congestion from multiple tasks communicating at once  
  - Lowers effective bandwidth and increases latency  

---

## Shared Memory Communication

### Mechanism
- Direct **read/write** access to shared variables  
- All threads/processes operate in a **common address space**  

### Advantages
- Simple programming model (easy to understand and implement)  
- **Low latency** data exchange  
- **Zero-copy**: No need to explicitly transfer data  

---

## Shared Memory Communication (Cont.)

### Challenges
- **Cache coherence overhead**  
  - Hardware ensures consistency between processor caches  
- **Synchronization required**  
  - To prevent race conditions when accessing shared data  
- **Limited scalability**  
  - Memory bandwidth becomes a bottleneck as the number of processors increases  
- **False sharing**  
  - Performance loss when threads modify different variables that reside on the same cache line  

### OpenMP Example
```c
#pragma omp parallel shared(result)
{
    // Each thread writes its result to a shared array
    result[omp_get_thread_num()] = compute();
}
```

---

## Message Passing Communication (MPI)

### Mechanism
- Communication through **explicit send/receive** operations  
- Processes have **private memory** and exchange data by messages  

### Characteristics
- Sender specifies **what to send** and **to whom**  
- Receiver provides **buffer** to store incoming data  
- Can be **synchronous** (blocking) or **asynchronous** (non-blocking)  

---

## Message Passing Communication (MPI) (Cont.)

### Communication Types
- **Point-to-Point**  
  - Direct exchange between specific processes  
  - Examples: `MPI_Send`, `MPI_Recv`  

- **Collective**  
  - Involves all processes in a group  
  - Examples: broadcast, scatter, gather, reduce  

### MPI Example (Point-to-Point)
```c
// Process 0 sends data to process 1
if (rank == 0) {
    MPI_Send(data, size, MPI_INT, 1, tag, comm);
} else if (rank == 1) {
    MPI_Recv(buffer, size, MPI_INT, 0, tag, comm, &status);
}
```

---

## Communication Patterns in SPMD Programs

- **Common patterns**:
  - **Broadcast**: One-to-all distribution
    - *Single process sends identical data to all other processes*
    - *Optimized implementations use tree-based algorithms*
    - *Examples: Distributing input parameters, configuration data*
  
  - **Scatter**: Divide and distribute data
    - *Root process divides data into chunks and sends each chunk to a different process*
    - *Each process receives distinct portion of the data*
    - *Example: Distributing array elements for parallel processing*
  
  - **Gather**: Collect distributed data
    - *Each process sends its local data to a root process*
    - *Root combines the data in a predetermined order*
    - *Example: Collecting partial results for final output*
  
  - **All-to-all**: Every process sends to every other
    - *Each process communicates with all other processes*
    - *Highest communication volume of all patterns*
    - *Examples: Matrix transpose, some FFT implementations*
  
  - **Reduction**: Combine data with operation (sum, max, etc.)
    - *Values from all processes combined using specified operation*
    - *Usually implemented as a tree-based algorithm*
    - *Examples: Finding global sum, global maximum, global minimum*
  
  - **Stencil/neighbor**: Exchange with adjacent processes
    - *Communication only between logically adjacent processes*
    - *Often used in grid/mesh-based computations*
    - *Examples: Finite difference methods, ghost/halo cell exchange*

- **Impact on performance**:
  - Communication pattern selection affects scalability
  - Different hardware may favor different patterns
  - Network topology can significantly influence performance

![Communication Patterns in SPMD Programs](images/communication_patterns.svg)

---

## Communication Overhead and Minimizing It

### Sources of Overhead
- **Latency**  
  - Time to initiate communication (setup time)  
- **Limited Bandwidth**  
  - Maximum data transfer rate  
- **Contention**  
  - Multiple tasks compete for communication channels  
- **Synchronization Delays**  
  - Waiting for coordination between tasks  
- **Protocol Processing**  
  - Extra time spent handling communication protocols  

---

## Communication Overhead and Minimizing It (Cont.)

### Minimization Strategies
- **Reduce communication frequency**  
  - Communicate only when necessary  
- **Batch communications**  
  - Group messages to minimize overhead (coarser granularity)  
- **Overlap communication with computation**  
  - Perform communication and useful work simultaneously  
- **Use asynchronous communication**  
  - Avoid blocking on message transfers  
- **Optimize data layouts**  
  - Arrange data to reduce the amount and frequency of communication  

---

## Example – Parallel Matrix Multiplication Communication

### 1D Decomposition (Row-Based)
- Each process owns a **block of rows**  
- All processes need the **entire second matrix (B)** to compute their part  
- Requires a **collective broadcast** of matrix B to all processes  

#### Communication Volume  
- O(n²), because each process requires a full copy of B 

![1D Decomposition Matrix Multiplication](images/matrix-multiplication-example-1d.svg)

---

## Example – Parallel Matrix Multiplication Communication (Cont.)

### 2D Decomposition (Block-Based)
- Each process owns a **submatrix block**  
- Communication happens **along rows and columns** of the process grid  
- Reduces data exchange compared to 1D decomposition  

#### Communication Volume  
- O(n² / √p), where **p** is the number of processors  
- More scalable than 1D decomposition for large **p**  

![2D Decomposition Matrix Multiplication](images/2d-matrix-multiplication.svg)

---

## 7. Performance and Efficiency

### How Do We Measure Performance?

- **Key metrics**:
  - **Speedup**: Performance gain vs. sequential
  - **Efficiency**: How well we use the processors
  - **Scalability**: Performance behavior as processors increase

- **Formulas**:
  - Speedup(n) = T₁ / Tₙ
  - Efficiency(n) = Speedup(n) / n
  - Cost = n × Tₙ

- **Performance comparison**:
  - Strong scaling: Fixed problem size
  - Weak scaling: Problem size grows with processors

---

## Performance Metrics in Parallel Computing

- **Speedup**: How much faster the parallel version is
  - *Absolute speedup*: Serial time / parallel time
  - *Relative speedup*: Best serial time / parallel time
  - *Ideal speedup*: Linear (Sp = p) - rarely achieved

- **Efficiency**: How well processors are utilized
  - *Efficiency = Speedup / Number of processors*
  - *Values range from 0 to 1 (or 0% to 100%)*
  - *Lower efficiency indicates resource underutilization*

- **Scalability**: How performance changes with system size
  - *Strong scaling*: Fixed problem size, increasing processors
  - *Weak scaling*: Increasing both problem size and processors
  - *Typically measured with efficiency at different scales*

- **Theoretical limits**:
  - *Amdahl's Law*: Speedup limited by sequential portion
    - *Speedup ≤ 1/(s + (1-s)/p) where s is sequential fraction*
  - *Gustafson's Law*: Scaled speedup with larger problems
    - *Scaled Speedup = p - α(p-1) where α is sequential fraction*

![Scalability Concepts in Parallel Programming](images/scalability_concepts.svg)

---

## Speedup and Efficiency Defined

- **Speedup**: 
  - Ratio of sequential time to parallel time
  - Speedup(n) = T₁ / Tₙ
  - Ideal speedup = n (linear)

- **Efficiency**: 
  - Ratio of speedup to number of processors
  - Efficiency(n) = Speedup(n) / n
  - Ideal efficiency = 1 (100%)

- **Factors affecting speedup**:
  - Serial portions
  - Communication overhead
  - Load imbalance
  - Resource contention

---

## Amdahl's Law

- **Formula**: Speedup(n) = 1 / (s + p/n)
  - s = serial fraction
  - p = parallel fraction (1-s)
  - n = number of processors

- **Key insight**: Serial portion limits maximum speedup
  - With 10% serial code (s=0.1), max speedup is 10x
  - With 1% serial code (s=0.01), max speedup is 100x

- **Implications**:
  - Focus on reducing serial portions
  - Diminishing returns with more processors

![Amdahl's Law](images/amdahls_law.svg)

---

## Gustafson's Law

- **Alternative scaling view**:
  - Amdahl's assumes fixed problem size
  - Gustafson assumes problem size grows with processors

- **Formula**: Speedup(n) = n - s × (n - 1)
  - s = serial fraction
  - n = number of processors

- **Key insight**: Larger problems can achieve better speedup
  - Parallel portion grows with problem size
  - Serial portion stays relatively constant

- **Weak scaling**: Problem size per processor stays constant

---

## Load Imbalance and Overheads

- **Load imbalance**:
  - Some processors finish early, wait for others
  - Efficiency loss = (Tmax - Tavg) / Tmax
  - Caused by: uneven data distribution, variable task complexity

- **Parallel overhead**:
  - Communication
  - Synchronization
  - Task management
  - Contention for shared resources

- **Total parallel time**: T_parallel = T_computation + T_overhead

---

## Optimizing Parallel Program Performance

- **Minimize communication**:
  - Increase computation/communication ratio
  - Use bulk transfers
  - Optimize data placement

- **Reduce synchronization points**:
  - Eliminate unnecessary barriers
  - Use asynchronous operations
  - Relax synchronization requirements when safe

- **Improve data locality**:
  - Cache-friendly data layouts
  - Memory-hierarchy-aware algorithms

- **Load balancing techniques**:
  - Dynamic work distribution
  - Over-decomposition
  - Work stealing

---

## Profiling and Analyzing Performance Bottlenecks

- **Tools**:
  - Timing instrumentation
  - Hardware performance counters
  - Parallel profilers (Intel VTune, NVIDIA NSight, TAU)
  - Communication analyzers (Vampir, Paraver)

- **What to look for**:
  - Load imbalance
  - Excessive synchronization
  - Communication hotspots
  - Memory access patterns
  - Cache behavior

---

## 8. Common Parallel Programming Patterns

### Parallel Loops (SPMD Example)

- **Essence**: Independent loop iterations executed in parallel
- **When to use**: No dependencies between iterations
- **OpenMP example**:
  ```c
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    results[i] = process(data[i]);
  }
  ```

- **MPI example**:
  ```c
  int chunk_size = N / num_procs;
  int start = rank * chunk_size;
  int end = (rank == num_procs-1) ? N : (rank+1) * chunk_size;
  
  for (int i = start; i < end; i++) {
    results[i] = process(data[i]);
  }
  ```

---

## Map-Reduce Pattern

- **Two phases**:
  - **Map**: Apply function to each data element (parallel)
  - **Reduce**: Combine results with associative operation

- **Example: Word count**:
  ```
  // Map phase (parallel)
  for each document in partition:
    for each word in document:
      emit(word, 1)
      
  // Reduce phase (parallel per key)
  for each unique word:
    count = sum(all 1's for this word)
    emit(word, count)
  ```

- **Properties**: Simple, scalable, fault-tolerant

---

## Fork-Join Model

- **Pattern**:
  - Main thread forks into multiple parallel tasks
  - Tasks execute concurrently
  - All tasks join (synchronize) at a barrier
  - Main thread continues

- **Example: OpenMP parallel sections**:
  ```c
  #pragma omp parallel sections
  {
    #pragma omp section
    { task_A(); }
    
    #pragma omp section
    { task_B(); }
    
    #pragma omp section
    { task_C(); }
  }
  // Implicit barrier here
  ```

- **When to use**: Heterogeneous tasks with barrier synchronization

---

## Pipeline Parallelism

- **Pattern**:
  - Split computation into ordered stages
  - Different data elements at different stages
  - Output of one stage is input to next

- **Example: Video processing**:
  ```
  Stage 1: Read video frame
  Stage 2: Apply filter
  Stage 3: Detect features
  Stage 4: Encode and save
  ```

- **Throughput**: Limited by slowest stage
- **Latency**: Sum of all stage processing times
- **When to use**: Stream processing, producer-consumer

---

## Example - Parallel Search Algorithm

- **Task**: Find element in unsorted array
- **Serial approach**: Linear scan
- **Parallel approach**:
  - Divide array into chunks
  - Each processor searches its chunk
  - If found, signal other processors to stop

- **Challenges**:
  - Early termination mechanism
  - Load balancing
  - Handling multiple matches

- **Speedup**: Nearly linear for large arrays

---

## Example - Parallel Sorting Algorithm

- **Parallel quicksort**:
  - Partition around pivot (serial or parallel)
  - Recursively sort subarrays in parallel
  - No merge step needed

- **Parallel merge sort**:
  - Divide array (serial or parallel)
  - Sort subarrays in parallel
  - Merge results (parallel merge possible)
- **Considerations**:
  - Communication overhead
  - Load balancing (pivot selection)
  - Cutoff to serial algorithm for small arrays

![Parallel Sorting](images/parallel-sorting-algorithms.svg)
---

## Combining Patterns for Complex Applications

- **Example: Image processing pipeline**:
  - **Map pattern**: Process chunks of image in parallel
  - **Pipeline pattern**: Input → Filter → Analyze → Output
  - **Reduction pattern**: Aggregate analysis results

- **Benefits of combined patterns**:
  - Exploit different forms of parallelism
  - Better resource utilization
  - Handle different parallelism granularities

- **Implementation approach**:
  - Identify pattern for each application component
  - Compose patterns with appropriate interfaces
  - Consider overall synchronization and data flow

---

## 9. Case Study: End-to-End Parallel Program

### Real-World Problem Setup

- **Problem**: Large image dataset processing
- **Operations**:
  - Load images
  - Apply multiple filters
  - Extract features
  - Classify images
  - Aggregate statistics

- **Constraints**:
  - 100,000+ high-resolution images
  - Process within reasonable time
  - Utilize multi-core and multi-node environment

![Case Study: Parallel Image Processing Pipeline](images/case_study_parallel_design.svg)

---

## Step 1: Decompose the Problem

- **Data decomposition**:
  - Split image dataset across nodes
    - *Distribute chunks of images across compute nodes*
    - *Can be done statically (equal portions) or dynamically (work queue)*
  - Further split within nodes to threads
    - *Each thread processes individual images or image tiles*
    - *Memory-conscious splitting to maximize cache utilization*

- **Task decomposition**:
  - Separate pipeline stages
    - *Input/output handling tasks*
    - *Image processing tasks*
    - *Analysis and classification tasks*
  - Specialized tasks for I/O, processing, analysis
    - *I/O threads to hide disk latency*
    - *Compute threads for CPU-intensive operations*
    - *Communication threads for network operations*

- **Hybrid approach**:
  - Data parallelism for image processing
    - *Process multiple images simultaneously*
    - *Process image regions in parallel*
  - Task parallelism for pipeline stages
    - *Different operations proceed in parallel*
    - *Producer-consumer relationships between stages*
  - Over-decomposition for load balancing
    - *Create more tasks than processors*
    - *Enables dynamic scheduling and better resource utilization*

---

## Step 2: Assign the Work

- **Between nodes (MPI)**:
  - Block distribution of images
    - *Initial static assignment of image sets to nodes*
    - *Each node gets a contiguous block of images*
  - Dynamic work stealing for better balance
    - *Nodes that finish early can request more work*
    - *Centralized master or distributed stealing protocols*
    - *Reduces impact of node-to-node performance variation*

- **Within nodes (OpenMP/threads)**:
  - Thread pool for image processing
    - *Persistent threads to avoid creation/destruction overhead*
    - *Task-based parallelism with work queues*
    - *Each thread processes multiple images over its lifetime*
  - Dedicated threads for I/O and communication
    - *Separate I/O threads prevent compute stalls*
    - *Prefetching to hide latency*
    - *Background communication threads for node coordination*

- **Assignment strategy**:
  - Initial static assignment
    - *Start with even distribution based on node capabilities*
    - *Accounts for known image size variations if available*
  - Work-stealing queue for dynamic balancing
    - *Double-ended queues for efficient stealing*
    - *Local LIFO, remote FIFO access patterns*
  - Priority for memory-intensive tasks
    - *Schedule memory-heavy tasks when bandwidth available*
    - *Group similar tasks to improve cache behavior*

---

## Step 3: Orchestrate Tasks

- **Synchronization strategy**:
  - Minimal global barriers
    - *Use only at major phase transitions if necessary*
    - *Prefer asynchronous coordination where possible*
  - Local synchronization through queues
    - *Producer-consumer queues between pipeline stages*
    - *Lock-free queues for high-throughput transfers*
  - Atomic operations for counters
    - *Track progress and completed work*
    - *Update statistics without locks when possible*

- **Communication patterns**:
  - Bulk transfers of image batches
    - *Aggregate multiple images for efficient network transfers*
    - *Amortize communication overhead across many images*
  - Reduce operation for aggregating results
    - *Hierarchical reduction for statistical results*
    - *Tree-based communication to minimize network contention*
  - Asynchronous I/O operations
    - *Non-blocking reads/writes to overlap I/O with computation*
    - *Double-buffering for continuous pipeline flow*

- **Resource management**:
  - Memory pools to reduce allocation overhead
    - *Pre-allocated buffers for images of similar sizes*
    - *Recycling of memory to avoid fragmentation*
  - Prefetching to hide I/O latency
    - *Read ahead for sequential image access patterns*
    - *Predictive prefetching based on processing order*

---

## Step 4: Optimize and Analyze

- **Initial performance**:
  - 60% efficiency on 128 cores
    - *Decent but not optimal resource utilization*
    - *Significant room for improvement*
  - I/O bottlenecks identified
    - *Disk access patterns causing contention*
    - *Network congestion during result collection*
  - Load imbalance during feature extraction
    - *Variable complexity of different images*
    - *Some nodes consistently finishing early*

- **Optimizations applied**:
  - Improved I/O with prefetching
    - *Added dedicated prefetch threads*
    - *Implemented image compression for network transfers*
  - Better work distribution algorithm
    - *Switched to work-stealing with history-based initial assignment*
    - *Implemented work chunking for better granularity control*
  - Memory layout optimization for cache utilization
    - *Reorganized image data for better spatial locality*
    - *Aligned buffers to cache line boundaries*

- **Final performance**:
  - 85% efficiency on 128 cores
    - *Significant improvement from optimizations*
    - *Most cores kept busy throughout execution*
  - Near-linear scaling up to 64 cores
    - *Almost perfect speedup until communication becomes significant*
  - 40x overall speedup
    - *From hours to minutes for the full dataset*

---

## Lessons Learned from the Case Study

- **Success factors**:
  - Hybrid decomposition approach
    - *Combined different parallelism types for maximum efficiency*
    - *Matched decomposition to problem characteristics*
  - Adaptive work assignment
    - *Dynamic adjustment prevented processor starvation*
    - *Handled unpredictable image processing times*
  - Minimizing global synchronization
    - *Local coordination reduced waiting time*
    - *Pipeline structure maintained continuous flow*
  - Careful attention to I/O and memory patterns
    - *Often the real bottleneck in data-intensive applications*
    - *Optimization focused on system-level performance*

- **Bottlenecks encountered**:
  - I/O subsystem limitations
    - *Disk bandwidth shared across processes*
    - *Network congestion during result collection*
  - Memory bandwidth contention
    - *Multiple cores competing for memory access*
    - *Cache thrashing with naive data layouts*
  - Global reduction operations
    - *Synchronization points causing idle time*
    - *Network topology affecting communication performance*
  - Task granularity tuning
    - *Finding optimal balance between overhead and parallelism*
    - *Required extensive experimentation and measurement*

- **General insights**:
  - No one-size-fits-all pattern
    - *Problem-specific considerations dominate*
    - *Combination of patterns often needed*
  - Multiple levels of parallelism needed
    - *Nodes, cores, vector units all contribute*
    - *Different strategies at each level*
  - Measurement and iteration are essential
    - *Performance analysis drove optimization decisions*
    - *Continuous refinement process yielded best results*

---

## 10. Parallel Programming Models and Tools

### Shared Memory Programming with OpenMP

- **Directive-based approach** for C, C++, Fortran
- **Programming model**:
  - Fork-join parallelism
  - Shared memory with thread-private variables
  - Implicit/explicit synchronization

- **Example: Parallel for loop**:
  ```c
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    result[i] = compute(data[i]);
  }
  ```

- **Strengths**: Easy to adopt, incremental parallelization
- **Limitations**: Single node, limited control

---

## Distributed Memory Programming with MPI

- **Message Passing Interface** standard
- **Programming model**:
  - SPMD with explicit messaging
  - Process-private memory
  - Explicit synchronization and data exchange

- **Example: Point-to-point communication**:
  ```c
  if (rank == 0) {
    MPI_Send(data, count, MPI_DOUBLE, 1, tag, comm);
  } else if (rank == 1) {
    MPI_Recv(data, count, MPI_DOUBLE, 0, tag, comm, &status);
  }
  ```

- **Strengths**: Scalable, portable, full control
- **Limitations**: Complex programming, explicit data management

---

## PGAS Languages: UPC, Chapel, and X10

- **Partitioned Global Address Space** model
- **Key concept**: 
  - Global memory space logically partitioned
  - Each portion has affinity to a processor
  - Global references with locality awareness

- **Example: Chapel distributed array**:
  ```chapel
  const D = {1..n} dmapped Block(boundingBox={1..n});
  var A: [D] real;
  
  forall i in D do
    A[i] = compute(i);
  ```

- **Strengths**: Simplified distributed programming, locality control
- **Limitations**: Performance tuning complexity, limited adoption

---

## GPU Programming with CUDA and SPMD

- **CUDA**: NVIDIA's GPU programming model
- **Core concepts**:
  - Kernels launch thousands of threads
  - Thread hierarchy (grids, blocks, threads)
  - Heterogeneous memory spaces

- **Example: CUDA kernel**:
  ```c
  __global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
      C[i] = A[i] + B[i];
  }
  
  // Launch with N threads
  vectorAdd<<<(N+255)/256, 256>>>(A, B, C, N);
  ```

- **Strengths**: Massive parallelism, high performance
- **Limitations**: Complex memory model, vendor-specific

---
## CUDA Kernel Launch: `vectorAdd<<<...>>>`

### 🚀 Launch Syntax

```cpp
vectorAdd<<<(N+255)/256, 256>>>(A, B, C, N);
```

- `<<<numBlocks, threadsPerBlock>>>` → **Execution configuration**
- Tells GPU how many threads to run and how to group them

---

### 🧮 Execution Breakdown

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

- `blockIdx.x`: ID of the current block
- `blockDim.x`: number of threads per block
- `threadIdx.x`: ID of the thread within its block

→ Computes **global index `i`** for each thread

---

### 📦 Example: `N = 1000`

```cpp
// Launch enough threads to cover all N elements
vectorAdd<<<(1000 + 255)/256, 256>>>(A, B, C, 1000);
```

- `numBlocks = 4` → 4 blocks
- `threadsPerBlock = 256`
- `Total threads = 4 × 256 = 1024`
- Threads with `i >= 1000` are skipped using:
  ```cpp
  if (i < n) C[i] = A[i] + B[i];
  ```

---

### Summary

- Each GPU thread computes one element of the result
- Threads are grouped into blocks for scheduling
- This pattern is common in **data-parallel GPU kernels**


## Big Data Parallelism with Apache Spark

- **RDD (Resilient Distributed Dataset)** abstraction
- **Programming model**:
  - Functional transformations on distributed data
  - Lazy evaluation
  - Implicit data distribution and fault tolerance

- **Example: Word count in Spark**:
  ```scala
  val wordCounts = textFile
    .flatMap(line => line.split(" "))
    .map(word => (word, 1))
    .reduceByKey(_ + _)
  ```

- **Strengths**: Ease of use, fault tolerance, scalability
- **Limitations**: JVM overhead, not for fine-grained tasks

---

## Heterogeneous Parallelism with SYCL and OpenCL

- **Single-source heterogeneous programming**
- **Target devices**: CPUs, GPUs, FPGAs, custom accelerators
- **Programming model**:
  - C++ with parallel patterns
  - Device-neutral code
  - Runtime selection of compute devices

- **Example: SYCL vector addition**:
  ```cpp
  queue q;
  buffer<float> buf_a(a, range<1>(n));
  buffer<float> buf_b(b, range<1>(n));
  buffer<float> buf_c(c, range<1>(n));
  
  q.submit([&](handler& h) {
    auto a = buf_a.get_access<access::mode::read>(h);
    auto b = buf_b.get_access<access::mode::read>(h);
    auto c = buf_c.get_access<access::mode::write>(h);
    
    h.parallel_for(range<1>(n), [=](id<1> i) {
      c[i] = a[i] + b[i];
    });
  });
  ```

---

## When to Use Which Model?

- **Single node, shared memory**:
  - OpenMP for simplicity
  - Threading libraries for more control

- **Multiple nodes, distributed memory**:
  - MPI for performance and control
  - PGAS for productivity with locality control

- **GPU acceleration**:
  - CUDA for NVIDIA GPUs (best performance)
  - OpenCL/SYCL for vendor-neutral code

- **Big data processing**:
  - Spark for ease of use and fault tolerance
  - MPI+X for custom, high-performance needs

---

## Parallel Programming Models Comparison

- **Shared Memory**:
  - *Global address space accessible by all processors*
  - *Examples: OpenMP, Pthreads, C++/Java threads*
  - *Advantages: Easier programming, efficient data sharing*
  - *Limitations: Limited scalability, potential for race conditions*

- **Distributed Memory**:
  - *Separate address spaces connected via message passing*
  - *Examples: MPI, UPC*
  - *Advantages: Excellent scalability, clear data ownership*
  - *Limitations: Higher complexity, explicit communication*

- **Hybrid**:
  - *Combines shared and distributed memory approaches*
  - *Examples: MPI+OpenMP, MPI+CUDA*
  - *Advantages: Maps well to modern HPC architectures*
  - *Limitations: Complex programming model, tuning challenges*

- **GPGPU/Accelerator**:
  - *Offloads computation to specialized hardware*
  - *Examples: CUDA, OpenCL, OpenACC*
  - *Advantages: Massive parallelism for specific workloads*
  - *Limitations: Memory transfer overhead, specialized programming*

![Parallel Programming Models Comparison](images/parallel_models_comparison.svg)

---

## 11. Advanced Topics

### Load Balancing Strategies

- **Static load balancing**:
  - Predetermined task distribution before execution
  - Block distribution: consecutive chunks to each process
  - Cyclic distribution: interleaved assignment (round-robin)
  - Good for homogeneous tasks/processors
  - Simple implementation with low overhead

- **Dynamic load balancing**:
  - Tasks assigned during runtime as processors become available
  - Work stealing: idle processors take work from busy ones
  - Centralized queue: central task pool for all processors
  - Better for irregular workloads and heterogeneous systems
  - Adds runtime overhead but improves resource utilization

- **Hybrid approaches**:
  - Combine static initial distribution with dynamic adjustments
  - Hierarchical strategies for different architecture levels
  - Example: static distribution between nodes, dynamic within nodes

![Load Balancing Strategies](images/load_balancing_strategies.svg)

---

## Scalability: Strong vs. Weak Scaling

- **Strong scaling**:
  - Fixed problem size
    - *Total amount of work remains the same as processors increase*
    - *Each processor gets less work as processor count grows*
    - *Goal: Solve the same problem faster with more processors*
  
  - Measure speedup as processors increase
    - *Ideal: Linear speedup (doubling processors halves execution time)*
    - *Reality: Sub-linear due to communication overhead and Amdahl's Law*
    - *Formula: Speedup(N) = T₁/T_N where T₁ is single-processor time and T_N is N-processor time*
  
  - Challenges: Communication overhead, Amdahl's law
    - *Communication costs grow relative to computation as work per processor shrinks*
    - *Serial portions become the dominant factor limiting performance*
    - *Eventually adding more processors yields diminishing or negative returns*

- **Weak scaling**:
  - Problem size increases with processors
    - *Work per processor remains constant as system scales*
    - *Total problem size grows linearly with processor count*
    - *Goal: Solve larger problems with more processors in the same time*
  
  - Each processor gets same amount of work
    - *Processor workload stays balanced as system grows*
    - *More practical for many real-world scenarios with large datasets*
  
  - Measure efficiency as processors increase
    - *Ideal: Constant execution time regardless of processor count (efficiency = 1.0)*
    - *Reality: Slight increase in time due to growing communication overhead*
    - *Formula: Efficiency(N) = T₁/(T_N) where T₁ is execution time on one processor and T_N is execution time on N processors*
  
  - Challenges: Memory scaling, increased communication
    - *Global communication operations often scale with log(N) or worse*
    - *Memory footprint must increase with problem size*
    - *May encounter system limitations as problem size grows*

- **Crossover point**: Where adding more processors hurts performance
  - *The number of processors beyond which performance degrades*
  - *Determined by problem characteristics, communication patterns, and hardware*
  - *Critical to identify for production deployments*

![Strong vs. Weak Scaling](images/strong_vs_weak_scaling.svg)

---

## Fault Tolerance in Parallel Systems

- **Why it matters**: Large systems have higher failure rates
- **Techniques**:
  - **Checkpointing**: Save state periodically
  - **Replication**: Multiple copies of tasks/data
  - **Message logging**: Record communication for replay
  - **Algorithm-based fault tolerance**: Exploit algorithm properties

- **Programming models support**:
  - Spark: Built-in resilience for RDDs
  - MPI: Fault tolerance extensions
  - Custom middleware for checkpointing

---

## Energy Efficiency in Parallel Computing

- **Growing concern** in high-performance computing
- **Hardware approaches**:
  - Dynamic voltage and frequency scaling
  - Power gating unused components
  - Heterogeneous computing (specialized accelerators)

- **Software approaches**:
  - Workload consolidation
  - Energy-aware scheduling
  - Algorithm redesign to reduce communication
  - Trading off performance for energy

- **Metrics**: FLOPS/watt, Energy Delay Product (EDP)

---

## Parallelism in the Cloud: Serverless and Autoscaling

- **Serverless computing model**:
  - Functions as building blocks
  - Automatic scaling
  - Event-driven parallelism

- **Auto-scaling services**:
  - Dynamic resource allocation
  - Horizontal vs. vertical scaling
  - Load-based triggers

- **Challenges**:
  - Stateless function limitations
  - Cold start overhead
  - Limited execution time
  - Resource allocation granularity

---

## Heterogeneous Parallelism (CPUs, GPUs, FPGAs, TPUs)

- **Diverse computing devices**:
  - CPUs: General purpose, complex cores
  - GPUs: Massively parallel, simpler cores
  - FPGAs: Reconfigurable hardware
  - TPUs/ASICs: Domain-specific acceleration

- **Programming challenges**:
  - Multiple programming models
  - Data movement overhead
  - Load balancing across different devices
  - Performance portability

- **Unified approaches**: OpenCL, SYCL, OneAPI, ROCm

---

## Future Trends: Quantum and Neuromorphic Computing

- **Quantum computing**:
  - Qubit-based systems
  - Quantum parallelism
  - Specialized algorithms (Shor's, Grover's)
  - Hybrid classical-quantum approaches

- **Neuromorphic computing**:
  - Brain-inspired architectures
  - Spiking neural networks
  - Low power, highly parallel
  - Event-driven processing

- **Impact on parallel programming**:
  - New programming models
  - Different decomposition strategies
  - Specialized algorithm designs

---

## 12. Wrap-Up

### Summary of Key Concepts

- **Parallel programming foundations**:
  - Decomposition: Breaking down problems
  - Assignment: Mapping tasks to processors
  - Orchestration: Coordinating execution
  - Communication: Exchanging data

- **Performance considerations**:
  - Amdahl's Law and scalability limits
  - Overhead vs. parallelism
  - Load balancing
  - Communication minimization

- **Programming models**: Choose based on application requirements

---

## Common Pitfalls to Avoid

- **Over-synchronization**: Unnecessary barriers and locks
  - *Adding too many synchronization points creates bottlenecks and reduces parallel efficiency*
  - *Example: Placing locks around operations that don't require protection*

- **Fine-grained parallelism** with high overhead
  - *When parallelism is too fine-grained, the management overhead exceeds the performance benefits*
  - *Example: Parallelizing tiny loops where thread creation costs more than the computation*

- **Ignoring locality**: Excessive remote memory access
  - *Remote memory access is significantly slower than local access in distributed systems*
  - *Failing to organize data for locality leads to performance degradation*

- **Over-optimistic scaling expectations**: Amdahl's Law
  - *Serial portions of your code fundamentally limit your maximum possible speedup*
  - *Even small serial fractions (1-5%) can severely restrict scaling at higher processor counts*

- **One-size-fits-all approach**: No universal solution
  - *Different problems require different parallelization strategies*
  - *Using the wrong pattern for your specific problem leads to suboptimal performance*

- **Premature optimization**: Measure before optimizing
  - *Optimizing without measurement often targets the wrong bottlenecks*
  - *Profile first to identify where time is actually being spent*

- **Ignoring serial bottlenecks**: Focus where it matters
  - *Per Amdahl's Law, optimizing parallel sections has diminishing returns if serial bottlenecks remain*
  - *A small serial bottleneck can negate benefits from even perfectly parallelized code*

- **Racing conditions and deadlocks**: Careful synchronization design
  - *Improper synchronization leads to data corruption or program freezes*
  - *Use higher-level synchronization constructs and validate concurrent code carefully*

---

## Further Learning Resources

- **Books**:
  - "Patterns for Parallel Programming" (Mattson, Sanders, Massingill)
  - "Programming Massively Parallel Processors" (Kirk, Hwu)
  - "Structured Parallel Programming" (McCool, Reinders, Robison)

- **Online courses**:
  - "Parallel Programming" on Coursera
  - "High Performance Computing" on edX

- **Documentation and tutorials**:
  - OpenMP, MPI, CUDA programming guides
  - Lawrence Livermore National Lab tutorials
  - Intel and NVIDIA developer resources 