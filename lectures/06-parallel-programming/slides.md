# Parallel Programming: Concepts and Strategies
## GWU ECE 6125: Parallel Computer Architecture

---

## 1. Introduction to Parallel Programming

### What is Parallel Programming?
- **Definition**: Simultaneous execution of multiple computations
- **Goal**: Increase speed and efficiency
- **Motivation**: Solve larger problems, reduce execution time
- **Core concept**: Breaking down problems into smaller parts that can run concurrently

![Sequential vs. parallel execution](images/sequential_vs_parallel.svg)

---

## Why Parallelism? The Need for Speed

- **Performance limits in sequential processing**
  - Clock speeds have plateaued
  - Power/thermal constraints ("Power Wall")
  - Memory access bottlenecks
- **Moore's Law transitions to multi-core**
  - Transistor count still increases
  - But single-core performance has flattened
  - Multiple cores allow continued performance scaling

![Moore's Law and multi-core transition](images/moores_law_multicore.svg)

---

## Brief History: From Single-Core to Many-Core

- **Early processors**: Sequential execution, von Neumann architecture
- **Vector processors** (1970s): SIMD instructions
- **Superscalar processors** (1980s-90s): Instruction-level parallelism
- **Multi-core CPUs** (2000s-present): Multiple cores on a single die
- **Many-core processors** (2010s): Tens to hundreds of cores
- **GPUs for general computing**: Thousands of simple cores
- **Heterogeneous computing**: Specialized accelerators for different workloads

---

## Challenges in Parallel Programming

- **Concurrency**: Managing simultaneous execution
  - *Coordinating multiple execution streams running at the same time*
  - *Requires careful design to avoid conflicts and race conditions*

- **Synchronization**: Coordinating access to shared resources
  - *Ensuring orderly access to data that multiple processes need to use or modify*
  - *Balancing protection with performance overhead*

- **Load balancing**: Distributing work evenly
  - *Ensuring all processors have similar amounts of work to avoid idle time*
  - *Particularly challenging with irregular or unpredictable workloads*

- **Communication overhead**: Data exchange between processors
  - *Time spent transferring data instead of computing results*
  - *Can become the dominant cost in distributed systems*

- **Scalability**: Performance as processors increase
  - *Maintaining efficiency as you add more processors*
  - *Limited by serial portions and increasing communication costs*

- **Debugging complexity**: Non-deterministic behavior
  - *Parallel bugs may appear intermittently and be difficult to reproduce*
  - *Traditional debugging techniques often inadequate*

- **Algorithm redesign**: Many sequential algorithms don't parallelize well
  - *Sequential algorithms often rely on state that's difficult to parallelize*
  - *May need fundamentally different approaches for parallel execution*

> "Making sequential programs run in parallel is so hard that it is considered one of computer science's grand challenges." — Tim Mattson, Intel

---

## 2. Thinking in Parallel

### The Parallel Programmer's Mindset

- **Shift from sequential to parallel thinking**
  - Identify independent operations
  - Understand data and control dependencies
  - Focus on concurrent execution paths
- **Key mental shift**:
  - Sequential: "What's the next step?"
  - Parallel: "What can be done at the same time?"
- **Different design approach**:
  - Break problem into concurrent units
  - Manage coordination between units
  - Consider data access patterns carefully

---

## Identifying Parallelism Opportunities

- **Look for independent tasks or data**
  - Data elements processed independently
  - Iterations with no dependencies
  - Separate functions that can run concurrently
- **Domain-specific parallelism**:
  - **Image processing**: Pixel-level parallelism
  - **Simulations**: Spatial decomposition
  - **AI/ML**: Batch processing, matrix operations
  - **Search/Sort**: Divide and conquer
  - **Graph algorithms**: Vertex/edge parallelism

---

## Types of Parallelism

- **Data Parallelism**: Same operation on different data
  - *Performs identical operations simultaneously on multiple data elements*
  - *Example: Applying a filter to each pixel in an image*
  - *Ideal when operations are independent and uniform across data*
  - *Scales well with data size and processor count*

- **Task Parallelism**: Different operations on same or different data
  - *Executes different functions simultaneously on potentially different data*
  - *Example: Rendering different parts of a 3D scene with specialized tasks*
  - *Good for heterogeneous workloads with distinct operations*
  - *Often requires more synchronization than data parallelism*

- **Pipeline Parallelism**: Assembly line of tasks
  - *Data flows through a series of processing stages*
  - *Different elements are at different stages simultaneously*
  - *Example: Video processing stages (decode, process, encode)*
  - *Throughput limited by the slowest stage (bottleneck)*
  - *Effective for streaming data with sequential dependencies*

![Types of parallelism](images/types_of_parallelism.svg)

---

## Case Study: Parallelism in Image Processing

- **Image as a 2D array of pixels**
- **Parallel processing approaches**:
  - **Pixel-level**: Apply operations to each pixel independently
  - **Block-level**: Process blocks of pixels in parallel
  - **Pipeline**: Stages for loading, processing, saving
- **Example operations**:
  - Blur, sharpen, edge detection, color transformations
  - Each can be done in parallel across the image
- **Speedup**: Nearly linear with processor count for embarrassingly parallel operations

---

## The SPMD Model (Single Program Multiple Data)

- **Core concept**: Same program runs on all processors, but each processes different data
- **Common implementation**: Use processor ID to determine which data to process
- **Widely used in**:
  - MPI applications
  - CUDA/GPU computing
  - OpenMP parallel sections
- **Example pseudocode**:
  ```
  function parallel_process():
    my_id = get_processor_id()
    my_data = get_data_chunk(my_id)
    result = process(my_data)
    return result
  ```

---

## 3. Decomposition

### What is Decomposition?

- **Definition**: Breaking the problem into parts that can be solved in parallel
- **Key considerations**:
  - Identify independent components
  - Balance computation across parts
  - Minimize communication between parts
  - Match problem structure to hardware capabilities
- **Goal**: Create enough parallelism to keep all processors busy
- **Importance**: Foundation of parallel algorithm design

![Problem decomposition](images/task_vs_data_decomposition.svg)

---

## Task Decomposition vs. Data Decomposition

- **Task Decomposition**
  - Divide by functionality
  - Different tasks perform different functions
  - Can be heterogeneous (different sizes)
  - Examples: UI rendering + physics + AI in a game
  
- **Data Decomposition**
  - Divide data among processors
  - Same operation on different data elements
  - Usually homogeneous (similar sizes)
  - Examples: Matrix multiplication, image processing

---

## Choosing the Right Decomposition Strategy

- **Consider these factors**:
  - Problem structure (data vs. function oriented)
  - Data access patterns and dependencies
  - Communication requirements
  - Load balance
  - Hardware architecture

- **Decision framework**:
  - Highly regular data operations → Data decomposition
  - Different computational phases → Task decomposition
  - Complex, irregular dependencies → Task/data hybrid
  - Streaming data → Pipeline decomposition

---

## Example - Summing an Array

- **Serial sum**:
  ```
  sum = 0
  for i from 0 to n-1:
    sum += array[i]
  ```

- **Parallel sum**:
  ```
  // Each processor does this:
  local_sum = 0
  for i in my_chunk:
    local_sum += array[i]
  
  // Then combine using tree reduction
  global_sum = reduce_sum(local_sum)
  ```

- **Tree reduction**: Logarithmic communication steps
- **Speedup**: Can approach linear with enough data

---

## Decomposition in SPMD Programs

- **Common approach**: Each processor handles a subset of the data
- **Data division methods**:
  - **Block division**: Contiguous chunks
  - **Cyclic division**: Round-robin assignment
  - **Block-cyclic**: Blocks in round-robin fashion
- **Example: Vector addition in SPMD**
  ```
  function parallel_vector_add(A, B):
    my_id = get_processor_id()
    my_chunk = get_my_data_range(my_id, A.length)
    
    for i in my_chunk:
      C[i] = A[i] + B[i]
  ```

---

## Decomposition Granularity: Fine vs. Coarse

- **Fine-grained decomposition**:
  - Many small tasks/data chunks
  - Better load balancing
  - Higher overhead (communication, management)
  - Example: One matrix element per task

- **Coarse-grained decomposition**:
  - Fewer, larger tasks/data chunks
  - Lower overhead
  - Potential load imbalance
  - Example: One matrix row per task

- **Optimal granularity**: Balance overhead vs. parallelism

---

## 4. Assignment

### Assigning Tasks to Processors

- **Definition**: Mapping decomposed tasks to physical processors
- **Considerations**:
  - Processor capabilities (heterogeneous systems)
  - Communication patterns and proximity
  - Memory access patterns
  - Load balance

- **Assignment policy impacts**:
  - Overall performance
  - Resource utilization
  - Communication overhead

---

## Static vs. Dynamic Assignment

- **Static Assignment**:
  - Tasks assigned at compile/start time
  - Fixed throughout execution
  - Low runtime overhead
  - Works well for predictable, uniform workloads
  - Examples: Regular domains in scientific computing

- **Dynamic Assignment**:
  - Tasks assigned during execution
  - Task queues, work stealing
  - Better load balancing for irregular workloads
  - Higher runtime overhead
  - Examples: Graph algorithms, recursive divide-and-conquer

---

## Load Balancing: Ensuring Equal Workload

- **Goal**: Equal computation on all processors
- **Challenges**:
  - Unpredictable task durations
  - Heterogeneous processor capabilities
  - Dynamic workloads

- **Techniques**:
  - **Work stealing**: Idle processors take work from busy ones
  - **Task queues**: Centralized or distributed queues of pending tasks
  - **Over-decomposition**: More tasks than processors
  - **Self-scheduling**: Processors request work as they become available

---

## Load Balancing Strategies

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

## Mapping Data to Processors in Data Parallelism

- **Block distribution**:
  - Contiguous chunks of data to each processor
  - Good spatial locality
  - Simple indexing

- **Cyclic distribution**:
  - Round-robin assignment
  - Better load balancing for non-uniform workloads
  - More complex indexing

- **Block-cyclic distribution**:
  - Blocks assigned in round-robin fashion
  - Balance between locality and load balancing
  - Used in scientific libraries (ScaLAPACK)

---

## Example - Work Assignment in Parallel Sorting

- **Parallel quicksort approach**:
  1. Select pivot
  2. Partition elements in parallel
  3. Recursively sort subarrays in parallel

- **Assignment strategies**:
  - Fixed processors per subarray (static)
  - Dynamic processor reassignment to larger subarrays
  - Work stealing for load balancing
  - Over-decomposition: create more subtasks than processors

---

## Granularity Revisited: Impact on Assignment

- **Too fine-grained**:
  - High task management overhead
  - Excessive communication/synchronization
  - Scheduling becomes a bottleneck

- **Too coarse-grained**:
  - Underutilized processors (idle time)
  - Poor load balancing
  - Limited scalability

- **Adaptive granularity**:
  - Start coarse, split if needed
  - Example: Parallel loops with dynamic chunk sizing

---

## 5. Orchestration

### What is Orchestration?

- **Definition**: Managing execution of parallel tasks
- **Includes**:
  - Synchronization between tasks
  - Communication mechanisms
  - Ensuring correct execution order
  - Managing shared resources
  - Handling dependencies

- **Design considerations**:
  - Minimize synchronization points
  - Balance parallelism and coordination
  - Select appropriate communication patterns

---

## Coordination Between Parallel Tasks

- **Types of dependencies**:
  - **Data dependencies**: Output from one task needed by another
    - *Occurs when one task produces data that another task requires*
    - *Example: Task B needs results computed by Task A to proceed*
  
  - **Control dependencies**: Task ordering requirements
    - *One task must complete before another can begin regardless of data*
    - *Example: Initialization must complete before main processing starts*
  
  - **Resource dependencies**: Access to shared resources
    - *Multiple tasks need access to the same limited resource*
    - *Example: Multiple threads writing to the same file or network connection*

- **Coordination mechanisms**:
  - **Barriers**:
    - *Force all threads/processes to wait until everyone reaches the barrier*
    - *Synchronizes all participating tasks at a specific point in execution*
    - *Useful for phase-based algorithms where all tasks must complete a phase before any can proceed*
  
  - **Locks/semaphores**:
    - *Locks (mutexes): Allow only one thread to access a resource at a time*
    - *Semaphores: Control access to a limited number of resources (counting semaphores)*
    - *Protect critical sections from concurrent access to prevent data corruption*
  
  - **Message passing**:
    - *Tasks explicitly send/receive data between each other*
    - *Synchronization is implicit in the communication (e.g., blocking receive)*
    - *Primary coordination mechanism in distributed memory systems (MPI)*
  
  - **Futures/promises**:
    - *Placeholders for values that will be computed asynchronously*
    - *Allow tasks to continue until they actually need the result*
    - *Example: Task A computes value X while Task B works on other things, then waits for X only when needed*
  
  - **Atomic operations**:
    - *Hardware-supported indivisible operations that cannot be interrupted*
    - *Examples: atomic increment, compare-and-swap, test-and-set*
    - *Useful for simple shared data updates without the overhead of locks*

---

## Synchronization: Why It's Needed

- **Purpose**:
  - Protect shared data
  - Enforce ordering constraints
  - Ensure task completion
  - Coordinate shared resource access

- **Common situations requiring synchronization**:
  - Shared variable updates
  - Producer-consumer relationships
  - Critical sections
  - Data aggregation points

---

## Race Conditions: What Can Go Wrong

- **Definition**: Outcome depends on relative timing of operations
- **Common causes**:
  - Unprotected shared data access
  - Missing synchronization
  - Incorrect locking protocols

- **Example: Shared counter**:
  ```
  // Incorrect parallel increment
  counter++;  // Actually: load, increment, store
  
  // Correct version
  atomic_increment(counter);  // Or use locks
  ```

- **Consequences**: Incorrect results, data corruption, system crashes

---

## Synchronization Tools

- **Locks**:
  - Mutex: Exclusive access to a resource
    - *A mutual exclusion (mutex) lock ensures only one thread can enter a critical section at a time*
    - *Thread must acquire the lock before entering protected code and release it when done*
    - *Other threads attempting to acquire the lock will wait (block) until it's available*
  
  - Reader-writer locks: Multiple readers, exclusive writers
    - *Allow concurrent read access but exclusive write access*
    - *Useful when reads are frequent but writes are rare*
    - *Example: Database systems where many queries read data but few update it*

- **Semaphores**:
  - Control access to a finite number of resources
    - *A counting mechanism that restricts access to a specified number of threads*
    - *Binary semaphore (value 0 or 1) is similar to a mutex*
    - *Counting semaphore can allow multiple threads (up to a limit) to access a resource*
  
  - Signal between threads/processes
    - *Used for producer-consumer problems and thread synchronization*
    - *Producer increases count, consumer decreases count*
    - *Can block when resources are unavailable (count = 0)*

- **Barriers**:
  - Force all threads to wait at a specific point
    - *A barrier blocks threads until a specified number have reached the barrier*
    - *Creates a synchronization point in parallel algorithms*
    - *Example: In an iterative solver, all threads must complete iteration N before any can start iteration N+1*
  
  - Continue only when all threads arrive
    - *Ensures all threads are at the same logical point in the program*
    - *Prevents threads from getting too far ahead or behind*
    - *Implementation: counter with a mutex and condition variable*

- **Atomic operations**:
  - Hardware-supported indivisible operations
    - *Operations that complete in a single, uninterruptible step*
    - *Guaranteed to be executed without interference from other threads*
    - *Much more efficient than using locks for simple operations*
  
  - Compare-and-swap, fetch-and-add, etc.
    - *Compare-and-swap (CAS): Atomically compare and change a value if it matches expected value*
    - *Fetch-and-add: Atomically increment a value and return the original value*
    - *Used to implement lock-free data structures and algorithms*

---

## Shared vs. Distributed Memory Architectures

- **Shared memory**:
  - All processors access same address space
  - Direct read/write to shared variables
  - Synchronization via locks, atomics

- **Distributed memory**:
  - Each processor has private memory
  - Communication through explicit messages
  - Synchronization via message passing

- **Hybrid systems**:
  - Shared memory within nodes
  - Distributed memory across nodes

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

- **Purpose**: Exchange data between parallel tasks
- **Types of communication**:
  - **Point-to-point**: Direct exchange between two tasks
    - *Communication occurs directly between a sender and a receiver*
    - *Examples: MPI_Send/MPI_Recv in MPI, direct message passing*
    - *Can be synchronous (blocking) or asynchronous (non-blocking)*
    - *Useful for nearest-neighbor communication patterns*
  
  - **Collective**: Coordinated among multiple tasks
    - *Operations involving a group of processes/threads*
    - *Examples: broadcast, gather, scatter, reduce, alltoall*
    - *More efficient than multiple point-to-point communications*
    - *Often optimized with tree-based or other advanced algorithms*
  
  - **One-sided**: Remote memory access
    - *One process directly accesses memory of another without its participation*
    - *Examples: MPI_Put, MPI_Get, PGAS operations*
    - *Can reduce synchronization overhead*
    - *Requires special hardware/software support*

- **Performance considerations**:
  - **Latency**: Time to initiate transfer
    - *Fixed overhead to start a communication*
    - *Dominates performance for small messages*
  
  - **Bandwidth**: Data volume per time
    - *Maximum rate at which data can be transferred*
    - *Dominates performance for large messages*
  
  - **Contention**: Multiple tasks communicating simultaneously
    - *When communication channels become congested*
    - *Reduces effective bandwidth and increases latency*

---

## Shared Memory Communication

- **Mechanism**: Direct read/write to shared variables
- **Advantages**:
  - Simple programming model
  - Low latency
  - Zero-copy data exchange

- **Challenges**:
  - Cache coherence overhead
  - Synchronization required
  - Limited scalability
  - False sharing

- **OpenMP example**:
  ```c
  #pragma omp parallel shared(result)
  {
    // All threads read/write to shared variable
    result[omp_get_thread_num()] = compute();
  }
  ```

---

## Message Passing Communication (MPI)

- **Mechanism**: Explicit send/receive operations
- **Characteristics**:
  - Sender specifies data and recipient
  - Receiver allocates buffer
  - Synchronous or asynchronous

- **Communication types**:
  - **Point-to-point**: Between specific processes
  - **Collective**: Involves all processes in a group

- **MPI example**:
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

- **Sources of overhead**:
  - Latency (setup time)
  - Limited bandwidth
  - Contention
  - Synchronization delays
  - Protocol processing

- **Minimization strategies**:
  - Reduce communication frequency
  - Batch communications (coarser granularity)
  - Overlap communication with computation
  - Use asynchronous communication
  - Optimize data layouts

---

## Example - Parallel Matrix Multiplication Communication

- **1D decomposition (row-based)**:
  - Each process has complete rows
  - All processes need the entire second matrix
  - Collective broadcast of second matrix

- **2D decomposition (block-based)**:
  - Each process has a submatrix block
  - Communication along rows and columns
  - Reduced communication volume

- **Communication volume comparison**:
  - 1D: O(n²)
  - 2D: O(n²/√p) where p is processor count

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