# Post-Moore Ideas
## Neuromorphic Computing · Photonic Computing and Communication
### Armin Mehrabian · Spring 2026

---

## Outline

1. Background & Motivations
2. Problem Statement & Objectives
3. Related Work
4. Approach, Accomplishments & Results
5. Conclusion

---

## Background & Motivations

### The End of Moore's Law

![Moore's Law scaling breakdown](images/image12.png)

- Computing performance has been challenged by the **end of Dennard scaling** in 2004
- "Will need non-traditional computing models and accelerators, such as **neuro-inspired** and **quantum** accelerators"

> *Source: researchcomputingservices.github.io/parallel-computing, July 2020*

Note:
Moore's Law predicted transistor count doubling every ~2 years. Dennard scaling — which kept power density constant as transistors shrank — broke down around 2004. Since then, increasing transistors no longer gives "free" performance.

---

## Background & Motivations

### Fundamental & Architectural Limits of Electronics

![Physical limits of scaling](images/image6.png)

**Physical limits:**
- Components approaching **atomic scale**
- Excessive **thermal noise** and **quantum tunneling**
- Capacitive effects become prohibitive

**Architectural limits:**
- **Von Neumann bottleneck** — memory and compute are separated
- Data movement dominates: bandwidth, fan-out constraints

Note:
The memory-compute separation in Von Neumann architecture means every operation requires moving data — increasingly expensive in energy and latency at scale.

---

## Background & Motivations

### Computation–Communication Disparity

![Computation vs communication bandwidth gap](images/image13.png)

- Node **computation capacity** increasing rapidly
- Node **bandwidth** increasing — but more slowly
- The disparity between compute and communication bandwidth **grows over time**
- Photonics can address **both** — it offers high-bandwidth, low-latency, energy-efficient interconnects

> Sunway TaihuLight (Nov 2017): B/F = 0.004 · Summit HPC (June 2018): B/F = 0.0005

*Bergman, K. "Empowering Flexible and Scalable High Performance Architectures with Embedded Photonics." IPDPS, 2018.*

---

## Background & Motivations

### Computational Fundamentals from Biology

![Biological neuron structure](images/image18.jpg)

- **Processing/memory proximity** — brain integrates memory and compute; contrary to Von Neumann
- **Heterogeneity in neurons** — sensory, motor, interneuron each with millions of variations
- **Large fan-in/fan-out** — ~10⁴ connections per neuron → photonics (WDM) handles this naturally
- **Long neurons** — optic nerve ~50 mm, motor nerves up to 1 m → photonics offers flat communication cost

> *"World is my cache"* — distance-agnostic optical communication

*Sources: faculty.washington.edu/chudler/facts.html · courses.lumenlearning.com*

---

## Background & Motivations

### Potential for Photonic Neuromorphic

![Photonic neuromorphic potential comparison](images/image25.png)

- Neuromorphic hardware (IBM TrueNorth, SpiNNaker) dramatically improves MAC efficiency
- Photonic integration can push this further — **attojoule/MAC** regime
- MAC = Multiply and Accumulate (the core operation in neural networks)

*Prucnal & Shastri, Neuromorphic Photonics, CRC Press, 2017 · Merolla et al. Science 345 (2014) · Furber et al. Proc. IEEE 102 (2014)*

---

## What Photonics Brings to the Table

![Distance-agnostic energy curve](images/image1.png)

![Optical bandwidth advantages](images/image16.png)

- **Distance-agnostic communication** — energy cost does not grow with distance (unlike copper)
- Flattens the energy curve for chip-to-chip and rack-to-rack links

*Cheng, Q. et al. "Recent advances in optical technologies for data centers." Optica 5.11 (2018): 1354–1370.*

---

## Background & Motivations

### Three Ways Photonics Can Assist

![Photonic links](images/image8.png)

![Bandwidth steering](images/image15.png)

![Photonic processors](images/image5.png)

1. **Photonic Links** — replacing copper wires for energy-efficient data movement
2. **Bandwidth Steering** — dynamic optical circuit switching
3. **Photonic Processors** — all-optical compute for MAC operations

---

## Background & Motivations

### Top Green500 Energy Efficiency

![Green500 energy efficiency trend](images/image9.png)

- Energy efficiency increasing rapidly from June 2016 to June 2018
- Fueled by GPU accelerators and heterogeneous design
- **Fundamental question:** can we continue improving efficiency beyond electronics?

---

## Background & Motivations

### Scaling Under Tight Energy Budget

- Increase **hit rates** of close memory banks (Cache, DRAM, ...)
- Improve **memory access energy efficiency**
- Improve **data movement energy efficiency**
- Decrease data movement through **locality exploitation**
- Increase **computing energy efficiency** — attojoule/MAC with photonics

*Miller, D. A. B. "Attojoule optoelectronics for low-energy information processing and communications." Journal of Lightwave Technology 35.3 (2017): 346–396.*

---

## Background & Motivations

### Machine Learning Workloads Dominating

![ML workload dominance](images/image19.png)

![GPU performance scaling](images/image27.png)

- Workloads are changing dramatically in favor of **Neural Networks**
- GPU design follows suit — NVIDIA introduced **Tensor Cores** for matrix multiply acceleration
- Tesla V100 with Tensor Cores: orders of magnitude faster for ML vs. P100

*NVIDIA News Center, Oct. 2017*

---

## Motivation Summary

- **Electronics approaching limits** — Dennard scaling ended, Moore's Law slowing
- **Optical advancements** bring significant promise in energy and bandwidth
- **Neuromorphic computing** is a viable alternative to Von Neumann architectures
- **Workloads are changing** — data analytics, neural networks dominate
- Next-generation tools must consider future workloads
- Moving from **homogeneous** to **heterogeneous** specialization — no one-size-fits-all

---

## Problem Statement

**Given:**
- Advances in **nanophotonics** and **neuromorphics** providing good energy efficiency (attojoule/MAC), high speed, and high bandwidth (WDM)
- Prior research focused only on **device-level or small circuit-level** neuromorphic nanophotonics

**Goals:**
- Explore design of **efficient specialized neuromorphic photonic computing systems** that:
  - Execute state-of-the-art neural networks
  - Exhibit great energy efficiency and speed
  - Are **weight programmable**
- Develop a **design and simulation framework** for neuromorphic photonics
- Highlight potentials, pitfalls, and limitations

---

## Related Work

### Evolution of Neuromorphic Computing

![Evolution of neuromorphic computing](images/image21.png)

From early spiking models to modern neuromorphic hardware — a 40-year trajectory toward brain-inspired computation.

---

## Related Work

### Biological Plausibility Spectrum

Different neuron models trade off **computational accuracy vs. hardware cost**:

| Model | Biological Plausibility (BP) |
|---|---|
| Artificial Neural Network | BP = 0 |
| Leaky Integrate-and-Fire (LI&F) | BP = 0.2 |
| FitzHugh-Nagumo | BP = 0.5 |
| Single-compartment Hodgkin-Huxley | BP = 0.6 |
| Full spiking model with membrane dynamics | BP = 1.0 |

Higher BP → more accurate, but **exponentially more compute** per neuron.

---

## Related Work

### Brain Neuron vs. Artificial Neuron

![Biological vs artificial neuron](images/image37.png)

![Perceptron model](images/image22.png)

- **Dendrites** receive electrical signals from axons of other neurons
- Signals are modulated — neuron fires only when total input **exceeds a threshold** → nonlinear activation
- In the **perceptron**, this maps to: weighted inputs + nonlinear activation function

*Kendall & Kumar. "The building blocks of a brain-inspired computer." Applied Physics Reviews 7.1 (2020).*

---

## Related Work

### Photonic MAC Operator (Neuron)

![Photonic MAC using MRR](images/image26.png)

![MRR weight bank](images/image35.png)

- Photonic MAC using **Microring Resonators (MRR)**
- Wavelength-sensitive → can benefit from **WDM** (Wavelength Division Multiplexing)
- MRR weight banks implement the weighted sum; photodetectors + electronics handle nonlinear activation

*Tait et al. "Neuromorphic photonic networks using silicon photonic weight banks." Scientific Reports 7 (2017).*

---

## Related Work

### Coherent 2-Layer Fully Connected Neural Network

![Coherent photonic neural network chip](images/image44.png)

![SVD matrix decomposition](images/image30.png)

- **Matrix decomposition** M = UΣVᵀ using optical Mach-Zehnder interferometer (MZI) meshes
- U, Σ, V implemented as separate optical matrix multiply cores
- First experimental demonstration of **deep learning with coherent nanophotonic circuits**

*Shen, Y. et al. "Deep learning with coherent nanophotonic circuits." Nature Photonics 11.7 (2017): 441.*

---

## Approach

### Architecture Design Principles

- Seeking **architecturally efficient** neural network designs through:
  - **Specialization** for common cases
  - **Transformations** to reduce hardware
  - Leveraging **novel devices**

**Photonic constraints to design around:**

| Property | Status |
|---|---|
| Footprint | Can be bulky — shrinking |
| Power | Extremely low, length-agnostic |
| Speed | Very high |
| Bandwidth | WDM ~10–100× parallelization |
| Memory | Research in progress |
| Interfacing | Requires DAC/ADC converters |

---

## Approach

### Specialization: Fully-Connected vs. Convolutional Networks

![FC vs CNN comparison](images/image31.png)

**Fully-Connected (FC):**
- All-to-all connections — impractical at scale
- Hard to train, poor for locally-structured data

**Convolutional (CNN):**
- Sparse connections between inputs and filters — far fewer MRRs needed
- Dominates modern architectures (vision, NLP, etc.)
- Good at **feature detection** in structured data

→ **CNNs are the target** for photonic specialization

---

## Approach

### Parallelism Opportunities in CNNs

![Kernel scanning animation](images/image68.gif)

![Multi-kernel parallel scanning](images/image48.gif)

- A **kernel** scans an input searching for a feature
- Multiple kernels (~hundreds) per CNN layer, each **independent** → parallelizable with WDM
- Kernel operations are small (3×3, 5×5) → fits naturally into photonic weight banks

---

## Approach

### Less Photonic Hardware with CNNs

![CNN hardware reduction](images/image43.png)

- Sparse connections → **fewer MRRs** needed
- Small kernel sizes → weight banks fit in realistic photonic chip areas
- Makes realization of full CNNs **feasible** with current photonic technology

---

## Approach

### Photonics Potential for CNNs — PCNNA

![PCNNA accelerator architecture](images/image24.png)

![AlexNet benchmark](images/image41.png)

- 10–100 WDM wavelengths map small kernel operations in a CNN layer
- Potential **3 orders of magnitude speedup** in execution time
- Input DAC and output ADC conversions are the bottleneck

*Mehrabian et al. "PCNNA: a photonic convolutional neural network accelerator." IEEE SOCC 2018.*

---

## Approach

### Transformations to Reduce Hardware Cost

Three algorithms for convolution — each with different hardware trade-offs:

| Algorithm | Best for | Trade-off |
|---|---|---|
| **GEMM (Toeplitz)** | Simple, general | Bloats matrix size → more overhead |
| **FFT** | Large kernels | Efficient for large kernel sizes |
| **Winograd** | Small kernels (3×3) | Trades multiplications for additions |

→ **Winograd** chosen: reduces multiplications at the cost of more additions, and additions are cheap in photonics

---

## Approach

### Winograd Filtering Algorithm

![Winograd algorithm structure](images/image34.png)

- Adopted to **perform convolution** with reduced computational complexity
- Reduces the number of **multiplications** (expensive in photonics) at the cost of increased additions
- Example: a 3×3 convolution with Winograd requires only **~2.25×** ops vs. ~9× for direct convolution

---

## Accomplishment

### A CNN Photonic Accelerator Architecture

![Winograd CNN photonic accelerator](images/image39.png)

- **Weights Path:** analog configurable memory (memristors) stores trained weights
- **Inputs Path:** optical signals carry input activations via WDM
- Winograd transform pre-processes weights and inputs before optical multiply

> Patent Pending: *Mehrabian, A., Sorger, V. J., El-Ghazawi, T., & Miscuglio, M. (2020). U.S. Patent Application No. 16/507,854.*

*Mehrabian et al. "A Winograd-based Integrated Photonics Accelerator for CNNs." IEEE J. Sel. Topics Quantum Electron. 2019.*

---

## Approach

### Novel Devices: Memristors as Analog Memory

![Memristor I-V characteristics](images/image38.png)

![Memristor structure](images/image33.png)

- **Memristors** (memory + resistors) — resistance is a function of charge history
- Serve as **analog, non-volatile** weight storage for photonic networks
- **6.5-bit** effective resolution demonstrated in metal-oxide bi-layer memristors
- Programmed by modulating pulse count, duration, or amplitude

*Stathopoulos et al. "Multibit memory operation of metal-oxide bi-layer memristors." Scientific Reports (2017).*

---

## Approach

### Lack of Design and Simulation Tools

![Simulation gap diagram](images/image45.png)

**The gap:** Photonic device simulators and deep learning frameworks (TensorFlow, PyTorch) do not speak the same language.

- Photonics tools model optical components (MRR transfer functions, noise, modulators)
- Deep learning tools model tensor operations

→ Need a **unified design methodology** bridging both worlds

---

## Accomplishment

### Design Methodology & Simulation Framework

A TensorFlow-integrated photonic neural network simulator:

- **Photonic Models** inserted as custom ops into TensorFlow's dataflow executor
- Models: MRR transfer functions, modulator nonlinearity, photodiode response, noise
- Enables **hardware-aware training** — train with photonic noise in the loop
- Benchmarked with VGG16 and AlexNet

*Mehrabian et al. "A Design Methodology for Post-Moore's Law Accelerators: The Case of a Photonic Neuromorphic Processor." IEEE ASAP 2020.*

---

## Results

### Design Space Exploration: MRR vs. MZI

![MRR device](images/image36.png)

![MZI device](images/image42.png)

- **MRR (Microring Resonator):** compact, wavelength-selective, suitable for WDM, but sensitive to temperature
- **MZI (Mach-Zehnder Interferometer):** robust, broadband, but larger footprint
- Benchmark the performance of photonic neural networks using both devices on state-of-the-art networks

---

## Results

### Nonlinear Activation Functions

![Nonlinear activation function examples](images/image51.png)

Commonly used functions: ReLU, sigmoid, tanh — each implemented using electro-absorption modulators

---

## Results

### Optimizing Nonlinear Activation Functions

![Activation function design space](images/image49.png)

![EAM activation implementation](images/image57.png)

- Dynamic range of voltage depends on **modulator length** and **laser power**
- Modulator length determines light absorption → shapes the activation curve
- Trade-off between **linearity**, **dynamic range**, and **energy consumption**

*George et al. "Neuromorphic photonics with electro-absorption modulators." Optics Express 27.4 (2019).*

---

## Results

### Performance of the Winograd CNN Accelerator

![Speed performance](images/image59.png)

![Power performance](images/image52.png)

![Speed/Power efficiency](images/image66.png)

- Benchmarked with **VGG16** (all convolutional layers)
- Metrics: Speed (GOP/s), Power (W), Speed/Power efficiency ((GOP/s)/W)
- Photonic implementation achieves significant efficiency advantages over electronic baselines

---

## Results

### Hardware-Aware Software Design: Effect of Noise

![Noise effect on accuracy (inference only)](images/image58.png)

- Gaussian noise (mean=0, sweeping standard deviation) injected only at **inference time**
- Photodiode and MRR noise degrade prediction accuracy as noise increases
- Baseline: trained on clean data, tested with noisy hardware

---

## Results

### Noise-Resilient Training

![Noise resilience with training](images/image54.png)

![Accuracy vs noise STD](images/image50.png)

- Adding a **small amount of noise during training** makes the network more resilient
- The network learns to be robust against hardware-level photonic noise
- Too much noise during training **deteriorates** inference performance → sweet spot exists

---

## Approach

### Envisioning an Optical Brain Simulator

![Brain simulator architecture](images/image65.png)

**Brain simulators** (NEURON, Genesis, BRIAN, NEST) model neuronal networks — but are communication-bound.

**Proposed approach:** Adaptive many-core brain simulator on a **hybrid reconfigurable optical NoC**

Key features:
- Simple photonic processing cores
- Custom optical interconnect fabric
- Departure from conventional parallel machine principles
- **Fault-tolerant** design

---

## Approach

### Simulation of the Hill-Tononi Model

The Hill-Tononi model models the brain's **wakefulness-to-sleep transition**:

- Two visual areas with associated thalamic and reticular thalamic nuclei
- **Wakefulness:** cortical neurons fire at irregular intervals
- **Slow-wave sleep:** all cortical neurons undergo slow oscillation (<1 Hz)
- Followed by burst firing even **higher** than wakefulness

A realistic benchmark for optical brain simulator design.

---

## Approach

### Hill-Tononi on Optical NoC

![HyPPI interconnect layout](images/image64.png)

![Neuron mapping to cores](images/image63.png)

- Each neuron processed on a **single core**
- Inter-layer neurons connect through **HyPPI** (Hybrid Plasmonics-Photonics Interconnect)
- Scales to large neuron counts with optical bandwidth

---

## Results

### Power Simulations: HyPPI vs. Electrical

![Power simulation results](images/image69.png)

- Comparison: HyPPI optical interconnect vs. conventional electrical (SpiNNaker: 18 cores, 1 Watt)
- Optical interconnect shows substantial **power savings** for brain simulation workloads
- Benefit grows as simulation scale increases

---

## Results

### Further Enhancements: Adaptive HyPPI NoC (D3NoC)

![D3NoC adaptive topology](images/image60.png)

![D3NoC power savings](images/image61.png)

- **D3NoC:** Dynamic Data-Driven hybrid photonic-plasmonic NoC
- Monitors and **predicts traffic**, augments topology with express optical bus
- Results:
  - Up to **67% savings in latency**
  - Up to **69% savings in power**

*Mehrabian et al. "D3NoC: a dynamic data-driven hybrid photonic plasmonic NoC." ACM Computing Frontiers, 2018.*

---

## Conclusion

- As transistors reach **atomic scale**, Moore's Law is **failing** — anticipated to flatten by 2025
- **Neuromorphic computing** is needed for problems not well addressed by digital computing
- We can exploit the **synergy between neuromorphic and photonic computing** for next-generation computers
- To realize full-scale photonic neural networks, we need both **mathematical transforms** (Winograd) and **novel physical devices** (memristors, EAMs)
- Photonic neural networks can **push the limits of electronics by orders of magnitude**
- Brain simulation is **communication-bound** — novel optical interconnects address the shortcomings of electrical interconnects

---

## Selected Publications & Patent

**Journal Papers:**
- Mehrabian et al. "A Winograd-based Integrated Photonics Accelerator for CNNs." *IEEE J. Sel. Topics Quantum Electron.* 26.1 (2019).
- George et al. "Neuromorphic photonics with electro-absorption modulators." *Optics Express* 27.4 (2019): 5181–5191.
- Miscuglio et al. "All-optical nonlinear activation function for photonic neural networks." *Optical Materials Express* 8.12 (2018): 3851–3863.

**Conference Papers:**
- Mehrabian et al. "PCNNA: a photonic convolutional neural network accelerator." IEEE SOCC 2018.
- Mehrabian et al. "A Design Methodology for Post-Moore's Law Accelerators." IEEE ASAP 2020.
- Mehrabian et al. "D3NoC: a dynamic data-driven hybrid photonic plasmonic NoC." ACM Computing Frontiers 2018.

**Patent:**
- Mehrabian, A., Sorger, V. J., El-Ghazawi, T., & Miscuglio, M. "Optical convolutional neural network accelerator." *U.S. Patent App. 16/507,854* (2020, Pending).
