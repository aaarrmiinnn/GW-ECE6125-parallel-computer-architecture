# Lecture 01 Update Summary

**Branch:** `feature/lecture-01-updates`
**Date:** January 2026
**Reviewer:** Claude (HPC Expert Review)

---

## Overview

This document summarizes updates made to Lecture 01 (Flynn's Taxonomy & Parallelism Intuition) to bring content up to date with current HPC landscape (as of late 2025/early 2026).

---

## Changes Made

### 1. Supercomputer Examples Updated

**Previous:** Fugaku, Summit
**Updated to:** El Capitan, Frontier, Aurora (exascale systems)

**Rationale:** Summit was decommissioned. The current TOP500 (November 2025) shows:
1. El Capitan (1.809 Exaflop/s) - LLNL
2. Frontier (1.353 Exaflop/s) - ORNL
3. Aurora (1.012 Exaflop/s) - ANL
4. JUPITER (1.000 Exaflop/s) - First European exascale
5. Fugaku dropped to #7 (442 Petaflop/s)

**Source:** [TOP500 November 2025](https://top500.org/lists/top500/2025/11/)

---

### 2. SIMD Extensions Updated

**Previous:** Intel AVX, ARM NEON
**Updated to:** Intel AVX-512/AVX10, ARM SVE/SVE2, ARM NEON

**Rationale:**
- AVX-512 is now mainstream (introduced 2016, widely adopted by 2024)
- AVX10 announced 2023, shipping in Granite Rapids (Q3 2024)
- ARM SVE/SVE2 is standard on modern ARM server chips (AWS Graviton, Apple Silicon)
- SVE2 supports scalable vector lengths (128-2048 bit)

**Sources:**
- [Intel AVX-512 Overview](https://www.intel.com/content/www/us/en/architecture-and-technology/avx-512-overview.html)
- [ARM SVE/SVE2 Blog](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/sve-sve2-enablement-in-simd-library)

---

### 3. GPU Architecture Examples Updated

**Previous:** NVIDIA CUDA cores, AMD Radeon
**Updated to:** NVIDIA Hopper/Blackwell, AMD Instinct MI300

**Rationale:**
- NVIDIA Blackwell (B200) announced March 2024, shipping Q1 2025
- Hopper (H100) was the dominant AI chip of 2023-2024
- AMD MI300X is a direct H100 competitor with superior memory capacity
- "AMD Radeon" is consumer branding; MI300 is the HPC/AI product line

**Sources:**
- [NVIDIA Blackwell Architecture](https://blog.us.fixstars.com/nvidia-blackwell-architecture-cto-guide/)
- [AI Chip Market Analysis](https://www.aichips.com/nvidia-blackwell-amd-mi-and-new-ai-chip-architectures-who-leads-in-2025/)

---

### 4. TPU Information Updated

**Previous:** "Google's TPUs use systolic arrays for matrix multiplications"
**Updated to:** "Google's TPU v6 (Trillium) uses 256×256 systolic arrays for matrix multiplications, delivering ~918 peak BF16 TFLOPS"

**Added:** NVIDIA Tensor Cores and AMD Matrix Cores as related technologies

**Rationale:**
- TPU v6 (Trillium) announced May 2024, GA December 2024
- TPU v7 (Ironwood) announced April 2025
- Specific performance numbers add educational value
- Tensor Cores/Matrix Cores use similar systolic concepts

**Sources:**
- [Google Cloud TPU Trillium](https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus)
- [TPU Architecture Guide](https://intuitionlabs.ai/articles/google-tpu-architecture-gemini-3)

---

### 5. Branch Prediction Accuracy Updated

**Previous:** "~90-95% accuracy in modern CPUs"
**Updated to:** "Modern TAGE predictors achieve 97-98% accuracy; misprediction rates below 2-3%"

**Rationale:**
- TAGE (Tagged Geometric) predictors are standard in modern CPUs (Intel, AMD, ARM)
- Modern predictors achieve significantly better than 95% accuracy
- Research shows misprediction rates below 2% are now the target

**Sources:**
- [Branch Prediction Wikipedia](https://en.wikipedia.org/wiki/Branch_predictor)
- [Branch Prediction 2025 Analysis](https://ai2.work/technology/ai-tech-branch-prediction-cpus-2025/)

---

### 6. Interconnect Technologies Updated

**Previous:** NUMAlink, InfiniBand, Ethernet / InfiniBand, Cray Aries
**Updated to:** HPE Slingshot, InfiniBand NDR/XDR, NVLink, CXL

**Rationale:**
- HPE Slingshot powers 71% of top 10 supercomputers' aggregate compute
- Cray Aries is legacy (replaced by Slingshot)
- NUMAlink is legacy SGI technology (SGI acquired by HPE 2016)
- InfiniBand NDR (400G) and XDR (800G) are current generations
- NVLink 5 delivers 1.8 TB/s bidirectional bandwidth
- CXL is emerging for memory pooling

**Sources:**
- [HPE Slingshot Dominance](https://www.sdxcentral.com/analysis/ethernets-exascale-victory-hpes-slingshot-continues-to-conquer-the-exascale-interconnect-battle/)
- [InfiniBand NDR/XDR](https://ascentoptics.com/blog/infiniband-ndr-xdr-for-ai-and-hpc-data-centers/)
- [NVLink Guide](https://intuitionlabs.ai/articles/nvidia-nvlink-gpu-interconnect)

---

### 7. MIMD System Examples Updated

**Previous:** IBM BlueGene, SGI Altix, Cray XC50, Fugaku
**Updated to:** HPE Cray EX (Frontier, Aurora), Eviden BullSequana (JUPITER), Fugaku

**Rationale:**
- IBM Blue Gene discontinued (last system: Mira, decommissioned 2019)
- SGI acquired by HPE in 2016; Altix product line discontinued
- Cray XC50 replaced by HPE Cray EX series
- HPE Cray EX powers 3 of top 4 supercomputers
- Eviden (Atos) BullSequana powers JUPITER (#4)

---

### 8. Multicomputer Examples Updated

**Previous:** IBM Blue Gene, Fugaku
**Updated to:** El Capitan, Frontier, Aurora, JUPITER, Fugaku

**Rationale:** Provides current, relevant examples of distributed memory systems.

---

## Items NOT Changed

The following content was reviewed and determined to be still accurate:

1. **Flynn's Taxonomy fundamentals** - Classification scheme from 1966 remains canonical
2. **Von Neumann Architecture** - Foundational concept, unchanged
3. **Pipeline hazard types** - RAW, structural, control hazards remain accurate
4. **Cache mapping policies** - Direct-mapped, set-associative, fully associative unchanged
5. **ILP concepts** - Superscalar, pipelining, OoO execution remain accurate
6. **Memory hierarchy latencies** - L1 (1-5 cycles), DRAM (100+ cycles) still reasonable approximations
7. **Historical examples** (Cray-1, ENIAC, Intel 8086) - Appropriate for teaching history

---

## Verification Checklist

- [x] All supercomputer examples reflect TOP500 November 2025
- [x] SIMD extensions include current Intel and ARM standards
- [x] GPU examples include 2024-2025 architectures
- [x] TPU information reflects v6 Trillium specifications
- [x] Branch prediction accuracy reflects modern TAGE predictors
- [x] Interconnects include current HPC standards
- [x] Deprecated systems (Blue Gene, SGI Altix) removed from current examples
- [x] All sources cited with URLs

---

## Recommendations for Future Updates

1. **Monitor TOP500** - List updates every June and November
2. **Watch for Aurora performance** - Intel system may improve with software maturation
3. **Track CXL adoption** - Emerging technology for memory disaggregation
4. **TPU v7 (Ironwood)** - Update when production benchmarks available
5. **NVIDIA Rubin** - Expected 2026, will be next major GPU architecture
