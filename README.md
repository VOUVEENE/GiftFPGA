# GiFt-FPGA: Graph Signal Processing-Based FPGA Placement Acceleration

GiFt-FPGA is an extension of **[DREAMPlaceFPGA](https://github.com/rachelselinar/DREAMPlaceFPGA)** that introduces a **Graph Signal Processing (GSP)-based initialization algorithm** to accelerate FPGA analytical placement.  
It integrates seamlessly with the existing DREAMPlaceFPGA framework and provides a high-quality initialization to reduce convergence time in large-scale heterogeneous FPGA designs.

---

## 🧭 Overview

Analytical FPGA placers such as DREAMPlaceFPGA typically start from random initialization and require hundreds of iterations to converge.  
A significant portion of runtime (≈20–25%) is spent in the initial expansion phase, where modules gradually spread out from the chip center.  

GiFt-FPGA replaces this random start with a **graph-filtered initialization**, allowing the placer to begin optimization from a density-balanced state.

---

## ⚙️ Key Features

- **Hybrid Graph Construction:** Combines Clique and Star connectivity models with a net-skipping mechanism to avoid densification from high-fanout nets.  
- **Weighted-Pin Initialization:** Uses weighted averages of pin coordinates to estimate smooth initial positions reflecting interconnect density.  
- **Fixed-Module Bounding-Box Constraint:** Applies FPGA-specific spatial constraints derived from fixed blocks to prevent divergence.  
- **GPU-Accelerated Graph Filtering:** Implements efficient sparse Laplacian filtering using pre-construction, pre-allocation, and batch pipelining for high throughput.

---

## 📂 Project Structure

```
DREAMPlaceFPGA/
│
├── benchmarks/                   # Benchmark circuits
├── dreamplacefpga/               # Core FPGA placer modules
│   ├── gift_init_placer.py       # GiFt-FPGA initialization module
│   ├── gift_adj_matrix.cpp       # Hybrid graph construction (C++/CUDA core)
│   ├── gift_adj_cpp.so           # Compiled shared object from gift_adj_matrix.cpp
│   ├── Placer.py                 # Main entry of DREAMPlaceFPGA (runs placement flow)
│   └── ...
│
├── run_gift.py                   # Entry script for GiFt initialization
├── run_placement.sh              # Baseline placement script
├── requirements.txt              # Python dependencies
└── CMakeLists.txt / Dockerfile   # Build configuration
```

---

## 🧰 Environment Setup

GiFt-FPGA shares the **same dependencies, environment, and build process** as the original [DREAMPlaceFPGA](https://github.com/rachelselinar/DREAMPlaceFPGA) project.  

Please follow the installation and compilation steps provided in the upstream repository.  
Once DREAMPlaceFPGA is successfully built, this extension can be used directly without additional dependencies.

---
## 🚀 Running the Algorithm

### GiFt-FPGA (GSP-based) Initialization

GiFt-FPGA uses JSON files where `"use_gift_init_place": 1`.

Run all test cases:
```bash
python run_gift.py test/*.json
```

Run a single case directly (the JSON must have `use_gift_init_place=1`):
```bash
python dreamplacefpga/Placer.py test/FPGA06.json
```

---

### Baseline (Random Initialization)

Baseline runs use the provided `*_no_gift.json` files where `"use_gift_init_place": 0`.

Run all baseline test cases:
```bash
python run_placement.sh test/*_no_gift.json
```

Run a single baseline case:
```bash
python dreamplacefpga/Placer.py test/FPGA06_no_gift.json
```


## ⚙️ GiFt-FPGA Configuration Notes

GiFt-FPGA introduces several additional configuration parameters in the JSON files under the `test/` directory.  
The key field `"use_gift_init_place": 1` enables the GSP-based initialization.  
The parameters prefixed with `"gift_"` control how the hybrid graph is constructed, how filtering is performed, and how placement boundaries are enforced.

Typical parameters include:

- `gift_scale` — overall scaling factor controlling how widely initial coordinates are distributed across the chip.  
- `gift_alpha0`, `gift_alpha1`, `gift_alpha2` — three weights defining the strength of multi-band graph filters for smoothness and local variation.  
- `gift_use_star_model` — enables the Star graph model for high-fanout nets to prevent graph densification.  
- `gift_max_net_size` and `net_skip_threshold` — set limits for large networks to skip or simplify dense connections.  
- `gift_enable_boundary_constraints` — keeps all initialized modules inside the chip boundary using a bounding-box constraint.  
- `gift_enable_resource_constraints` — optionally accounts for heterogeneous FPGA resources (e.g., DSP/RAM regions).  
- `gift_bbox_margin` — defines a small safety margin from the chip edge.  
- `gift_center_method` — specifies how each node’s center is computed, typically `"weighted_pin"`.  

These parameters can be tuned to balance initialization quality and runtime.  
For most experiments, the default settings in `test/FPGAxx.json` work well.

---

## 🧩 Implementation Details

| Component | Description |
|------------|-------------|
| `gift_init_placer.py` | Implements the overall GiFt initialization flow: graph construction, graph filtering, boundary constraint application, and exporting coordinates to DREAMPlaceFPGA. |
| `gift_adj_matrix.cpp` | Core C++/CUDA module responsible for hybrid graph generation. It merges clique and star connectivity models, applies fanout skipping, and outputs a sparse adjacency matrix optimized for GPU filtering. |
| `gift_adj_cpp.so` | The compiled shared object generated from `gift_adj_matrix.cpp`, loaded dynamically by Python via PyTorch’s C++ extension API. |
| `Placer.py` | The main entry of DREAMPlaceFPGA; it coordinates global placement, legalization, and detailed placement. GiFt-FPGA provides initialization data that are directly injected into this flow to improve convergence speed. |

---

## 📜 License

This project is released under the **MIT License**.  
See the `LICENSE` file for details.

---