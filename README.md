# HeRMES: Heterogeneity Aware Resource Mapping for Efficient Scheduling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-Compatible-green)](https://github.com/microsoft/DeepSpeed)

**HeRMES** (Heterogeneity Aware Resource Mapping for Efficient Scheduling of Deep Learning Workloads) is an intelligent scheduling framework designed to optimize deep learning training workloads across heterogeneous GPU clusters. By leveraging Mixed-Integer Programming (MIP) formulations, HeRMES maximizes training throughput while preventing Out-of-Memory (OOM) errors and ensuring efficient resource utilization across all gpu in a HPC cluster.

## 🚀 Key Features

- **🎯 Heterogeneous Hardware Support**: Native support for mixed GPU specifications (H100, A100, V100, T4, etc.)
- **🧠 Intelligent Workload Distribution**: Optimal allocation of microbatches and model parameter shards
- **🛡️ OOM Prevention**: Built-in memory constraint handling to prevent training failures  
- **⚡ Multiple Solver Support**: MILP, MIQP, and convex relaxation (ECOS) solvers
- **🔌 Framework Agnostic**: Compatible with PyTorch FSDP, DeepSpeed, and other distributed training frameworks
- **🌐 Bandwidth-Aware Scheduling**: Considers full interconnect topology for communication optimization
- **🌡️ Thermal-Aware Optimization**: Dynamic adaptation to GPU thermal states and throttling
- **📊 Real-time Monitoring**: GPU utilization, temperature, and memory monitoring

## 📊 Performance Highlights

- **62x throughput improvement** over naive FSDP in heterogeneous environments
- **30-80% gains** over manually tuned baselines
- Effective utilization of all GPU tiers without bottlenecks
- Automatic gradient accumulation step optimization
- Dynamic replanning based on thermal conditions

## 🛠️ Installation

### Prerequisites

```bash
# Install Python dependencies
pip install torch torchvision deepspeed
pip install numpy scipy cvxpy
pip install datasets transformers wandb
pip install psutil pynvml

# Optional: Install Gurobi for optimal performance (requires license)
pip install gurobipy
```

### Quick Install

```bash
git clone https://github.com/yourusername/HeRMES.git
cd HeRMES
pip install -e .
```

## 🎯 Quick Start

### Basic Usage

```python
from hermes import DynamicHO3Scheduler
import torch

# Auto-detect GPU configuration
scheduler = DynamicHO3Scheduler(
    monitoring_interval=5.0,  # Monitor every 5 seconds
    replan_interval=300.0,    # Replan every 5 minutes
    debug=True
)

# Start a training job
batch_size = 1024
num_layers = 32
scheduler.start_job(batch_size=batch_size, num_layers=num_layers)

# Get current scheduling plan
plan = scheduler.get_current_plan()
print(f"Microbatch distribution: {plan['m']}")
print(f"Parameter sharding: {plan['r']}")
print(f"Estimated throughput: {plan['tp']:.1f} samples/s")
```

### Integration with DeepSpeed

```python
import deepspeed
from hermes import DynamicHO3Scheduler

# Initialize scheduler
scheduler = DynamicHO3Scheduler(debug=True)
scheduler.start_job(batch_size=1024, num_layers=32)

# Initialize DeepSpeed engine
model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters()
)

# Integrate with HeRMES
scheduler.integrate_deepspeed(model_engine)

# Training loop with dynamic adaptation
for step, batch in enumerate(dataloader):
    outputs = model_engine(**batch)
    loss = outputs.loss
    
    model_engine.backward(loss)
    model_engine.step()
    
    # Check for replanning every 50 steps
    if step % 50 == 0:
        scheduler.check_replan()
```

### Advanced Configuration

```python
# Manual GPU specification Examples
gpu_types = {
    "H100": {"count": 4, "flops": 500, "mem": 80},
    "A100": {"count": 4, "flops": 312, "mem": 40}, 
    "V100": {"count": 4, "flops": 125, "mem": 16},
    "T4": {"count": 4, "flops": 65, "mem": 16}
}

scheduler = DynamicHO3Scheduler(
    gpu_types=gpu_types,
    monitoring_interval=10.0,
    replan_interval=60.0,
    thermal_threshold=80,
    debug=True
)

# Compare with ZeRO-3 baseline
comparison = scheduler.scheduler.compare_with_zero3(
    batch=1024, 
    L=32,
    zero3_params={'offload_param': True}
)

print(f"HeRMES vs ZeRO-3 speedup: {comparison['speedup']:.2f}x")
print(f"Efficiency gain: {comparison['efficiency_gain']:.2f}x")
```

## 📈 Experimental Results

Our comprehensive evaluation demonstrates significant performance improvements:

| Method | Throughput (samples/s) | Step Time (s) | Memory Efficiency | GPU Utilization |
|--------|------------------------|---------------|-------------------|-----------------|
| **HeRMES-MIQP** | **6,094.7** | **0.179** | **Optimal** | **Balanced** |
| HeRMES-MILP | 3,249.1 | 0.611 | High | Good |
| ECOS (theoretical) | 4,430.0 | 0.116 | High | Perfect |
| FSDP Baseline | 98.2 | 5.368 | Poor | Imbalanced |

### Sample Output from HeRMES Scheduler

```
=== HeRMES-MIQP Schedule (batch=1088, L=32) ===
GPU      | m_i | r_i   | mem(GB) | comp(s) | grad(s) | tot(s) | util(%)
---------|-----|-------|---------|---------|---------|--------|--------
H100_0   | 3   | 0.142 | 20.57   | 0.064   | 0.114   | 0.179  | 100.0
H100_1   | 3   | 0.142 | 20.57   | 0.064   | 0.114   | 0.179  | 100.0  
A100_0   | 3   | 0.095 | 13.91   | 0.102   | 0.076   | 0.179  | 100.0
V100_0   | 2   | 0.000 | 0.44    | 0.169   | 0.000   | 0.170  | 95.2
T4_0     | 1   | 0.000 | 0.22    | 0.163   | 0.000   | 0.163  | 91.5

Predicted throughput: 6,094.7 samples/s
Thermal states: [1.0, 1.0, 0.95, 0.87, 0.82, ...]
```

## 🧪 Running Experiments

### Large Language Model Training

```bash
# Train GPT-NeoX 20B with HeRMES optimization
python examples/train_ho3_large.py \
    --model_name EleutherAI/gpt-neox-20b \
    --dataset_name c4 \
    --max_steps 1000 \
    --per_device_batch_size 1 \
    --gradient_accumulation 16 \
    --monitoring_interval 5.0 \
    --replan_interval 300.0 \
    --log_wandb
```

### Benchmark Against Baselines

```bash
# Run comprehensive benchmarks
python experiments/benchmark_schedulers.py \
    --cluster_config configs/heterogeneous_16gpu.json \
    --model_sizes 10B 20B 70B \
    --batch_sizes 512 1024 2048 \
    --compare_baselines fsdp zero3 uniform
```

### Synthetic Cluster Evaluation

```bash
# Reproduce paper results
python experiments/synthetic_evaluation.py \
    --gpu_config synthetic_16gpu \
    --workload_config llm_workloads \
    --solvers miqp milp ecos \
    --output_dir results/
```

## 🔬 Methodology

HeRMES formulates heterogeneous scheduling as an optimization problem:

**Objective**: Minimize maximum per-GPU step time with fairness regularization
```
min T - λ Σ(rᵢ - r̄)²
```

**Subject to constraints**:
- **Memory limits**: `rᵢ × P_total + mᵢ × m_act ≤ Mᵢ`
- **Workload coverage**: `Σmᵢ = B_global/l`  
- **Model completeness**: `Σrᵢ = 1`
- **Thermal awareness**: `F_thermal = F × thermal_state`

Where:
- `mᵢ`: microbatches assigned to GPU i
- `rᵢ`: fraction of model parameters on GPU i  
- `T`: bottleneck step time
- `λ`: fairness regularization weight

### Solver Architecture

```python
# MIQP formulation (optimal)
objective = T + idle_penalty + fairness_penalty + thermal_penalty

# MILP formulation (faster) 
objective = T + linearized_penalties

# CVXPY fallback
objective = relaxed_continuous_formulation
```


## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/HeRMES.git
cd HeRMES
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black hermes/ examples/ tests/
flake8 hermes/
```

## 📄 Citation

If you use HeRMES in your research, please cite:

```bibtex
@article{singh2025hermes,
  title={HeRMES: Heterogeneity Aware Resource Mapping for Efficient Scheduling of Deep Learning Workloads},
  author={Singh, Tushar},
  journal={University of Maryland, College Park},
  year={2025}
}
```

## 🔧 Solver Requirements

HeRMES supports multiple optimization solvers:

- **Gurobi**: Commercial solver with academic licenses (best performance for MIQP)
- **CVXPY**: Open-source convex optimization (good performance, widely available)
- **SciPy**: Fallback linear programming solver (basic functionality)

### Installation Guide

```bash
# Academic Gurobi license (recommended)
pip install gurobipy
# Follow Gurobi academic license setup

# Open-source alternative  
pip install cvxpy[CBC,GLPK]

# Basic fallback (included with SciPy)
pip install scipy
```

## ⚠️ Current Limitations

- Static cluster topology assumption during planning intervals
- Simplified communication modeling (congestion, compression)
- Focus on data/model parallelism (pipeline parallelism in development)
- Synthetic validation (real-world benchmarking ongoing)

## 🗺️ Roadmap

- [ ] **Real-world Validation**: Extensive testing on cloud platforms (AWS, GCP, Azure)
- [ ] **Pipeline Parallelism**: Integration with advanced pipeline strategies
- [ ] **Dynamic Topology**: Support for elastic and spot instance clusters  
- [ ] **Advanced Monitoring**: Network congestion and bandwidth monitoring
- [ ] **Framework Integration**: Native support for more ML frameworks
- [ ] **Web Dashboard**: Real-time scheduling visualization interface
- [ ] **Auto-tuning**: ML-based hyperparameter optimization for solver configs


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **University of Maryland, College Park** - Computer Science Department
- **Open Source Community** - For optimization libraries and distributed training frameworks

---

**Built by the HeRMES Team at University of Maryland**

*Making heterogeneous GPU training efficient, one schedule at a time.*
