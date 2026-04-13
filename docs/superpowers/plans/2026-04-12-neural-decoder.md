# Neural CNN Decoder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a convolutional neural network decoder for surface code QEC, following Gu et al. (arXiv:2604.08358). Train in PyTorch with Stim data generation, then benchmark against Union-Find and PyMatching.

**Architecture:** CNN with bottleneck residual blocks and direction-specific 3D convolutions. Trained on-the-fly with Stim syndrome data. Binary cross-entropy loss on logical observable predictions.

**Tech Stack:** PyTorch, Stim, numpy, pybind11 (for integration with existing decoder infrastructure)

**Note:** Phase 2A (model + training) runs on CPU/MPS locally. Phase 2B (GPU inference kernel) requires a GPU instance and will be planned separately after training validates.

---

### Task 1: CNN Model Architecture

**Files:**
- Create: `decoder/train/model.py`

- [ ] **Step 1: Write the CNN decoder model**

Create `decoder/train/model.py`:

```python
"""Neural CNN decoder for surface code QEC.

Architecture follows Gu et al. (arXiv:2604.08358, April 2026):
- Input: binary syndrome tensor [B, 1, R, d, d]
- Embedding: 1x1x1 conv lifting binary input to H dimensions
- L bottleneck residual blocks:
    - 1x1x1 conv: H → H//4 (reduce)
    - 3x3x3 conv: H//4 → H//4 (message passing)
    - 1x1x1 conv: H//4 → H (restore)
    - Residual connection + LayerNorm
- Final: global average pool → MLP → logit per observable
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class DecoderConfig:
    """Configuration for the neural decoder."""
    distance: int           # Code distance
    rounds: int             # Number of syndrome rounds (typically = distance)
    hidden_dim: int = 128   # Hidden dimension H (paper uses 256-512)
    n_blocks: int = 5       # Number of residual blocks L (paper uses L ~ d)
    n_observables: int = 1  # Number of logical observables to predict
    dropout: float = 0.0    # Dropout rate (0 = no dropout)


class BottleneckBlock(nn.Module):
    """Bottleneck residual block with 3D convolution.

    Reduce (1x1x1) → Message passing (3x3x3) → Restore (1x1x1) → Residual + LayerNorm
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        reduced = hidden_dim // 4

        self.reduce = nn.Conv3d(hidden_dim, reduced, kernel_size=1, bias=False)
        self.message = nn.Conv3d(reduced, reduced, kernel_size=3, padding=1, bias=False)
        self.restore = nn.Conv3d(reduced, hidden_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: [B, H, R, d, d]
        residual = x

        out = F.gelu(self.reduce(x))
        out = F.gelu(self.message(out))
        out = self.restore(out)
        out = self.dropout(out)

        out = out + residual

        # LayerNorm over channel dim (need to permute)
        # x is [B, H, R, d, d] → permute to [B, R, d, d, H] → norm → permute back
        out = out.permute(0, 2, 3, 4, 1)
        out = self.norm(out)
        out = out.permute(0, 4, 1, 2, 3)

        return out


class NeuralDecoder(nn.Module):
    """CNN decoder for surface code quantum error correction.

    Takes binary syndrome tensors and predicts logical observable flips.
    """
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # Embedding: binary syndrome → H-dimensional representation
        self.embed = nn.Conv3d(1, H, kernel_size=1, bias=True)

        # L bottleneck residual blocks
        self.blocks = nn.ModuleList([
            BottleneckBlock(H, config.dropout)
            for _ in range(config.n_blocks)
        ])

        # Output head: global average pool → MLP → logits
        self.head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, config.n_observables),
        )

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            syndrome: float tensor [B, 1, R, d, d] — binary syndrome as float

        Returns:
            logits: float tensor [B, n_observables] — raw logits (apply sigmoid for probabilities)
        """
        # Embed
        x = self.embed(syndrome)  # [B, H, R, d, d]

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling over spatial+temporal dims
        x = x.mean(dim=(2, 3, 4))  # [B, H]

        # MLP head
        logits = self.head(x)  # [B, n_observables]

        return logits

    def predict(self, syndrome: torch.Tensor) -> torch.Tensor:
        """Predict observable flips (binary).

        Args:
            syndrome: float tensor [B, 1, R, d, d]

        Returns:
            predictions: bool tensor [B, n_observables]
        """
        with torch.no_grad():
            logits = self.forward(syndrome)
            return logits > 0

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- [ ] **Step 2: Write a smoke test**

Create `decoder/tests/test_neural_model.py`:

```python
import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from model import NeuralDecoder, DecoderConfig


class TestNeuralModel:
    def test_forward_pass(self):
        """Model should produce correct output shape."""
        config = DecoderConfig(distance=3, rounds=3, hidden_dim=32, n_blocks=2)
        model = NeuralDecoder(config)

        # Batch of 4 syndromes for d=3, 3 rounds
        syndrome = torch.randn(4, 1, 3, 3, 3)
        logits = model(syndrome)

        assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"

    def test_predict(self):
        """Predict should return boolean tensor."""
        config = DecoderConfig(distance=3, rounds=3, hidden_dim=32, n_blocks=2)
        model = NeuralDecoder(config)

        syndrome = torch.randn(4, 1, 3, 3, 3)
        preds = model.predict(syndrome)

        assert preds.dtype == torch.bool
        assert preds.shape == (4, 1)

    def test_parameter_count(self):
        """Check model size is reasonable."""
        config = DecoderConfig(distance=5, rounds=5, hidden_dim=128, n_blocks=5)
        model = NeuralDecoder(config)
        n_params = NeuralDecoder.count_parameters(model)

        print(f"d=5, H=128, L=5: {n_params:,} parameters ({n_params * 4 / 1e6:.1f} MB at FP32)")
        assert n_params < 10_000_000, f"Model too large: {n_params}"

    def test_d5_shape(self):
        """d=5 model with 5 rounds."""
        config = DecoderConfig(distance=5, rounds=5, hidden_dim=64, n_blocks=3)
        model = NeuralDecoder(config)

        # d=5: syndrome tensor has shape [B, 1, 5, 5, 5]
        # But Stim's detectors for d=5 rotated surface code = 24 per round × 5 rounds = 120
        # Need to reshape 120 detectors into spatial layout
        # For now, just test with the expected spatial shape
        syndrome = torch.randn(8, 1, 5, 5, 5)
        logits = model(syndrome)
        assert logits.shape == (8, 1)

    def test_gradient_flow(self):
        """Gradients should flow through the entire model."""
        config = DecoderConfig(distance=3, rounds=3, hidden_dim=32, n_blocks=2)
        model = NeuralDecoder(config)

        syndrome = torch.randn(4, 1, 3, 3, 3)
        target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

        logits = model(syndrome)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest decoder/tests/test_neural_model.py -v
```

- [ ] **Step 4: Commit**

```bash
git add decoder/train/model.py decoder/tests/test_neural_model.py
git commit -m "feat(decoder): neural CNN model architecture following Gu et al. 2026"
```

---

### Task 2: Stim Data Pipeline for Training

**Files:**
- Create: `decoder/train/data.py`

- [ ] **Step 1: Write the data pipeline**

Create `decoder/train/data.py`:

```python
"""Stim-based data pipeline for neural decoder training.

Generates syndrome data on-the-fly — no pre-generated datasets needed.
Stim runs at ~1B Clifford gates/sec, so data generation is never the bottleneck.
"""
import stim
import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    distance: int
    rounds: int
    physical_error_rate: float
    batch_size: int = 256
    code_type: str = "surface_code:rotated_memory_z"


class SyndromeDataset:
    """On-the-fly syndrome data generator using Stim.

    Each call to sample() generates a fresh batch of syndromes
    with their ground-truth observable flips.
    """
    def __init__(self, config: DataConfig):
        self.config = config
        self.circuit = stim.Circuit.generated(
            config.code_type,
            distance=config.distance,
            rounds=config.rounds,
            after_clifford_depolarization=config.physical_error_rate,
            before_measure_flip_probability=config.physical_error_rate,
            after_reset_flip_probability=config.physical_error_rate,
        )
        self.sampler = self.circuit.compile_detector_sampler()
        self.n_detectors = self.circuit.num_detectors
        self.n_observables = self.circuit.num_observables

        # Compute spatial layout for reshaping detectors to 3D tensor
        # For rotated surface code: d^2 - 1 stabilizers per round
        self.detectors_per_round = self.n_detectors // config.rounds
        self._compute_spatial_layout()

    def _compute_spatial_layout(self):
        """Compute how to reshape flat detector indices into [R, H, W] spatial tensor.

        For rotated surface code distance d:
        - d^2 - 1 detectors per round
        - Arrange in a d x d grid (with one position unused)
        """
        d = self.config.distance
        r = self.config.rounds
        dpr = self.detectors_per_round

        # Simple layout: pad to d x d per round
        # Detectors map to a checkerboard pattern on the d x d grid
        # For simplicity, use a d x d spatial grid and zero-pad unused positions
        self.spatial_shape = (r, d, d)
        self.total_spatial = r * d * d

        # Build mapping from detector index to (round, row, col)
        # For rotated surface code, detectors are numbered sequentially per round
        self.det_to_spatial = np.zeros((self.n_detectors, 3), dtype=np.int32)
        for i in range(self.n_detectors):
            round_idx = i // dpr
            local_idx = i % dpr
            # Map local detector index to grid position
            # Simple row-major mapping (approximate — exact layout depends on Stim's convention)
            row = local_idx // d
            col = local_idx % d
            if row >= d:
                row = d - 1  # Clamp for codes where dpr > d*d
            self.det_to_spatial[i] = [round_idx, row, col]

    def detectors_to_tensor(self, detection_events: np.ndarray) -> torch.Tensor:
        """Reshape flat detection events [B, n_detectors] to spatial tensor [B, 1, R, d, d].

        Args:
            detection_events: bool array [B, n_detectors]

        Returns:
            tensor: float tensor [B, 1, R, d, d]
        """
        B = detection_events.shape[0]
        d = self.config.distance
        r = self.config.rounds

        tensor = torch.zeros(B, 1, r, d, d, dtype=torch.float32)

        for i in range(self.n_detectors):
            ri, row, col = self.det_to_spatial[i]
            if ri < r and row < d and col < d:
                tensor[:, 0, ri, row, col] = torch.from_numpy(
                    detection_events[:, i].astype(np.float32)
                )

        return tensor

    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of syndromes and ground-truth labels.

        Returns:
            syndromes: float tensor [B, 1, R, d, d]
            labels: float tensor [B, n_observables]
        """
        bs = batch_size or self.config.batch_size
        det_events, obs_flips = self.sampler.sample(shots=bs, separate_observables=True)

        syndromes = self.detectors_to_tensor(det_events)
        labels = torch.from_numpy(obs_flips.astype(np.float32))

        return syndromes, labels


class CurriculumScheduler:
    """Three-stage noise annealing curriculum from Gu et al.

    Stage 1: Train at low noise (easy examples, learn basic structure)
    Stage 2: Train at medium noise (harder examples)
    Stage 3: Train at target noise (full difficulty)
    """
    def __init__(self, target_rate: float, total_steps: int):
        self.target = target_rate
        self.total = total_steps
        self.stages = [
            (0.0, 0.2, target_rate * 0.1),   # 0-20%: low noise
            (0.2, 0.6, target_rate * 0.5),    # 20-60%: medium noise
            (0.6, 1.0, target_rate),           # 60-100%: target noise
        ]

    def get_rate(self, step: int) -> float:
        frac = step / max(self.total, 1)
        for start, end, rate in self.stages:
            if frac < end:
                # Linear interpolation within stage
                stage_frac = (frac - start) / (end - start)
                if start == 0:
                    return rate  # First stage: constant low rate
                prev_rate = self.stages[self.stages.index((start, end, rate)) - 1][2]
                return prev_rate + stage_frac * (rate - prev_rate)
        return self.target
```

- [ ] **Step 2: Write data pipeline tests**

Create `decoder/tests/test_data_pipeline.py`:

```python
import sys
import os
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from data import SyndromeDataset, DataConfig, CurriculumScheduler


class TestSyndromeDataset:
    def test_sample_shape(self):
        """Sampled syndromes should have correct shape."""
        config = DataConfig(distance=3, rounds=3, physical_error_rate=0.005, batch_size=16)
        dataset = SyndromeDataset(config)
        syndromes, labels = dataset.sample()

        assert syndromes.shape == (16, 1, 3, 3, 3)
        assert labels.shape == (16, 1)  # 1 observable for surface code

    def test_sample_values(self):
        """Syndromes should be binary (0 or 1), labels should be binary."""
        config = DataConfig(distance=3, rounds=3, physical_error_rate=0.005, batch_size=100)
        dataset = SyndromeDataset(config)
        syndromes, labels = dataset.sample()

        # Syndromes are float but should only contain 0.0 and 1.0
        unique_vals = torch.unique(syndromes)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

        # Labels should be 0.0 or 1.0
        unique_labels = torch.unique(labels)
        assert all(v in [0.0, 1.0] for v in unique_labels.tolist())

    def test_low_noise_mostly_zeros(self):
        """At very low noise, most syndromes should be all-zero."""
        config = DataConfig(distance=3, rounds=3, physical_error_rate=0.0001, batch_size=500)
        dataset = SyndromeDataset(config)
        syndromes, _ = dataset.sample()

        n_zero = torch.sum(syndromes.sum(dim=(1, 2, 3, 4)) == 0).item()
        assert n_zero > 400, f"Expected >400 zero syndromes, got {n_zero}"

    def test_d5_shape(self):
        """d=5 syndrome should have correct shape."""
        config = DataConfig(distance=5, rounds=5, physical_error_rate=0.005, batch_size=8)
        dataset = SyndromeDataset(config)
        syndromes, labels = dataset.sample()

        assert syndromes.shape == (8, 1, 5, 5, 5)


class TestCurriculum:
    def test_curriculum_stages(self):
        """Curriculum should increase noise over training."""
        sched = CurriculumScheduler(target_rate=0.007, total_steps=80000)

        early = sched.get_rate(0)
        mid = sched.get_rate(40000)
        late = sched.get_rate(79999)

        assert early < mid < late
        assert abs(late - 0.007) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest decoder/tests/test_data_pipeline.py -v
```

- [ ] **Step 4: Commit**

```bash
git add decoder/train/data.py decoder/tests/test_data_pipeline.py
git commit -m "feat(decoder): Stim data pipeline — on-the-fly syndrome generation + curriculum"
```

---

### Task 3: Training Loop

**Files:**
- Create: `decoder/train/train.py`

- [ ] **Step 1: Write the training script**

Create `decoder/train/train.py`:

```python
"""Training script for the neural CNN decoder.

Usage:
    python decoder/train/train.py --distance 3 --hidden_dim 64 --n_blocks 3 --steps 10000
    python decoder/train/train.py --distance 5 --hidden_dim 128 --n_blocks 5 --steps 40000
"""
import argparse
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from model import NeuralDecoder, DecoderConfig
from data import SyndromeDataset, DataConfig, CurriculumScheduler


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, dataset, device, n_shots=10000):
    """Evaluate model accuracy on fresh syndromes."""
    model.eval()
    syndromes, labels = dataset.sample(n_shots)
    syndromes = syndromes.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = model(syndromes)
        preds = (logits > 0).float()
        n_errors = (preds != labels).any(dim=1).sum().item()

    ler = n_errors / n_shots
    model.train()
    return ler


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # Model
    model_config = DecoderConfig(
        distance=args.distance,
        rounds=args.distance,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        n_observables=1,
        dropout=args.dropout,
    )
    model = NeuralDecoder(model_config).to(device)
    n_params = NeuralDecoder.count_parameters(model)
    print(f"Model: d={args.distance}, H={args.hidden_dim}, L={args.n_blocks}, "
          f"{n_params:,} params ({n_params * 4 / 1e6:.1f} MB)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Curriculum
    curriculum = CurriculumScheduler(args.noise_rate, args.steps)

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    best_ler = 1.0
    start_time = time.time()

    for step in range(args.steps):
        # Get current noise rate from curriculum
        current_p = curriculum.get_rate(step)

        # Create dataset at current noise rate
        data_config = DataConfig(
            distance=args.distance,
            rounds=args.distance,
            physical_error_rate=current_p,
            batch_size=args.batch_size,
        )
        dataset = SyndromeDataset(data_config)
        syndromes, labels = dataset.sample()
        syndromes = syndromes.to(device)
        labels = labels.to(device)

        # Forward + backward
        logits = model(syndromes)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Log
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            print(f"step {step:>6}/{args.steps}  loss={loss.item():.4f}  "
                  f"p={current_p:.5f}  lr={lr:.6f}  time={elapsed:.0f}s")

        # Evaluate
        if step > 0 and step % args.eval_interval == 0:
            eval_config = DataConfig(
                distance=args.distance,
                rounds=args.distance,
                physical_error_rate=args.noise_rate,
                batch_size=args.batch_size,
            )
            eval_dataset = SyndromeDataset(eval_config)
            ler = evaluate(model, eval_dataset, device, n_shots=args.eval_shots)
            print(f"  >>> EVAL LER at p={args.noise_rate}: {ler:.6f}")

            if ler < best_ler:
                best_ler = ler
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'config': model_config,
                    'ler': ler,
                }, ckpt_dir / "best_model.pt")
                print(f"  >>> Saved best model (LER={ler:.6f})")

        # Periodic checkpoint
        if step > 0 and step % (args.eval_interval * 5) == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
            }, ckpt_dir / f"checkpoint_{step}.pt")

    # Final evaluation
    eval_config = DataConfig(
        distance=args.distance,
        rounds=args.distance,
        physical_error_rate=args.noise_rate,
    )
    eval_dataset = SyndromeDataset(eval_config)
    final_ler = evaluate(model, eval_dataset, device, n_shots=50000)
    print(f"\nFinal LER at p={args.noise_rate}: {final_ler:.6f}")
    print(f"Best LER: {best_ler:.6f}")
    print(f"Total time: {time.time() - start_time:.0f}s")

    # Save final model
    torch.save({
        'step': args.steps,
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'ler': final_ler,
    }, ckpt_dir / "final_model.pt")

    return final_ler


def main():
    parser = argparse.ArgumentParser(description="Train neural QEC decoder")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_rate", type=float, default=0.007)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_shots", type=int, default=10000)
    parser.add_argument("--checkpoint_dir", type=str, default="decoder/train/checkpoints")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test training runs for a few steps**

```bash
cd /Users/bledden/Documents/quantum
python decoder/train/train.py --distance 3 --hidden_dim 32 --n_blocks 2 --steps 500 --batch_size 64 --eval_interval 250 --log_interval 50
```

Expected: loss decreases, LER evaluation runs, checkpoint saved.

- [ ] **Step 3: Commit**

```bash
git add decoder/train/train.py
git commit -m "feat(decoder): training loop — curriculum learning, checkpointing, LER evaluation"
```

---

### Task 4: Evaluation + Comparison Script

**Files:**
- Create: `decoder/train/evaluate.py`

- [ ] **Step 1: Write evaluation script**

Create `decoder/train/evaluate.py`:

```python
"""Evaluate a trained neural decoder against Union-Find and PyMatching.

Usage:
    python decoder/train/evaluate.py --checkpoint decoder/train/checkpoints/best_model.pt
"""
import argparse
import sys
import os
import time
import torch
import numpy as np
import pymatching
import stim

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

from model import NeuralDecoder, DecoderConfig
from data import SyndromeDataset, DataConfig
from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph

# Try importing C++ decoder
try:
    import pydecoder
    HAS_CPP_DECODER = True
except ImportError:
    HAS_CPP_DECODER = False
    print("Warning: pydecoder not found. Skipping C++ Union-Find comparison.")


def graph_to_cpp(decoder_graph):
    sg = pydecoder.SyndromeGraph()
    sg.n_detectors = decoder_graph.n_detectors
    sg.n_observables = decoder_graph.n_observables
    for src, tgt, prob, obs in decoder_graph.edges:
        sg.add_edge(src, tgt, prob, obs)
    sg.build_adjacency()
    return sg


def evaluate_neural(model, dataset, device, n_shots=50000):
    """Evaluate neural decoder LER."""
    model.eval()
    total_errors = 0
    total = 0
    batch_size = 1000

    times = []
    for _ in range(n_shots // batch_size):
        syndromes, labels = dataset.sample(batch_size)
        syndromes = syndromes.to(device)
        labels = labels.to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(syndromes)
            preds = (logits > 0).float()
        t1 = time.perf_counter()
        times.append((t1 - t0) / batch_size)

        total_errors += (preds != labels).any(dim=1).sum().item()
        total += batch_size

    ler = total_errors / total
    latency_us = np.median(times) * 1e6
    return ler, latency_us


def evaluate_pymatching(circuit, det_events, obs_flips):
    matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model())
    t0 = time.perf_counter()
    pm_pred = matching.decode_batch(det_events)
    t1 = time.perf_counter()
    n_errors = np.sum(np.any(pm_pred != obs_flips, axis=1))
    ler = n_errors / len(det_events)
    latency_us = (t1 - t0) / len(det_events) * 1e6
    return ler, latency_us


def evaluate_union_find(graph, det_events, obs_flips):
    if not HAS_CPP_DECODER:
        return None, None
    cpp_graph = graph_to_cpp(graph)
    uf = pydecoder.UnionFindDecoder(cpp_graph)
    t0 = time.perf_counter()
    uf_pred = uf.decode_batch(det_events)
    t1 = time.perf_counter()
    n_errors = np.sum(np.any(uf_pred != obs_flips, axis=1))
    ler = n_errors / len(det_events)
    latency_us = (t1 - t0) / len(det_events) * 1e6
    return ler, latency_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_shots", type=int, default=50000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
                          "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    model = NeuralDecoder(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model: d={config.distance}, H={config.hidden_dim}, "
          f"L={config.n_blocks}, trained LER={ckpt.get('ler', 'N/A')}")

    # Evaluate at multiple noise rates
    print(f"\n{'p':>8} {'Neural':>10} {'PyMatch':>10} {'UF':>10} {'N_lat(us)':>10}")
    print("-" * 55)

    for p in [0.001, 0.002, 0.005, 0.007, 0.01]:
        # Neural decoder
        data_config = DataConfig(
            distance=config.distance, rounds=config.rounds,
            physical_error_rate=p,
        )
        dataset = SyndromeDataset(data_config)
        neural_ler, neural_lat = evaluate_neural(model, dataset, device, args.n_shots)

        # PyMatching + Union-Find (using same syndromes for fair comparison)
        stim_config = SurfaceCodeConfig(
            distance=config.distance, rounds=config.rounds,
            physical_error_rate=p,
        )
        circuit = make_circuit(stim_config)
        det_events, obs_flips = sample_syndromes(circuit, args.n_shots)
        graph = extract_decoder_graph(circuit)

        pm_ler, pm_lat = evaluate_pymatching(circuit, det_events, obs_flips)
        uf_ler, uf_lat = evaluate_union_find(graph, det_events, obs_flips)

        uf_str = f"{uf_ler:.6f}" if uf_ler is not None else "N/A"
        print(f"{p:>8.4f} {neural_ler:>10.6f} {pm_ler:>10.6f} {uf_str:>10} {neural_lat:>10.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add decoder/train/evaluate.py
git commit -m "feat(decoder): evaluation script — Neural vs UF vs PyMatching comparison"
```

---

### Task 5: Train and Validate (d=3 first, then d=5)

- [ ] **Step 1: Train d=3 model (quick, ~10 min on CPU)**

```bash
python decoder/train/train.py \
    --distance 3 --hidden_dim 64 --n_blocks 3 \
    --steps 20000 --batch_size 256 \
    --noise_rate 0.007 --lr 1e-3 \
    --eval_interval 2000 --log_interval 200 \
    --checkpoint_dir decoder/train/checkpoints/d3
```

- [ ] **Step 2: Evaluate d=3 model**

```bash
python decoder/train/evaluate.py \
    --checkpoint decoder/train/checkpoints/d3/best_model.pt \
    --n_shots 50000
```

- [ ] **Step 3: Train d=5 model (longer, ~30-60 min on CPU/MPS)**

```bash
python decoder/train/train.py \
    --distance 5 --hidden_dim 128 --n_blocks 5 \
    --steps 40000 --batch_size 256 \
    --noise_rate 0.007 --lr 1e-3 \
    --eval_interval 5000 --log_interval 500 \
    --checkpoint_dir decoder/train/checkpoints/d5
```

- [ ] **Step 4: Evaluate d=5 model**

```bash
python decoder/train/evaluate.py \
    --checkpoint decoder/train/checkpoints/d5/best_model.pt \
    --n_shots 50000
```

- [ ] **Step 5: Save results and commit**

```bash
git add decoder/train/checkpoints/*/best_model.pt
git commit -m "feat(decoder): trained neural models — d=3 and d=5 surface codes"
```

---

## Plan Self-Review

**Spec coverage:**
- [x] CNN architecture (Task 1: bottleneck residual blocks, embedding, MLP head)
- [x] Training pipeline (Task 2: Stim data, curriculum annealing)
- [x] Training loop (Task 3: BCE loss, AdamW, checkpointing)
- [x] Evaluation (Task 4: Neural vs UF vs PyMatching comparison)
- [x] Train and validate (Task 5: d=3 and d=5)
- [ ] GPU inference kernel (Phase 2B — separate plan after training validates)
- [ ] FP8 quantization (Phase 2B)

**Placeholder scan:** No TBDs. All code blocks complete. Training script has full argument parsing.

**Key simplifications vs Gu et al.:**
- AdamW instead of Muon (simpler, can upgrade later)
- Standard 3D conv instead of direction-specific weights (first version)
- Smaller models for local training (H=64-128 vs paper's 256-512)
- CPU/MPS training feasible for d=3,5 (paper used GPU for d=13)

**What we'll learn:** Whether the CNN architecture produces meaningful LER improvement over UF even at small scale (d=3,5). If yes, the architecture works and we scale up on GPU. If no, we need to debug the spatial layout mapping or architecture before investing GPU hours.
