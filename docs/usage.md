# Usage Guide

## Python API

OptiPFair provides a simple Python API for pruning models.

### Basic Usage

```python
from transformers import AutoModelForCausalLM
from optipfair import prune_model

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Prune the model with default settings (10% pruning, MAW method)
pruned_model = prune_model(model=model)

# Save the pruned model
pruned_model.save_pretrained("./pruned-model")
```

### Advanced Usage

```python
# Prune with custom settings
pruned_model, stats = prune_model(
    model=model,
    pruning_type="MLP_GLU",              # Type of pruning to apply
    neuron_selection_method="MAW",       # Method to calculate neuron importance
    pruning_percentage=20,               # Percentage of neurons to prune
    # expansion_rate=140,                # Alternatively, specify target expansion rate
    show_progress=True,                  # Show progress during pruning
    return_stats=True                    # Return pruning statistics
)

# Print pruning statistics
print(f"Original parameters: {stats['original_parameters']:,}")
print(f"Pruned parameters: {stats['pruned_parameters']:,}")
print(f"Reduction: {stats['reduction']:,} parameters ({stats['percentage_reduction']:.2f}%)")
```

## Command-Line Interface

OptiPFair provides a command-line interface for pruning models:

### Basic Usage

```bash
# Prune a model with default settings (10% pruning, MAW method)
optipfair prune --model-path meta-llama/Llama-3.2-1B --output-path ./pruned-model
```

### Advanced Usage

```bash
# Prune with custom settings
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type MLP_GLU \
  --method MAW \
  --pruning-percentage 20 \
  --output-path ./pruned-model \
  --device cuda \
  --dtype float16
```

### Analyzing a Model

```bash
# Analyze a model's architecture and parameter distribution
optipfair analyze --model-path meta-llama/Llama-3.2-1B
```

## Neuron Selection Methods

OptiPFair supports three methods for calculating neuron importance:

### MAW (Maximum Absolute Weight)

The MAW method identifies neurons based on the maximum absolute weight values in their connections. This is typically the most effective method for GLU architectures.

```python
pruned_model = prune_model(
    model=model,
    neuron_selection_method="MAW",
    pruning_percentage=20
)
```

### VOW (Variance of Weights)

The VOW method identifies neurons based on the variance of their weight values. This can be useful for certain specific architectures.

```python
pruned_model = prune_model(
    model=model,
    neuron_selection_method="VOW",
    pruning_percentage=20
)
```

### PON (Product of Norms)

The PON method uses the product of L1 norms to identify important neurons. This is an alternative approach that may be useful in certain contexts.

```python
pruned_model = prune_model(
    model=model,
    neuron_selection_method="PON",
    pruning_percentage=20
)
```

## Pruning Percentage vs Expansion Rate

OptiPFair supports two ways to specify the pruning target:

### Pruning Percentage

Directly specify what percentage of neurons to remove:

```python
pruned_model = prune_model(
    model=model,
    pruning_percentage=20  # Remove 20% of neurons
)
```

### Expansion Rate

Specify the target expansion rate (ratio of intermediate size to hidden size) as a percentage:

```python
pruned_model = prune_model(
    model=model,
    expansion_rate=140  # Target 140% expansion rate
)
```

This approach is often more intuitive when comparing across different model scales.

## Depth Pruning

OptiPFair also supports depth pruning, which removes entire transformer layers from models. This is more aggressive than neuron-level pruning but can lead to significant efficiency gains.

### Python API

#### Basic Depth Pruning

```python
from optipfair import prune_model

# Remove 2 layers from the end of the model
pruned_model = prune_model(
    model=model,
    pruning_type="DEPTH",
    num_layers_to_remove=2
)
```

#### Depth Pruning by Percentage

```python
# Remove 25% of layers
pruned_model = prune_model(
    model=model,
    pruning_type="DEPTH",
    depth_pruning_percentage=25.0
)
```

#### Depth Pruning with Specific Layer Indices

```python
# Remove specific layers (e.g., layers 2, 5, and 8)
pruned_model = prune_model(
    model=model,
    pruning_type="DEPTH",
    layer_indices=[2, 5, 8]
)
```

### Command-Line Interface

#### Basic Depth Pruning

```bash
# Remove 2 layers from the end of the model
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type DEPTH \
  --num-layers-to-remove 2 \
  --output-path ./depth-pruned-model
```

#### Depth Pruning by Percentage

```bash
# Remove 25% of layers
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type DEPTH \
  --pruning-percentage 25 \
  --output-path ./depth-pruned-model
```

#### Depth Pruning with Specific Layers

```bash
# Remove specific layers
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type DEPTH \
  --layer-indices "2,5,8" \
  --output-path ./depth-pruned-model
```

## Comparing Pruning Types

### MLP GLU vs Depth Pruning

| Feature | MLP GLU Pruning | Depth Pruning |
|---------|-----------------|---------------|
| **Granularity** | Neuron-level | Layer-level |
| **Aggressiveness** | Moderate | High |
| **Parameter Reduction** | Gradual | Significant |
| **Model Structure** | Preserved | Layers removed |
| **Fine-tuning Need** | Minimal | Recommended |
| **Efficiency Gains** | Moderate | High |

### When to Use Each Method

**Use MLP GLU Pruning when:**
- You want gradual parameter reduction
- You need to preserve model structure
- You have limited time for fine-tuning
- You need precise control over expansion rates

**Use Depth Pruning when:**
- You need significant efficiency gains
- You can afford to fine-tune the model
- You have very large models with many layers
- You need maximum inference speed improvement

## Evaluating Pruned Models

After pruning, you can use OptiPFair's evaluation tools to assess the performance of the pruned model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair.evaluation.benchmarks import time_inference, compare_models_inference

# Load original and pruned models
original_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
pruned_model = AutoModelForCausalLM.from_pretrained("./pruned-model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Compare inference speed
comparison = compare_models_inference(
    original_model,
    pruned_model,
    tokenizer,
    prompts=["Paris is the capital of", "The speed of light is approximately"],
    max_new_tokens=50
)

print(f"Speedup: {comparison['speedup']:.2f}x")
print(f"Tokens per second improvement: {comparison['tps_improvement_percent']:.2f}%")
```