# Examples

![Optimize LLMs](/images/optiPfair.png)

This directory contains practical examples demonstrating how to use OptiPFair for structured pruning and bias analysis of large language models.

## Overview

The examples showcase different aspects of OptiPFair's capabilities, from basic pruning operations to model compatibility checking. Each example is designed to be educational and immediately usable.

## 📁 Available Examples

### 🐍 Python Scripts

#### `prune_llama.py`
A complete Python script demonstrating structured pruning of LLaMA models using OptiPFair.

**Features:**
- Loads and prunes LLaMA models with configurable parameters
- Compares performance before and after pruning
- Includes benchmarking and text generation testing
- Saves the pruned model for future use

**Usage:**
```bash
python prune_llama.py
```

**What it demonstrates:**
- Loading models with proper device/dtype handling
- Pruning with MAW (Maximum Absolute Weight) method
- Performance benchmarking and comparison
- Text generation quality assessment
- Model saving and cleanup

### 📓 Jupyter Notebooks

#### `basic_pruning_mlp.ipynb`
An interactive notebook perfect for learning the fundamentals of structured pruning with OptiPFair.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/basic_pruning_mlp.ipynb)

**Features:**
- **Two pruning approaches:** Percentage-based and expansion-rate-based pruning
- **Interactive learning:** Step-by-step explanations with code
- **Model comparison:** Before/after generation examples
- **Statistics tracking:** Detailed pruning results and metrics

**What you'll learn:**
- How to configure pruning parameters
- The difference between pruning percentage and expansion rate
- How to evaluate pruned model performance
- Best practices for memory management in Colab

**Recommended for:**
- First-time users of OptiPFair
- Understanding structured pruning concepts
- Learning different pruning strategies

#### `pruning_compatibility_check.ipynb`
A quick compatibility checker that determines if your model can be pruned with OptiPFair.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/pruning_compatibility_check.ipynb.ipynb)

**Features:**
- **30-second analysis:** Quick compatibility assessment
- **Architecture detection:** Identifies model structure and GLU layers
- **Expansion ratio calculation:** Analyzes MLP dimensions
- **Detailed reporting:** Clear compatibility status and recommendations

**What it checks:**
- GLU architecture presence (gate_proj, up_proj, down_proj)
- MLP expansion ratios
- Supported model types (LLaMA, Mistral, Gemma, Qwen, Phi)
- Layer structure compatibility

**Perfect for:**
- Verifying new models before pruning
- Understanding model architecture
- Troubleshooting compatibility issues

## 🚀 Getting Started

### Prerequisites

```bash
# Install OptiPFair
pip install optipfair

# For notebooks (optional visualization dependencies)
pip install optipfair[viz]
```

### Recommended Order

1. **Start with compatibility checking:** Run `pruning_compatibility_check.ipynb` to verify your model is supported
2. **Learn the basics:** Work through `basic_pruning_mlp.ipynb` to understand pruning concepts
3. **Try the Python script:** Use `prune_llama.py` for production-ready pruning workflows

## 🎯 Key Concepts Demonstrated

### Pruning Methods
- **MAW (Maximum Absolute Weight):** Default method, typically best results
- **VOW (Variance of Weights):** Alternative approach for specific use cases
- **PON (Product of Norms):** Experimental method for research

### Pruning Targets
- **Percentage-based:** Direct specification (e.g., "remove 20% of neurons")
- **Expansion-rate-based:** Target specific ratios (e.g., "reduce to 200% expansion")

### Model Compatibility
- **GLU Architecture:** Required for structured pruning
- **Supported Models:** LLaMA, Mistral, Gemma, Qwen, Phi families
- **Architecture Analysis:** Automatic detection and validation

## 📊 Expected Results

### Typical Pruning Results
- **Parameter Reduction:** 10-50% fewer parameters
- **Memory Savings:** Proportional to parameter reduction
- **Speed Improvements:** 1.2-2x faster inference
- **Quality Preservation:** Minimal impact on generation quality

### Example Metrics
```
Original parameters: 1,235,814,400
Pruned parameters: 1,074,792,448
Reduction: 161,021,952 parameters (13.03%)
Final expansion rate: 320.02%
```

## 🔧 Configuration Options

### Common Parameters
- `pruning_percentage`: 10-50% (start with 20%)
- `expansion_rate`: 140-300% (alternative to percentage)
- `neuron_selection_method`: "MAW", "VOW", or "PON"
- `show_progress`: Enable progress bars
- `return_stats`: Get detailed statistics

### Device Settings
```python
# Auto-detect best device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if "cuda" in device else torch.float32
```

## 🎓 Learning Path

### Beginners
1. Run compatibility check on your model
2. Follow the basic pruning notebook step-by-step
3. Experiment with different pruning percentages
4. Try the Python script for automation

### Advanced Users
1. Customize pruning parameters for your use case
2. Integrate pruning into your model training pipeline
3. Benchmark different neuron selection methods
4. Contribute new examples or improvements

## 📚 Additional Resources

- **Documentation:** [https://peremartra.github.io/optipfair/](https://peremartra.github.io/optipfair/)
- **GitHub Repository:** [https://github.com/peremartra/optipfair](https://github.com/peremartra/optipfair)
- **LLM Reference Manual:** `optipfair_llm_reference_manual.txt` in the root directory

## 🤝 Contributing

Found an issue or want to add a new example? Contributions are welcome!

1. Fork the repository
2. Create your feature branch
3. Add your example with clear documentation
4. Submit a pull request

## 📄 License

These examples are provided under the Apache 2.0 license, same as the main OptiPFair project.