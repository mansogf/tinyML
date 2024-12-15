# Breaking Down W4A4 Quantization: A Systematic Analysis of Post-Training Quantization Methods

This repository contains the implementation and analysis of various post-training quantization (PTQ) techniques for achieving W4A4 (4-bit weights and activations) quantization in Large Language Models (LLMs). Our work systematically evaluates existing methods, their limitations, and proposes improvements for extreme quantization scenarios.

## 🎯 Key Objectives

- Push the boundaries of W4A4 quantization using post-training techniques
- Identify specific challenges in W4A4 quantization
- Analyze why current methods struggle with extreme quantization
- Propose and evaluate novel solutions for improved quantization performance

## 📊 Main Results

Our best configuration achieves:
- 75% reduction in model size (631.71MB → 157.93MB)
- Only 7.9% perplexity degradation on WikiText-2
- 23.8% increase in inference latency

## 🔬 Implemented Methods

1. **AWQ with Activation Quantization**
   - Extension of original AWQ algorithm
   - Perplexity: +19% (22.06 → 26.26)
   - Memory reduction: 73.2%

2. **SmoothQuant Implementation**
   - Scales weights up and activations down
   - Perplexity: +20% (22.06 → 26.54)
   - Memory reduction: 73.3%

3. **Layer-wise Quantization**
   - Variable precision across layers
   - Perplexity: +23% (22.06 → 27.06)
   - Memory reduction: 73.0%

4. **Activation Clipping Analysis**
   - Various clipping strategies evaluated
   - Best results with careful threshold selection
   - Detailed analysis of different clipping methods

5. **Adaptive Rounding**
   - Combines outlier-aware weight quantization
   - Includes mixed-precision activation handling
   - Memory reduction: 91.6% (631.71MB → 53.29MB)
   - Perplexity: 24.31

6. **AWQ with Dynamic Group Sizing**
   - Best overall performance
   - Perplexity: 23.96
   - Memory usage: 157.93MB

## 🛠️ Setup and Installation

```bash
# Clone the repository
git clone https://github.com/mansogf/tinyML.git
cd tinyML

# Install dependencies
pip install -r requirements.txt
```

## 📈 Usage

```python
# Example usage of the quantization methods
from quantization import AWQ, SmoothQuant, LayerwiseQuant

# Load your model
model = load_model("path/to/model")

# Apply quantization
quantized_model = AWQ(
    model,
    group_size=128,
    percentile_threshold=0.998,
    importance_threshold=0.95,
    scale_factor=1.0
)
```

## 📊 Evaluation

We evaluate our methods on:
- Dataset: WikiText-2-raw-v1
- Model: OPT-350M
- Metrics: Perplexity and Memory Usage

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## 👥 Authors

- Albert Hung (azhung@mit.edu)
- Gabriel Manso (gmanso@mit.edu)
- Jeremy Carin (jcarin@mit.edu)
- Jophy Ye (jophyyh@mit.edu)
- Pakaphol Thadawanin (big536@mit.edu)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
