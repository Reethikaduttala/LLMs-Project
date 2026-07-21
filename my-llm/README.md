# my-llm

> A tiny 5M parameter model for prototyping and small projects on CPU or basic GPU

Created with [create-llm](https://github.com/theaniketgiri/create-llm) ✨

---

## 📋 Project Overview

| Property | Value |
|----------|-------|
| **Template** | TINY |
| **Model** | GPT (~5.0M parameters) |
| **Tokenizer** | BPE |
| **Hardware** | None (CPU-friendly) |
| **Training Time** | 10-30 minutes |
| **Min Data** | 1,000+ examples |
| **CPU Compatible** | ✅ Yes |

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Add Your Training Data

Place your text files in `data/raw/`:

```bash
# Example: Download Shakespeare
curl https://www.gutenberg.org/files/100/100-0.txt > data/raw/shakespeare.txt

# Or copy your own files
cp /path/to/your/data.txt data/raw/
```

**Data Requirements:**
- Format: Plain text (.txt files)
- Encoding: UTF-8
- Minimum: 1,000+ examples
- Recommended: Clean, well-formatted text

### Step 3: Train Tokenizer

```bash
python tokenizer/train.py --data data/raw/
```

This creates a vocabulary from your data. You'll see:
- Vocabulary size
- Sample encoding
- Tokenizer statistics

### Step 4: Prepare Dataset

```bash
python data/prepare.py
```

This tokenizes and prepares your data. You'll see:
- Number of training examples
- Number of validation examples
- Total tokens processed

### Step 5: Start Training

```bash
python training/train.py
```

**Training will show:**
- Real-time loss
- Learning rate schedule
- Tokens per second
- Estimated time remaining

**After training completes**, you'll see a menu with options:
1. **Continue training** - Add more training steps
2. **Launch chat interface** - Test your model in a web UI
3. **Exit** - Finish and exit

### Step 6: Test Your Model (Post-Training Chat)

After training, select option 2 to launch the interactive chat interface:

```bash
# Or launch manually anytime:
python chat_interface.py
```

This opens a web interface where you can:
- Chat with your trained model in real-time
- Adjust generation parameters (temperature, top-k, top-p)
- Test different prompts and see responses
- Clear conversation and start fresh

The interface automatically loads your best checkpoint and runs at http://localhost:7860

### Step 7: Monitor Training

Watch for these indicators:

**Good Training:**
- Loss steadily decreasing
- Perplexity: 5-20 (depends on data)
- No warnings

**Potential Issues:**
- Perplexity < 1.5: Possible overfitting
- Loss not decreasing: Check learning rate
- "Model too large" warning: Add more data or use smaller template

### Step 8: Evaluate Your Model (Optional)

```bash
python evaluation/evaluate.py --checkpoint checkpoints/checkpoint-best.pt
```

Output includes:
- Perplexity score
- Loss metrics
- Performance statistics

### Step 9: Generate Text (Optional)

```bash
python evaluation/generate.py \
  --checkpoint checkpoints/checkpoint-best.pt \
  --prompt "Once upon a time" \
  --temperature 0.8
```

**Temperature Guide:**
- 0.1-0.3: Focused, deterministic
- 0.7-0.9: Balanced, creative
- 1.0-1.5: Very creative, diverse

### Step 10: Terminal Chat (Optional)

```bash
python chat.py --checkpoint checkpoints/checkpoint-best.pt
```

**Chat Commands:**
- `/temp <value>`: Adjust temperature
- `/clear`: Clear conversation
- `/quit`: Exit chat

---

## 📁 Project Structure

```
my-llm/
│
├── 📂 data/
│   ├── raw/                    # ← Put your .txt files here
│   ├── processed/              # Tokenized data (auto-generated)
│   ├── dataset.py              # PyTorch dataset classes
│   └── prepare.py              # Data preprocessing script
│
├── 📂 models/
│   ├── architectures/
│   │   ├── gpt.py             # GPT architecture implementation
│   │   ├── nano.py            # 1M parameter model
│   │   ├── tiny.py            # 6M parameter model
│   │   ├── small.py           # 100M parameter model
│   │   └── base.py            # 1B parameter model
│   ├── __init__.py
│   └── config.py              # Configuration loader
│
├── 📂 tokenizer/
│   ├── train.py               # Tokenizer training script
│   └── tokenizer.json         # Trained tokenizer (auto-generated)
│
├── 📂 training/
│   ├── train.py               # Main training script ⭐
│   ├── trainer.py             # Trainer class
│   ├── callbacks/             # Training callbacks
│   │   ├── base.py
│   │   ├── checkpoint.py      # Checkpoint management
│   │   ├── logging.py         # TensorBoard logging
│   │   └── checkpoint_manager.py
│
├── 📂 evaluation/
│   ├── evaluate.py            # Model evaluation
│   └── generate.py            # Text generation
│
├── 📂 plugins/                # Optional integrations
│   ├── wandb_plugin.py        # Weights & Biases
│   └── huggingface_plugin.py  # HuggingFace Hub
│
├── 📂 checkpoints/            # Saved models (auto-generated)
├── 📂 logs/                   # Training logs (auto-generated)
│
├── 📄 llm.config.js           # Main configuration ⚙️
├── 📄 requirements.txt        # Python dependencies
├── 📄 chat.py                 # Interactive chat interface
├── 📄 deploy.py               # Deployment script
├── 📄 compare.py              # Model comparison tool
└── 📄 README.md               # This file
```

---

## ⚙️ Configuration

All settings are in `llm.config.js`:

```javascript
module.exports = {
  model: {
    type: 'gpt',
    size: 'tiny',
    vocab_size: 10000,  // Auto-detected from tokenizer
    max_length: 512,
    layers: 4,
    heads: 4,
    dim: 256,
    dropout: 0.2,
  },
  training: {
    batch_size: 16,
    learning_rate: 0.0006,
    max_steps: 10000,
    // ... more options
  },
};
```

**Common Adjustments:**
- `batch_size`: Reduce if out of memory
- `learning_rate`: Adjust if loss unstable
- `dropout`: Increase if overfitting (0.2-0.4)
- `max_steps`: Increase for better quality

### 📖 Understanding Vocabulary Size

**What is vocab_size?**
- The number of unique tokens your model can understand
- Must match your trained tokenizer's vocabulary
- Automatically detected and synchronized by the system

**How it works:**
1. You train a tokenizer on your data → creates vocabulary
2. System reads actual vocab size from `tokenizer/tokenizer.json`
3. Model is initialized with the correct vocab size
4. Training validates that everything matches

**Important:**
- ✅ **DO:** Let the system auto-detect vocab size (default behavior)
- ✅ **DO:** Train tokenizer before training model
- ❌ **DON'T:** Manually override vocab_size unless you know what you're doing
- ❌ **DON'T:** Change vocab_size after training starts

**Typical vocab sizes:**
- Small datasets (shakespeare.txt): 3,000-10,000 tokens
- Medium datasets: 10,000-32,000 tokens
- Large datasets: 32,000-50,000 tokens

**If you see "vocab size mismatch":**
- This is automatically corrected
- No action needed
- The model will use the tokenizer's actual vocab size

---

## 💡 Training Tips

- Good balance between size and capability
- Requires 1,000+ examples to avoid overfitting
- Can handle small to medium datasets (1-50MB)
- Use for prototyping before scaling to SMALL
- Watch for overfitting: if perplexity < 1.5, add more data
- Recommended: 5,000+ training examples for best results

---

## 🔧 Advanced Usage

### Resume Training

If training was interrupted:

```bash
python training/train.py --resume checkpoints/checkpoint-1000.pt
```

### Model Comparison

Compare multiple trained models:

```bash
python compare.py checkpoints/model-v1/ checkpoints/model-v2/
```

Shows:
- Side-by-side metrics
- Sample generations
- Performance comparison

### Custom Generation

```bash
# Adjust creativity
python evaluation/generate.py \
  --checkpoint checkpoints/checkpoint-best.pt \
  --prompt "Your prompt" \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95 \
  --max-length 200
```

### Deploy to Hugging Face

```bash
python deploy.py \
  --checkpoint checkpoints/checkpoint-best.pt \
  --to huggingface \
  --repo-id username/my-model
```

---

## 🔌 Plugins

### No Plugins Enabled

To enable plugins, edit `llm.config.js`:

```javascript
plugins: [
  'wandb',        // Experiment tracking
  'huggingface',  // Model sharing
  'synthex',      // Synthetic data
]
```

---

## 🐛 Troubleshooting

### "Vocab size mismatch detected"
✅ **This is normal!** The tool auto-detects and uses the correct vocab size from your tokenizer.

**What it means:**
- Your `llm.config.js` has a different vocab_size than your trained tokenizer
- The system automatically uses the tokenizer's actual vocabulary size
- This prevents training issues and poor generation quality

**No action needed** - the mismatch is automatically corrected!

### Repetitive text generation ("which which which...")
❌ **Vocabulary mismatch issue**

**Symptoms:**
- Model generates the same word repeatedly
- Output looks like: "which which which which..."
- Happens especially with small datasets (e.g., shakespeare.txt)

**Root Cause:**
- Tokenizer vocabulary size doesn't match model embedding layer
- Model can't properly learn token representations

**Solution:**
1. **Check vocab sizes match:**
   ```bash
   # The training script validates this automatically
   python training/train.py
   ```

2. **If you see a mismatch error:**
   - The model was auto-corrected during loading
   - Training should work correctly
   - If issues persist, retrain from scratch

3. **For existing checkpoints with wrong vocab:**
   - Cannot be fixed - must retrain
   - Delete checkpoints/ directory
   - Retrain with correct vocab size

**Prevention:**
- Always train tokenizer before training model
- Let the system auto-detect vocab size (don't override manually)
- Verify "✓ Vocabulary sizes match" message during training

### "Position embedding index error" or sequences too long
✅ **Automatically handled!** Sequences exceeding max_length are truncated with warnings.
- Check data preprocessing if you see frequent truncation warnings
- Consider increasing `max_length` in config if needed (requires retraining)

### "Model may be too large for dataset"
⚠️ **Warning:** Risk of overfitting
- **Solution 1:** Add more training data (recommended)
- **Solution 2:** Use smaller template (nano/tiny)
- **Solution 3:** Increase dropout in llm.config.js

### "Perplexity < 1.5"
❌ **Overfitting detected**
- Model memorized the data
- Add much more data or use smaller model

### "CUDA out of memory"
- Reduce `batch_size` in llm.config.js
- Enable `mixed_precision: true`
- Increase `gradient_accumulation_steps`

### "Training loss not decreasing"
- Check learning rate (try 1e-4 to 1e-3)
- Verify data loaded correctly
- Try longer warmup period

### "Tokenizer not found"
- Run `python tokenizer/train.py --data data/raw/` first
- Make sure data/raw/ contains .txt files

---

## 📊 Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| **RAM** | 4GB minimum |
| **GPU** | None (CPU-friendly) |
| **Storage** | 10GB+ free space |
| **Training Time** | 10-30 minutes |

---

## 📚 Resources

- [create-llm Documentation](https://github.com/theaniketgiri/create-llm)
- [Training Best Practices](https://github.com/theaniketgiri/create-llm/docs/training.md)
- [API Reference](https://github.com/theaniketgiri/create-llm/docs/api.md)
- [Troubleshooting Guide](https://github.com/theaniketgiri/create-llm/docs/troubleshooting.md)

---

## 📝 License

MIT

---

## 🙏 Acknowledgments

This project was created with [create-llm](https://github.com/theaniketgiri/create-llm) - The fastest way to start training your own Language Model.

**Built with:**
- PyTorch
- Transformers
- Tokenizers
- TensorBoard

---

**Happy Training! 🚀**

If you encounter any issues, please check the troubleshooting section above or visit the [create-llm repository](https://github.com/theaniketgiri/create-llm/issues).
