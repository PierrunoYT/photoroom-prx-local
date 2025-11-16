# PRX-1024 Text-to-Image Web UI

A beautiful Gradio web interface for Photoroom's **PRX-1024-t2i-beta** text-to-image model.

![PRX Model](https://img.shields.io/badge/Model-PRX--1024--t2i--beta-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Model License](https://img.shields.io/badge/Model%20License-Apache%202.0-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## 🌟 Features

- **Easy-to-use Interface**: Clean and intuitive Gradio UI
- **Advanced Controls**: Adjust inference steps, guidance scale, and seed
- **Example Prompts**: Pre-loaded cinematic examples from the model card
- **Real-time Generation**: Generate high-quality 1024px images
- **Reproducible Results**: Control randomness with seed settings

## 📋 Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- At least 8GB RAM (16GB+ recommended for GPU)
- ~5GB disk space for model weights

## 🚀 Quick Start

### 1. Clone or Download

```bash
git clone https://github.com/PierrunoYT/photoroom-prx-local.git
cd photoroom-prx-local
```

### 2. Install Dependencies

**Option A: Using the setup script (Recommended)**

On Windows (PowerShell):
```powershell
cmd /c setup.bat
```

On Windows (Command Prompt):
```cmd
setup.bat
```

On Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh
```

**Option B: Manual installation**

1. Install PyTorch 2.6+ with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

   **Note**: PyTorch 2.6 or higher is required for PRX model support.

2. Install diffusers from GitHub (required for PRX support):
```bash
pip install --upgrade git+https://github.com/huggingface/diffusers.git
```

3. Upgrade transformers (required for T5Gemma support):
```bash
pip install --upgrade transformers
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

**Important Notes:**
- PRX requires the latest development version of diffusers from GitHub
- The first run will download the model weights (~5GB) from Hugging Face
- A CUDA-compatible GPU is highly recommended for reasonable generation times
- The app includes a temporary monkey patch for 1024x1024 resolution bins (will be removed once official bins are added to diffusers)

### 3. Run the Application

```bash
python app.py
```

The web interface will be available at: `http://localhost:7860`

## 🎨 Usage

1. **Enter a Prompt**: Describe the image you want to generate
2. **Adjust Settings** (optional):
   - **Inference Steps** (1-50): More steps = higher quality but slower (default: 28)
   - **Guidance Scale** (1-15): How closely to follow the prompt (default: 5.0)
   - **Seed**: Set for reproducible results, or randomize for variety
3. **Click Generate**: Wait for the model to create your image
4. **Download**: Right-click the generated image to save

## 💡 Tips for Better Results

- **Be Descriptive**: Include details about lighting, composition, mood, and style
- **Cinematic Language**: Terms like "cinematic 4K", "shallow depth of field", "golden hour" work well
- **Guidance Scale**: 
  - 3-5: More creative, artistic freedom
  - 5-7: Balanced (recommended)
  - 7+: Very literal interpretation
- **Inference Steps**: 
  - 20-28: Good balance of speed and quality
  - 28-40: Higher quality, slower generation

## 📝 Example Prompts

```
A front-facing portrait of a lion in the golden savanna at sunset
```

```
A massive black monolith standing alone in a mirror-like salt flat after rainfall, 
horizon dissolving into pastel pink and cyan, reflections perfect and infinite, 
minimalist 2.39:1 frame, cinematic atmosphere of silence
```

```
Ancient pagoda rising above clouds, morning mist rolling over forested mountains, 
golden sunrise light illuminating temple roof tiles, cinematic wide-angle composition, 
ethereal atmosphere, ultra-detailed realism with painterly undertone
```

## 🔧 Configuration

### Running on CPU

If you don't have a CUDA GPU, the app will automatically use CPU. Note that generation will be significantly slower.

### Custom Port

Edit `app.py` and change the port in the launch configuration:

```python
demo.launch(
    server_port=7860,  # Change this to your desired port
    ...
)
```

### Share Publicly

To create a public link (via Gradio tunneling):

```python
demo.launch(
    share=True,  # Change this to True
    ...
)
```

## 📦 Model Information

**PRX (Photoroom Experimental)** is a 1.3-billion-parameter text-to-image model:

- **Architecture**: MMDiT-like diffusion transformer variant
- **Resolution**: 1024 pixels
- **Text Encoder**: T5-Gemma-2B-2B-UL2 (multilingual)
- **Latent Backbone**: Flux VAE
- **Training**: Supervised fine-tuning (SFT)
- **License**: Apache 2.0

### Model Links

- 🤗 [Hugging Face Model Card](https://huggingface.co/Photoroom/prx-1024-t2i-beta)
- 📚 [PRX Collection](https://huggingface.co/collections/Photoroom/prx)
- 📝 [Announcement Blog Post](https://www.photoroom.com/tech/introducing-prx)

## 🐛 Troubleshooting

### Error: Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6

This error means you need to upgrade PyTorch to version 2.6 or higher:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

For CPU only:
```bash
pip install --upgrade torch torchvision torchaudio
```

After upgrading, verify your PyTorch version:
```bash
python -c "import torch; print(torch.__version__)"
```

### ModuleNotFoundError: No module named 'diffusers.pipelines.prx'

This means you need the latest development version of diffusers:

```bash
pip install --upgrade git+https://github.com/huggingface/diffusers.git
```

### ValueError: text_encoder/prx.py does not exist

This error occurs when using an older version of diffusers. Install from GitHub:

```bash
pip install --upgrade git+https://github.com/huggingface/diffusers.git
```

### ModuleNotFoundError: No module named 'transformers.models.t5gemma'

This means you need to upgrade transformers to the latest version:

```bash
pip install --upgrade transformers
```

### Out of Memory Error

If you encounter CUDA out of memory errors:

1. Reduce the inference steps
2. Close other GPU applications
3. Try running on CPU (slower but uses system RAM)

### Model Download Issues

If the model fails to download:

1. Check your internet connection
2. Ensure you have ~5GB free disk space
3. Try manually downloading from Hugging Face

### Import Errors

Make sure all dependencies are installed correctly:

1. Install PyTorch with CUDA
2. Install diffusers from GitHub
3. Upgrade transformers
4. Install remaining requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade git+https://github.com/huggingface/diffusers.git
pip install --upgrade transformers
pip install -r requirements.txt
```

**Note**: Make sure you have PyTorch 2.6 or higher installed.

## 📄 License

This project (the web interface and application code) is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

**Important**: The PRX-1024-t2i-beta model itself is licensed under **Apache 2.0** by Photoroom. When using this application, you are subject to both:
- The MIT License for this application code
- The Apache 2.0 License for the PRX model

### Use Restrictions

You must not use this application or PRX models for:

1. Any of the restricted uses set forth in the Gemma Prohibited Use Policy
2. Any activity that violates applicable laws or regulations

For full model license details, see the [PRX Model Card](https://huggingface.co/Photoroom/prx-1024-t2i-beta)

## 🙏 Credits

- **Model**: [Photoroom](https://www.photoroom.com/) - PRX-1024-t2i-beta
- **Framework**: [Gradio](https://gradio.app/) for the web interface
- **Diffusion Library**: [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- **Resolution bins fix**: Thanks to the Photoroom team for the [temporary workaround](https://huggingface.co/spaces/Photoroom/PRX-1024-beta-version/blob/main/app.py#L40)

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

---

**Enjoy creating beautiful images with PRX! 🎨✨**

