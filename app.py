import gradio as gr
import torch
import os
from diffusers import DiffusionPipeline

# Initialize the model
print("Loading PRX model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device} with dtype: {dtype}")

# Load the model with trust_remote_code=True to allow custom pipeline code
pipe = DiffusionPipeline.from_pretrained(
    "Photoroom/prx-1024-t2i-beta",
    torch_dtype=dtype,
    trust_remote_code=True
).to(device)

print(f"Model loaded successfully!")

# Example prompts from the model card
EXAMPLE_PROMPTS = [
    "A front-facing portrait of a lion in the golden savanna at sunset",
    "A close-up portrait in a photography studio, multiple soft light sources creating gradients of shadow on her face, minimal background, cinematic 4K realism, artistic focus on light and emotion rather than glamour",
    "A massive black monolith standing alone in a mirror-like salt flat after rainfall, horizon dissolving into pastel pink and cyan, reflections perfect and infinite, minimalist 2.39:1 frame, cinematic atmosphere of silence",
    "Rain has just ended on a green plain, puddles glistening under soft sunlight, an astronaut on horseback rides slowly through the mist, a vivid rainbow curving behind distant mountains, cinematic clarity, detailed wet suit reflections, volumetric light",
    "A woman standing ankle-deep in the ocean at dawn, gentle waves touching her feet, mist and pastel horizon, cinematic wide composition, calm and contemplative mood",
    "Hundreds of paper lanterns drifting along a quiet river at dusk, soft orange light piercing cold blue mist, reflections trembling across rippled water, camera at water level with shallow DOF",
    "Wide aerial shot over a black sand beach in Iceland, massive waves crashing with white foam, dramatic clouds opening to reveal a ray of sunlight, cinematic 16:9 composition, ultra-detailed texture of basalt cliffs",
    "Ancient pagoda rising above clouds, morning mist rolling over forested mountains, golden sunrise light illuminating temple roof tiles, cinematic wide-angle composition, ethereal atmosphere",
]

def generate_image(prompt, num_inference_steps, guidance_scale, seed, randomize_seed):
    """Generate an image using the PRX model."""
    if not prompt:
        return None, "Please enter a prompt"
    
    try:
        # Set seed for reproducibility
        if randomize_seed:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate image
        print(f"Generating image with prompt: {prompt[:50]}...")
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        return image, f"Generated successfully! Seed: {seed}"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="PRX-1024 Text-to-Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 PRX-1024 Text-to-Image Generator
    
    Generate high-quality images using Photoroom's **PRX-1024-t2i-beta** model - a 1.3B parameter 
    text-to-image model trained from scratch and released under Apache 2.0 license.
    
    **Model:** [Photoroom/prx-1024-t2i-beta](https://huggingface.co/Photoroom/prx-1024-t2i-beta)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=4,
                value=EXAMPLE_PROMPTS[0]
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                num_steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=28,
                    step=1,
                    label="Number of Inference Steps",
                    info="More steps = higher quality but slower generation"
                )
                
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    value=5.0,
                    step=0.5,
                    label="Guidance Scale",
                    info="How closely to follow the prompt (higher = more literal)"
                )
                
                seed = gr.Number(
                    label="Seed",
                    value=42,
                    precision=0,
                    info="Set seed for reproducible results"
                )
                
                randomize_seed = gr.Checkbox(
                    label="Randomize Seed",
                    value=True,
                    info="Generate a random seed for each generation"
                )
            
            generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
            
            gr.Markdown("### 💡 Example Prompts")
            gr.Examples(
                examples=[[prompt] for prompt in EXAMPLE_PROMPTS],
                inputs=[prompt_input],
                label=None
            )
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                height=600
            )
            output_status = gr.Textbox(
                label="Status",
                interactive=False
            )
    
    gr.Markdown("""
    ---
    ### About PRX
    
    **PRX (Photoroom Experimental)** is a 1.3-billion-parameter text-to-image model:
    - ✅ Trained entirely from scratch
    - ✅ Apache 2.0 license (fully open-source)
    - ✅ Fast inference with high-quality outputs
    - ✅ Uses T5-Gemma-2B for multilingual text encoding
    - ✅ Flow matching with discrete scheduling
    
    **Tips for better results:**
    - Be descriptive and specific in your prompts
    - Include details about lighting, composition, and style
    - Cinematic and photographic terms work well
    - Try different guidance scales (3-7 typically works best)
    """)
    
    # Connect the generate button
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, num_steps, guidance_scale, seed, randomize_seed],
        outputs=[output_image, output_status]
    )
    
    # Also allow Enter key to generate
    prompt_input.submit(
        fn=generate_image,
        inputs=[prompt_input, num_steps, guidance_scale, seed, randomize_seed],
        outputs=[output_image, output_status]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )

