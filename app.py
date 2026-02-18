
import gradio as gr
from image_generation import (
    z_image_turbo, 
    load_z_models, 
    unload_z_models, 
    is_low_gpu, 
    has_two_gpus
)


# Condition: If we have LOW RAM (<=15GB) AND only 1 GPU, use Safe Mode.
use_safe_mode = is_low_gpu() and not has_two_gpus()

MODEL_DATA = None
if not use_safe_mode:
    # FAST MODE (Dual GPU or High VRAM)
    # We load models ONCE here. They stay in RAM for the whole loop.
    print("🚀 High Performance Detected: Loading models once for batch...")
    MODEL_DATA = load_z_models()
else:
    # SAFE MODE (Single Low GPU)
    # We keep my_models as None.
    # The 'z_image_turbo' function will see 'None', load the model, generate, 
    # and then UNLOAD it immediately for every single image.
    print("🛡️ Safe Mode Detected (Low VRAM): Will load/unload per image.")
    MODEL_DATA = None


def generate_image(
    positive_prompt,
    negative_prompt,
    width,
    height,
    steps,
    cfg,
    seed,
):
    img_path = z_image_turbo(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed,
        model_data=MODEL_DATA   # <- keep models loaded
    )

    return img_path


custom_css = """.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"""



ASPECTS = {
    "1024x1024 (1:1)": (1024, 1024),
    "1152x896 (9:7)": (1152, 896),
    "896x1152 (7:9)": (896, 1152),
    "1152x864 (4:3)": (1152, 864),
    "864x1152 (3:4)": (864, 1152),
    "1248x832 (3:2)": (1248, 832),
    "832x1248 (2:3)": (832, 1248),
    "1280x720 (16:9)": (1280, 720),
    "720x1280 (9:16)": (720, 1280),
    "1344x576 (21:9)": (1344, 576),
    "576x1344 (9:21)": (576, 1344),
}

def update_dims(aspect):
    return ASPECTS[aspect]

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
          <div style="text-align: center; margin: 20px auto; max-width: 800px;">
              <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎨 Z Image Turbo Gradio API</h1>
          </div>""")
    with gr.Row():

        # LEFT SIDE INPUTS
        with gr.Column(scale=1):
            positive = gr.Textbox(
                label="Positive Prompt",
                placeholder="Describe your image...",
                lines=4   
            )
            aspect = gr.Dropdown(
                list(ASPECTS.keys()),
                value="1024x1024 (1:1)",
                label="Aspect Ratio"
            )
            

        
            generate_btn = gr.Button("🚀 Generate Image")
            with gr.Accordion("Settings", open=False):
              width = gr.Slider(256, 2048, value=1024, step=64, label="Width")
              height = gr.Slider(256, 2048, value=1024, step=64, label="Height")

              # auto update width/height from dropdown
              aspect.change(
                  update_dims,
                  inputs=aspect,
                  outputs=[width, height]
              )
              steps = gr.Slider(1, 30, value=9, label="Steps")
              cfg = gr.Slider(
                        0.1, 10,
                        value=1.0,
                        step=0.1,
                        label="CFG Scale"
                    )
              seed = gr.Number(
                        value=0,
                        label="Seed (0 = Random)"
                    )
              negative = gr.Textbox(label="Negative Prompt",lines=4)

        # RIGHT SIDE OUTPUT
        with gr.Column(scale=1):
            output_img = gr.Image(label="Generated Image",type="filepath")

    generate_btn.click(
        generate_image,
        inputs=[positive, negative, width, height, steps, cfg, seed],
        outputs=output_img
    )



# demo.launch(debug=True)
import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def run_demo(share,debug):
    demo.queue().launch(share=share,debug=debug)
if __name__ == "__main__":
    run_demo()
