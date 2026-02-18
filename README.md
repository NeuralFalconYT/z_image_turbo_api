
# Z Image Turbo via Gradio Public API, Run Locally Without GPU

Generate high-quality images using the ```Z Image Turbo``` text-to-image generation model via API. If you want to build a small project but can’t use an image generation model due to GPU or storage limitations, you can run this on a free cloud server like Google Colab, Kaggle, Modal, lightning.ai etc and access it through an API. It may be slower, but it’s better than having no local image generation option at all.

---

## ✅ Features

### 🛡️ Low GPU Mode (≤15GB VRAM)
Automatically reloads the model for each generation to prevent crashes and CUDA out-of-memory errors.

### ⚡ Dual GPU Support
If two GPUs are available, the workload is split across both GPUs for smoother performance without CUDA memory crashes.

### 🚀 Single High-VRAM GPU Mode
On a single powerful GPU, models load once and run smoothly for faster image generation.

---

## 🚀 Usage Steps

### Step 1

Run the server on any platform that provides free GPU access, such as Google Colab, Kaggle, or other cloud environments.  
<br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/z_image_turbo_api/blob/main/server.ipynb)

---

### Step 2

Copy the public Gradio URL.

<img width="1781" height="965" alt="Gradio URL Example" src="https://github.com/user-attachments/assets/a815fe6c-297f-4978-a1bb-5aa306e8331e" />

---

### Step 3

Paste the public Gradio URL into the client code and change the prompts as needed.  
Make sure **Gradio is installed on your local system** (if you are using Python).

You can also access the API from other languages as well, such as **JavaScript, R, and more**.  
To learn how, click **"Use via API"** on the Gradio app interface.

---

### 🧑‍💻 Client-Side Python Code:

```python
import os
import shutil
from gradio_client import Client


def generate_and_save_image(
    positive_prompt,
    negative_prompt="",
    width=1024,
    height=1024,
    steps=9,
    save_dir="z_image_turbo_images"
): 
    global client
    # Call API
    result = client.predict(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        api_name="/generate_image"
    )

    # Ensure save folder exists
    os.makedirs(save_dir, exist_ok=True)

    # Extract filename from returned temp path
    filename = os.path.basename(result)

    # Final save path
    final_path = os.path.join(save_dir, filename)

    # Copy file locally
    shutil.copy(result, final_path)

    return final_path


# Copy paste the public gradio url
gradio_url="https://54a0319f1d589128ff.gradio.live/"
client = Client(gradio_url)

img_path = generate_and_save_image(
    "a cute girl with short hair in a flower field"
)

print("Saved to:", img_path)
````

---

### Generated Sample Results:

<p align="center">
  <img src="https://github.com/user-attachments/assets/6942492b-d52a-4f22-b1d0-59485042a6ac" width="48%" />
  <img src="https://github.com/user-attachments/assets/441c863c-3317-43f6-92ee-d020dcff9963" width="48%" />
</p>


