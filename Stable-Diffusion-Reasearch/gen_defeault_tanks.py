!pip install diffusers transformers accelerate

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount('/content/drive')

# Set up the model ReV Animated
model_id = "stablediffusionapi/rev-animated"

# Download and load the model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Set up DPM++ 2M Karras
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)

# Move the model to GPU if available
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Generate the image
prompt = "t-72, tank, forest"

# Desired number of simulations
image_number = 4

# Desired number for preview
prev_num = 4

# Generation of images
gen_image = []

for i in range(image_number):
  image = pipe(
      prompt,
      width=768,
      height=432,
      num_inference_steps=30,
  ).images[0]
  image.save(f"/content/drive/MyDrive/Tanks-Default/tank-default-{i}.png")

  gen_image.append(image)


plt.figure(figsize=(15, 15))
for i in range(prev_num):
  plt.subplot(1, prev_num, i + 1)
  plt.imshow(gen_image[i])
  plt.axis('off')
plt.show()
