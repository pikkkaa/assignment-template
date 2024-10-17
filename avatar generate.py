from diffusers import AutoPipelineForText2Image, LCMScheduler
import torch

model = 'lykon/dreamshaper-8-lcm'
pipe = AutoPipelineForText2Image.from_pretrained(model)

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

def generate_avatar(gender, personality, color):
    prompt = f"A {gender} avatar with {color} color scheme, exhibiting {personality} personality traits."
    images = pipe(prompt, num_inference_steps=8, guidance_scale=1.5).images
    images[0].show()

while True:
    gender = input("Enter gender (male/female):\n>>> ")
    
    personality = input("Enter personality traits (e.g., cheerful, serious, adventurous):\n>>> ")

    color = input("Enter preferred color (e.g., blue, red, green):\n>>> ")
    
    generate_avatar(gender, personality, color)