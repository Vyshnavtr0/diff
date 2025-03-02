from flask import Flask, request, jsonify
import torch
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

# Load Stable Diffusion model (this may take time)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "a futuristic city")
    image = pipe(prompt).images[0]
    image.save("output.png")
    return jsonify({"image_url": "output.png"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)