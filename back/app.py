# from flask import Flask, request, jsonify, send_file
# from diffusers import StableDiffusionPipeline
# import torch
# from PIL import Image
# import io
# from flask_cors import CORS


# # Initialize the app and load the model
# app = Flask(__name__)
# CORS(app, resources={r"/generate": {"origins": "http://localhost:3000"}})
# model_path = "./stable-diffusion-2-1-base"
# pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
# pipeline = pipeline.to("cuda")  # Send to GPU

# @app.route("/generate", methods=["POST"])
# def generate_image():
#     try:
#         # Get the text prompt from the request
#         data = request.get_json()
#         prompt = data.get("prompt", "")

#         if not prompt:
#             return jsonify({"error": "Prompt is required"}), 400

#         # Generate the image
#         image = pipeline(prompt).images[0]

#         # Save the image to a BytesIO object
#         img_io = io.BytesIO()
#         image.save(img_io, "PNG")
#         img_io.seek(0)

#         return send_file(img_io, mimetype="image/png")

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)




# from flask import Flask, request, jsonify, send_file
# from diffusers import StableDiffusionPipeline
# import torch
# from PIL import Image
# import io
# from flask_cors import CORS
# import logging

# # Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize the app and load the model
# app = Flask(__name__)
# CORS(app, resources={r"/generate": {"origins": "http://localhost:3000"}})

# model_path = "./stable-diffusion-2-1-base"
# logger.info("Loading model...")

# pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
# pipeline.enable_attention_slicing()  # Reduce memory usage
# pipeline = pipeline.to("cuda")  # Send to GPU
# logger.info("Model loaded successfully.")

# @app.route("/generate", methods=["POST"])
# def generate_image():
#     try:
#         logger.info("Received request to generate image.")
        
#         # Get the text prompt from the request
#         data = request.get_json()
#         prompt = data.get("prompt", "")

#         if not prompt:
#             logger.error("No prompt provided.")
#             return jsonify({"error": "Prompt is required"}), 400

#         if len(prompt) > 500:
#             logger.error("Prompt too long.")
#             return jsonify({"error": "Prompt too long"}), 400

#         logger.info(f"Generating image for prompt: {prompt}")
#         image = pipeline(prompt).images[0]

#         # Save the image to a BytesIO object
#         img_io = io.BytesIO()
#         image.save(img_io, "PNG")
#         img_io.seek(0)

#         logger.info("Image generated successfully.")
#         return send_file(img_io, mimetype="image/png")

#     except Exception as e:
#         logger.error(f"Error during image generation: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)






from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import logging   
import numpy as np
from sam_model import SamHandler  # Import SAM handler
from google_api import search_products_google  # Import Google API logic

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": "http://localhost:3000"}})

# Stable Diffusion model path
model_path = "./stable-diffusion-2-1-base"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load Stable Diffusion model
try:
    logger.info("Loading Stable Diffusion model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipeline.enable_attention_slicing()
    pipeline = pipeline.to(device)
    logger.info("Stable Diffusion model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Stable Diffusion model: {e}")
    raise e


@app.route("/generate", methods=["POST"])
def generate_image():
    try:
        logger.info("Received request to generate image.")

        # Get the text prompt from the request
        data = request.get_json()
        prompt = data.get("prompt", "")

        if not prompt:
            logger.error("No prompt provided.")
            return jsonify({"error": "Prompt is required"}), 400

        if len(prompt) > 500:
            logger.error("Prompt too long.")
            return jsonify({"error": "Prompt too long"}), 400

        logger.info(f"Generating image for prompt: {prompt}")

        # Ensure dimensions are divisible by 8
        height = 512  # Set height divisible by 8
        width = 512   # Set width divisible by 8

        # Generate the image
        generated_image = pipeline(prompt, height=height, width=width).images[0]

        # Save the generated image temporarily in memory
        generated_img_io = io.BytesIO()
        generated_image.save(generated_img_io, "PNG")
        generated_img_io.seek(0)

        logger.info("Image generation completed successfully.")

        # Return generated image as a separate file
        return send_file(generated_img_io, mimetype="image/png", as_attachment=True, download_name="generated_image.png")

    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        return jsonify({"error": f"Failed to generate image: {str(e)}"}), 500


@app.route("/segment", methods=["POST"])
def segment_image():
    try:
        logger.info("Received request to segment image.")

        # Get the generated image from the request
        data = request.get_json()
        image_data = data.get("image", "")

        if not image_data:
            logger.error("No image data provided.")
            return jsonify({"error": "Image data is required"}), 400

        # Convert image data from hex to bytes
        image_bytes = bytes.fromhex(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Initialize SAM handler and segment the image
        logger.info("Segmenting the generated image using SAM...")
        sam_handler = SamHandler()
        segmented_image_pil = sam_handler.segment_image(image)

        # Save the segmented image temporarily in memory
        segmented_img_io = io.BytesIO()
        segmented_image_pil.save(segmented_img_io, "PNG")
        segmented_img_io.seek(0)

        logger.info("Image segmentation completed successfully.")

        # Return segmented image as a separate file
        return send_file(segmented_img_io, mimetype="image/png", as_attachment=True, download_name="segmented_image.png")

    except Exception as e:
        logger.error(f"Error during image segmentation: {str(e)}")
        return jsonify({"error": f"Failed to segment image: {str(e)}"}), 500


@app.route("/search_products", methods=["POST"])
def search_products():
    try:
        data = request.get_json()
        product_query = data.get("query", "")

        if not product_query:
            return jsonify({"error": "Product query is required"}), 400

        # Search products using Google API
        products = search_products_google(product_query)

        if not products:
            return jsonify({"error": "No products found."}), 404

        return jsonify({"products": products})

    except Exception as e:
        logger.error(f"Error during product search: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)















    #google client id = 343599201601-4gko8jvakbv8lhdom1mjvdm8ruh1t7h3.apps.googleusercontent.com
