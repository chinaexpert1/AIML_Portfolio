from modules.extraction.embedding import Embedding
from modules.extraction.preprocessing import Preprocessing
from modules.retrieval.search import FaissSearch
from modules.retrieval.index.bruteforce import FaissBruteForce
from modules.retrieval.index.lsh import FaissLSH
from modules.retrieval.index.hnsw import FaissHNSW
import os
import pickle
from PIL import Image
from flask import Flask, request, jsonify
# OMP workaround
os.environ["KMP_DUPLICATE_LIB_OK"]="True" 

"""
Flask app for processing images.

This script provides two endpoints:
1. /identify: Processes an image and returns the top-k identities.
2. /add: Adds a provided image to the gallery with an associated name.

Usage:
    Run the app with: python app.py
    Sample curl command for /identify:
        curl -X POST -F "image=@/path/to/image.jpg" -F "k=3" http://localhost:5000/identify
    Sample curl command for /add:
        curl -X POST -F "image=@/path/to/image.jpg" -F "name=Firstname_Lastname" http://localhost:5000/add
"""

import numpy as np

from PIL import Image

app = Flask(__name__)

## List of designed parameters: 
# (Configure these parameters according to your design decisions)
DEFAULT_K = '3'
MODEL = 'vggface2'
INDEX = 'bruteforce'
SIMILARITY_MEASURE = 'euclidean'
EMBED_DIM = 512
GALLERY_PATH = f"storage/multi_image_gallery"

# Initialize components


embedder = Embedding(pretrained=MODEL)
preprocess = Preprocessing(image_size=160)


def initialize_index():
    # If GALLERY_PATH is a directory, build the index from its subfolders.
    if os.path.isdir(GALLERY_PATH):
        if INDEX == 'bruteforce':
            new_index = FaissBruteForce(dim=EMBED_DIM, metric=SIMILARITY_MEASURE)
        elif INDEX == 'lsh':
            new_index = FaissLSH(dim=EMBED_DIM, nbits=128)
        elif INDEX == 'hnsw':
            new_index = FaissHNSW(dim=EMBED_DIM, M=32, efConstruction=40)
        else:
            raise ValueError("Unsupported index type specified.")

        # Iterate over each subfolder (each identity).
        for identity in os.listdir(GALLERY_PATH):
            identity_dir = os.path.join(GALLERY_PATH, identity)
            if os.path.isdir(identity_dir):
                for image_file in os.listdir(identity_dir):
                    # Skip files that begin with ._
                    if image_file.startswith("._"):
                        continue
                    image_path = os.path.join(identity_dir, image_file)
                    try:
                        img = Image.open(image_path)
                        tensor = preprocess.process(img)
                        vector = embedder.encode(tensor)
                        new_index.add_embeddings([vector], [identity])
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
        return new_index
    else:
        # If GALLERY_PATH is not a directory, assume it's a pickle file and load it.
        if INDEX == 'bruteforce':
            return FaissBruteForce.load(GALLERY_PATH)
        elif INDEX == 'lsh':
            return FaissLSH.load(GALLERY_PATH)
        elif INDEX == 'hnsw':
            return FaissHNSW.load(GALLERY_PATH)

# Initialize components
index = initialize_index()
retrieval = FaissSearch(index, metric=SIMILARITY_MEASURE)




@app.route('/identify', methods=['POST'])
def identify():
    """
    Process the probe image to identify top-k identities in the gallery.

    Expects form-data with:
      - image: Image file to be processed.
      - k: (optional) Integer specifying the number of top identities 
           (default is 3).

    Returns:
      JSON response with a success message and the provided value of k.
      If errors occur, returns a JSON error message with the appropriate status code.
    """
    # Check if the request has the image file
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Retrieve and validate the integer parameter "k"
    try:
        k = int(request.form.get('k', DEFAULT_K))
    except ValueError:
        return jsonify({"error": "Invalid integer for parameter 'k'"}), 400

    try:
        # Open the image using Pillow and convert it to a NumPy array (for logging)
        pil_image = Image.open(file)
        np_image = np.array(pil_image)
        print(np_image)
    except Exception as e:
        return jsonify({
            "error": "Failed to convert image to numpy array",
            "details": str(e)
        }), 500
    ########################################
    # TASK 1a: Implement /identify endpoint
    #         to return the top-k identities
    #         of the provided probe.
    ########################################
    try:
        # Process the original PIL image through the preprocessing pipeline
        tensor = preprocess.process(pil_image)
        vector = embedder.encode(tensor)
    except Exception as e:
        return jsonify({
            "error": "Failed to process image",
            "details": str(e)
        }), 500


    try:
        distances, indices, identities = retrieval.search(vector, k=int(request.form.get('k', DEFAULT_K)))
    except Exception as e:
        return jsonify({
            "error": "Search failed",
            "details": str(e)
        }), 500



    return jsonify({
        "message": f"Returned top-{request.form.get('k', DEFAULT_K)} identities using index={INDEX} and metric={SIMILARITY_MEASURE}",
        "ranked identities": identities
    }), 200


@app.route("/add", methods=['POST'])
def add():
    """
    Add a provided image to the gallery with an associated name.

    Expects form-data with:
      - image: Image file to be added.
      - name: String representing the identity associated with the image.

    Returns:
      JSON response confirming the image addition.
      If errors occur, returns a JSON error message with the appropriate status code.
    """
    # Check if the request has the image file
    if 'image' not in request.files:
        return jsonify({"Error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"Error": "No file selected for uploading"}), 400

    # Convert the image into a NumPy array
    try:
        pil_image = Image.open(file)
        image = np.array(Image.open(file))
        print(image)
    except Exception as e:
        return jsonify({
            "error": "Failed to convert image to numpy array",
            "details": str(e)
        }), 500

    # Retrieve the 'name' parameter
    name = request.form.get('name')
    if not name:
        return jsonify({"Error": "Must have associated 'name'"}), 400

    ########################################
    # TASK 1b: Implement `/add` endpoint to
    #         add the provided image to the 
    #         catalog.
    ########################################

    # Determine the subfolder (using the provided name) inside the gallery directory.
    person_folder = os.path.join(GALLERY_PATH, name)
    try:
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
    except Exception as e:
        return jsonify({
            "error": "Failed to create subfolder for gallery",
            "details": str(e)
        }), 500

    # Save the image into the subfolder with a unique filename.
    import time
    timestamp = int(time.time())
    new_filename = f"{name}_{timestamp}.jpg"
    save_path = os.path.join(person_folder, new_filename)
    try:
        pil_image.save(save_path)
    except Exception as e:
        return jsonify({
            "error": "Failed to save image to disk",
            "details": str(e)
        }), 500

    try:
        pil_image = Image.open(file)
        # Preprocess the image (resizing, normalization, etc.)
        tensor = preprocess.process(pil_image)
        # Generate an embedding vector for the image
        vector = embedder.encode(tensor)
    except Exception as e:
        return jsonify({"error": "Failed to process image", "details": str(e)}), 500

    # Add the embedding to the index and save the updated index to a file.
    try:
        index.add_embeddings([vector], [name])
        # Save the index to a file inside the gallery directory rather than the directory itself.
        GALLERY_INDEX_FILE = os.path.join(GALLERY_PATH, "index.pkl")
        index.save(GALLERY_INDEX_FILE)
    except Exception as e:
        return jsonify({
            "error": "Failed to add image to gallery",
            "details": str(e)
        }), 500

    return jsonify({
        "message": f"New image added to gallery (as {name}) and indexed into catalog."
    }), 200


if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')
