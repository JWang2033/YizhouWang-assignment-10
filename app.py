import os
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from open_clip import create_model_and_transforms, tokenizer

# Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')


# Configuration
UPLOAD_FOLDER = 'output'
IMAGES_FOLDER = 'coco_images_resized'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/32"
pretrained = "openai"
embedding_file = "image_embeddings.pickle"

# Load the CLIP model and preprocess function
model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()

# Load image embeddings
if os.path.exists(embedding_file):
    print(f"Loading precomputed embeddings from {embedding_file}...")
    image_embeddings = pd.read_pickle(embedding_file)
    print(f"Loaded {len(image_embeddings)} image embeddings.")
else:
    print(f"Embedding file '{embedding_file}' not found. Please compute embeddings.")
    image_embeddings = None

@app.route('/')
def index():
    return render_template('index.html')  # Serve the index.html file from the static directory

# Serve images from coco_images_resized directory
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_FOLDER, filename)

@app.route('/output/<path:filename>')
def serve_output_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Route for image-to-image search
@app.route('/search/image', methods=['POST'])
def search_by_image():
    """
    Perform image-to-image search by finding the most similar image
    to the uploaded query image in the dataset.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    # Save the uploaded image
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(uploaded_image_path)

    # Ensure precomputed embeddings are available
    if image_embeddings is None or image_embeddings.empty:
        return jsonify({"error": "No embeddings available for similarity comparison"}), 400

    # Compute embedding for the uploaded image
    image = Image.open(uploaded_image_path).convert("RGB")
    image_tensor = preprocess_val(image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = model.encode_image(image_tensor)
        query_embedding = F.normalize(query_embedding, p=2, dim=1).cpu().numpy()

    # Compute similarities
    similarities = {
        row['file_name']: float(F.cosine_similarity(torch.tensor(query_embedding), torch.tensor(row['embedding']).unsqueeze(0)))
        for _, row in image_embeddings.iterrows()
    }

    # Find the most similar image
    top_result = max(similarities.items(), key=lambda x: x[1])
    retrieved_image_name = top_result[0]
    similarity_score = top_result[1]

    # Retrieve and copy the most similar image to the output folder
    retrieved_image_path = os.path.join(IMAGES_FOLDER, retrieved_image_name)
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"image_to_image_{retrieved_image_name}")
    from shutil import copyfile
    copyfile(retrieved_image_path, output_image_path)

    # Return result
    return jsonify({
        "query_image": file.filename,
        "retrieved_image": retrieved_image_name,
        "similarity_score": similarity_score,
        "output_image_path": output_image_path
    })

@app.route('/search/text', methods=['POST'])
def search_by_text():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    if image_embeddings is None or image_embeddings.empty:
        return jsonify({"error": "No embeddings available for similarity comparison"}), 400

    # Compute text embedding
    text_tokens = tokenizer.tokenize(query)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens.to(device))
        text_embedding = F.normalize(text_embedding, p=2, dim=1).cpu().numpy()

    # Compute similarities
    similarities = {
        row['file_name']: float(F.cosine_similarity(torch.tensor(text_embedding), torch.tensor(row['embedding']).unsqueeze(0)))
        for _, row in image_embeddings.iterrows()
    }

    if not similarities:
        return jsonify({"error": "No similar images found"}), 404

    # Get the most similar image
    top_result = max(similarities.items(), key=lambda x: x[1])
    retrieved_image_url = f"/images/{top_result[0]}"
    return jsonify({
        "query": query,
        "retrieved_image": retrieved_image_url,
        "similarity_score": top_result[1]
    })




# Route for hybrid search (text + image)
@app.route('/search/hybrid', methods=['POST'])
def search_hybrid():
    """
    Perform hybrid search using both a text query and an image query.
    Combine the embeddings based on a user-provided lambda (lam) value.
    """
    if 'image' not in request.files or 'query' not in request.form:
        return jsonify({"error": "Image and text query are both required"}), 400

    text_query = request.form['query']
    image_file = request.files['image']

    # Lambda input for weight adjustment
    try:
        lam = float(request.form.get('lambda', 0.5))  # Default lambda is 0.5
        if not 0.0 <= lam <= 1.0:
            raise ValueError("Lambda must be between 0 and 1.")
    except ValueError as e:
        return jsonify({"error": f"Invalid lambda value: {str(e)}"}), 400

    # Save the uploaded image
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(uploaded_image_path)

    # Ensure precomputed embeddings are available
    if image_embeddings is None or image_embeddings.empty:
        return jsonify({"error": "No embeddings available for similarity comparison"}), 400

    # Compute text embedding
    text_tokens = tokenizer.tokenize(text_query)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens.to(device))
        text_embedding = F.normalize(text_embedding, p=2, dim=1).cpu().numpy()

    # Compute image embedding
    image = Image.open(uploaded_image_path).convert("RGB")
    image_tensor = preprocess_val(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image_tensor)
        image_embedding = F.normalize(image_embedding, p=2, dim=1).cpu().numpy()

    # Combine text and image embeddings using lambda
    combined_embedding = F.normalize(
        lam * torch.tensor(text_embedding) + (1.0 - lam) * torch.tensor(image_embedding),
        p=2, dim=1
    ).numpy()

    # Compute similarities
    similarities = {
        row['file_name']: float(F.cosine_similarity(torch.tensor(combined_embedding), torch.tensor(row['embedding']).unsqueeze(0)))
        for _, row in image_embeddings.iterrows()
    }

    # Find the most similar image
    top_result = max(similarities.items(), key=lambda x: x[1])
    retrieved_image_name = top_result[0]
    similarity_score = top_result[1]

    # Retrieve and copy the most similar image to the output folder
    retrieved_image_path = os.path.join(IMAGES_FOLDER, retrieved_image_name)
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"hybrid_search_{retrieved_image_name}")
    from shutil import copyfile
    copyfile(retrieved_image_path, output_image_path)

    # Return result
    return jsonify({
        "query_image": image_file.filename,
        "query_text": text_query,
        "lambda": lam,
        "retrieved_image": retrieved_image_name,
        "similarity_score": similarity_score,
        "output_image_path": output_image_path
    })

if __name__ == '__main__':
    # Ensure necessary directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Verify that embeddings are loaded
    if image_embeddings is None or image_embeddings.empty:
        print("No precomputed embeddings found. Please generate 'image_embeddings.pickle' using the notebook.")
    else:
        print(f"Using {len(image_embeddings)} precomputed embeddings.")

    # Launch the Flask server on port 3000
    app.run(host='0.0.0.0', port=3000, debug=True)