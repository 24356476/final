from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import base64
from io import BytesIO
from PIL import Image
from SinglePairDataset import SinglePairDataset
from flask import send_file


app = Flask(__name__)

# Folder paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_FOLDER = os.path.join(UPLOAD_FOLDER, 'model')
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'cloth')
OUTPUT_FOLDER = os.path.join(UPLOAD_FOLDER, 'output')  # Change output path
# Ensure upload and output folders exist
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(CLOTH_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Fixed opt configuration function
def get_fixed_opt():
    class Opt:
        dataroot = UPLOAD_FOLDER
        fine_height = 384  # Example fixed height
        fine_width = 256   # Example fixed width
        semantic_nc = 13   # Number of channels for the segmentation map
        output_dir = OUTPUT_FOLDER
        # Add other fixed options as needed for the dataset and model
    return Opt()


# Route to serve the HTML page
@app.route('/')
def index():
    return send_from_directory('client', 'index.html')


# Route to handle image uploads and virtual try-on processing
@app.route('/virtual_tryon', methods=['POST'])
def virtual_tryon():
    opt = get_fixed_opt()  # Use the fixed configuration
    opt.data_root = "C:/Users/Noyal/PycharmProjects/VITON"  # Add data_root directly here

    # Clear previous uploads
    clear_folder(MODEL_FOLDER)
    clear_folder(CLOTH_FOLDER)
    clear_folder(OUTPUT_FOLDER)


    # Save the uploaded files
    person_image = request.files['personImage']
    cloth_image = request.files['shirtImage']
    person_image_path = os.path.join(MODEL_FOLDER, person_image.filename)
    cloth_image_path = os.path.join(CLOTH_FOLDER, cloth_image.filename)
    person_image.save(person_image_path)
    cloth_image.save(cloth_image_path)

    # Construct the pair using filenames
    model_filename = person_image.filename
    cloth_filename = cloth_image.filename
    pair = f"{model_filename} + {cloth_filename}"
    print("Constructed Pair:", pair)  # For debugging

    # Temporary response; replace with actual processing in SinglePairDataset
    #return jsonify({"message": "Pair constructed and printed successfully."})


    # Step 1: Initialize the SinglePairDataset with the provided images and options
    dataset = SinglePairDataset(person_image_path, cloth_image_path, opt)

    result = dataset[0]  # Get the single pair processed data

    # Define path to save the output image in OUTPUT_FOLDER
    output_path = os.path.join(OUTPUT_FOLDER, "output_image.png")

    # Assuming result['output_image'] is the PIL Image or tensor; save it directly to output_path
    result['output_image'].save(output_path)



    # Serve the output image from OUTPUT_FOLDER
    return send_file(output_path, mimetype='image/png')



    # Step 2: Fetch the processed result from the dataset (we assume there’s only one item, so we use index 0)
    #output_path = os.path.join(MODEL_FOLDER, "output_image.png")
    #return send_file(output_path, mimetype='image/png')

    # Step 3: For testing purposes, print the keys of the result dictionary to ensure processing works
    #print("Processed Result Keys:", result.keys())

    # Step 4: Return a JSON response with the processed result keys
    #return jsonify({
    #    "message": "Pair constructed and processed successfully.",
    #    "result_keys": list(result.keys())  # Send back the keys as confirmation
    #})

# Helper function to clear the folder
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


# Serve static files (CSS, JS, images)
@app.route('/client/<path:filename>')
def serve_client_file(filename):
    return send_from_directory('client', filename)


# Serve uploaded files (e.g., generated output image)
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
