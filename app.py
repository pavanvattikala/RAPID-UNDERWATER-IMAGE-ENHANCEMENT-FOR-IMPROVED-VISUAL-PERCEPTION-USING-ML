from flask import Flask, render_template, request, send_file, send_from_directory,jsonify
import cv2
import os
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
from keras.models import model_from_json
from utils.data_utils import getPaths, read_and_resize, preprocess, deprocess
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

model = None
filename = None

# Define the path to the uploadedImages folder
UPLOAD_FOLDER = 'uploadedImages'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Define the path to the Generated folder
GENERATED_FOLDER = 'generatedImages'
if not os.path.exists(GENERATED_FOLDER):
    os.makedirs(GENERATED_FOLDER)

app.config['GENERATED_FOLDER'] = GENERATED_FOLDER


# Define the path to the Graphs folder
GRAPHS_FOLDER = 'graphs'
if not os.path.exists(GRAPHS_FOLDER):
    os.makedirs(GRAPHS_FOLDER)

app.config['GRAPHS_FOLDER'] = GRAPHS_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/loadModel", methods=["GET", "POST"])
def loadModel():
    global model
    if request.method == "POST":
        with open('models/gen_p/model_15320_.json', "r") as json_file:
            loaded_model_json = json_file.read()
        json_file.close()    
        model = model_from_json(loaded_model_json)
        model.load_weights('models/gen_p/model_15320_.h5')
        return "FUnIE-GAN Model loaded"

@app.route("/uploadImage", methods=["GET", "POST"])
def uploadImage():
    if request.method == "POST" and 'image' in request.files:
        image_file = request.files['image']
        if image_file.filename != '':
            # Save the uploaded image to the UPLOAD_FOLDER
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            return image_path
    return "No image uploaded or invalid file"

@app.route("/generateImprovedImage", methods=["POST"])
def generateImprovedImage():
    global model

    # Get the path to the uploaded image from the request
    image_path = request.form.get("image_path")
    if not image_path:
        return "No image provided"

    # Read and preprocess the uploaded image
    image_name = image_path.replace("\\", "/")
    image_path = "uploadedImages/"+image_name

    inp_img = read_and_resize(image_path, (256, 256))
    im = preprocess(inp_img)
    im = np.expand_dims(im, axis=0) #adding as a batch of 1
    gen = model.predict(im)
    gen_img = deprocess(gen)[0]
    inputImg = cv2.imread(image_path)
    inputImg = cv2.resize(inputImg,(256,256))
    gen_img = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)
    generated_image_path = os.path.join(app.config['GENERATED_FOLDER'], image_name)
    cv2.imwrite(generated_image_path,gen_img)

    return generated_image_path


@app.route('/uploadedImages/<path:filename>')
def uploaded_image(filename):
    return send_from_directory('uploadedImages', filename)

@app.route('/generatedImages/<filename>')
def generated_images(filename):
    # Path to the directory containing generated images
    generated_dir = 'generatedImages'
    # Construct the full path to the image file
    image_path = os.path.join(generated_dir, filename)
    # Serve the image file
    return send_file(image_path, mimetype='image/jpeg')  # Adjust mimetype as per your image type


import seaborn as sns

@app.route("/generateGraphs", methods=["POST"])
def generateGraphs():
    
    # Get the path to the uploaded image from the request
    image_path = request.form.get("image_path")
    if not image_path:
        return "No image provided"

    # Read and preprocess the uploaded image
    image_name = image_path.replace("\\", "/")
    uploaded_image_path = "uploadedImages/"+image_name
    generated_image_path = "generatedImages/"+image_name
    
    # Read the original and generated images
    original_img = cv2.imread(uploaded_image_path)
    original_img = cv2.resize(original_img,(256,256))

    generated_img = cv2.imread(generated_image_path)
    generated_img = cv2.resize(generated_img,(256,256))

    # Generate the correlation matrix
    generateCorrelationMatrix(original_img,generated_img)

    # Generate the density plot
    densityPlot(original_img,generated_img)


    correlation_plot  = os.path.join(app.config['GRAPHS_FOLDER'], 'correlation_plot.png')
    density_plot = os.path.join(app.config['GRAPHS_FOLDER'], 'density_plot.png')

    return jsonify({'correlation_plot': correlation_plot, 'density_plot': density_plot})


def generateCorrelationMatrix(original_img,generated_img):
    global model

    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    generated_gray = cv2.cvtColor(generated_img, cv2.COLOR_BGR2GRAY)

    # Flatten the pixel intensities
    original_flat = original_gray.flatten()
    generated_flat = generated_gray.flatten()

    # Create a DataFrame with pixel intensities
    data = {'Original': original_flat, 'Generated': generated_flat}
    df = pd.DataFrame(data)

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Generate a heatmap using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix between Original and Generated Images')
    plt.xlabel('Original Image Pixel Intensity')
    plt.ylabel('Generated Image Pixel Intensity')

    # Save the plot
    correlation_plot_path = os.path.join(app.config['GRAPHS_FOLDER'], 'correlation_plot.png')
    plt.savefig(correlation_plot_path)

    # Close the plot to avoid showing it in the Flask app
    plt.close()

def densityPlot(original_img,generated_img):
    # Generate graphs
    data = {'Original': original_img.flatten(), 'Generated': generated_img.flatten()}
    df = pd.DataFrame(data)
    df.plot(kind='density')
    plt.title('Density Plot')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    graph_path = os.path.join(app.config['GRAPHS_FOLDER'], 'density_plot.png')
    plt.savefig(graph_path)

    #  # Save data as CSV
    # csv_path = os.path.join(app.config['GRAPHS_FOLDER'], 'density_data.csv')
    # df.to_csv(csv_path, index=False)


@app.route('/graphs/<filename>')
def generated_graphs(filename):
    # Path to the directory containing generated graphs
    graph_dir = 'graphs'
    # Construct the full path to the graph file
    graph_path = os.path.join(graph_dir, filename)
    # Serve the graph file
    return send_file(graph_path, mimetype='image/png')

if __name__ == "__main__":
    app.run()
