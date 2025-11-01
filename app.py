
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image, ImageOps, ImageChops
import numpy as np
import io

app = Flask(__name__)
CORS(app)

try:
    model = tf.keras.models.load_model('mnist_cnn_model.keras')
    print("* Model (mnist_cnn_model.keras) loaded successfully")
except Exception as e:
    print(f"* Error loading model: {e}")
    model = None

def preprocess_image(image_file):
    """
    Advanced preprocessing to make uploaded images look like MNIST data.
    """
    try:
        img = Image.open(image_file.stream).convert('L')

      
        img_array_for_check = np.array(img)
        mean_brightness = np.mean(img_array_for_check)

        if mean_brightness > 128:
            print("* Detected black-on-white image, inverting.")
            img = ImageOps.invert(img)
        else:
            print("* Detected white-on-black image, no inversion needed.")

        try:
           
            img = ImageOps.autocontrast(img, cutoff=2)
            print("* Image contrast automatically stretched.")
        except Exception as e:
            print(f"Error during autocontrast: {e}")

   
        try:
            threshold = 128
            img = img.point(lambda p: 255 if p > threshold else 0, 'L')
            print("* Image binarized (pure black and white).")
        except Exception as e:
            print(f"Error during thresholding: {e}")
   
        bg = Image.new('L', img.size, 0)
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        
        if not bbox:
            print("Warning: Bounding box not found, using simple resize.")
            img = img.resize((28, 28), Image.LANCZOS)
        else:
           
            img = img.crop(bbox)

        max_dim = max(img.width, img.height)
        square_img = Image.new('L', (max_dim, max_dim), 0)
        paste_x = (max_dim - img.width) // 2
        paste_y = (max_dim - img.height) // 2
        square_img.paste(img, (paste_x, paste_y))
        
        
        padding = int(max_dim * 0.2)
        padded_img = ImageOps.expand(square_img, border=padding, fill=0) # 0 is black
        
        
        img = padded_img.resize((28, 28), Image.LANCZOS)
        
        
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            processed_image = preprocess_image(file)
            if processed_image is None:
                return jsonify({'error': 'Could not process image'}), 400

            prediction = model.predict(processed_image)
            predicted_digit = int(np.argmax(prediction))
            confidence = float(np.max(prediction) * 100)

            return jsonify({
                'predicted_digit': predicted_digit,
                'confidence': f"{confidence:.2f}"
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)