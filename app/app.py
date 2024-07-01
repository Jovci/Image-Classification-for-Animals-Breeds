import sys
import os
from flask import Flask, request, render_template
from torchvision import transforms

# Ensure the src directory is in the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction import predict

app = Flask(__name__)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('static', 'uploads', file.filename)
            file.save(file_path)
            class_id, class_name = predict(file_path, '../saved_models/cat_breed_model.pth', transform)
            return render_template('index.html', prediction=class_name, image_path=file_path)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
