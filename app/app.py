from flask import Flask, request, render_template
from src.predict import predict
from torchvision import transforms

app = Flask(__name__)

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
            file_path = 'static/uploads/' + file.filename
            file.save(file_path)
            prediction = predict(file_path, 'saved_models/cat_breed_model.pth', transform, num_classes=12)
            return render_template('index.html', prediction=prediction, image_path=file_path)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
