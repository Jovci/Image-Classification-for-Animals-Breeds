# Animal Breed Identification with PyTorch

This project aims to develop an image classification model using PyTorch to identify different animal breeds. The model is trained on the Oxford-IIIT Pet Dataset and can classify images of animals into various breeds.

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/cat_breed_classification.git
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download and preprocess the dataset (instructions in `notebooks/data_preprocessing.ipynb`).
4. Train the model:
   ```sh
   python src/train.py
   ```
5. Evaluate the model:
   ```sh
   python src/evaluate.py
   ```
6. Run the web application:
   ```sh
   python app/app.py
   ```

## Usage

To predict the breed of a cat in an image, upload the image through the web application and get the predicted breed.

## Project Structure

- `data/`: Contains raw and processed dataset files.
- `notebooks/`: Jupyter Notebooks for data preprocessing.
- `src/`: Source code for dataset handling, model training, evaluation, and prediction.
- `app/`: Web application code.
- `saved_models/`: Directory for saved model weights.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## Results

- Model accuracy: 71%



## Contributing

Contributions are welcome! Please create a pull request or open an issue for any bugs or feature requests.
