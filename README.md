# MNIST CNN Fullstack Application

A full-stack application with a CNN model trained on MNIST data for digit recognition.

## Architecture

- **Frontend**: Simple HTML/CSS/JS interface with canvas for drawing digits
- **Backend**: FastAPI server with PyTorch model for digit recognition
- **Proxy**: Nginx for serving the frontend and proxying API requests to the backend
- **Model**: CNN with two convolutional blocks and a fully connected classifier

## Model Architecture

The model is a Convolutional Neural Network (CNN) with:
- Two convolutional blocks, each with:
  - Conv2D layers with 5x5 and 3x3 kernels
  - ReLU activation
  - MaxPooling
  - Dropout for regularization
- Fully connected classifier with:
  - 512 -> 128 -> 10 neurons
  - Dropout between layers

## Training the Model

The model was trained on the MNIST dataset using:
- Data augmentation (rotation, translation)
- Adam optimizer
- Learning rate scheduling
- Early stopping

You can explore the training process in:
- `main.ipynb`: Jupyter notebook with the original training code and experiments
- `train.py`: Standalone script extracted from the notebook for retraining the model

To retrain the model:

```bash
python train.py
```

## Running the Application

### With Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the applications at:
# - Frontend: http://localhost (port 80)
# - Backend API: http://localhost/predict/ (proxied through Nginx)
```

### Without Docker

#### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Usage

1. Draw a digit (0-9) on the canvas
2. Click "Recognize" to get the model's prediction
3. Click "Clear" to reset the canvas

## API Endpoints

- `POST /predict/`: Accepts an image and returns the predicted digit
- `GET /health`: Health check endpoint

## Performance

The model achieves approximately 99.6% accuracy on the MNIST test set.

## Project Structure

- **backend/**: FastAPI backend serving the CNN model
- **frontend/**: Next.js frontend application with Tailwind CSS
- **nginx/**: Nginx configuration for proxying requests
- **train.py**: Script for training the CNN model on MNIST dataset
- **main.ipynb**: Jupyter notebook exploring the model and dataset

```
.
├── docker-compose.yaml   # Docker Compose configuration
├── main.ipynb            # Original training notebook
├── train.py              # Training script for the model
├── backend/
│   ├── server.py         # FastAPI server with model
│   ├── best_model.pt     # Trained CNN model
│   ├── Dockerfile        # Backend Dockerfile
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── src/              # Next.js application source code
│   ├── public/           # Static assets
│   ├── Dockerfile        # Frontend Dockerfile
│   └── package.json      # Frontend dependencies
└── nginx/
    └── nginx.conf        # Nginx configuration
```