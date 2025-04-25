# MNIST CNN Fullstack Application

A full-stack application with a CNN model trained on MNIST data for digit recognition.

> **Note:** This is my first machine learning project using Convolutional Neural Networks (not counting previous work with Artificial Neural Networks).

## Architecture

- **Frontend**: Next.js application with React and Tailwind CSS featuring a drawing canvas
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


## Demo

![MNIST Digit Recognizer Demo](media/mnist.gif)

The MNIST Digit Recognizer correctly identifies handwritten digits from 0-9

- Displaying confidence scores for all possible digits
- Real-time processing with the CNN model

This is a great way to test my trained model and get feedback on how I can improve it in the future.

## Running the Application with Docker 

```bash
# Build and run with Docker Compose
docker compose up -d
```

Once the application is running, access it at http://localhost in your browser.

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
- **media/**: Contains demonstration GIFs and images