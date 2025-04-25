import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import torch.nn as nn


# Define the model structure to match the saved model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x


# Load model
model = CNNModel()
model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu")))
model.eval()

# Set up FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Transformation
transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        logits = outputs[0]
        relu_logits = F.relu(logits)  # relu

        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)[0]
        prediction = outputs.argmax(dim=1).item()

        # Convert tensor values to Python float for JSON serialization
        probs = [float(p) for p in probabilities.tolist()]
        processed_logits = [float(l) for l in relu_logits.tolist()]

        # Calculate min and max logit values for proper scaling in frontend
        min_logit = min(processed_logits)
        max_logit = max(processed_logits)

    return {
        "prediction": prediction,
        "probabilities": probs,
        "logits": processed_logits,
        "min_logit": min_logit,
        "max_logit": max_logit,
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
