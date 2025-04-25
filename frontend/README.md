# MNIST Digit Recognizer Frontend

A minimalist Next.js application for recognizing handwritten digits using a CNN model.

## Features

- Clean, modern UI with Tailwind CSS
- Interactive canvas for drawing digits
- Real-time digit recognition through the backend API
- Confidence score visualization for predictions

## Getting Started

### Development

```bash
# Install dependencies
npm install

# Run the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
# Build for production
npm run build

# Start the production server
npm start
```

### Docker

```bash
# Build the Docker image
docker build -t mnist-frontend .

# Run the container
docker run -p 3000:3000 mnist-frontend
```

## Demo

![MNIST Digit Recognizer Demo](../media/mnist_demo.gif)

The interface allows users to:
- Draw digits on a responsive canvas
- Get real-time predictions from the model
- See confidence scores for each possible digit (0-9)
- Clear and redraw as needed

## API Integration

The application connects to a FastAPI backend that hosts the CNN model for digit recognition. 

In development mode, it expects the backend to be running at `http://localhost:8000`, but in production, the requests are proxied through Nginx which routes them to the backend service.

## Production Setup with Nginx

In the production environment, the application is served through Nginx which:

1. Serves the frontend on port 80
2. Proxies API requests from `/predict/` to the backend service
3. Handles CORS and other HTTP settings

This setup is configured in the root `docker-compose.yaml` file and `nginx/nginx.conf`.

## Technologies Used

- Next.js 13
- React
- TypeScript
- Tailwind CSS 