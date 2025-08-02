"use client"; // This is required for components with hooks like useState and useEffect

import { useState, useRef, useEffect } from 'react';

// --- DrawingCanvas Component ---
// This component encapsulates all the drawing logic.
const DrawingCanvas = ({ onPredict, isLoading }) => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // Initialize the canvas with a black background
  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.fillStyle = 'black';
    context.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  // Function to get correct coordinates for both mouse and touch events
  const getCoordinates = (event) => {
    if (event.touches && event.touches.length > 0) {
      const rect = canvasRef.current.getBoundingClientRect();
      return { offsetX: event.touches[0].clientX - rect.left, offsetY: event.touches[0].clientY - rect.top };
    }
    return { offsetX: event.nativeEvent.offsetX, offsetY: event.nativeEvent.offsetY };
  };

  const startDrawing = (event) => {
    const { offsetX, offsetY } = getCoordinates(event);
    const context = canvasRef.current.getContext('2d');
    context.strokeStyle = 'white';
    context.lineWidth = 18; // Slightly thicker line for better recognition
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.beginPath();
    context.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const draw = (event) => {
    if (!isDrawing) return;
    // Prevents scrolling on mobile while drawing
    event.preventDefault(); 
    const { offsetX, offsetY } = getCoordinates(event);
    const context = canvasRef.current.getContext('2d');
    context.lineTo(offsetX, offsetY);
    context.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.fillStyle = 'black';
    context.fillRect(0, 0, canvas.width, canvas.height);
    // Also clear the prediction in the parent component
    onPredict(null, true); 
  };

  const handlePredictClick = () => {
    onPredict(canvasRef.current, false);
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <canvas
        ref={canvasRef}
        width={420}
        height={420}
        className="bg-black rounded-2xl shadow-2xl cursor-crosshair border-2 border-cyan-500/50"
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
      />
      <div className="flex w-full justify-center gap-4">
        <button
          onClick={handlePredictClick}
          className="w-1/2 px-6 py-3 bg-cyan-600 text-white font-semibold rounded-lg shadow-lg hover:bg-cyan-700 transition-transform transform hover:scale-105 disabled:bg-gray-500 disabled:cursor-not-allowed"
          disabled={isLoading}
        >
          {isLoading ? 'Thinking...' : 'Predict'}
        </button>
        <button
          onClick={clearCanvas}
          className="w-1/2 px-6 py-3 bg-red-600 text-white font-semibold rounded-lg shadow-lg hover:bg-red-700 transition-transform transform hover:scale-105"
        >
          Clear
        </button>
      </div>
    </div>
  );
};


// --- Main Home Component ---
export default function Home() {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async (canvas, isClearing) => {
    // If the clear button was pressed, just reset the state
    if (isClearing) {
        setPrediction(null);
        setError(null);
        return;
    }

    setIsLoading(true);
    setPrediction(null);
    setError(null);

    // 1. Create a temporary canvas to downscale the image to 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    // 2. Draw the large canvas image onto the small one
    // --- BUG FIX: Use the canvas's actual width and height ---
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
    
    // 3. Get the pixel data from the small canvas
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const pixelData = imageData.data;
    
    // 4. Process the pixel data to match the model's input format
    const processedData = [];
    for (let i = 3; i < pixelData.length; i += 4) { // We use the alpha channel
      processedData.push(pixelData[i] / 255.0);
    }

    // 5. Send the data to the API
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: processedData }),
      });

      if (!response.ok) {
        const errorBody = await response.json();
        throw new Error(errorBody.detail || 'Network response was not ok');
      }

      const result = await response.json();
      setPrediction(result.prediction);
    } catch (err) {
      console.error("Failed to fetch prediction:", err);
      setError("Could not connect to the model. Is the API running?");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="bg-gray-900 text-white flex min-h-screen flex-col items-center justify-center gap-8 p-8 font-sans bg-gradient-to-br from-gray-900 via-gray-800 to-black">
      <div className="text-center">
        <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
          Handwritten Digit Classifier
        </h1>
        <p className="text-lg text-gray-400 mt-2">
          Draw a digit from 0 to 9 and see the prediction from a model built with Python & NumPy.
        </p>
      </div>

      <div className="flex flex-col lg:flex-row items-center justify-center gap-12">
        <DrawingCanvas onPredict={handlePredict} isLoading={isLoading} />
        
        <div className="w-64 h-64 bg-gray-800/50 rounded-2xl flex items-center justify-center border-2 border-gray-700 shadow-2xl relative">
          {/* Glowing effect */}
          {prediction !== null && <div className="absolute inset-0 bg-cyan-500/20 blur-2xl animate-pulse rounded-2xl"></div>}
          
          <div className="text-center z-10">
            {isLoading && <p className="text-gray-400 text-2xl">Analyzing...</p>}
            {error && <p className="text-red-400 text-center p-4">{error}</p>}
            {prediction !== null && (
              <>
                <p className="text-gray-400 text-2xl">Prediction</p>
                <p className="text-9xl font-bold text-cyan-300">{prediction}</p>
              </>
            )}
            {!isLoading && !error && prediction === null && (
                <p className="text-gray-500 text-2xl">Draw a digit</p>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
