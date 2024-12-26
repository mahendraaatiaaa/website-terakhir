// URL model ONNX
const MODEL_URL = './model3 copy.onnx';

// DOM Elements
const imageUpload = document.getElementById('imageUpload');
const classifyButton = document.getElementById('classifyButton');
const imageCanvas = document.getElementById('imageCanvas');
const resultDiv = document.getElementById('result');

// Variables
let model;

// Load the ONNX model
async function loadModel() {
    try {
        model = await ort.InferenceSession.create(MODEL_URL);
        console.log('Model loaded successfully');

        // Log input dan output names
        console.log('Input Names:', model.inputNames); // ['input']
        console.log('Output Names:', model.outputNames); // ['output']

        classifyButton.disabled = false; // Enable the classify button
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

// Preprocess image to match model input
function preprocessImage(image) {
    const canvas = imageCanvas;
    const ctx = canvas.getContext('2d');

    // Set canvas size and draw image
    canvas.width = 177;
    canvas.height = 177;
    ctx.drawImage(image, 0, 0, 177, 177);

    // Extract image data and normalize
    const imageData = ctx.getImageData(0, 0, 177, 177);
    const { data } = imageData;
    const input = new Float32Array(3 * 177 * 177);

    for (let i = 0; i < data.length; i += 4) {
        const pixelIndex = i / 4;
        input[pixelIndex] = data[i] / 255.0;        // R
        input[pixelIndex + 177 * 177] = data[i + 1] / 255.0; // G
        input[pixelIndex + 2 * 177 * 177] = data[i + 2] / 255.0; // B
    }

    return new ort.Tensor('float32', input, [1, 3, 177, 177]); // Batch size = 1
}

// Classify image
async function classifyImage() {
    resultDiv.textContent = 'Classifying...';

    try {
        const image = new Image();
        image.src = URL.createObjectURL(imageUpload.files[0]);
        image.onload = async () => {
            const tensor = preprocessImage(image);
            console.log('Tensor:', tensor);

            // Periksa input dan output model
            const feeds = { input: tensor }; // Nama input dari model
            const output = await model.run(feeds);
            console.log('Model output:', output);

            // Ambil data output dari nama output
            const probabilities = output.output.data; // Nama output dari model
            const classNames = ['anggur', 'apel', 'belimbing', 'jeruk', 'kiwi', 'mangga', 'nanas', 'pisang', 'semangka', 'stroberi'];

            // Temukan kelas dengan probabilitas tertinggi
            const maxIndex = probabilities.indexOf(Math.max(...probabilities));
            resultDiv.textContent = `Predicted Class: ${classNames[maxIndex]} (${(probabilities[maxIndex] * 100).toFixed(2)}%)`;
        };
    } catch (error) {
        console.error('Error classifying image:', error);
        resultDiv.textContent = 'Error classifying image. Check console for details.';
    }
}

// Event Listeners
imageUpload.addEventListener('change', () => {
    const file = imageUpload.files[0];

    if (file) {
        // Display the image preview
        const reader = new FileReader();
        reader.onload = function(e) {
            const imagePreview = document.getElementById('imagePreview');
            imagePreview.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Enable the classify button
        classifyButton.disabled = false;
    } else {
        classifyButton.disabled = true;
    }
});

classifyButton.addEventListener('click', classifyImage);

// Load model on page load
loadModel();
