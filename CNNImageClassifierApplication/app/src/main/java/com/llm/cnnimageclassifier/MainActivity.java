package com.llm.cnnimageclassifier;

import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.util.Log;

import android.Manifest;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import com.google.common.util.concurrent.ListenableFuture;

import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import android.widget.Button;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import androidx.camera.core.CameraSelector;
import androidx.exifinterface.media.ExifInterface;


public class MainActivity extends AppCompatActivity {

    private static final String LENET_BASELINE_MODEL = "lenet5_baseline2.pt";
    private static final String LENET_BEST_MODEL = "lenet5_pruned_quant.pt";
    private static final String RESNET_BASELINE_MODEL = "resnet50_baseline.pt";
    private static final String RESNET_BEST_MODEL = "resnet50_structured_sparse40.pt";

    private Module model;
    private String currentModelPath = LENET_BASELINE_MODEL; // Default model
    private TextView resultText;
    private TextView inferenceText;
    private ImageView imageView;
    private RadioGroup modelSelector;
    private RadioGroup typeSelector;

    private static final String TAG = "MainActivity";  // For logging purposes

    private ExecutorService cameraExecutor;
    private ImageCapture imageCapture;
    private boolean cameraReady = false;  // Flag to track camera readiness

    private final List<String> IMAGENET_CLASSES = new ArrayList<>();
    private Map<String, String> IMAGENET_CLASS_MAP = new HashMap<>();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultText = findViewById(R.id.resultText);  // TextView to show the result
        imageView = findViewById(R.id.imageView);    // ImageView to display the image
        inferenceText = findViewById(R.id.inferenceTime);

        try {
            Log.d(TAG, "Loading the model...");
            model = Module.load(assetFilePath(currentModelPath));
            Log.d(TAG, "Model loaded successfully");
            loadClassLabels();
            loadImageNetIDs();
            // Check if the app has the camera permission
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 100);
            } else {
                // Start Camera
                startCamera();
            }

            // Handle Radio Button Selection
            modelSelector = findViewById(R.id.modelSelector);
            typeSelector = findViewById(R.id.typeSelector);

            RadioGroup.OnCheckedChangeListener modelChangeListener = (group, checkedId) -> updateModel();
            RadioGroup.OnCheckedChangeListener typeChangeListener = (group, checkedId) -> updateModel();

            // Set listeners
            modelSelector.setOnCheckedChangeListener(modelChangeListener);
            typeSelector.setOnCheckedChangeListener(typeChangeListener);

            // Button to take a picture
            Button captureButton = findViewById(R.id.captureButton);
            captureButton.setOnClickListener(v -> {
                if (cameraReady) {
                    takePhoto();  // Take photo only if camera is ready
                } else {
                    Log.d(TAG, "Camera is not ready yet.");
                }
            });

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Error loading the model", e);
        }
    }

    // Function to handle both changes
    private void updateModel() {
        int selectedModel = modelSelector.getCheckedRadioButtonId();
        int selectedType = typeSelector.getCheckedRadioButtonId();

        Log.d(TAG, "Model Changed: " + selectedModel + ", Type Changed: " + selectedType);

        if (selectedModel == R.id.lenet) {
            if (selectedType == R.id.best) {
                currentModelPath = LENET_BEST_MODEL;
            } else if (selectedType == R.id.baseline) {
                currentModelPath = LENET_BASELINE_MODEL;
            }
        } else if (selectedModel == R.id.resnet) {
            if (selectedType == R.id.best) {
                currentModelPath = RESNET_BEST_MODEL;
            } else if (selectedType == R.id.baseline) {
                currentModelPath = RESNET_BASELINE_MODEL;
            }
        }

        loadModel(currentModelPath);
    }

    private void loadModel(String modelPath) {
        resultText.setText("Predicted Class: ");
        try {
            Log.d(TAG, "Loading model: " + modelPath);
            model = Module.load(assetFilePath(modelPath));
            Log.d(TAG, "Model loaded successfully: " + modelPath);
        } catch (IOException e) {
            Log.e(TAG, "Error loading model: " + modelPath, e);
        }
    }

    private void startCamera() {
        // Set up CameraX to capture images
        cameraExecutor = Executors.newSingleThreadExecutor();

        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                imageCapture = new ImageCapture.Builder().build();

                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture);

                preview.setSurfaceProvider(((PreviewView) findViewById(R.id.previewView)).getSurfaceProvider());
                ((PreviewView) findViewById(R.id.previewView)).setScaleType(PreviewView.ScaleType.FIT_CENTER);
                cameraReady = true;

            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void takePhoto() {
        resultText.setText("Predicted Class: ");
        if (imageCapture == null) {
            Log.e(TAG, "ImageCapture is not initialized.");
            return;
        }

        File photoFile = new File(getExternalFilesDir(null), "photo.jpg");

        ImageCapture.OutputFileOptions outputOptions = new ImageCapture.OutputFileOptions.Builder(photoFile).build();
        imageCapture.takePicture(outputOptions, cameraExecutor, new ImageCapture.OnImageSavedCallback() {
            @Override
            public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                Log.d(TAG, "Image captured successfully");

                long startTime = System.nanoTime();
                Bitmap bitmap = BitmapFactory.decodeFile(photoFile.getAbsolutePath());

                Bitmap rotatedBitmap = rotateImageIfRequired(bitmap, photoFile.getAbsolutePath());

                runOnUiThread(() -> {
                    imageView.setImageBitmap(rotatedBitmap);

                    // Preprocess and classify image
                    Bitmap processedBitmap;
                    if (currentModelPath.contains("lenet")) {
                        Log.d(TAG, "processing LENET Bitmap ");
                        processedBitmap = preprocessImageForLeNet(rotatedBitmap);
                    } else {
                        Log.d(TAG, "processing RESNET Bitmap ");
                        processedBitmap = preprocessImageForResNet(rotatedBitmap);
                    }

                    Tensor inputTensor = bitmapToTensor(processedBitmap);

                    Log.d(TAG, "Input tensor shape: " + inputTensor.shape());
                    long inferenceStartTime = System.nanoTime();
                    Log.d(TAG, "Running inference on the model...");

                    Tensor output = model.forward(IValue.from(inputTensor)).toTensor();
                    float[] scores = output.getDataAsFloatArray();
                    int predictedClass = argmax(scores);
                    Log.d(TAG, "System.nanoTime(: " + System.nanoTime());
                    double inferenceEndTime = System.nanoTime();
                    double totalInferenceTimeMs = (inferenceEndTime - inferenceStartTime) /  1_000_000.0; // Convert to milliseconds
                    double totalProcessingTimeMs = (inferenceEndTime - startTime) / 1_000_000.0; // Full processing time

                    Log.d(TAG, "Predicted Class: " + predictedClass);
                    if (currentModelPath.contains("lenet")) {
                    resultText.setText("Predicted Class: " + predictedClass);}
                    else{
                        // Get top 3 predictions
                        List<Integer> top3Indices = getTopNIndices(scores, 3);

                        // Display top 3 predicted classes and their corresponding labels
                        StringBuilder top3Result = new StringBuilder();
                        for (int i = 0; i < 3; i++) {
                            int newpredictedClass = top3Indices.get(i);
                            top3Result.append("Top " + (i + 1) + ": ").append(newpredictedClass).append(" - ")
                                    .append(getClassLabel(newpredictedClass)).append("\n");
                        }

                        Log.d(TAG, top3Result.toString());
                        resultText.setText("Top 3 Result: " + top3Result.toString());
                    }
                    Log.d(TAG, "Inference Time: " + totalInferenceTimeMs + " ms");
                    Log.d(TAG, "Total Processing Time: " + totalProcessingTimeMs + " ms");

                    inferenceText.setText("Inference Time: "+ totalInferenceTimeMs + " ms");
                });
            }

            // Helper function to get class label
            private String getClassLabel(int classIndex) {
                Log.e(TAG, "classIndex" + (classIndex-1));
                if (classIndex-1 >= 0 && classIndex-1 < IMAGENET_CLASSES.size()) {
                    Log.e(TAG, "IMAGENET_CLASSES NAME" + IMAGENET_CLASSES.get(classIndex-1));
                    return IMAGENET_CLASS_MAP.getOrDefault(IMAGENET_CLASSES.get(classIndex-1), "Unknown Class").split(",")[0];
                } else {
                    return "Unknown Class"; // In case the index is out of bounds
                }

            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e(TAG, "Error capturing image", exception);
            }
        });
    }

    private Bitmap preprocessImageForLeNet(Bitmap bitmap) {
        Log.d(TAG, "Applying transformations to image for LeNet");
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true);
        Bitmap grayscaleBitmap = toGrayscale(resizedBitmap);
        return normalizeForLeNet(grayscaleBitmap);
    }

    private Bitmap preprocessImageForResNet(Bitmap bitmap) {
        Log.d(TAG, "Applying transformations to image for ResNet");

        // Resize to 224x224
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

        // Convert grayscale image to RGB (for ResNet)
        return convertToRGB(resizedBitmap); // Convert to RGB for ResNet
    }


    private Bitmap normalizeForLeNet(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Bitmap normalizedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int pixel = bitmap.getPixel(i, j);
                int gray = Color.red(pixel); // Grayscale image, red, green, and blue are same
                float normalizedPixel = (gray / 255.0f - 0.5f) / 0.5f;
                int normalizedColor = Color.rgb((int) (normalizedPixel * 255), (int) (normalizedPixel * 255), (int) (normalizedPixel * 255));
                normalizedBitmap.setPixel(i, j, normalizedColor);
            }
        }
        return normalizedBitmap;
    }

    private Bitmap toGrayscale(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Bitmap grayscaleBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int pixel = bitmap.getPixel(i, j);
                int red = Color.red(pixel);
                int green = Color.green(pixel);
                int blue = Color.blue(pixel);
                int gray = (int) (0.299 * red + 0.587 * green + 0.114 * blue);
                grayscaleBitmap.setPixel(i, j, Color.rgb(gray, gray, gray));
            }
        }
        return grayscaleBitmap;
    }

    private Bitmap convertToRGB(Bitmap grayscaleBitmap) {
        int width = grayscaleBitmap.getWidth();
        int height = grayscaleBitmap.getHeight();
        Bitmap rgbBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

        // Convert each pixel to RGB format
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int pixel = grayscaleBitmap.getPixel(i, j);
                int gray = Color.red(pixel);  // Use grayscale value for R, G, B
                rgbBitmap.setPixel(i, j, Color.rgb(gray, gray, gray)); // Set RGB with same value
            }
        }

        return rgbBitmap;
    }

    private Tensor bitmapToTensor(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        // Determine if it's for LeNet or ResNet based on the model
        if (currentModelPath.contains("lenet")) {
            // Preprocessing for LeNet (28x28 and grayscale)
            return preprocessForLeNet(bitmap);
        } else {
            // Preprocessing for ResNet (224x224 and RGB)
            return preprocessForResNet(bitmap);
        }
    }

    private Tensor preprocessForLeNet(Bitmap bitmap) {
        // Resize the image to 28x28 for LeNet
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true);
        Bitmap grayscaleBitmap = toGrayscale(resizedBitmap); // Convert to grayscale
        return bitmapToTensorForLeNet(grayscaleBitmap);
    }

    private Tensor preprocessForResNet(Bitmap bitmap) {
        // Resize the image to 224x224 for ResNet
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        // Convert to RGB for ResNet
        return bitmapToTensorForResNet(resizedBitmap);
    }

    private Tensor bitmapToTensorForLeNet(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] tensorData = new float[width * height]; // For grayscale image
        int index = 0;
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int gray = Color.red(pixel); // Grayscale image, so use red value for all channels
            // Normalize pixel values to [-1, 1]
            tensorData[index++] = (gray / 255.0f - 0.5f) / 0.5f;
        }

        // Create the tensor with the shape [1, 1, 28, 28] for LeNet (grayscale image)
        return Tensor.fromBlob(tensorData, new long[]{1, 1, height, width});
    }

    private Tensor bitmapToTensorForResNet(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] tensorData = new float[width * height * 3]; // For RGB (3 channels)
        int index = 0;
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int r = Color.red(pixel);
            int g = Color.green(pixel);
            int b = Color.blue(pixel);

            // Normalize RGB values to [-1, 1] for ResNet
            tensorData[index++] = (r / 255.0f - 0.5f) / 0.5f;
            tensorData[index++] = (g / 255.0f - 0.5f) / 0.5f;
            tensorData[index++] = (b / 255.0f - 0.5f) / 0.5f;
        }

        // Create the tensor with the shape [1, 3, 224, 224] for ResNet (RGB image)
        return Tensor.fromBlob(tensorData, new long[]{1, 3, height, width});
    }

    private int argmax(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private String assetFilePath(String assetName) throws IOException {
        File file = new File(getFilesDir(), assetName);
        if (!file.exists()) {
            try (InputStream is = getAssets().open(assetName);
                 FileOutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
            }
        }
        return file.getAbsolutePath();
    }

    private Bitmap rotateImageIfRequired(Bitmap img, String imagePath) {
        try {
            ExifInterface exif = new ExifInterface(imagePath);
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);

            Matrix matrix = new Matrix();

            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    matrix.postRotate(90);
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    matrix.postRotate(180);
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    matrix.postRotate(270);
                    break;
                case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                    matrix.postScale(-1, 1);
                    break;
                case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                    matrix.postScale(1, -1);
                    break;
                default:
                    // No rotation needed
                    return img;
            }

            return Bitmap.createBitmap(img, 0, 0, img.getWidth(), img.getHeight(), matrix, true);
        } catch (IOException e) {
            e.printStackTrace();
            return img;  // Return the original image if there's an error
        }
    }

    private void loadClassLabels() {

        try (InputStream is = getAssets().open("imagenet_class_labels.txt")) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            String line;
            while ((line = reader.readLine()) != null) {
                // Split each line by spaces (first part is ID, rest is the label)
                String[] parts = line.split("\\s+", 2); // Split into max 2 parts
                if (parts.length > 1) {
                    String id = parts[0];     // The unique ImageNet ID (e.g., n03444034)
                    String label = parts[1];  // The label (e.g., "entity")
                    IMAGENET_CLASS_MAP.put(id, label); // Store in map
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Log to check the labels are correctly loaded
        Log.d(TAG, "Class Labels Loaded: " + IMAGENET_CLASS_MAP);
    }

    private void loadImageNetIDs() {

        try (InputStream is = getAssets().open("imagenet_ids.txt")) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            String line;
            while ((line = reader.readLine()) != null) {
                IMAGENET_CLASSES.add(line.trim()); // Trim to remove extra spaces or newlines
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Log to check that the IDs are correctly loaded
        Log.d(TAG, "ImageNet IDs Loaded: " + IMAGENET_CLASSES);
    }

    // Helper function to get top N indices based on scores
    private List<Integer> getTopNIndices(float[] scores, int topN) {
        List<Integer> topIndices = new ArrayList<>();
        float[] sortedScores = scores.clone();
        Arrays.sort(sortedScores); // Sort the scores

        for (int i = 0; i < topN; i++) {
            float score = sortedScores[sortedScores.length - 1 - i]; // Get the highest score
            for (int j = 0; j < scores.length; j++) {
                if (scores[j] == score && !topIndices.contains(j)) {
                    topIndices.add(j);
                    break;
                }
            }
        }
        return topIndices;
    }

}
