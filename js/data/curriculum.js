const generatePlaceholder = (title) => ({
    title: title,
    content: `
        <h3>${title}</h3>
        <p>Detailed notes for <strong>${title}</strong> are currently being curated.</p>
        <p>Check back soon for:</p>
        <ul>
            <li>Comprehensive explanations</li>
            <li>Real-world examples</li>
            <li>Code snippets</li>
        </ul>
    `
});

window.curriculumData = [
    {
        id: "unit-1",
        title: "Unit 1: Python Programming - II",
        description: "Advanced Python concepts utilizing NumPy and Pandas for data manipulation.",
        color: "#f59e0b", // Amber
        learningOutcomes: [
            {
                title: "Recap of NumPy library",
                content: `
                    <h3>NumPy (Numerical Python)</h3>
                    <p><strong>NumPy</strong> is the fundamental package for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.</p>
                    
                    <h4>Key Features:</h4>
                    <ul>
                        <li><strong>ndarray</strong>: A fast and space-efficient multidimensional array providing vectorized arithmetic operations.</li>
                        <li><strong>Broadcasting</strong>: Functions that act on arrays element-wise without explicit loops.</li>
                        <li><strong>Linear Algebra</strong>: Built-in linear algebra, Fourier transform, and random number capabilities.</li>
                    </ul>

                    <div class="code-snippet">
                        <pre><code>import numpy as np

# Creating an array
arr = np.array([1, 2, 3, 4, 5])

# Vectorized operation (multiply all by 2)
print(arr * 2) 
# Output: [2 4 6 8 10]</code></pre>
                    </div>
                `
            },
            {
                title: "Recap of Pandas Library",
                content: `
                    <h3>Pandas</h3>
                    <p><strong>Pandas</strong> is an open-source library providing high-performance, easy-to-use data structures and data analysis tools for Python.</p>
                    
                    <h4>Core Data Structures:</h4>
                    <ul>
                        <li><strong>Series</strong>: One-dimensional labeled array capable of holding any data type.</li>
                        <li><strong>DataFrame</strong>: Two-dimensional labeled data structure with columns of potentially different types (like a spreadsheet).</li>
                    </ul>

                    <div class="code-snippet">
                        <pre><code>import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

# Load data into a DataFrame
df = pd.DataFrame(data)

print(df)</code></pre>
                    </div>
                `
            },
            {
                title: "Importing and Exporting Data between CSV Files and DataFrames",
                content: `
                    <h3>Importing & Exporting: The Data Bridge</h3>
                    <p>Think of the DataFrame as a middleman. You aren't just moving files; you're converting "Flat Text" into a "Labeled Grid."</p>
                    <ul>
                        <li><strong>In:</strong> <code>df = pd.read_csv('file.csv')</code></li>
                        <li><strong>Out:</strong> <code>df.to_csv('new_file.csv', index=False)</code> <em>(Note: Setting index to False prevents an extra unnamed column from appearing!)</em></li>
                    </ul>
                `
            },
            {
                title: "Handling missing value",
                content: `
                    <h3>Handling Missing Values: The "Gap" Strategy</h3>
                    <p>Missing data isn't just an empty space; it’s a decision-making point.</p>
                    <ul>
                        <li><strong>Detection:</strong> <code>df.isnull()</code> (A "heat map" showing holes in a fabric)</li>
                        <li><strong>Dropping (< 5% missing):</strong> <code>df.dropna()</code> (Cutting out the broken row)</li>
                        <li><strong>Imputation (> 5% missing):</strong> <code>df.fillna(value)</code> (Patching with Mean or Median)</li>
                    </ul>
                `
            },
            {
                title: "Linear Regression algorithm (**For Advanced Learners)",
                content: `
                    <h3>Linear Regression: The Line of Best Fit</h3>
                    <p>This isn't just drawing a line through dots. It’s about minimizing the "error" (the distance between the dots and the line).</p>
                    <p>The goal is to find the values for <strong>slope (m)</strong> and <strong>intercept (b)</strong> in the equation: <code>y = mx + b + ε</code></p>
                    <p>The algorithm uses <strong>Gradient Descent</strong> to "roll down" the hill until it finds the lowest point of error.</p>
                `
            }
        ],
        activities: [
            {
                title: "Apply the fundamental concepts of NumPy and Pandas",
                content: `
                    <h3>Applying NumPy and Pandas</h3>
                    <p>Think of NumPy as the engine (raw speed, numbers) and Pandas as the dashboard (labels, tables).</p>
                    <div class="code-snippet">
                        <pre><code>import numpy as np
import pandas as pd
arr = np.array([10, 20, 30])
df = pd.DataFrame({'Score': arr}, index=['Math', 'Science', 'Art'])</code></pre>
                    </div>
                `
            },
            {
                title: "Import and export data between CSV files and Pandas Data Frames",
                content: `
                    <h3>CSV Data Exchange</h3>
                    <p>When moving data between CSV and DataFrames, check the shape and head immediately after importing.</p>
                    <div class="code-snippet">
                        <pre><code>df = pd.read_csv('data.csv')
df.head()
df.to_csv('out.csv', index=False)</code></pre>
                    </div>
                `
            },
            {
                title: "Implement Linear Regression algorithm",
                content: `
                    <h3>Implementing Linear Regression</h3>
                    <p>We split the data to ensure the model actually learned and didn't just memorize.</p>
                    <div class="code-snippet">
                        <pre><code>from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)</code></pre>
                    </div>
                `
            }
        ]
    },
    {
        id: "unit-2",
        title: "Unit 2: Data Science Methodology",
        description: "An analytic approach to capstone projects, focusing on model validation and evaluation.",
        color: "#10b981", // Emerald
        learningOutcomes: [
            {
                title: "Introduction to Data Science Methodology",
                content: `
                    <h3>Introduction to Data Science Methodology</h3>
                    <p>The methodology is a circular, iterative process. The core purpose is to answer four questions:</p>
                    <ul>
                        <li>What is the problem? (Business Understanding)</li>
                        <li>What data do we need? (Data Requirements)</li>
                        <li>Does the data answer the question? (Data Preparation)</li>
                        <li>Can we predict the future? (Modeling/Evaluation)</li>
                    </ul>
                `
            },
            {
                title: "Steps for Data Science Methodology",
                content: `
                    <h3>10 Steps of Data Science Methodology</h3>
                    <p>A logical flow where skipping a step makes the whole model collapse.</p>
                    <ul>
                        <li><strong>Business Understanding & Analytic Approach</strong></li>
                        <li><strong>Data Requirements & Collection</strong></li>
                        <li><strong>Data Understanding & Preparation:</strong> Clean and format (takes 70-80% of time).</li>
                        <li><strong>Modeling & Evaluation:</strong> Run algorithms and check accuracy.</li>
                        <li><strong>Deployment & Feedback:</strong> Put the model into the real world and learn from it.</li>
                    </ul>
                `
            },
            {
                title: "Model Validation Techniques",
                content: `
                    <h3>Model Validation Techniques</h3>
                    <p>How do you know your model isn't just "memorizing" the answers? You need to hide some data from it during training.</p>
                    <ul>
                        <li><strong>Train/Test Split:</strong> The simplest method (e.g., 80% to train, 20% to test).</li>
                        <li><strong>K-Fold Cross-Validation:</strong> Divide the data into K "folds". The model trains K times, using a different fold as the test set each time to ensure reliability across all data.</li>
                    </ul>
                `
            },
            {
                title: "Model Performance- Evaluation Metrics",
                content: `
                    <h3>Model Performance: Evaluation Metrics</h3>
                    <p>A model's "accuracy" can be a lie if the data is imbalanced. We use different metrics:</p>
                    <h4>For Regression:</h4>
                    <ul>
                        <li><strong>MSE & RMSE:</strong> Error metrics indicating how far off predictions are.</li>
                        <li><strong>R-Squared:</strong> How much of the variation is explained by your variables.</li>
                    </ul>
                    <h4>For Classification:</h4>
                    <ul>
                        <li><strong>Precision:</strong> "Of all predicted positives, how many were actually positive?"</li>
                        <li><strong>Recall:</strong> "Of all actual positives, how many did we catch?"</li>
                        <li><strong>F1-Score:</strong> The balance between Precision and Recall.</li>
                    </ul>
                `
            }
        ],
        activities: [
            {
                title: "Integrate Data Science Methodology steps into the Capstone Project",
                content: `
                    <h3>Integrating Methodology</h3>
                    <p>A Capstone isn't just a script; it’s a story:</p>
                    <ul>
                        <li><strong>The Hook:</strong> Business Understanding (Problem Statement)</li>
                        <li><strong>The Mess:</strong> Data Preparation (Handling missing values/outliers)</li>
                        <li><strong>The Engine:</strong> Modeling (Trying at least 2 algorithms)</li>
                        <li><strong>The Proof:</strong> Evaluation (Testing metrics)</li>
                    </ul>
                `
            },
            {
                title: "Calculate MSE and RMSE values",
                content: `
                    <h3>Calculating Error Metrics</h3>
                    <div class="code-snippet">
                        <pre><code>from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"MSE: {mse:.2f} | RMSE: {rmse:.2f}")</code></pre>
                    </div>
                `
            },
            {
                title: "Calculate Precision, Recall, F1 score",
                content: `
                    <h3>Calculating Classification Metrics</h3>
                    <p>We use the Confusion Matrix trio via the classification report.</p>
                    <div class="code-snippet">
                        <pre><code>from sklearn.metrics import classification_report

# Gives precision, recall, f1-score and accuracy
print(classification_report(y_test, predictions))</code></pre>
                    </div>
                `
            }
        ]
    },
    {
        id: "unit-3",
        title: "Unit 3: Making Machines See",
        description: "Understanding Computer Vision, its process, applications, and challenges.",
        color: "#3b82f6", // Blue
        learningOutcomes: [
            {
                title: "How Machines See",
                content: `
                    <h3>How Machines See: The Digital Grid</h3>
                    <p>While we see colors and shapes, a computer sees a <strong>Matrix</strong>.</p>
                    <ul>
                        <li><strong>Pixels:</strong> Each image is broken into tiny squares.</li>
                        <li><strong>Grayscale:</strong> Each pixel is a number from 0 (Black) to 255 (White).</li>
                        <li><strong>Color (RGB):</strong> Each pixel is a stack of three numbers representing Red, Green, and Blue intensity.</li>
                    </ul>
                `
            },
            {
                title: "Working of Computer Vision",
                content: `
                    <h3>Working of Computer Vision</h3>
                    <p>The "magic" happens through <strong>Pattern Recognition</strong>. Modern CV uses <strong>Convolutional Neural Networks (CNNs)</strong>.</p>
                    <p>Think of a CNN like a series of filters:</p>
                    <ul>
                        <li><strong>Low-level filters:</strong> Detect simple edges (vertical, horizontal).</li>
                        <li><strong>Mid-level filters:</strong> Combine edges into shapes (circles, squares).</li>
                        <li><strong>High-level filters:</strong> Combine shapes into features (eyes, nose, wheels).</li>
                    </ul>
                `
            },
            {
                title: "Computer Vision Process",
                content: `
                    <h3>The Computer Vision Process</h3>
                    <p>A 4-step linear pipeline:</p>
                    <ol>
                        <li><strong>Image Acquisition:</strong> Capturing the data (Camera, Sensor, Video).</li>
                        <li><strong>Preprocessing:</strong> Cleaning the "noise." Resizing images, converting to grayscale, or enhancing contrast.</li>
                        <li><strong>Feature Extraction:</strong> The machine identifies the "defining marks" (e.g., the whiskers on a cat).</li>
                        <li><strong>Classification/Detection:</strong> The final decision—"This is a cat" or "This is a stop sign."</li>
                    </ol>
                `
            },
            {
                title: "Applications of Computer Vision",
                content: `
                    <h3>Applications of Computer Vision</h3>
                    <p>CV is transforming every industry:</p>
                    <ul>
                        <li><strong>Healthcare:</strong> MRI/X-ray analysis (Identifying tumors faster than human eyes).</li>
                        <li><strong>Automotive:</strong> Self-Driving Cars (Detecting pedestrians, lanes, and traffic lights).</li>
                        <li><strong>Security:</strong> Facial Recognition (Unlocking phones or identifying suspects).</li>
                        <li><strong>Retail:</strong> Amazon Go / Auto-checkout (Tracking what items a customer picks up).</li>
                    </ul>
                `
            },
            {
                title: "Challenges of Computer Vision",
                content: `
                    <h3>Challenges of Computer Vision</h3>
                    <p>Why is it so hard? Because machines are literal, and the real world is messy.</p>
                    <ul>
                        <li><strong>Variations in Lighting:</strong> A cat in the sun looks like a different "matrix" than a cat in the shadows.</li>
                        <li><strong>Viewpoint Variation:</strong> A chair from the top looks nothing like a chair from the side.</li>
                        <li><strong>Occlusion:</strong> If a person is behind a tree, the machine might only see a "half-human" and fail to recognize them.</li>
                        <li><strong>Scale:</strong> Is that a tiny toy car close up, or a real car far away?</li>
                    </ul>
                `
            }
        ],
        activities: [
            {
                title: "Explain computer vision and its significance",
                content: `
                    <h3>Computer Vision & Its Significance</h3>
                    <p>Computer Vision is the field of Artificial Intelligence that enables computers to derive meaningful information from digital images or videos.</p>
                    <p><strong>Why it matters:</strong> Humans see a "dog." Computers see a <strong>3D Tensor</strong> (height, width, color channels). CV automates tasks that previously required human sight—but at a scale and speed no human can match.</p>
                `
            },
            {
                title: "Binary Art - Recreating Images with 0s and 1s",
                content: `
                    <h3>Binary Art: Recreating Images</h3>
                    <p>In a <strong>Binary Image</strong>, every pixel is represented by a single bit: 0 = Black (Off), 1 = White (On).</p>
                    <p><strong>Thresholding:</strong> To turn a colorful world into Binary Art:</p>
                    <ol>
                        <li>Take a grayscale image (values 0–255).</li>
                        <li>Pick a "Threshold" (e.g., 127).</li>
                        <li>Any pixel > 127 becomes 1; any pixel < 127 becomes 0.</li>
                    </ol>
                `
            },
            {
                title: "Working with OpenCV",
                content: `
                    <h3>Working with OpenCV</h3>
                    <p>OpenCV is the industry-standard "Swiss Army Knife" for image processing. It treats images as NumPy arrays.</p>
                    <div class="code-snippet">
                        <pre><code>import cv2 # The standard import
# 1. Load the image
img = cv2.imread('image.jpg')
# 2. Convert to Grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 3. Apply Canny Edge Detection
edges = cv2.Canny(gray_img, 100, 200)
# 4. Show the result
cv2.imshow('Edges', edges)
cv2.waitKey(0)</code></pre>
                    </div>
                `
            }
        ]
    },
    {
        id: "unit-4",
        title: "Unit 4: Orange Data Mining",
        description: "Using Orange for visual programming in data mining and machine learning.",
        color: "#f97316", // Orange
        learningOutcomes: [
            {
                title: "What is Data Mining?",
                content: `
                    <h3>What is Data Mining?</h3>
                    <p>Data Mining is the process of discovering hidden patterns, correlations, and anomalies within large datasets to predict outcomes.</p>
                    <p>It is the intersection of Statistics, Machine Learning, and Database Systems. The Goal: Turning Raw Data → Information → Knowledge.</p>
                `
            },
            {
                title: "Introduction to Orange Data Mining Tool",
                content: `
                    <h3>Introduction to Orange Data Mining Tool</h3>
                    <p>Orange is an open-source, component-based software for data mining, machine learning, and data visualization. Built on Python libraries like NumPy and Scikit-learn.</p>
                    <ul>
                        <li><strong>Visual Programming:</strong> Connect "widgets" with lines (wires) without writing code.</li>
                        <li><strong>Interactive:</strong> Change a setting and downstream visualizations update instantly.</li>
                    </ul>
                `
            },
            {
                title: "Components of Orange",
                content: `
                    <h3>Components of Orange</h3>
                    <ul>
                        <li><strong>Widgets:</strong> Circular icons that perform specific tasks (e.g., File, Scatter Plot, Tree).</li>
                        <li><strong>Links (Wires):</strong> Lines connecting widgets, representing Data Flow.</li>
                        <li><strong>Canvas:</strong> The "Workflow" area to build your pipeline.</li>
                    </ul>
                `
            },
            {
                title: "Default Widget Catalogue",
                content: `
                    <h3>Default Widget Catalogue</h3>
                    <ul>
                        <li><strong>Data (White):</strong> Input & Cleaning (File, CSV Import, Select Columns)</li>
                        <li><strong>Visualize (Blue):</strong> Seeing the Data (Scatter Plot, Box Plot, Distributions)</li>
                        <li><strong>Model (Orange):</strong> The "Brains" (Linear Regression, k-NN, Random Forest)</li>
                        <li><strong>Evaluate (Green):</strong> Testing Success (Test and Score, Confusion Matrix, ROC Analysis)</li>
                        <li><strong>Unsupervised (Purple):</strong> Finding Patterns (Distances, K-Means)</li>
                    </ul>
                `
            }
        ],
        activities: [
            {
                title: "Load and visualize the Iris dataset",
                content: `
                    <h3>Load and visualize the Iris dataset</h3>
                    <p>The first step is moving from raw numbers to visual patterns.</p>
                    <ol>
                        <li>Use the "File" Widget to load the Iris dataset ("iris.tab").</li>
                        <li>Connect the File widget to a Scatter Plot.</li>
                        <li>Set X-axis to Petal Length and Y-axis to Petal Width. Notice how the "Setosa" species clusters away from the others.</li>
                    </ol>
                `
            },
            {
                title: "Use classification widgets",
                content: `
                    <h3>Use classification widgets</h3>
                    <p>Connect your data to learners to predict species:</p>
                    <ul>
                        <li><strong>Logistic Regression:</strong> A linear approach.</li>
                        <li><strong>k-Nearest Neighbors (k-NN):</strong> Classifies based on how close a point is to its "neighbors."</li>
                        <li><strong>Random Forest:</strong> A committee of decision trees that vote on the outcome.</li>
                    </ul>
                `
            },
            {
                title: "Evaluating the Classification Model",
                content: `
                    <h3>Evaluating the Classification Model</h3>
                    <p>Connect both Data and Model outputs to the "Test and Score" widget.</p>
                    <ul>
                        <li><strong>AUC (Area Under Curve):</strong> Closer to 1.0 is better for distinguishing classes.</li>
                        <li><strong>CA (Classification Accuracy):</strong> The percentage of correct guesses.</li>
                        <li><strong>F1 Score:</strong> Balance of Precision and Recall.</li>
                    </ul>
                    <p>Use the Confusion Matrix widget to see where the model got confused.</p>
                `
            }
        ]
    },
    {
        id: "unit-5",
        title: "Unit 5: Introduction to Big Data",
        description: "Concepts of Big Data, analytics, and mining data streams.",
        color: "#ec4899", // Pink
        learningOutcomes: [
            {
                title: "Introduction to Big Data",
                content: `
                    <h3>Introduction to Big Data</h3>
                    <p>Big Data refers to massive, complex datasets that traditional data-processing software cannot handle.</p>
                    <p>We no longer look just for "why" things happen (causation) but "what" is happening (patterns/correlations). Generated by GPS, social media, sensors, etc.</p>
                `
            },
            {
                title: "Types of Big Data",
                content: `
                    <h3>Types of Big Data</h3>
                    <ul>
                        <li><strong>Structured:</strong> Fixed schema; fits perfectly in rows and columns (SQL, Excel).</li>
                        <li><strong>Semi-Structured:</strong> Uses "tags" to separate elements (JSON, XML, HTML).</li>
                        <li><strong>Unstructured:</strong> No internal structure, hardest to analyze but makes up 80% of Big Data (Images, Videos, PDFs, Voice).</li>
                    </ul>
                `
            },
            {
                title: "Characteristics of Big Data (5 Vs)",
                content: `
                    <h3>Characteristics of Big Data (The 5 Vs)</h3>
                    <ul>
                        <li><strong>Volume:</strong> The sheer scale (Terabytes to Zettabytes).</li>
                        <li><strong>Velocity:</strong> The speed of accumulation (streaming real-time).</li>
                        <li><strong>Variety:</strong> Different formats.</li>
                        <li><strong>Veracity:</strong> The "truthiness" or quality of data.</li>
                        <li><strong>Value:</strong> Can we turn this data into a decision or profit?</li>
                    </ul>
                `
            },
            {
                title: "Big Data Analytics",
                content: `
                    <h3>Big Data Analytics</h3>
                    <p>The science of examining large datasets to uncover hidden patterns.</p>
                    <ol>
                        <li><strong>Descriptive:</strong> "What happened?"</li>
                        <li><strong>Diagnostic:</strong> "Why did it happen?"</li>
                        <li><strong>Predictive:</strong> "What will happen?"</li>
                        <li><strong>Prescriptive:</strong> "How can we make it happen?"</li>
                    </ol>
                `
            }
        ],
        activities: [
            {
                title: "Analyze future trends in Big Data",
                content: `
                    <h3>Future Trends in Big Data</h3>
                    <ul>
                        <li><strong>Edge AI & Computing:</strong> Processing locally on devices to reduce latency.</li>
                        <li><strong>Data Fabric & Mesh:</strong> Decentralized architecture treating data as a product.</li>
                        <li><strong>Generative AI Integration:</strong> Using RAG for real-time accurate answers on private data.</li>
                        <li><strong>Quantum Analytics:</strong> Speeding up complex problem solving.</li>
                    </ul>
                `
            },
            {
                title: "Mining Data Streams",
                content: `
                    <h3>Mining Data Streams</h3>
                    <p>Extracting knowledge from continuous, high-speed data records that can only be read once.</p>
                    <ul>
                        <li><strong>Single Pass:</strong> Algorithm learns as data flies by.</li>
                        <li><strong>Limited Memory:</strong> Store only summaries (Synopses).</li>
                        <li><strong>Concept Drift:</strong> Models must adapt as "rules" change over time.</li>
                        <li><strong>Real-time Response:</strong> Insight must be delivered immediately.</li>
                    </ul>
                `
            }
        ]
    },
    {
        id: "unit-6",
        title: "Unit 6: Understanding Neural Networks",
        description: "Structure, working, and types of Neural Networks.",
        color: "#8b5cf6", // Violet
        learningOutcomes: [
            {
                title: "Parts of a Neural Network",
                content: `
                    <h3>Anatomy of a Neural Network</h3>
                    <p>Neural networks are composed of layers of node, behaving like neurons in the human brain.</p>
                    
                    <h4>Key Components:</h4>
                    <ul>
                        <li><strong>Input Layer</strong>: Receives the initial data signal.</li>
                        <li><strong>Hidden Layers</strong>: Where the "magic" happens. Nodes here apply weights and biases to inputs.</li>
                        <li><strong>Output Layer</strong>: Produces the final result (prediction or classification).</li>
                        <li><strong>Weights & Biases</strong>: Parameters that the network "learns" during training.</li>
                    </ul>
                `
            },
            {
                title: "Working of a Neural Network",
                content: `
                    <h3>How it Works</h3>
                    <ol>
                        <li><strong>Forward Propagation</strong>: Data flows from input -> hidden -> output.</li>
                        <li><strong>Loss Calculation</strong>: The difference between the predicted output and actual target is calculated.</li>
                        <li><strong>Backpropagation</strong>: The error is sent back through the network to update weights.</li>
                    </ol>
                    <p><em>(Check out the interactive visualizer above to see this in action!)</em></p>
                `
            },
            {
                title: "Components of a Neural Network",
                content: `
                    <h3>Components of a Neural Network</h3>
                    <ul>
                        <li><strong>Neuron (Node):</strong> The basic unit that stores a value and performs a calculation. (Cell Body)</li>
                        <li><strong>Weights (w):</strong> Determines how much "importance" to give an input. (Synapse Strength)</li>
                        <li><strong>Bias (b):</strong> Adjusts flexibility allowing the neuron to activate even if inputs are low. (Threshold)</li>
                        <li><strong>Activation Function:</strong> A gatekeeper (like ReLU or Sigmoid) deciding if the signal passes.</li>
                    </ul>
                `
            },
            {
                title: "Types of Neural Networks",
                content: `
                    <h3>Types of Neural Networks</h3>
                    <ul>
                        <li><strong>Feedforward (ANN):</strong> Simplest type, one direction. Best for simple tabular data.</li>
                        <li><strong>Convolutional (CNN):</strong> Uses filters to scan grids. Best for Images.</li>
                        <li><strong>Recurrent (RNN):</strong> Has loops to remember previous inputs. Best for Sequences (Time-series, speech).</li>
                        <li><strong>Transformers:</strong> Uses "Attention" mechanism. Powers LLMs like GPT-4.</li>
                        <li><strong>Generative Adversarial (GAN):</strong> Two networks "fight" each other to generate realistic new data.</li>
                    </ul>
                `
            },
            {
                title: "Future of Neural Networks",
                content: `
                    <h3>Future of Neural Networks</h3>
                    <ul>
                        <li><strong>Neuromorphic Computing:</strong> Chips mimicking brain structure with low power.</li>
                        <li><strong>AI Agents:</strong> Networks acting autonomously using tools.</li>
                        <li><strong>Quantum-Neural Hybrids:</strong> Quantum computers speeding up training.</li>
                        <li><strong>Liquid Neural Networks:</strong> Adapting parameters after training for real-time environments.</li>
                    </ul>
                `
            }
        ],
        activities: [
            {
                title: "Build a TensorFlow model to convert Celsius to Fahrenheit",
                content: `
                    <h3>TensorFlow: Celsius to Fahrenheit</h3>
                    <p>A regression problem predicting continuous numbers.</p>
                    <div class="code-snippet">
                        <pre><code>import tensorflow as tf
import numpy as np

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer_0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius, fahrenheit, epochs=500, verbose=False)</code></pre>
                    </div>
                `
            },
            {
                title: "Classification problem using TensorFlow playground",
                content: `
                    <h3>TensorFlow Playground: Classification</h3>
                    <p>Visual sandbox strategy:</p>
                    <ul>
                        <li><strong>Features:</strong> Start with X1 and X2. For circular, add X1² and X2².</li>
                        <li><strong>Hidden Layers:</strong> 0-1 for linear, 2-3 for complex (spirals).</li>
                        <li><strong>Activation Function:</strong> Switch from Linear to ReLU or Tanh for curved boundaries.</li>
                        <li><strong>Loss Graph:</strong> Watch Test Loss. If it goes up, you are Overfitting.</li>
                    </ul>
                `
            }
        ]
    },
    {
        id: "unit-7",
        title: "Unit 7: Generative AI",
        description: "Generative vs Discriminative models, LLMs, and applications.",
        color: "#06b6d4", // Cyan
        learningOutcomes: [
            {
                title: "Introduction to Generative AI",
                content: `
                    <h3>Introduction to Generative AI</h3>
                    <p>GenAI uses unsupervised or semi-supervised machine learning to create new content (text, images, audio) that mimics human-made data.</p>
                    <p>The core shift is from AI that analyzes data (Discriminative) to AI that synthesizes it (Generative). We are moving toward <strong>Multimodal AI</strong> processing text, video, and audio simultaneously.</p>
                `
            },
            {
                title: "Generative vs Discriminative models",
                content: `
                    <h3>Generative vs Discriminative models</h3>
                    <p>Think of them as the <strong>Artist</strong> vs. the <strong>Judge</strong>:</p>
                    <ul>
                        <li><strong>Discriminative (Judge):</strong> Goal is to classify/label data. Learns the boundary between classes. "Is this a 0 or a 1?" (e.g., Logistic Regression, CNN).</li>
                        <li><strong>Generative (Artist):</strong> Goal is to create new data samples. Learns the distribution of the data. "What does a 0 usually look like?" (e.g., GANs, Transformers).</li>
                    </ul>
                `
            },
            {
                title: "LLM- Large Language Model",
                content: `
                    <h3>Large Language Models (LLMs)</h3>
                    <p>LLMs are massive statistical engines that predict the next token (word) in a sequence.</p>
                    <p>They rely on the <strong>Transformer architecture</strong> using a mechanism called <strong>Self-Attention</strong>, which allows the model to "attend" to every word in a sentence at once to understand context.</p>
                `
            },
            {
                title: "Ethical and Social Implications",
                content: `
                    <h3>Ethical and Social Implications</h3>
                    <ul>
                        <li><strong>Hallucinations:</strong> GenAI models are probability machines and can confidently state falsehoods as truth.</li>
                        <li><strong>Bias & Fairness:</strong> Models amplify prejudices found in training data.</li>
                        <li><strong>Deepfakes & Misinformation:</strong> Hyper-realistic fake media disrupting social trust.</li>
                        <li><strong>Environmental Impact:</strong> Massive power consumption during training.</li>
                    </ul>
                `
            }
        ],
        activities: [
            {
                title: "Use the Gemini API to design a chatbot",
                content: `
                    <h3>Designing a Chatbot with the Gemini API</h3>
                    <p>A professional chatbot is a managed state with Memory, using the exact chat loop pattern:</p>
                    <div class="code-snippet">
                        <pre><code>from google import genai
import os

client = genai.Client(api_key="YOUR_API_KEY")
chat_session = client.chats.create(model="gemini-3-flash")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]: break
    response = chat_session.send_message(user_input)
    print(f"Gemini Bot: {response.text}")</code></pre>
                    </div>
                `
            },
            {
                title: "Prompt Engineering with Google Gemini",
                content: `
                    <h3>Prompt Engineering</h3>
                    <p>Use the <strong>"Bento-Box" Framework</strong> with XML-style tags to structure the context:</p>
                    <ul>
                        <li><code>&lt;persona&gt;</code> Sets the expertise level</li>
                        <li><code>&lt;task&gt;</code> Specific action</li>
                        <li><code>&lt;data&gt;</code> Raw input</li>
                        <li><code>&lt;limit&gt;</code> Constraints</li>
                    </ul>
                    <p><strong>Advanced Techniques:</strong> Chain-of-Thought (forcing step-by-step logic) and Few-Shot Prompting (giving examples of the exact output format).</p>
                `
            }
        ]
    },
    {
        id: "unit-8",
        title: "Unit 8: Data Storytelling",
        description: "The art of telling stories through data visualization and narrative.",
        color: "#f43f5e", // Rose
        learningOutcomes: [
            {
                title: "Introduction to Storytelling",
                content: `
                    <h3>Introduction to Data Storytelling</h3>
                    <p>The practice of building a narrative around data and its visualizations to convey meaning powerfully.</p>
                    <p>The "Magic Trio":</p>
                    <ul>
                        <li><strong>Data:</strong> The raw evidence (the "What").</li>
                        <li><strong>Visuals:</strong> The clarity (the "Aha!").</li>
                        <li><strong>Narrative:</strong> The context (the "Why it matters").</li>
                    </ul>
                `
            },
            {
                title: "Narrative Structure (Freytag’s Pyramid)",
                content: `
                    <h3>Narrative Structure (Freytag’s Pyramid)</h3>
                    <ol>
                        <li><strong>Exposition:</strong> Setting the scene (The Status Quo).</li>
                        <li><strong>Inciting Incident:</strong> The problem starts (The Pivot).</li>
                        <li><strong>Rising Action:</strong> Building tension (The Investigation).</li>
                        <li><strong>Climax:</strong> The turning point (The Insight/Aha! Moment).</li>
                        <li><strong>Falling Action:</strong> Result of the climax (The Validation/Fix).</li>
                        <li><strong>Resolution:</strong> Tying up loose ends (The Recommendation).</li>
                    </ol>
                `
            },
            {
                title: "Ethics in Data Storytelling",
                content: `
                    <h3>Ethics in Data Storytelling</h3>
                    <ul>
                        <li><strong>Avoid Cherry-Picking:</strong> Don't just show data points that support your theory.</li>
                        <li><strong>Avoid Truncated Y-Axis:</strong> Don't start a bar chart at 50 instead of 0 to make small differences look massive.</li>
                        <li><strong>Causation vs. Correlation:</strong> Don't imply A caused B just because graphs look similar.</li>
                        <li><strong>Ensure Transparency & Accuracy:</strong> Always cite sources, mention missing data, and use proper scales.</li>
                    </ul>
                `
            }
        ],
        activities: [
            {
                title: "Create an effective data story using given data",
                content: `
                    <h3>Creating a Data Story</h3>
                    <p>Follow the "3-Second Rule": If the audience can’t understand the main takeaway of a slide in 3 seconds, the visual is too complex.</p>
                    <p>Identify the "Protagonist" (the main metric), find the "Antagonist" (the problem/outlier), and write the Climax (the core insight) ending with a clear Call to Action.</p>
                `
            }
        ]
    }
];
