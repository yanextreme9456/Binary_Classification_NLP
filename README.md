📚 Binary Classification with NLP (IMDB Reviews)

This project demonstrates how to perform binary classification on natural language data using the IMDB dataset. It uses Keras with a simple feedforward neural network to predict whether a movie review is positive or negative.

The neural network is trained with different architectures and optimizers, and its performance is visualized through loss and accuracy plots.

🚀 Features

-Loads and preprocesses the IMDB dataset
-One-hot encodes the input data for neural network processing
-Builds and trains different neural network models for binary classification
-Visualizes training and validation loss/accuracy
-Evaluates the model on test data
-Generates predictions (probabilities of positive reviews)
-Tests different architectures and loss functions for performance improvement

🛠️ Technologies Used

Python, Keras, Tensorflow, NumPy, Matplotlib.

⚙️ How It Works

1-.Data Loading and Preprocessing

    IMDB reviews are loaded, keeping only the top 10,000 most frequent words.
    Data is transformed into one-hot encoded vectors.
    
2.-Model Building and Compilation

    Multiple feedforward neural networks (Dense layers) are created.
    Activations: ReLU for hidden layers, Sigmoid for output.
    Loss functions: binary_crossentropy and mse.
    Optimizer: rmsprop.

3.-Training

    The model is trained on partial training data, with validation data to monitor overfitting.
    Several variations are trained with different epochs, batch sizes, and layer configurations.

4.-Evaluation and Visualization

    Model performance is evaluated on the test set.
    Loss and accuracy curves are plotted.
    Predictions are made on sample data.