import os
from argparse import ArgumentParser  # To handle command-line arguments
from data import Dataset  # Custom class for dataset handling
from model import CharCNN  # Custom CharCNN model class
import tensorflow as tf  # TensorFlow library for deep learning
from constant import *  # Import constants from a separate module
from tensorflow.keras.optimizers import Adam  # Adam optimizer

# Enable eager execution in TensorFlow (useful for debugging)
tf.config.run_functions_eagerly(True)

if __name__ == "__main__":
    parser = ArgumentParser()  # Create a parser for command-line arguments
    home_dir = os.getcwd()  # Get the current working directory

    # Define the command-line arguments and their default values
    parser.add_argument("--batch-size", default=128, type=int, help="Batch size for training")
    parser.add_argument("--mode", default="small", type=str, help="Model mode: 'small', 'large', or 'all'")
    parser.add_argument("--vocab-folder", default=f'{home_dir}/saved_vocab/CharCNN/', type=str,
                        help="Folder to save the vocabulary")
    parser.add_argument("--train-file", default='data.csv', type=str, help="Path to the training file")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs for training")
    parser.add_argument("--embedding-size", default=100, type=int, help="Size of the embeddings")
    parser.add_argument("--test-size", default=0.3, type=float, help="Percentage of data for testing")
    parser.add_argument("--num-classes", default=2, type=int, help="Number of output classes")
    parser.add_argument("--learning-rate", default=0.001, type=float, help="Learning rate for the optimizer")
    parser.add_argument("--smallCharCNN-folder", default="smallCharCNN", type=str, 
                        help="Folder to save the Small CharCNN model")
    parser.add_argument("--largeCharCNN-folder", default="largeCharCNN", type=str, 
                        help="Folder to save the Large CharCNN model")
    parser.add_argument("--padding", default="same", type=str, help="Padding mode: 'same' or 'valid'")

    args = parser.parse_args()  # Parse the arguments provided by the user

    # Display configured parameters
    print('--------------------- Welcome to CharCNN -------------------')
    print('---------------------------------------------------------------------')
    print('Training CharCNN model with the following hyperparameters:')
    for i, arg in enumerate(vars(args)):  # Iterate through arguments and print them
        print(f'{i + 1}. {arg}: {vars(args)[arg]}')
    print('=====================================================================')

    # Load the dataset
    print("------------- LOADING TRAINING DATA ------------")
    dataset = Dataset(vocab_folder=args.vocab_folder)  # Initialize the dataset with the saved vocabulary
    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = dataset.build_dataset(data_path=args.train_file, test_size=args.test_size)

    # Limit the data size (to reduce computational load during testing)
    slice_idx = 5000
    x_train = x_train[:slice_idx]
    x_val = x_val[:slice_idx]
    y_train = y_train[:slice_idx]
    y_val = y_val[:slice_idx]

    # Initialize Small and Large CharCNN models
    print("------------- INITIALIZING MODELS ------------")
    small_CharCNN = CharCNN(
        vocab_size=dataset.vocab_size,  # Vocabulary size
        embedding_size=args.embedding_size,  # Embedding size
        max_length=dataset.max_len,  # Maximum sequence length
        num_classes=args.num_classes,  # Number of output classes
        feature="small",  # Specify the type of model
        padding=args.padding  # Padding mode
    )

    large_CharCNN = CharCNN(
        vocab_size=dataset.vocab_size,
        embedding_size=args.embedding_size,
        max_length=dataset.max_len,
        num_classes=args.num_classes,
        feature="large",
        padding=args.padding
    )

    # Define the loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # Create the Adam optimizer with the specified learning rate
    adam = Adam(learning_rate=args.learning_rate)

    # Compile the models
    small_CharCNN.compile(optimizer=adam, loss=loss, metrics=["accuracy"])
    large_CharCNN.compile(optimizer=adam, loss=loss, metrics=["accuracy"])

    # Train the Small CharCNN model (if requested by the user)
    if args.mode in ['small', 'all']:
        print("------------- TRAINING SMALL CHARCNN ------------")
        small_CharCNN.fit(
            x_train, y_train,  # Training data
            validation_data=(x_val, y_val),  # Validation data
            epochs=args.epochs,  # Number of epochs
            batch_size=args.batch_size,  # Batch size for training
            validation_batch_size=args.batch_size  # Batch size for validation
        )
        print("---------- FINISHED TRAINING SMALL CHARCNN --------")
        # Save the Small CharCNN model
        os.makedirs(args.smallCharCNN_folder, exist_ok=True)  # Create the folder if it does not exist
        small_CharCNN.save(f"{args.smallCharCNN_folder}/model.keras")  # Save the model

    # Train the Large CharCNN model (if requested by the user)
    if args.mode in ['large', 'all']:
        print("------------- TRAINING LARGE CHARCNN ------------")
        large_CharCNN.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_batch_size=args.batch_size
        )
        print("---------- FINISHED TRAINING LARGE CHARCNN --------")
        # Save the Large CharCNN model
        os.makedirs(args.largeCharCNN_folder, exist_ok=True)  # Create the folder if it does not exist
        large_CharCNN.save(f"{args.largeCharCNN_folder}/model.keras")  # Save the model
