import os
from argparse import ArgumentParser
from data import Dataset
from model import CharCNN
import tensorflow as tf
from constant import *
from tensorflow.keras.optimizers import Adam

# Abilita il funzionamento eager (se necessario)
tf.config.run_functions_eagerly(True)

if __name__ == "__main__":
    parser = ArgumentParser()
    home_dir = os.getcwd()

    # Definizione dei parametri accettati da linea di comando
    parser.add_argument("--batch-size", default=128, type=int, help="Dimensione del batch per l'addestramento")
    parser.add_argument("--mode", default="small", type=str, help="Modalità del modello: 'small', 'large' o 'all'")
    parser.add_argument("--vocab-folder", default=f'{home_dir}/saved_vocab/CharCNN/', type=str, help="Cartella per salvare il vocabolario")
    parser.add_argument("--train-file", default='data.csv', type=str, help="Percorso del file di addestramento")
    parser.add_argument("--epochs", default=3, type=int, help="Numero di epoche")
    parser.add_argument("--embedding-size", default=100, type=int, help="Dimensione degli embeddings")
    parser.add_argument("--test-size", default=0.3, type=float, help="Percentuale di dati di test")
    parser.add_argument("--num-classes", default=2, type=int, help="Numero di classi di output")
    parser.add_argument("--learning-rate", default=0.001, type=float, help="Learning rate per l'ottimizzatore")
    parser.add_argument("--smallCharCNN-folder", default="smallCharCNN", type=str, help="Cartella per salvare il modello Small CharCNN")
    parser.add_argument("--largeCharCNN-folder", default="largeCharCNN", type=str, help="Cartella per salvare il modello Large CharCNN")
    parser.add_argument("--padding", default="same", type=str, help="Modalità di padding: 'same' o 'valid'")

    args = parser.parse_args()

    # Messaggio introduttivo
    print('--------------------- Welcome to CharCNN -------------------')
    print('---------------------------------------------------------------------')
    print('Training CharCNN model with the following hyperparameters:')
    for i, arg in enumerate(vars(args)):
        print(f'{i + 1}. {arg}: {vars(args)[arg]}')
    print('=====================================================================')

    # Caricamento dei dati
    print("------------- LOADING TRAINING DATA ------------")
    dataset = Dataset(vocab_folder=args.vocab_folder)
    x_train, x_val, y_train, y_val = dataset.build_dataset(data_path=args.train_file, test_size=args.test_size)

    slice_idx = 70
    x_train = x_train[:slice_idx]
    x_val = x_val[:slice_idx]
    y_train = y_train[:slice_idx]
    y_val = y_val[:slice_idx]

    # Inizializzazione dei modelli
    print("------------- INITIALIZING MODELS ------------")
    small_CharCNN = CharCNN(
        vocab_size=dataset.vocab_size,
        embedding_size=args.embedding_size,
        max_length=dataset.max_len,
        num_classes=args.num_classes,
        feature="small",
        padding=args.padding
    )

    large_CharCNN = CharCNN(
        vocab_size=dataset.vocab_size,
        embedding_size=args.embedding_size,
        max_length=dataset.max_len,
        num_classes=args.num_classes,
        feature="large",
        padding=args.padding
    )

    # Definizione della funzione di perdita
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # Definizione dell'ottimizzatore
    adam = Adam(learning_rate=args.learning_rate)

    # Compilazione dei modelli
    small_CharCNN.compile(optimizer=adam, loss=loss, metrics=["accuracy"])
    large_CharCNN.compile(optimizer=adam, loss=loss, metrics=["accuracy"])

    # Addestramento del modello
    if args.mode in ['small', 'all']:
        print("------------- TRAINING SMALL CHARCNN ------------")
        small_CharCNN.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_batch_size=args.batch_size
        )
        print("---------- FINISHED TRAINING SMALL CHARCNN --------")
        # Salvataggio del modello Small CharCNN
        os.makedirs(args.smallCharCNN_folder, exist_ok=True)  # Crea la cartella se non esiste
        small_CharCNN.save(f"{args.smallCharCNN_folder}/model.keras")


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
        # Salvataggio del modello Large CharCNN
        os.makedirs(args.largeCharCNN_folder, exist_ok=True)  # Crea la cartella se non esiste
        large_CharCNN.save(f"{args.largeCharCNN_folder}/model.keras")
