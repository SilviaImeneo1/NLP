import os
from argparse import ArgumentParser
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data import Dataset

if __name__ == "__main__":
    # Parser degli argomenti
    parser = ArgumentParser()
    home_dir = os.getcwd()
    parser.add_argument("--smallCharCNN-folder", default="smallCharCNN", type=str, help="Directory del modello Small-CharCNN")
    parser.add_argument("--largeCharCNN-folder", default="largeCharCNN", type=str, help="Directory del modello Large-CharCNN")
    parser.add_argument("--vocab-folder", default=f'{home_dir}/saved_vocab/CharCNN/', type=str, help="Directory del vocabolario salvato")
    parser.add_argument("--test-file", default="test.csv", type=str, help="File CSV contenente i dati di test")
    parser.add_argument("--model", default="small", type=str, choices=["small", "large"], help="Modello da utilizzare: 'small' o 'large'")
    parser.add_argument("--result-file", default="result.csv", type=str, help="File CSV per salvare i risultati")
    args = parser.parse_args()

    # Informazioni di benvenuto
    print('----------------------------------------------------------')
    print('Hyperparameters:')
    for i, arg in enumerate(vars(args)):
        print(f'{i+1}. {arg}: {vars(args)[arg]}')
    print('==========================================================')

    # Caricamento del modello
    print('Loading the model')
    if args.model == "small":
        model_path = os.path.join(args.smallCharCNN_folder, "model.keras")
        model = tf.keras.models.load_model(model_path)
    else:
        model_path = os.path.join(args.largeCharCNN_folder, "model.keras")
        model = tf.keras.models.load_model(model_path)
    print(f'Model {args.model} loaded.')

    # Caricamento del tokenizer
    print('Loading tokenizer')
    dataset = Dataset(vocab_folder=args.vocab_folder)
    label_dict = dataset.label_dict
    print('Tokenizer loaded')

    # Caricamento e preprocessing dei dati di test
    print('Preprocessing of test data')
    try:
        test_data = pd.read_csv(args.test_file, header=None, names=['sentence'])
    except FileNotFoundError:
        print(f"Errore: Il file {args.test_file} non esiste. Verifica il percorso e riprova.")
        exit(1)

    sentences = test_data['sentence'].values
    preprocessed_sentences = [dataset.preprocess_data(s) for s in sentences]
    tokenized_sentences = dataset.tokenizer.texts_to_sequences(preprocessed_sentences)
    #max_len = model.input_shape[1]  # Lunghezza massima dall'input del modello
    max_len = model.max_length
    padded_sentences = pad_sequences(tokenized_sentences, maxlen=max_len, padding='post')

    # Predizione
    print('Prediction')
    predictions = model.predict(padded_sentences)
    predicted_labels = np.argmax(predictions, axis=1)

    # Decodifica delle predizioni
    reverse_label_dict = {i: label for label, i in label_dict.items()}
    decoded_labels = [reverse_label_dict[label] for label in predicted_labels]

    # Salvataggio dei risultati
    print('Saving predictions')
    result_df = pd.DataFrame({'sentence': sentences, 'label': decoded_labels})
    result_df.to_csv(args.result_file, index=False)
    print(f"Predictions saved in {args.result_file}")
