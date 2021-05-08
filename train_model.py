from Dataset import *
from Dataloader import *
from Model import *

import os.path as p
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training and evaluating script')

    parser.add_argument('--data_directory', type=str, default='sentiment-datasets', help='Path to data directory')
    parser.add_argument('--hug_path', type=str, default='nlptown/bert-base-multilingual-uncased-sentiment',
                        help='Path to model on hugginface library')
    parser.add_argument('--batch_size', type=int, default=10, help='Size of batches')
    parser.add_argument('--lr', type=float, default=0.00002, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--warmup', type=int, default=100, help='Number of warmup steps')
    parser.add_argument('--val_steps', type=int, default=1000, help='Number of steps before validation')
    parser.add_argument('--loss_print', type=int, default=200, help='Number of steps before print loss')

    parser.add_argument("--to_lower", type=bool, const='True', nargs='?', default=True, help="Set text to lower")
    parser.add_argument("--do_filter", type=bool, const='True', nargs='?', default=True, help="Do filter by dictionary")
    parser.add_argument("--remove_rep", type=bool, const='True', nargs='?', default=True, help="Remove repeating symbols")
    parser.add_argument("--remove_stops", type=bool, const='True', nargs='?', default=False, help="Remove stopwords")
    parser.add_argument("--remove_punkt", type=bool, const='True', nargs='?', default=True, help="Remove punktuations")
    parser.add_argument("--lemmatize", type=bool, const='True', nargs='?', default=False, help="Lemmatize words")
    parser.add_argument("--stemming", type=bool, const='True', nargs='?', default=False, help="Use stemming")


    args = parser.parse_args()

    nltk.download('punkt')
    nltk.download('stopwords')

    train_dataset = Dataset(tokenizer=args.hug_path,
                            to_lower=args.to_lower,
                            do_filter=args.do_filter,
                            remove_repetitions=args.remove_rep,
                            remove_stopwords=args.remove_stops,
                            remove_punktuations=args.remove_punkt,
                            lemmatize=args.lemmatize,
                            stemming=args.stemming,
                            n_classes=3)

    train_dataset.load_dataset(path2json=p.join(args.data_directory, 'train.jsonl'))

    val_dataset = Dataset(tokenizer=args.hug_path,
                          to_lower=args.to_lower,
                          do_filter=args.do_filter,
                          remove_repetitions=args.remove_rep,
                          remove_stopwords=args.remove_stops,
                          remove_punktuations=args.remove_punkt,
                          lemmatize=args.lemmatize,
                          stemming=args.stemming,
                          n_classes=3)

    val_dataset.load_dataset(path2json=p.join(args.data_directory, 'val.jsonl'))

    train_loader = Dataloader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = Dataloader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    model = Model(args.hug_path)

    model.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=args.epochs, lr=args.lr, warmup=args.warmup, validate_steps=args.val_steps,
              loss_print=args.loss_print)
