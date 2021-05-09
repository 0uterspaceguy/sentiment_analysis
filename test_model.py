from Dataset import *
from Dataloader import *
from Model import *
import os.path as p
import argparse
import nltk

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model testing script')

    parser.add_argument('--data_directory', type=str, default='sentiment-datasets', help='Path to data directory')
    parser.add_argument('--hug_path', type=str, default='nlptown/bert-base-multilingual-uncased-sentiment',
                        help='Path to model on hugginface library')
    parser.add_argument('--local_path', type=str, default='model/model.pth',
                        help='Path to your trained model')
    parser.add_argument('--batch_size', type=int, default=10, help='Size of batches')

    parser.add_argument("--to_lower", type=bool, const='True', nargs='?', default=True, help="Set text to lower")
    parser.add_argument("--do_filter", type=bool, const='True', nargs='?', default=True, help="Do filter by dictionary")
    parser.add_argument("--remove_rep", type=bool, const='True', nargs='?', default=True,
                        help="Remove repeating symbols")
    parser.add_argument("--remove_stops", type=bool, const='True', nargs='?', default=False, help="Remove stopwords")
    parser.add_argument("--remove_punkt", type=bool, const='True', nargs='?', default=True, help="Remove punktuations")
    parser.add_argument("--lemmatize", type=bool, const='True', nargs='?', default=False, help="Lemmatize words")
    parser.add_argument("--stemming", type=bool, const='True', nargs='?', default=False, help="Use stemming")

    args = parser.parse_args()

    nltk.download('punkt')
    nltk.download('stopwords')

    test_dataset = Dataset(tokenizer=args.hug_path,
                           to_lower=args.to_lower,
                           do_filter=args.do_filter,
                           remove_repetitions=args.remove_rep,
                           remove_stopwords=args.remove_stops,
                           remove_punktuations=args.remove_punkt,
                           lemmatize=args.lemmatize,
                           stemming=args.stemming,
                           n_classes=3)

    test_dataset.load_dataset(path2json=p.join(args.data_directory, 'val.jsonl'))

    test_loader = Dataloader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    model = Model(args.hug_path, args.local_path)

    model.test(test_loader, with_save = False)
