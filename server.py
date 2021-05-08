from flask import Flask, request, Response
from Dataset import *
from Model import *
import os.path as p
import argparse
import json
import nltk

app = Flask(__name__)

@app.route('/api/get_sentiment', methods=['POST'])
def send_sentiment():
    data = request.json
    text = data['text']

    prepared_text = dataset.prepare_text(text)
    ids, mask = dataset.tokenize(prepared_text)
    sentiment = model.predict(ids, mask)[0]
    json_sentiment = {'label': sentiments[sentiment]}

    response = app.response_class(
        response=json.dumps(json_sentiment),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Server script for sentiment recognition model')

    parser.add_argument('--hug_path', type=str, default='nlptown/bert-base-multilingual-uncased-sentiment',
                        help='Path to model on hugginface library')
    parser.add_argument('--local_path', type=str, default='model.pth',
                        help='Path to your trained model')

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

    sentiments = ['negative', 'neutral', 'positive']

    dataset = Dataset(tokenizer=args.hug_path,
                      to_lower=args.to_lower,
                      do_filter=args.do_filter,
                      remove_repetitions=args.remove_rep,
                      remove_stopwords=args.remove_stops,
                      remove_punktuations=args.remove_punkt,
                      lemmatize=args.lemmatize,
                      stemming=args.stemming,
                      n_classes=3)

    print('Load model...')
    model = Model(args.hug_path, args.local_path)
    print('Model loaded succesfully')

    app.run()