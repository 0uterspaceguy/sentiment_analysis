import requests
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example client script for sentiment recognition model')
    parser.add_argument('-u', '--url', type=str, default='http://127.0.0.1:5000/api/get_sentiment',
                        help='web server adress')

    args = parser.parse_args()

    def make_predicition(text):
        data = {'text': text}
        r = requests.post(args.url, json=data)
        return r.json()['label']

    while True:
        passage = input('Input some text: ')
        print(make_predicition(passage))