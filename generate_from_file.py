"""
Generate a random sentence given an input file.
"""

import nltk.data
from nltk import word_tokenize
from sentence_generator import Generator

import random
import logging
import time

import os

class SentenceGenerator:
    def __init__(self, chain_length, nbooks):
        logging.info(f"setting up sentence generator from max {nbooks} books")
        logging.info("this may take a while...")
        ta=time.time()
        books_path = './books/'
        (_, _, filenames) = next(os.walk(books_path))
        random.shuffle(filenames)
        sent_tokens=[]
        logging.info("-"*78)
        for index, fname in enumerate(filenames):
            logging.info(f"processing book {index+1}: {fname}")
            with open(books_path+fname, 'r') as f:
                sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
                sents = sent_detector.tokenize(f.read().strip())
                sent_tokens += [word_tokenize(sent.replace('\n', ' ').lower()) for sent in sents]
            if index >= nbooks:
                break

        logging.info("-"*78)
        tb=time.time()
        logging.info(f"book processing took {tb-ta}s")

        logging.info("generating tokens")
        self.generator = Generator(sent_tokens, args.chain_length)
        logging.info("done")

    def gen_sentence(self):
        return self.generator.generate()
