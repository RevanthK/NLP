from assignment2_nlm import FFLM, FF
import math
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
import pandas as pd
from collections import OrderedDict
import io

def main(args):
    model = torch.load(args.model)
    emb_weights = model.E.weight

    print('Loaded Embeddings: ', emb_weights, '\nShape:', emb_weights.shape)

    tokenizer = util.Tokenizer(tokenize_type='nltk', lowercase=True)
    train_toks = tokenizer.tokenize(open('data/gigaword_subset.val').read())
    train_ngram_counts = tokenizer.count_ngrams(train_toks)
    vocab = [tup[0] for tup, _ in train_ngram_counts[0].most_common(v)]

    print('Tokenized vocabulary')

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')

    for num, word in enumerate(vocab):
        vecs = emb_weights[num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x.item()) for x in vecs]) + "\n")

    out_v.close()
    out_m.close()

    print('...Created files for embedding visualization')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', type=int, default=10000, help='Vocabulary size of model')
    parser.add_argument('--model', type=str, default='model.pt', help='path/name of your saved model')
    parser.add_argument('--vec_file', type=str, default='vecs.tsv', help='output file name for vector info')
    parser.add_argument('--word_file', type=str, default='meta.tsv', help='output file name for word info')
    args = parser.parse_args()
    main(args)
