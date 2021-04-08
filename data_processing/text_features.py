### COPY-PASTA from audio_features.py (only temporary to adopt the structure)
import numpy as np
import sys
import os
import math

from transformers import BertTokenizer, BertModel
from bert_text_features.parse_json_transcript import encode_json_transcript_with_bert



def create_embedding(name):
    if name == "BERT":
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        bert_model = BertModel.from_pretrained('bert-base-cased')

        return tokenizer, bert_model
    elif name == "FastText":
        return FastText()
    else:
        print(f"ERROR: Unknown embedding type '{args.text_embedding}'! Supported embeddings: 'BERT' and 'FastText'.")
        exit(-1)


def extract_bert_features(textpath, files, destpath, fps):
    embedding_model = create_embedding("BERT")
    fps_encoding = 10
    # TODO check if fps % fps_encoding == 0. If not, output a warning
    multiplier = int(fps/fps_encoding)
    for f in files:
        file = os.path.join(textpath, f + ".json")
        outfile = destpath + '/' + f + '.npy'
                
        print('{}\t->\t{}'.format(file,outfile))

        if isinstance(embedding_model, tuple):
            text_encoding = encode_json_transcript_with_bert(
                file, tokenizer = embedding_model[0], bert_model = embedding_model[1])
        else:
            raise Exception('Something is wrong with the BERT embedding model')
        # TODO Is this naive approach of doubling each entry correct? Try to understand G's approach
        text_encoding = [val for val in text_encoding for _ in range(multiplier)]
        text_encoding = np.array(text_encoding)

        print(text_encoding.shape)
        print(np.min(text_encoding),np.max(text_encoding))
        np.save(outfile, text_encoding)

