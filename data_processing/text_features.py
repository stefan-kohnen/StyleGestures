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
    assert fps % fps_encoding == 0, "target fps for transcript features not a multiple of 10"
    multiplier = int(fps/fps_encoding)
    print(f"multiplier: {multiplier}")
    for f in files:
        file = os.path.join(textpath, f + ".json")
        outfile = destpath + '/' + f + '.npy'
                
        print('{}\t->\t{}'.format(file,outfile))

        if isinstance(embedding_model, tuple):
            text_encoding = encode_json_transcript_with_bert(
                file, tokenizer = embedding_model[0], bert_model = embedding_model[1])
        else:
            raise Exception('Something is wrong with the BERT embedding model')

        assert type(text_encoding) == np.ndarray, "Before duplication: Type of `text_encoding` wrong."

        print(f"Shape before duplication: {text_encoding.shape}")
        orig_dim_samples = text_encoding.shape[0]
        cols = np.linspace(0, text_encoding.shape[0], dtype=int, endpoint=False, num=text_encoding.shape[0] * 2)
        # NOTE: because of the dtype, 'cols' contains each index in 0:text.shape[0] twice
        text_encoding = text_encoding[cols, :]
        print(f"Shape after duplication: {text_encoding.shape}")

        assert type(text_encoding) == np.ndarray, "After duplication: Type of `text_encoding` wrong."
        # TODO adapt the following assertion to np.array (use shape) 
        assert 2 * orig_dim_samples == text_encoding.shape[0], "BERT: Entry duplicating failed"

        # print(text_encoding.shape)
        print(np.min(text_encoding),np.max(text_encoding))
        np.save(outfile, text_encoding)

