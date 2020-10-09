import argparse
import json
import os

import S3
import word_embeddings


def main(args):
    word_embs = word_embeddings.load_embeddings(args.embs_path)

    dirname = os.path.dirname(args.output_jsonl)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(args.output_jsonl, 'w') as out:
        with open(args.input_jsonl, 'r') as f:
            for line in f:
                instance = json.loads(line)
                summary = instance['summary']
                references = instance['references']
                pyr, resp = S3.S3(references, summary, word_embs, model_folder)
                out.write(json.dumps({
                    'pyr': pyr,
                    'resp': resp
                }) + '\n')


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl')
    argp.add_argument('output_jsonl')
    argp.add_argument('embs_path')
    argp.add_argument('model_folder')
    args = argp.parse_args()
    main(args)
