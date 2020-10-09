import argparse
import json
import os

import S3
import word_embeddings


def main(args):
    word_embs = word_embeddings.load_embeddings(args.embs_path)

    system_summaries = []
    references_list = []
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            instance = json.loads(line)
            system_summaries.append(instance['summary'])
            references_list.append(instance['references'])

    scores_pyr, scores_resp = S3.S3_batch(references_list, system_summaries, word_embs, model_folder)

    dirname = os.path.dirname(args.output_jsonl)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(args.output_jsonl, 'w') as out:
        for pyr, resp in zip(scores_pyr, scores_resp):
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
