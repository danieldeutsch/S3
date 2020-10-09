import argparse
import json

import S3
import word_embeddings


def main(args):
    word_embs = word_embeddings.load_embeddings(args.embs_path)

    system_summaries = []
    references_list = []
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            instance = json.loads(line)
            # The input summaries to S3 are lists of sentences. The example
            # just passes the whole text in as 1 sentence without pre-sentence tokenizing
            # it, so we will do the same. But the input summaries are expected
            # to just be 1 string each, so we wrap them in an extra list
            summary = [instance['summary']]
            references = [[reference] for reference in instance['references']]
            system_summaries.append(summary)
            references_list.append(references)

    scores_pyr, scores_resp = S3.S3_batch(references_list, system_summaries, word_embs, args.model_folder)

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
