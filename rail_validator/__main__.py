import argparse
import json
from collections import Counter
from .data import read_applications

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['inspect','evaluate','evaluate-ml','process','resolve','train-ml','serve'])
    parser.add_argument('input')
    parser.add_argument('--output', default='output')
    parser.add_argument('--confidence', default='output/confidence.json')
    parser.add_argument('--method', choices=['baseline','g3','ml'], default='g3')
    parser.add_argument('--split', choices=['train','test','all','holdout'], default='test')
    parser.add_argument('--decisions')
    parser.add_argument('--model', default='output/ml/model.joblib')
    args = parser.parse_args()
    if args.command == 'serve':
        from .web import serve
        serve(args.input)
        return
    if args.command == 'train-ml':
        from .ml import train_model
        print(json.dumps(train_model(args.input,args.output),ensure_ascii=False,indent=2))
        return
    if args.command == 'resolve':
        from .process import resolve
        if not args.decisions: parser.error('resolve requires --decisions')
        resolve(args.input,args.decisions,args.output)
        return
    if args.command in ('evaluate','evaluate-ml'):
        from .evaluate import run_evaluation
        print(json.dumps(run_evaluation(args.input,args.output,args.model if args.command=='evaluate-ml' else None),ensure_ascii=False,indent=2))
        return
    if args.command == 'process':
        from .process import run_processing
        run_processing(args.input,args.output,args.confidence,args.method,args.split,args.model)
        return
    splits, channels, blanks, ids = Counter(), Counter(), Counter(), Counter()
    for row in read_applications(args.input):
        splits[row['split']] += 1
        channels[row['channel']] += 1
        ids[row['app_id']] += 1
        blanks.update(k for k, v in row.items() if not v.strip())
    print(json.dumps({'rows':sum(splits.values()),'splits':splits,'channels':channels,'blank_fields':blanks,'repeated_ids':sum(v-1 for v in ids.values())}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
