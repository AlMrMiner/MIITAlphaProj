import argparse
import json
from collections import Counter
from .data import read_applications

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['inspect'])
    parser.add_argument('input')
    args = parser.parse_args()
    splits, channels, blanks, ids = Counter(), Counter(), Counter(), Counter()
    for row in read_applications(args.input):
        splits[row['split']] += 1
        channels[row['channel']] += 1
        ids[row['app_id']] += 1
        blanks.update(k for k, v in row.items() if not v.strip())
    print(json.dumps({'rows':sum(splits.values()),'splits':splits,'channels':channels,'blank_fields':blanks,'repeated_ids':sum(v-1 for v in ids.values())}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
