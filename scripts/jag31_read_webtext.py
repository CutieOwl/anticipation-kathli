import json
import gzip


DATA_DIR = '/jagupard31/scr1/kathli/openwebtext'

json_content = []
with gzip.open(f'{DATA_DIR}/openwebtext_val.1-of-8.jsonl.gz', 'rb') as gzip_file:
    for line in gzip_file:  # Read one line.
        line = line.rstrip()
        if line:  # Any JSON data on it?
            obj = json.loads(line)
            json_content.append(obj)
            print(obj['text'])
            break

print(len(json_content))

#print(json.dumps(json_content, indent=4))  # Pretty-print data parsed.    