import json

with open('caption_spaceom_train.json') as f1, open('vqa_spaceom_train_multiframe.json') as f2:
    captions = json.load(f1)
    vqa = json.load(f2)

combined = captions + vqa

with open('train_all.json', 'w') as out:
    json.dump(combined, out, indent=2)