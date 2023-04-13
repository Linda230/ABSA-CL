# -*- coding: utf-8 -*-
# file: generate.py

import nlpaug.augmenter.word as naw

def main():
    dataset_files = {
        'acl2014': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'res2014': {
            'train': './datasets/semeval14/restaurant_train.raw',
            'test': './datasets/semeval14/restaurant_test.raw',
        },
        'laptop2014': {
            'train': './datasets/semeval14/laptop_train.raw',
            'test': './datasets/semeval14/laptop_test.raw'
        },
        'Twitter15':{
            'train': './datasets/Twitter15/twitter15_train.txt',
            'test': './datasets/Twitter15/twitter15_test.txt',
        },
        'Twitter17': {
            'train': './datasets/Twitter17/twitter17_train.txt',
            'test': './datasets/Twitter17/twitter17_test.txt',
        }
    }

    def extract_data(fname):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_text = []

        for i in range(0, len(lines), 3):
            text = lines[i].lower().strip()
            all_text.append(text)
        return all_text

    fname = dataset_files['res2014']['train']

    Text = extract_data(fname=fname)

    def aug_data(dataset, type):
        augment_data = []
        if type == 'bta':
            aug = naw.BackTranslationAug()
        elif type == 'sr':
            aug = naw.SynonymAug()

        for i in range(0, len(dataset)):
            augmented_data = aug.augment(dataset[i])
            augment_data.append(augmented_data)
        return augment_data

    Aug_data = aug_data(dataset=Text, type='sr')
    print(len(Aug_data))
    f = open('./datasets/Aug_Data/Aug_acl14/sr_aug.txt', "w", encoding='utf-8')
    for line in Aug_data:
        f.write(line+"\r")
    f.close


if __name__ == '__main__':
    main()
