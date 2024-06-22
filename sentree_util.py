'''
Copyright 2024 Yaguang Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from argparse import ArgumentParser
import yaml

import bintree
import sentree

if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('config')
    argp.add_argument('-seed', default=0, type=int, help='sets all random seeds for (within-machine) reproducibility')
    argp.add_argument('-token_internal', default='', help='token in sequence representing internal node, \'<ITN>\' for default')
    argp.add_argument('-token_vacancy', default='', help='token in sequence representing vacancy node, \'<VAC>\' for default')
    cli_args = argp.parse_args()
    yaml_args = yaml.full_load(open(cli_args.config))
    sentree_instance = sentree.SenTree(yaml_args, cli_args.seed)
    bintree.BinTree.set_special_tokens(cli_args.token_internal, cli_args.token_vacancy)

    print("SenTree Util. (2024 Yaguang Li)")

    prompt = input("\nEnter s to convert a sentence to a binary tree implied sequence, or\n" +\
                     "Enter t to convert a binary tree implied sequence to a sentence, or\n" +\
                     "Enter q to quit: ")

    while prompt != 'q':
        while prompt == 's':
            prompt = input("SENTENCE:\n> ")
            if prompt == 'q' or prompt == 't':
                break
            print(' '.join(sentree_instance.sentence_to_sequence(prompt)))
            prompt = 's'
        while prompt == 't':
            prompt = input("SEQUENCE:\n> ")
            if prompt == 'q' or prompt == 's':
                break
            print(bintree.BinTree.build_from_sequence(prompt.strip().split()))
            prompt = 't'
    
