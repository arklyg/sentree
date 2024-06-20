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

import torch
import numpy as np

import data
import model
import probe
import bintree

from pytorch_pretrained_bert import BertTokenizer, BertModel

class SenTree(object):
    def __init__(self, yaml_args, seed=None, token_intenal='<int>', token_vacancy='<vac>'):
        # Define seed
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.args = yaml_args
        self.args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define the BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        self.model = BertModel.from_pretrained('bert-large-cased')
        self.model.to(self.args['device'])
        self.model.eval()

        # Define the depth probe
        self.depth_probe = probe.OneWordPSDProbe(self.args)
        self.depth_probe.load_state_dict(torch.load(self.args['probe']['depth_params_path'], map_location=self.args['device']))

        # Define the special tokens
        self.token_internal = token_intenal
        self.token_vacancy = token_vacancy

    def sentence_to_sequence(self, sentence):
        return self.tree_to_sequence(self.sentence_to_tree(sentence))

    def sentence_to_tree(self, sentence):
        # Tokenize the sentence and create tensor inputs to BERT
        untokenized_sent = sentence.strip().split()
        tokenized_sent = self.tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + ' '.join(untokenized_sent) + ' [SEP]')
        untok_tok_mapping = data.SubwordDataset.match_tokenized_to_untokenized(tokenized_sent, untokenized_sent)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sent)
        segment_ids = [1 for x in tokenized_sent]

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segment_ids])

        tokens_tensor = tokens_tensor.to(self.args['device'])
        segments_tensors = segments_tensors.to(self.args['device'])

        with torch.no_grad():
            # Run sentence tensor through BERT after averaging subwords for each token
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            single_layer_features = encoded_layers[self.args['model']['model_layer']]
            representation = torch.stack([torch.mean(single_layer_features[0,untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], dim=0) for i in range(len(untokenized_sent))], dim=0)
            representation = representation.view(1, *representation.size())

            depth_predictions = self.depth_probe(representation).detach().cpu()[0][:len(untokenized_sent)].numpy()

            word_depth = []
            for i in range(len(untokenized_sent)):
                word_depth.append((untokenized_sent[i], depth_predictions[i]))

            return bintree.BinTree.build_from_sentence(word_depth, 0, len(word_depth))

    def tree_to_sequence(cls, tree, truncate=True):
        sequence = tree.trav_root()

        if truncate:
            while len(sequence) and sequence[-1] == bintree.BinTree.token_vacancy:
                sequence = sequence[:-1]

        return sequence
