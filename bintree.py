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

import sys

class BinTree(object):
    # class variables
    token_internal = '<ITN>'
    token_vacancy = '<VAC>'

    def __init__(self, data, left=None, right=None):
        # instance variables
        self.data, self.left, self.right = data, left, right

    def trav_root(self, seq=None):
        seq = (seq + [self.data]) if seq else [self.data]
        if self.left or self.right:
            seq += [BinTree.token_internal]
            if self.left:
                seq = self.left.trav_root(seq)
            else:
                seq += [BinTree.token_vacancy]
            if self.right:
                seq = self.right.trav_root(seq)
            else:
                seq += [BinTree.token_vacancy]
        return seq

    def __str__(self):
        return (str(self.left) if self.left else '').strip() + ' ' + self.data + ' ' + (str(self.right) if self.right else '').strip()

    @classmethod
    def set_special_tokens(cls, interval, vacancy):
        if interval:
            cls.token_internal = interval
        if vacancy:
            cls.token_vacancy = vacancy

    @classmethod
    def build_from_sentence(cls, segs, start, end):
        # end is excluded
        top_i, top_v = 0, sys.float_info.max
        for i in range(start, end):
            (_, cur_v) = segs[i]
            if cur_v < top_v:
                top_v = cur_v
                top_i = i

        if start < top_i:
            left = cls.build_from_sentence(segs, start, top_i)
        else:
            left = None
        if top_i + 1 < end:
            right = cls.build_from_sentence(segs, top_i + 1, end)
        else:
            right = None
        (data, _) = segs[top_i]
        return cls(data, left, right)

    @classmethod
    def build_from_sequence(cls, segs):
        (ret, _) = cls.build_from_sequence_recursively(segs, 0)
        return ret

    @classmethod
    def build_from_sequence_recursively(cls, segs, index):
        if index >= len(segs):
            return (None, index)
        if segs[index] == cls.token_vacancy:
            return (None, index + 1)

        data, left, right = segs[index], None, None
        index += 1
        if index < len(segs) and segs[index] == cls.token_internal:
            index += 1
            (left, index) = cls.build_from_sequence_recursively(segs, index)
            (right, index) = cls.build_from_sequence_recursively(segs, index)

        return (cls(data, left, right), index)

