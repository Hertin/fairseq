# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import BaseWrapperDataset, data_utils


class AddTargetDatasetBart(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
        pad,
        eos,
        batch_targets,
        process_label=None,
        add_to_input=False,
    ):
        super().__init__(dataset)
        self.labels = labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.add_to_input = add_to_input

    def get_label(self, index):
        return (
            self.labels[index]
            if self.process_label is None
            else self.process_label(self.labels[index])
        )

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index)

        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            collated["ntokens"] = collated["target_lengths"].sum().item()
        else:
            collated["ntokens"] = sum([len(t) for t in target])

        collated["target"] = target

        if self.add_to_input:
            prev_output_tokens = target.clone()
            prev_output_tokens[:, 0] = target.gather(
                1,
                (target.ne(self.pad).sum(dim=1) - 1).unsqueeze(-1),
            ).squeeze()
            prev_output_tokens[:, 1:] = target[:, :-1]
            collated["target"] = target.long()            
            collated["net_input"]["prev_output_tokens"] = prev_output_tokens.long()
            collated["net_input"]["bart_input_tokens"] = collated["target"].clone()
            collated["ntokens"] += target.size(0)

        return collated
