"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import numpy as np
import torch

from lightning_modules.data_modules.collates.base import BaseCollate
from utils.constants import COMMON_SPECIAL_TOKENS, HeadTypes, Tasks
from utils.misc import is_otor


class TransformerDecoderCollate(BaseCollate):
    def __init__(self, decoder_dict):
        super().__init__()
        self.decoder_dict = decoder_dict

        self.task2block_token = {
            Tasks.OCR_READ_2HEAD: "[END_TEXT_BLOCK]",
        }
        self.end_prompt_token = "[END_PROMPT]"
        self.dontcare_token = "[DONTCARE]"
        self.ignore_index = -100

    def __call__(self, instances, decoder_name, decoder_example_indices):
        dec_batch = {
            "metas": [],
            "dataset_names": [],
            "task_names": [],
            "decoder_names": [],
            "multiview_imgs": [],  # for contrastive learning
            "second_imgs": [],  # for otor
            "gt_jsons": [],  # for donut evaluation
            "all_answers": [],  # for vqa evaluation
        }

        self.stack_batch(dec_batch, instances, decoder_example_indices)
        self.tensorize(dec_batch)

        decoder = self.decoder_dict[decoder_name]
        tokenizer = decoder.tokenizer
        end_prompt_token_id = tokenizer.convert_tokens_to_ids(self.end_prompt_token)

        end_prompt_token_indices = []
        for example_idx in decoder_example_indices:
            instance = instances[example_idx]
            end_prompt_token_indices.append(
                instance["input_ids"].index(end_prompt_token_id)
            )
        max_end_prompt_token_idx = max(end_prompt_token_indices)

        # update labels
        (
            input_ids,
            origin_input_ids,
            attention_mask,
            labels,
            origin_labels,
            gt_sequences,
            origin_gt_sequences,
        ) = self.get_labels(
            instances,
            decoder,
            decoder_example_indices,
            end_prompt_token_indices,
            max_end_prompt_token_idx,
        )
        dec_batch.update(
            {
                "input_ids": torch.from_numpy(input_ids),
                "origin_input_ids": torch.from_numpy(origin_input_ids),
                "attention_mask": torch.from_numpy(attention_mask),
                "labels": torch.from_numpy(labels),
                "origin_labels": torch.from_numpy(origin_labels),
                "gt_sequences": gt_sequences,
                "origin_gt_sequences": origin_gt_sequences,
                "max_end_prompt_token_idx": max_end_prompt_token_idx,
            }
        )

        # For 2head model
        original_block_quads, block_quads, block_first_tokens = self.get_2head_labels(
            instances,
            decoder,
            decoder_example_indices,
            input_ids,
        )
        dec_batch["original_block_quads"] = original_block_quads
        dec_batch["block_quads"] = torch.from_numpy(block_quads)
        dec_batch["block_first_tokens"] = torch.from_numpy(block_first_tokens)
        return dec_batch

    def get_labels(
        self,
        instances,
        decoder,
        decoder_example_indices,
        end_prompt_token_indices,
        max_end_prompt_token_idx,
    ):
        decoder_bsz = len(decoder_example_indices)
        decoder_max_length = decoder.decoder_max_length
        tokenizer = decoder.tokenizer
        pad_token_id = tokenizer.pad_token_id
        dontcare_token_id = tokenizer.convert_tokens_to_ids(self.dontcare_token)

        input_ids = np.ones((decoder_bsz, decoder_max_length), dtype=int) * pad_token_id
        origin_input_ids = (
            np.ones((decoder_bsz, decoder_max_length), dtype=int) * pad_token_id
        )
        attention_mask = np.zeros((decoder_bsz, decoder_max_length), dtype=int)

        labels = np.zeros((decoder_bsz, decoder_max_length), dtype=int)
        origin_labels = np.zeros((decoder_bsz, decoder_max_length), dtype=int)
        gt_sequences = []
        origin_gt_sequences = []

        for i, (example_idx, end_prompt_token_idx) in enumerate(
            zip(decoder_example_indices, end_prompt_token_indices)
        ):
            instance = instances[example_idx]

            if end_prompt_token_idx < max_end_prompt_token_idx:
                n_insert_tokens = max_end_prompt_token_idx - end_prompt_token_idx
                input_ids[i, :] = (
                    instance["input_ids"][:end_prompt_token_idx]
                    + [pad_token_id] * n_insert_tokens
                    + instance["input_ids"][end_prompt_token_idx:-n_insert_tokens]
                )
                attention_mask[i, :] = (
                    instance["attention_mask"][:end_prompt_token_idx]
                    + [1] * n_insert_tokens
                    + instance["attention_mask"][end_prompt_token_idx:-n_insert_tokens]
                )
                if is_otor(instances[0]["task_names"], oracle=True):
                    origin_input_ids[i, :] = (
                        instance["origin_input_ids"][:end_prompt_token_idx]
                        + [pad_token_id] * n_insert_tokens
                        + instance["origin_input_ids"][
                            end_prompt_token_idx:-n_insert_tokens
                        ]
                    )
            else:  # end_prompt_token_idx == max_end_prompt_token_idx
                input_ids[i, :] = instance["input_ids"]
                attention_mask[i, :] = instance["attention_mask"]
                if is_otor(instances[0]["task_names"], oracle=True):
                    origin_input_ids[i, :] = instance["origin_input_ids"]

            labels[i, :] = input_ids[i, :].copy()
            origin_labels[i, :] = origin_input_ids[i, :].copy()

            gt_sequences.append(instance["gt_sequences"])
            if is_otor(instances[0]["task_names"], oracle=True):
                origin_gt_sequences.append(instance["origin_gt_sequences"])

        if is_otor(instances[0]["task_names"], or_oracle=True):
            origin_input_ids = input_ids.copy()
            pos_indices = np.logical_and(
                input_ids >= tokenizer.convert_tokens_to_ids("[POS_0]"),
                input_ids <= tokenizer.convert_tokens_to_ids("[POS_9]"),
            )
            rand_mask = np.random.rand(*input_ids.shape) < 0.1
            rand_mask = np.logical_and(rand_mask, pos_indices)

            input_ids[rand_mask] = tokenizer.convert_tokens_to_ids("[MASK]")
            origin_input_ids[pos_indices] = tokenizer.convert_tokens_to_ids("[MASK]")

        labels[:, : max_end_prompt_token_idx + 1] = self.ignore_index
        labels[labels == pad_token_id] = self.ignore_index

        origin_labels[:, : max_end_prompt_token_idx + 1] = self.ignore_index
        origin_labels[origin_labels == pad_token_id] = self.ignore_index

        # DontCare Processing
        # Based on DC token, search left and right until common special token appears.
        # If a special token appears, all areas are processed with DC tokens.
        common_spe_set = set(tokenizer.convert_tokens_to_ids(COMMON_SPECIAL_TOKENS))
        for dc_example_idx, dc_token_idx in zip(*np.where(labels == dontcare_token_id)):
            dc_start_idx = dc_end_idx = dc_token_idx

            for idx in reversed(range(dc_token_idx)):
                if labels[dc_example_idx, idx] in common_spe_set:
                    dc_start_idx = idx
                    break
            for idx in range(dc_token_idx + 1, len(labels[dc_example_idx])):
                if labels[dc_example_idx, idx] in common_spe_set:
                    dc_end_idx = idx
                    break

            labels[dc_example_idx, dc_start_idx : dc_end_idx + 1] = self.ignore_index
            origin_labels[
                dc_example_idx, dc_start_idx : dc_end_idx + 1
            ] = self.ignore_index
        return (
            input_ids,
            origin_input_ids,
            attention_mask,
            labels,
            origin_labels,
            gt_sequences,
            origin_gt_sequences,
        )

    def get_2head_labels(
        self,
        instances,
        decoder,
        decoder_example_indices,
        input_ids,
    ):
        decoder_bsz = len(decoder_example_indices)
        decoder_max_length = decoder.decoder_max_length
        tokenizer = decoder.tokenizer

        original_block_quads = []
        block_quads = np.zeros((decoder_bsz, decoder_max_length, 8))
        block_target_tokens = np.zeros((decoder_bsz, decoder_max_length), dtype=int)
        for example_idx in decoder_example_indices:
            instance = instances[example_idx]
            task_name = instance["task_names"]
            if task_name.endswith(HeadTypes.TWO_HEAD):
                original_block_quads.append(instance["original_block_quads"])
                target_block_token_id = tokenizer.convert_tokens_to_ids(
                    self.task2block_token[task_name]
                )
                target_block_indices = np.nonzero(
                    input_ids[example_idx] == target_block_token_id
                )[0]

                max_blocks = len(target_block_indices)
                block_quads[example_idx, target_block_indices] = np.array(
                    instance["original_block_quads"][:max_blocks]
                )

                # Set as target only if it is a valid text block
                # Non-valid text block example: [DONTCARE]
                for target_block_idx, is_valid_block in zip(
                    target_block_indices, instance["is_valid_blocks"][:max_blocks]
                ):
                    if is_valid_block == 1:
                        block_target_tokens[example_idx, target_block_idx] = 1
            else:
                original_block_quads.append([])

        return original_block_quads, block_quads, block_target_tokens
