"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from asian_bart import AsianBartForCausalLM
from Levenshtein import distance
from overrides import overrides
from transformers import AutoConfig, AutoModelForCausalLM

from model.decoders.base import BaseDecoder
from model.losses import WeightLoss, get_loss
from model.model_utils import get_tokenizer
from utils.constants import COMMON_SPECIAL_TOKENS, DecoderTypes, HeadTypes, Tasks
from utils.misc import get_file, is_otor


class TransformerDecoder(BaseDecoder):
    def __init__(self, cfg, decoder_cfg, decoder_name):
        super().__init__()
        self.cfg = cfg
        self.decoder_cfg = decoder_cfg
        self.decoder_name = decoder_name

        self.model = _get_huggingface_decoder(decoder_cfg)
        if self.decoder_cfg.kwargs.n_prune_layers > 0:
            _prune_decoder_layers(self.model, self.decoder_cfg)

        self.tokenizer = get_tokenizer(
            decoder_cfg.huggingface_path,
            decoder_cfg.kwargs.tokenizer_name,
            self.decoder_name,
            cfg.model.resume_model_path,
            cfg.model.w_pretrained_model_path,
            decoder_cfg.kwargs.tokenizer_path,
        )
        self.decoder_max_length = decoder_cfg.kwargs.decoder_max_length
        self._add_special_tokens(cfg.dataset_items)
        self.end_prompt_token = "[END_PROMPT]"
        self.end_prompt_token_id = self.tokenizer.convert_tokens_to_ids(
            self.end_prompt_token
        )
        self.decoder_end_token = "[END]"
        self.end_token_id = self.tokenizer.convert_tokens_to_ids(self.decoder_end_token)

        self.target_block_start_token_id_dict = {
            Tasks.OCR_READ: self.tokenizer.convert_tokens_to_ids("[START_BOX]"),
            Tasks.OCR_READ_2HEAD: self.tokenizer.convert_tokens_to_ids(
                "[START_TEXT_BLOCK]"
            ),
            Tasks.OCR_READ_TEXTINSTANCEPADDING: self.tokenizer.convert_tokens_to_ids(
                "[START_BOX]"
            ),
            Tasks.OTOR: self.tokenizer.convert_tokens_to_ids("[START_BOX]"),
            Tasks.OTOR_ORACLE: self.tokenizer.convert_tokens_to_ids("[START_BOX]"),
        }

        self.target_block_token_id_dict = {
            Tasks.OCR_READ: self.tokenizer.convert_tokens_to_ids("[END_BOX]"),
            Tasks.OCR_READ_TEXTINSTANCEPADDING: self.tokenizer.convert_tokens_to_ids(
                "[END_BOX]"
            ),
            Tasks.OCR_READ_2HEAD: self.tokenizer.convert_tokens_to_ids(
                "[END_TEXT_BLOCK]"
            ),
            Tasks.OTOR: self.tokenizer.convert_tokens_to_ids("[END_BOX]"),
            Tasks.OTOR_ORACLE: self.tokenizer.convert_tokens_to_ids("[END_BOX]"),
        }

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference

        self.weight_loss = WeightLoss(cfg.dataset_items)
        self.loss_func = get_loss(decoder_cfg.loss_func)
        self.ignore_index = -100
        self.calc_val_loss = decoder_cfg.calc_val_loss
        self.calc_confidence = decoder_cfg.calc_confidence

        self.connector_size = self.model.config.hidden_size
        if decoder_cfg.head_type == HeadTypes.TWO_HEAD:
            self.loc_loss_2head_func = get_loss(decoder_cfg.loc_loss_2head_func)
            self.reg_head = self.__get_reg_head(
                decoder_cfg.kwargs.n_loc_head_hidden_layers
            )

        if decoder_cfg.scob.use:
            self.projector_q = self._get_projector(decoder_cfg.scob.project_dim)
            self.projector_k = self._get_projector(decoder_cfg.scob.project_dim)
            self.otor_scale_factor = decoder_cfg.scob.loss_weight

    def _get_projector(self, output_dim):
        hidden_size = self.model.config.hidden_size
        projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_dim),
        )
        return projection_head

    @overrides
    def forward(self, dec_batch, connector_out, enc_kwargs):
        if self.training:
            output_dict, losses_dict, kwargs = self.train_forward(
                dec_batch, connector_out
            )
        else:
            output_dict, losses_dict, kwargs = self.valtest_forward(
                dec_batch, connector_out
            )
        return output_dict, losses_dict, kwargs

    def train_forward(self, dec_batch, connector_out):

        input_ids = dec_batch["input_ids"]
        attention_mask = dec_batch["attention_mask"]
        labels = dec_batch["labels"]

        if is_otor(dec_batch["task_names"][0], oracle=True):
            input_ids = torch.cat((input_ids, dec_batch["origin_input_ids"]), 0)
            attention_mask = torch.cat((attention_mask, attention_mask), 0)
            labels = torch.cat((labels, dec_batch["origin_labels"]), 0)
        elif is_otor(dec_batch["task_names"][0]):
            input_ids = torch.cat((input_ids, dec_batch["origin_input_ids"]), 0)
            attention_mask = torch.cat((attention_mask, attention_mask), 0)
            labels = torch.cat((labels, labels), 0)

        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()

        decoder_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=connector_out,
            # labels=labels,  # comment out to get per-example loss
            output_hidden_states=True,
        )

        projection = None
        if self.decoder_cfg.scob.use:
            bsz = decoder_outputs.hidden_states[-1].shape[0]
            projection = torch.cat(
                [
                    self.projector_q(decoder_outputs.hidden_states[-1][: int(bsz / 2)]),
                    self.projector_k(decoder_outputs.hidden_states[-1][int(bsz / 2) :]),
                ],
                dim=0,
            )

        pr_sequences = None  # Not used in Train phase
        token_loss = self.get_per_example_loss(
            dec_batch, decoder_outputs.logits, labels, projection
        )

        supcon_loss = 0
        if (
            is_otor(dec_batch["task_names"][0], or_oracle=True)
            and self.decoder_cfg.scob.use
        ):
            supcon_loss = token_loss["supcon_loss"]
            token_loss = token_loss["token_loss"]

        aux_loss = 0
        if self.decoder_cfg.aux_loss:
            aux_loss = self.get_per_example_aux_loss(
                dec_batch, decoder_outputs.hidden_states[1:-1], labels
            )

        if self.do_2head_regression(dec_batch):
            loc_loss = self.get_per_example_2head_loc_loss(dec_batch, decoder_outputs)
        else:
            loc_loss = 0

        token_loss *= self.decoder_cfg.token_loss_weight
        loc_loss *= self.decoder_cfg.loc_loss_weight

        decoder_loss = token_loss + loc_loss + aux_loss + supcon_loss

        output_dict = {
            "pr_sequences": pr_sequences,
            "pr_loc_outputs": None,  # Not used in Train phase
            "pr_confs": None,  # Not used in Train phase
        }
        losses_dict = {
            "decoder_token_loss": token_loss,
            "decoder_aux_loss": aux_loss,
            "decoder_supcon_loss": supcon_loss,
            "decoder_loc_loss": loc_loss,
            "decoder_loss": decoder_loss,
        }
        kwargs = {}

        return output_dict, losses_dict, kwargs

    def do_2head_regression(self, dec_batch):
        do_reg = False
        if self.decoder_cfg.head_type == HeadTypes.TWO_HEAD and (
            "block_quads" in dec_batch or "od_block_quads" in dec_batch
        ):
            do_reg = True
        return do_reg

    def valtest_forward(self, dec_batch, connector_out):
        input_ids = dec_batch["input_ids"]

        if is_otor(dec_batch["task_names"][0], or_oracle=True):
            bsz, _, _ = connector_out.size()
            connector_out = connector_out[: int(bsz / 2)]

        bsz = input_ids.shape[0]
        do_reg = self.do_2head_regression(dec_batch)

        # https://github.com/huggingface/transformers/blob/master/src/transformers/generation_utils.py#L640
        decoder_outputs = self.model.generate(
            input_ids=input_ids[:, : dec_batch["max_end_prompt_token_idx"] + 1],
            encoder_hidden_states=connector_out,
            max_length=self.decoder_max_length + 1,
            num_beams=1,
            early_stopping=False if self.calc_val_loss else True,
            use_cache=True,
            eos_token_id=self.end_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=do_reg,
            forced_eos_token_id=False,
        )

        if self.decoder_cfg.kwargs.tokenizer_name.startswith("char_"):
            pr_sequences = []
            for pr_idx in range(decoder_outputs.sequences.shape[0]):
                pr_sequence = "".join(
                    self.tokenizer.convert_ids_to_tokens(
                        decoder_outputs.sequences[pr_idx, :-1]
                    )
                )
                pr_sequences.append(pr_sequence)
        else:
            pr_sequences = self.tokenizer.batch_decode(
                decoder_outputs.sequences[:, :-1]
            )

        if self.calc_val_loss:
            scores = torch.stack(decoder_outputs.scores).permute(1, 0, 2).contiguous()
            scores = scores[:, :-1, :].contiguous()
            labels = dec_batch["labels"][
                :, dec_batch["max_end_prompt_token_idx"] + 1 :
            ].contiguous()
            decoder_token_loss = self.get_per_example_loss(dec_batch, scores, labels)
        else:
            decoder_token_loss = 0  # dummy value

        if do_reg:
            decoder_hidden_states = self.get_decoder_hidden_states(decoder_outputs)
            if torch.is_tensor(decoder_hidden_states):
                reg_outputs = self.reg_head(decoder_hidden_states)
                pr_loc_outputs = self.get_pr_loc_outputs(
                    dec_batch, decoder_outputs, reg_outputs
                )
            else:
                pr_loc_outputs = [[] for _ in range(bsz)]

            if self.calc_val_loss:
                decoder_loc_loss = 0  # Not implemented yet
            else:
                decoder_loc_loss = 0  # dummy value
        else:
            pr_loc_outputs = [[] for _ in range(bsz)]
            decoder_loc_loss = 0

        if self.calc_confidence:
            pr_confs = self.get_pr_confs(dec_batch, decoder_outputs)
        else:
            pr_confs = [[] for _ in range(bsz)]

        decoder_token_loss *= self.decoder_cfg.token_loss_weight
        decoder_loc_loss *= self.decoder_cfg.loc_loss_weight

        decoder_loss = decoder_token_loss + decoder_loc_loss
        output_dict = {
            "pr_sequences": pr_sequences,
            "pr_loc_outputs": pr_loc_outputs,
            "pr_confs": pr_confs,
        }
        losses_dict = {
            "decoder_token_loss": decoder_token_loss,
            "decoder_loc_loss": decoder_loc_loss,
            "decoder_loss": decoder_loss,
        }
        kwargs = {}

        return output_dict, losses_dict, kwargs

    def get_per_example_2head_loc_loss(self, dec_batch, decoder_outputs):
        reg_outputs = self.reg_head(decoder_outputs.hidden_states[-1])
        block_quads = dec_batch["block_quads"][:, 1:].contiguous()
        block_first_tokens = dec_batch["block_first_tokens"][:, 1:].contiguous()

        loc_loss = self.loc_loss_2head_func(reg_outputs, block_quads)
        loc_loss = loc_loss.mean(-1)
        loc_loss = loc_loss * block_first_tokens
        loc_loss = loc_loss.sum(1) / block_first_tokens.sum(1)
        # To avoid NaN loss
        loc_loss = torch.nan_to_num(loc_loss, nan=0.0, posinf=0.0, neginf=0.0)
        loc_loss = self.weight_loss(dec_batch, loc_loss).mean()

        return loc_loss

    def get_decoder_hidden_states(self, decoder_outputs):
        decoder_hidden_states = []
        for token_idx, all_layers_hidden_states in enumerate(
            decoder_outputs.hidden_states[:-1]
        ):
            if token_idx == 0:
                decoder_hidden_states.append(
                    all_layers_hidden_states[-1][:, -1, :].unsqueeze(1).contiguous()
                )
            else:
                decoder_hidden_states.append(all_layers_hidden_states[-1].contiguous())

        if len(decoder_hidden_states) == 0:
            return 0
        decoder_hidden_states = torch.cat(decoder_hidden_states, 1)

        return decoder_hidden_states

    def get_pr_loc_outputs(self, dec_batch, decoder_outputs, reg_outputs):
        bsz = reg_outputs.shape[0]
        pr_loc_outputs = [[] for _ in range(bsz)]
        for example_idx, sequence in enumerate(decoder_outputs.sequences):
            task_name = dec_batch["task_names"][example_idx]
            target_block_token_id = self.target_block_token_id_dict[task_name]
            is_target_block_token_indices = (
                sequence[:-1][dec_batch["max_end_prompt_token_idx"] + 1 :]
                == target_block_token_id
            )

            pr_loc_outputs[example_idx] = reg_outputs[
                example_idx, is_target_block_token_indices, :
            ]
        return pr_loc_outputs

    def get_pr_confs(self, dec_batch, decoder_outputs):
        logits = torch.stack(decoder_outputs.scores).permute(1, 0, 2).contiguous()
        probs = nn.functional.softmax(logits, -1).cpu().numpy()

        pr_confs = []
        for example_idx, sequence in enumerate(decoder_outputs.sequences):
            task_name = dec_batch["task_names"][example_idx]
            target_block_start_token_id = self.target_block_start_token_id_dict[
                task_name
            ]
            target_block_token_id = self.target_block_token_id_dict[task_name]

            wo_prompt_seq = sequence[
                torch.nonzero(sequence == self.end_prompt_token_id) + 1 :
            ]

            seq_pr_confs = []
            token_probs = []
            for token_idx, token_id in enumerate(wo_prompt_seq):
                if task_name == "object_detection":
                    if token_id == target_block_token_id:
                        seq_pr_confs.append(
                            probs[example_idx, token_idx - 1].max()
                        )  # cls confidence
                        # seq_pr_confs.append(probs[example_idx, token_idx-5:token_idx-1].max(1).mean()) # box confidence mean
                else:
                    if token_id == target_block_start_token_id:  # [START_TEXT_BLOCK]
                        token_probs.clear()
                    elif token_id == target_block_token_id:  # [END_TEXT_BLOCK]
                        # Since pr_reg_outputs is generated on every [END_TEXT_BLOCK]
                        if len(token_probs) == 0:
                            block_prob = 0.0
                        else:
                            block_prob = sum(token_probs) / len(token_probs)
                        seq_pr_confs.append(block_prob)
                        token_probs.clear()
                    else:
                        token_probs.append(probs[example_idx, token_idx, token_id])

            pr_confs.append(seq_pr_confs)
        return pr_confs

    def __get_reg_head(self, n_loc_head_hidden_layers):
        # https://github.com/facebookresearch/detr/blob/main/models/detr.py#L38
        # https://github.com/facebookresearch/detr/blob/091a817eca74b8b97e35e4531c1c39f89fbe38eb/models/detr.py#L289
        reg_heads = []
        hidden_dim = self.model.config.hidden_size
        for _ in range(n_loc_head_hidden_layers):
            reg_heads.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        reg_heads.append(nn.Linear(hidden_dim, 8))
        return nn.Sequential(*reg_heads)

    def prepare_inputs_for_inference(
        self, input_ids, past=None, attention_mask=None, use_cache=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        if past is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
            "encoder_hidden_states": model_kwargs["encoder_hidden_states"],
        }

    def _add_special_tokens(self, dataset_items):
        special_tokens = []
        for dataset_item in dataset_items:

            special_token_file = get_file(
                dataset_path=dataset_item.dataset_root_path,
                prefix="special_tokens",
                postfix=dataset_item.dataset_label,
                ext=".txt",
            )
            this_special_tokens = (
                open(special_token_file, "r", encoding="utf-8")
                .read()
                .strip()
                .split("\n")
            )
            special_tokens.extend(this_special_tokens)

        # Add common special tokens (e.g. [START_PROMPT], ...)
        special_tokens.extend(COMMON_SPECIAL_TOKENS)

        special_tokens = sorted(list(set(special_tokens)))

        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def _otor_masking_labels(self, labels, task_name):
        """origin_img Generate label value for predicted part, masking pos part"""
        bsz, _ = labels.size()
        pos_indices = torch.logical_and(
            labels >= self.tokenizer.convert_tokens_to_ids("[POS_0]"),
            labels <= self.tokenizer.convert_tokens_to_ids("[POS_9]"),
        )

        pos_indices[: int(bsz / 2)] = False
        labels[pos_indices] = self.ignore_index
        return labels

    def _get_supcon_loss(self, projection, labels):
        bsz, n_decoder_tokens, dim_size = projection.shape

        projection_gen = nn.functional.normalize(
            projection[: int(bsz / 2)], dim=1
        ).view(-1, dim_size)
        projection_origin = nn.functional.normalize(
            projection[int(bsz / 2) :], dim=1
        ).view(-1, dim_size)

        labels_for_supcon = labels[: int(bsz / 2)].clone().view(-1)
        not_char_indices = torch.logical_not(
            torch.logical_and(  # Finding char index with char tokenizer
                labels_for_supcon >= self.tokenizer.convert_tokens_to_ids(" "),
                labels_for_supcon < self.tokenizer.convert_tokens_to_ids("[CHAR_PAD]"),
            )
        )

        labels_for_mask = labels_for_supcon
        labels_for_mask[not_char_indices] = self.ignore_index
        char_indices = (labels_for_mask != self.ignore_index).nonzero(as_tuple=True)[0]

        projection_gen = projection_gen[char_indices]
        projection_origin = projection_origin[char_indices]
        labels_for_mask = labels_for_mask[char_indices]

        anchor_feature = torch.cat([projection_gen, projection_origin], dim=0)
        labels = torch.cat([labels_for_mask, labels_for_mask], dim=0)
        mask = torch.eq(labels[:, None], labels[:, None].T)  # nk

        logits = torch.einsum("nc,ck->nk", [anchor_feature, anchor_feature.T])
        logits /= 0.07  # tau

        supcon_loss = (
            (-torch.log_softmax(logits, dim=-1) * mask)
            .sum(dim=-1, keepdim=True)
            .div(mask.sum(dim=-1, keepdim=True) + 1e-5)
        )
        supcon_loss = supcon_loss.mean() * self.decoder_cfg.scob.loss_weight

        return supcon_loss

    def get_per_example_loss(self, dec_batch, logits, labels, projection=None):
        bsz, n_decoder_tokens, vocab_size = logits.shape
        if is_otor(dec_batch["task_names"][0]):
            labels = self._otor_masking_labels(labels, dec_batch["task_names"][0])

        mask = labels != self.ignore_index
        if self.decoder_cfg.loss_func.name == "focal":
            # To circumvent the error in focal_loss,
            # the labels corresponding to ignore_index are replaced with 0.
            # Afterwards, it is not a problem becuase the loss is calculated
            # element-wisely using the ignore_index.
            target_labels = labels * mask
        else:
            target_labels = labels

        loss = self.loss_func(logits.view(-1, vocab_size), target_labels.view(-1))
        loss = loss.view(bsz, n_decoder_tokens)

        loss *= mask if mask.any() else 0.0

        # mask.sum(1) will never be zero
        # For that to happen, all labels must be ignore_index (-100).
        loss = loss.sum(1) / mask.sum(1)
        loss = self.weight_loss(dec_batch, loss).mean()

        if (
            is_otor(dec_batch["task_names"][0], or_oracle=True)
            and projection is not None
            and self.decoder_cfg.scob.use
        ):
            supcon_loss = self._get_supcon_loss(
                projection,
                labels,
            )
            loss = {"token_loss": loss, "supcon_loss": supcon_loss}

        return loss

    def get_per_example_aux_loss(self, dec_batch, hidden_states, labels):
        hidden_states_tensor = torch.cat(hidden_states, dim=0)
        logits = self.model.cls(hidden_states_tensor)

        bsz, n_decoder_tokens, vocab_size = logits.shape
        clone_labels = labels.clone()
        labels_for_aux = torch.cat(
            [clone_labels for _ in range(len(hidden_states))],
            dim=0,
        )

        mask = labels_for_aux != self.ignore_index
        if self.decoder_cfg.loss_func.name == "focal":
            # To circumvent the error in focal_loss,
            # the labels corresponding to ignore_index are replaced with 0.
            # Afterwards, it is not a problem becuase the loss is calculated
            # element-wisely using the ignore_index.
            target_labels = labels_for_aux * mask
        else:
            target_labels = labels_for_aux

        loss = self.loss_func(logits.view(-1, vocab_size), target_labels.view(-1))
        loss = loss.view(bsz, n_decoder_tokens)

        loss *= mask if mask.any() else 0.0

        # mask.sum(1) will never be zero
        # For that to happen, all labels must be ignore_index (-100).
        loss = loss.sum(1) / mask.sum(1)
        loss = self.weight_loss(dec_batch, loss).mean()
        return loss

    @overrides
    def get_step_out_dict(
        self, batch, batch_idx, loader_idx, dec_out_dict, enc_kwargs, dec_kwargs
    ):
        dec_batch = batch[self.decoder_name]

        pr_sequences = dec_out_dict["pr_sequences"]
        gt_sequences = dec_batch["gt_sequences"]

        pr_loc_outputs = dec_out_dict["pr_loc_outputs"]
        gt_loc_outputs = dec_batch["original_block_quads"]

        pr_confs = dec_out_dict["pr_confs"]

        norm_ed_dict = defaultdict(list)
        score_dict = defaultdict(list)

        gt_jsons = []  # for evaluation

        for example_idx, (
            pr_sequence,
            gt_sequence,
            pr_loc_output,
            gt_loc_output,
            pr_conf,
        ) in enumerate(
            zip(pr_sequences, gt_sequences, pr_loc_outputs, gt_loc_outputs, pr_confs)
        ):
            verbose = batch_idx == 0 and example_idx == 0
            dataset_name: str = dec_batch["dataset_names"][example_idx]
            task_name: str = dec_batch["task_names"][example_idx]

            dtd_key = (dataset_name, task_name, self.decoder_name)
            pr_sequence, gt_sequence = self.seq_truncate_end(pr_sequence, gt_sequence)
            norm_ed = self.calc_norm_ed(pr_sequence, gt_sequence)
            norm_ed_dict[dtd_key].append(norm_ed)

            if verbose:
                out_str = f"pr:\t{pr_sequence}\n"
                out_str += f"gt:\t{gt_sequence}\n"
                out_str += f"norm_ed:\t{norm_ed:.4f}"
                print(out_str, flush=True)

            if isinstance(pr_loc_output, torch.Tensor):  # two-head model
                pr_loc_output = pr_loc_output.cpu().numpy().tolist()

            if len(gt_loc_output) == 0:
                pr_loc_output = None
                gt_loc_output = None

            pr_dict = {"pr_seq": pr_sequence, "pr_loc_output": pr_loc_output}
            all_answers = dec_batch["all_answers"][example_idx]

            gt_dict = {
                "gt_seq": gt_sequence,
                "gt_loc_output": gt_loc_output,
                "all_answers": all_answers,
            }

            kwargs = {
                "img_path": dec_batch["metas"][example_idx]["img_path"],
                "width": dec_batch["metas"][example_idx]["img_size"]["width"],
                "height": dec_batch["metas"][example_idx]["img_size"]["height"],
                "task_name": task_name,
                "tokenizer_name": self.decoder_cfg.kwargs.tokenizer_name,
                "char_pad_token": "[CHAR_PAD]",
            }

            gt_json = None
            if dataset_name.startswith("donut_") and task_name == Tasks.DONUT_KIE:
                kwargs["gt_json"] = dec_batch["gt_jsons"][example_idx]
                gt_json = kwargs["gt_json"]
                gt_jsons.append(gt_json)

            result = self.downstream_evaluator.downstream_score(
                dtd_key,
                DecoderTypes.TRANSFORMER,
                pr_dict,
                gt_dict,
                kwargs,
                verbose=verbose,
            )
            score_dict[dtd_key].append(result)

        step_out_dict = {}
        for dtd_key in score_dict.keys():
            # contstruct step-out for accumulation
            step_out = {
                "loader_idx": loader_idx,
                "pr_sequences": pr_sequences,
                "gt_sequences": gt_sequences,
                "gt_jsons": gt_jsons,
                "norm_eds": norm_ed_dict[dtd_key],
                "scores": score_dict[dtd_key],
            }
            step_out_dict[dtd_key] = step_out

        return step_out_dict

    def seq_truncate_end(self, pr_sequence, gt_sequence):
        if self.decoder_end_token in pr_sequence:
            pr_dec_end_idx = pr_sequence.find(self.decoder_end_token)
            pr_sequence = pr_sequence[:pr_dec_end_idx].strip()
        gt_dec_end_idx = gt_sequence.find(self.decoder_end_token)
        gt_sequence = gt_sequence[:gt_dec_end_idx].strip()

        return pr_sequence, gt_sequence

    def calc_norm_ed(self, pr_sequence, gt_sequence):
        pr_end_prompt_idx = pr_sequence.find(self.end_prompt_token)
        gt_end_prompt_idx = gt_sequence.find(self.end_prompt_token)
        without_prompt_pr = pr_sequence[
            pr_end_prompt_idx + len(self.end_prompt_token) :
        ].strip()
        without_prompt_gt = gt_sequence[
            gt_end_prompt_idx + len(self.end_prompt_token) :
        ].strip()

        norm_factor = max(len(without_prompt_pr), len(without_prompt_gt))
        if norm_factor == 0:
            # When gt sequence is empty (e.g. 0 objects in object_detection task)
            norm_ed = 0.0
        else:
            norm_ed = distance(without_prompt_pr, without_prompt_gt) / norm_factor

        return norm_ed

    def init_weights(self, weights_init_fn):
        for module_name, module in self.__dict__["_modules"].items():
            if module_name != "model":  # pretrained weight from huggingface
                module.apply(weights_init_fn)

        if self.decoder_cfg.kwargs.tokenizer_name == "char_en__bert-base-cased":
            # https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L731
            word_embeddings = self.model.bert.embeddings.word_embeddings
            word_embeddings.weight.data.normal_(mean=0.0, std=0.02)
            word_embeddings.weight.data[0].zero_()  # padding_idx == 0


def _get_huggingface_decoder(decoder_cfg):
    """get huggingface transformer decoder"""
    transformers.logging.set_verbosity_error()

    decoder_model_path = decoder_cfg.kwargs.decoder_model_name

    if decoder_cfg.kwargs.decoder_model_name in [
        "bert-base-cased",
        "bert-base-multilingual-cased",
        "roberta-base",
        "xlm-roberta-base",
        "facebook/bart-base",
        "facebook/mbart-large-cc25",
    ]:
        config = AutoConfig.from_pretrained(decoder_model_path)

        if decoder_cfg.kwargs.decoder_model_name in [
            "bert-base-cased",
            "bert-base-multilingual-cased",
            "roberta-base",
            "xlm-roberta-base",
        ]:
            config.is_decoder = True
            config.add_cross_attention = True
            config.position_embedding_type = decoder_cfg.kwargs.pe_type

        elif decoder_cfg.kwargs.decoder_model_name in [
            "facebook/bart-base",
            "facebook/mbart-large-cc25",
        ]:
            config.is_decoder = True
            config.add_cross_attention = True
            config.is_encoder_decoder = False

        if decoder_cfg.kwargs.decoder_max_length <= config.max_position_embeddings:
            decoder = AutoModelForCausalLM.from_pretrained(
                decoder_model_path, config=config
            )
        else:
            # If the decoder_max_length written in config file
            # is longer than length in pre-trained model,
            # interpoate the position_embeddings.
            pretrained_decoder = AutoModelForCausalLM.from_pretrained(
                decoder_model_path, config=config
            )

            config.max_position_embeddings = decoder_cfg.kwargs.decoder_max_length
            if decoder_cfg.kwargs.decoder_model_name in [
                "roberta-base",
                "xlm-roberta-base",
            ]:
                config.max_position_embeddings += 2
            decoder = AutoModelForCausalLM.from_config(config)

            interpolate_pos_emb_and_load(
                pretrained_decoder, decoder, config.max_position_embeddings
            )

    elif decoder_cfg.kwargs.decoder_model_name in ["hyunwoongko/asian-bart-ecjk"]:
        config = AutoConfig.from_pretrained(decoder_model_path)
        config.is_decoder = True
        config.add_cross_attention = True
        config.is_encoder_decoder = False

        if decoder_cfg.kwargs.decoder_max_length <= config.max_position_embeddings:
            decoder = AsianBartForCausalLM.from_pretrained(
                decoder_model_path, config=config
            )
        else:
            # If the decoder_max_length written in config file
            # is longer than length in pre-trained model,
            # interpoate the position_embeddings.
            pretrained_decoder = AsianBartForCausalLM.from_pretrained(
                decoder_model_path, config=config
            )

            config.max_position_embeddings = decoder_cfg.kwargs.decoder_max_length
            decoder = AsianBartForCausalLM(config)

            interpolate_pos_emb_and_load(
                pretrained_decoder, decoder, config.max_position_embeddings
            )

    else:
        raise ValueError(
            f"Invalid decoder_model_name={decoder_cfg.kwargs.decoder_model_name}"
        )

    return decoder


def _prune_decoder_layers(decoder, decoder_cfg):
    if decoder_cfg.kwargs.decoder_model_name in [
        "bert-base-cased",
        "bert-base-multilingual-cased",
    ]:
        layers = decoder.bert.encoder.layer
    elif decoder_cfg.kwargs.decoder_model_name in ["roberta-base", "xlm-roberta-base"]:
        layers = decoder.roberta.encoder.layer
    elif decoder_cfg.kwargs.decoder_model_name in [
        "facebook/bart-base",
        "facebook/mbart-large-cc25",
        "hyunwoongko/asian-bart-ecjk",
    ]:
        layers = decoder.model.decoder.layers
    else:
        raise ValueError(
            f"Invalid decoder_model_name={decoder_cfg.kwargs.decoder_model_name}"
        )

    n_prune_layers = decoder_cfg.kwargs.n_prune_layers
    n_decoder_layers = decoder.config.num_hidden_layers

    if n_prune_layers >= n_decoder_layers:
        raise ValueError(
            f"n_prune_layers ({n_prune_layers}) >= n_decoder_layers ({n_decoder_layers})"
        )

    if decoder_cfg.kwargs.prune_layer_position == "lower":
        for _ in range(n_prune_layers):
            del layers[0]
    elif decoder_cfg.kwargs.prune_layer_position == "upper":
        for _ in range(n_prune_layers):
            del layers[-1]
    else:
        raise ValueError(
            f"Unknown prune_layer_position={decoder_cfg.prune_layer_position}"
        )

    decoder.config.num_hidden_layers = len(layers)


def interpolate_pos_emb_and_load(pretrained_decoder, decoder, _max_position_embeddings):
    """
    When load the weights in pretrained_decoder to decoder,
    resize the position_embeddings in pretrained_decoder through interpolation
    and load weights.
    """

    decoder_class_name = decoder.__class__.__name__
    max_position_embeddings = _max_position_embeddings
    if decoder_class_name == "BertLMHeadModel":
        position_ids_key_name = "bert.embeddings.position_ids"
        position_embeddings_key_name = "bert.embeddings.position_embeddings.weight"
    elif decoder_class_name in ["RobertaForCausalLM", "XLMRobertaForCausalLM"]:
        position_ids_key_name = "roberta.embeddings.position_ids"
        position_embeddings_key_name = "roberta.embeddings.position_embeddings.weight"
    elif decoder_class_name in [
        "BartForCausalLM",
        "MBartForCausalLM",
        "AsianBartForCausalLM",
    ]:
        position_embeddings_key_name = "model.decoder.embed_positions.weight"
        max_position_embeddings += 2
    else:
        raise ValueError(f"Unknown decoder_class_name={decoder_class_name}")

    pretrained_decoder_state_dict = pretrained_decoder.state_dict()
    if decoder_class_name not in [
        "BartForCausalLM",
        "MBartForCausalLM",
        "AsianBartForCausalLM",
    ]:
        # position_ids
        pretrained_decoder_state_dict[position_ids_key_name] = torch.arange(
            max_position_embeddings
        ).expand((1, -1))

    # position_embeddings
    resized_pos_emb = resize_position_embeddings(
        pretrained_decoder_state_dict[position_embeddings_key_name],
        max_position_embeddings,
    )
    pretrained_decoder_state_dict[position_embeddings_key_name] = resized_pos_emb

    decoder.load_state_dict(pretrained_decoder_state_dict)


def resize_position_embeddings(pos_emb, decoder_max_length):
    resized_pos_emb = pos_emb.transpose(0, 1).contiguous().unsqueeze(0)
    resized_pos_emb = F.interpolate(
        resized_pos_emb, size=decoder_max_length, mode="linear", align_corners=False
    )
    resized_pos_emb = resized_pos_emb.squeeze(0).transpose(0, 1).contiguous()
    return resized_pos_emb
