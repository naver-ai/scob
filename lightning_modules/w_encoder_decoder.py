"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import time

import torch
import torch.utils.data
from overrides import overrides

from lightning_modules.result_extractors import get_result_extractors
from lightning_modules.w import W
from model.model_utils import load_state_dict
from utils.constants import Tasks
from utils.saver import change_permissions_recursive, prepare_checkpoint_dir


class WEncDec(W):
    def __init__(self):
        super().__init__()

        if self.cfg.model.w_pretrained_model_path:
            model_path = self.cfg.model.w_pretrained_model_path
            state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
            kwargs = {}
            if self.cfg.model.encoder.encoder_model_name.startswith("swin_transformer"):
                kwargs["window_size"] = self.cfg.model.encoder[
                    self.cfg.model.encoder.encoder_model_name
                ].window_size
            load_state_dict(self.net, state_dict, "w_pretrained_model", kwargs)

        self.time_tracker = None
        self.val_result_extractors = None
        self.test_result_extractors = None
        self.masked_img_logged = False
        self.ignore_index = -100

    @overrides
    def setup(self, stage):
        self.time_tracker = time.time()
        if stage == "fit" and self.global_rank == 0:
            prepare_checkpoint_dir()
            change_permissions_recursive(self.cfg.tensorboard_dir, 0o777)
            change_permissions_recursive(self.cfg.save_weight_dir, 0o777)
        self.val_result_extractors = get_result_extractors(self.cfg.dataset_items)
        self.test_result_extractors = get_result_extractors(self.cfg.dataset_items)

    @overrides
    def training_step(self, batch, batch_idx):  # pylint: disable=W0613,W0221
        """training_step"""
        loss, _, decs_losses_dict, _, _ = self.net(batch)

        log_dict_input = {"train_loss": loss}
        for decoder_name, losses_dict in decs_losses_dict.items():
            for k, v in losses_dict.items():
                log_dict_input[f"train_{decoder_name}_{k}"] = v
        self.log_dict(log_dict_input, sync_dist=True)

        if self.global_step % self.cfg.train.log_interval == 0:
            self._log_shell(log_dict_input, prefix="train ")

        return loss

    def _val_test_step(self, batch, batch_idx, loader_idx, mode):
        total_loss, decs_out_dict, decs_losses_dict, enc_kwargs, decs_kwargs = self.net(
            batch
        )
        decoder_name = batch["decoder_names"][0]
        decoder = self.net.decoders.decoder_dict[decoder_name]
        dec_out_dict = decs_out_dict[decoder_name]
        dec_kwargs = decs_kwargs[decoder_name]

        dec_step_out_dict = decoder.get_step_out_dict(
            batch, batch_idx, loader_idx, dec_out_dict, enc_kwargs, dec_kwargs
        )

        for dtd_key, dec_step_out in dec_step_out_dict.items():
            dec_step_out["loss"] = total_loss
            for k, v in decs_losses_dict[decoder_name].items():
                dec_step_out[k] = v

            if mode == "val":
                self.val_result_extractors[dtd_key].extract(dec_step_out)
            elif mode == "test":
                self.test_result_extractors[dtd_key].extract(dec_step_out)

        return {"loss": total_loss}

    @overrides
    def on_validation_start(self):
        self.masked_img_logged = False
        for decoder in self.net.decoders.decoder_dict.values():
            if hasattr(decoder, "rgb2gray_vec"):
                decoder.rgb2gray_vec = decoder.rgb2gray_vec.to(self.device)
            if (
                hasattr(decoder, "normalize_mean")
                and decoder.normalize_mean is not None
            ):
                decoder.normalize_mean = decoder.normalize_mean.to(self.device)
            if hasattr(decoder, "normalize_std") and decoder.normalize_std is not None:
                decoder.normalize_std = decoder.normalize_std.to(self.device)

    @torch.no_grad()
    def _extract_results(self, step_outputs, mode):
        """Extract validation / test results and logging."""
        assert mode in ["val", "test"]

        final_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        final_norm_ed = torch.tensor(0, dtype=torch.float32, device=self.device)
        final_loss_denom, final_norm_ed_denom = 0, 0
        extractors = (
            self.val_result_extractors if mode == "val" else self.test_result_extractors
        )
        for dtd_key, extractor in extractors.items():
            dataset_name, task_name, decoder_name = dtd_key
            prefix = f"{dataset_name}/{task_name}/{decoder_name}/{mode}_"
            log_info = extractor.get_log_info(prefix)

            self.log_dict(log_info, sync_dist=True)
            self._log_shell(log_info, prefix=f"{mode} ")

            # Since CLEval works with torchmetric, we simply get a macro result with it.
            if task_name in [
                Tasks.OCR_READ,
                Tasks.OCR_READ_2HEAD,
                Tasks.OCR_READ_TEXTINSTANCEPADDING,
                Tasks.OTOR,
                Tasks.OTOR_ORACLE,
            ]:
                decoder = self.net.decoders.decoder_dict[decoder_name]
                evaluator = decoder.downstream_evaluator
                cleval_result = evaluator.get_cleval_score(
                    dtd_key, self.device
                )
                cleval_dict = {
                    prefix + "det_r": cleval_result["det_r"],
                    prefix + "det_p": cleval_result["det_p"],
                    prefix + "det_h": cleval_result["det_h"],
                    prefix + "e2e_r": cleval_result["e2e_r"],
                    prefix + "e2e_p": cleval_result["e2e_p"],
                    prefix + "e2e_h": cleval_result["e2e_h"],
                }
                self.log_dict(cleval_dict, sync_dist=False)
                self._log_shell(cleval_dict, prefix=f"{mode} ")

            final_loss += log_info[prefix + "loss"]
            final_loss_denom += 1
            if prefix + "norm_ed" in log_info:
                final_norm_ed += log_info[prefix + "norm_ed"]
                final_norm_ed_denom += 1

        # macro averages
        log_info = {
            f"{mode}_loss": final_loss / final_loss_denom,
        }
        if final_norm_ed_denom > 0:
            log_info[f"{mode}_norm_ed"] = final_norm_ed / final_loss_denom

        self.log_dict(log_info, sync_dist=True)
        self._log_shell(log_info, prefix=f"{mode} ")

        for extractor in extractors.values():
            extractor.initialize()
