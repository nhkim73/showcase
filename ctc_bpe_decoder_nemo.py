# @name:        ctc_bpe_decoder_nemo.py
# @author:      nam (hoon) kim
# @date:        2023-01-02 04:18:54
# @updated:     2023-02-09 06:13:20
# @description: <description>

from enum import Enum
from pathlib import Path
from typing import *

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from lego.asr.decoder.asr_decoder import *
from lego.base.logger import logger
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

__all__ = ["CTCBPEDecoderNemo"]


class _ConfigSanity(object):
    def __check_adc_cfg_sanity(self, adc_cfg: Dict) -> None:
        if adc_cfg["tgt_spr"] <= 0.0:
            raise ValueError("invalid adc.tgt_spr ... %.2f" % adc_cfg["tgt_spr"])

    # def __check_adc_cfg_sanity(self, adc_cfg: Dict) -> None:

    def __check_fea_cfg_sanity(self, fea_cfg: Dict) -> None:
        for x in fea_cfg.keys():
            if fea_cfg[x] < 0:
                raise ValueError("invalid feature.%s ... %d" % (x, fea_cfg[x]))

        if fea_cfg["trim_frame_len"] <= fea_cfg["trim_shift_len"]:
            raise ValueError(
                "feature.trim_frame_len should be LARGER than feature.trim_shift_len ... (%d, %d)"
                % (fea_cfg["trim_frame_len"], fea_cfg["trim_shift_len"])
            )

    # def __check_fea_cfg_sanity(self, fea_cfg: Dict) -> None:

    def __check_decoder_cfg_sanity(self, decoder_cfg: Dict) -> None:
        if decoder_cfg["output_type"] not in [x.name for x in OutputType]:
            raise ValueError(
                "invalid decoder.output_type ... %s" % decoder_cfg["output_type"]
            )
        if not Path(decoder_cfg["mdl_name"]).exists():
            raise FileNotFoundError(
                "not found decoder.mdl_name ... %s" % decoder_cfg["mdl_name"]
            )

    # def __check_decoder_cfg_sanity(self, decoder_cfg: Dict) -> None:

    def __check_sys_cfg_sanity(self, sys_cfg: Dict) -> None:
        if sys_cfg["use_gpu"] and sys_cfg["decoder_cuda_id"] < 0:
            raise ValueError(
                "invalid sys.decoder_cuda_id ... %d" % sys_cfg["decoder_cuda_id"]
            )

    # def __check_sys_cfg_sanity(self, sys_cfg: Dict) -> None:

    def check(self, adc_cfg: Dict, fea_cfg: Dict, decoder_cfg: Dict, sys_cfg: Dict):
        try:
            self.__check_adc_cfg_sanity(adc_cfg)
            self.__check_fea_cfg_sanity(fea_cfg)
            self.__check_decoder_cfg_sanity(decoder_cfg)
            self.__check_sys_cfg_sanity(sys_cfg)
        except Exception:
            raise

    # def check(self, adc_cfg: Dict, fea_cfg: Dict, decoder_cfg: Dict, sys_cfg: Dict):


# class _ConfigSanity(Object):


class CTCBPEDecoderNemo(ASRDecoder):
    def __init__(
        self, adc_cfg: Dict, fea_cfg: Dict, decoder_cfg: Dict, sys_cfg: Dict
    ) -> None:
        # check config sanity
        cfg_sanity = _ConfigSanity()
        cfg_sanity.check(adc_cfg, fea_cfg, decoder_cfg, sys_cfg)
        logger.info("checked configuration sanity for CTCBPEDecoderNemo ...")

        # set up adc
        self.__tgt_spr = adc_cfg["tgt_spr"]

        # set up sys
        if sys_cfg["use_gpu"] and torch.cuda.is_available():
            logger.info("use_gpu = true, cuda:%d" % sys_cfg["decoder_cuda_id"])
            device = torch.device("cuda:%d" % sys_cfg["decoder_cuda_id"])
        else:
            logger.info("use_gpu = false, cpu utilised, instead")
            device = torch.device("cpu")

        # set up feature
        self.__trim_top_db = fea_cfg["trim_top_db"]
        self.__trim_frame_len = fea_cfg["trim_frame_len"]
        self.__trim_shift_len = fea_cfg["trim_shift_len"]
        self.__feat = WaveformFeaturizer(
            sample_rate=self.__tgt_spr, int_values=False, augmentor=None
        )

        # set up decoder
        self.__output_type = OutputType[decoder_cfg["output_type"]]
        self.__mdl_name = decoder_cfg["mdl_name"]
        self.__subword_mdl = nemo_asr.models.ASRModel.restore_from(self.__mdl_name)

        if self.__output_type == OutputType.timestamp:
            mdl_decoder_cfg = self.__subword_mdl.cfg.decoding
            mdl_decoder_cfg.preserve_alignments = True
            mdl_decoder_cfg.compute_timestamps = True
            self.__subword_mdl.change_decoding_strategy(mdl_decoder_cfg)
            logger.info("time stamp is applied ...")

        self.__subword_mdl = self.__subword_mdl.to(device)
        self.__device = next(self.__subword_mdl.parameters()).device
        self.__subword_mdl.eval()
        self.__subword_mdl.encoder.freeze()
        self.__subword_mdl.decoder.freeze()
        logger.info("loaded ctc bpe mdl ... %s" % decoder_cfg["mdl_name"])

    # def __init__(self, adc_cfg: Dict, fea_cfg: Dict, decoder_cfg: Dict, sys_cfg: Dict) -> None:

    def __prep_entire_feature(self, adcfea: torch.Tensor):
        tok = torch.tensor([]).long()  # it's always []
        tok_ln = torch.tensor(len(tok)).long()
        adcfea_ln = torch.tensor(adcfea.shape[0]).long()
        dur_sec = adcfea_ln / self.__tgt_spr
        return [(adcfea, adcfea_ln, tok, tok_ln, torch.tensor([0.0, dur_sec]))]

    # def __prep_entire_feature(self, adcfea: torch.Tensor):

    def __prep_epd_feature(self, adcfea: torch.Tensor, epd_ts: torch.Tensor):
        tok = torch.tensor([]).long()  # it's always []
        tok_ln = torch.tensor(len(tok)).long()
        spr = self.__tgt_spr
        epd_data = []
        for x in epd_ts:
            beg_pt = int(x[0] * spr)
            end_pt = int(x[1] * spr)
            segm = adcfea[beg_pt:end_pt]
            segm_ln = torch.tensor(len(segm)).long()
            epd_data.append((segm, segm_ln, tok, tok_ln, x))

        return epd_data

    # def __prep_epd_feature(self, adcfea: torch.Tensor, epd_ts: torch.Tensor):

    def __prep_datalayer(
        self, adcfea: torch.Tensor, epd_ts: torch.Tensor = None
    ) -> List[torch.Tensor]:
        if epd_ts == None:
            batch = self.__prep_entire_feature(adcfea)
        else:
            batch = self.__prep_epd_feature(adcfea, epd_ts)

        item_wise = list(zip(*batch))
        _, adcfea_lns, _, tok_lns, _ = item_wise

        # REMARK
        # not sure why it just checks the length of the 1st item.
        # it throws all samples in batch if the length of the 1st item is zero.
        has_adc = adcfea_lns[0] is not None
        max_adcfea_ln = max(adcfea_lns).item() if has_adc else 0
        max_tok_ln = max(tok_lns).item()

        logger.info("max audio feature length ... %d" % max_adcfea_ln)
        logger.info("max token length ... %d" % max_tok_ln)

        feaseq = []
        tokseq = []
        epdseq = []
        pad_id = 0

        for x in batch:
            cur_fea, cur_fea_ln, cur_tok, cur_tok_ln, epd_ts = x

            if has_adc:
                cur_ln = cur_fea_ln.item()
                if cur_ln < max_adcfea_ln:
                    padding = (0, max_adcfea_ln - cur_ln)
                    cur_fea = torch.nn.functional.pad(cur_fea, padding)
                feaseq.append(cur_fea)

            cur_tok_ln = cur_tok_ln.item()
            if cur_tok_ln < max_tok_ln:
                padding = (0, max_tok_len - cur_tok_ln)
                cur_tok = torch.nn.functional.pad(cur_tok, padding, value=pad_id)
            tokseq.append(cur_tok)
            epdseq.append(epd_ts)

        if has_adc:
            feaseq = torch.stack(feaseq)
            feaseq_ln = torch.stack(adcfea_lns)
        else:
            feaseq, feaseq_ln = None, None
        tokseq = torch.stack(tokseq)
        tokseq_ln = torch.stack(tok_lns)
        epdseq = torch.stack(epdseq)
        return (feaseq, feaseq_ln, tokseq, tokseq_ln, epdseq)

    # def __prep_datalayer(self, adcfea: torch.Tensor, epd_ts: torch.Tensor = None) -> List[torch.Tensor]:

    def run(self, adc: np.ndarray[int], epd_ts: torch.Tensor = None) -> List[Any]:
        # adc should be converted into AudioSegment obj to use
        # nemo wave feature.
        proc_adc = AudioSegment(
            adc,
            self.__tgt_spr,
            trim=False,
            trim_ref=np.max,
            trim_top_db=self.__trim_top_db,
            trim_frame_length=self.__trim_frame_len,
            trim_hop_length=self.__trim_shift_len,
        )
        adcfea = self.__feat.process_segment(proc_adc)
        fea = self.__prep_datalayer(adcfea, epd_ts)
        logits, logits_ln, greedy_predicts = self.__subword_mdl.forward(
            input_signal=fea[0].to(self.__device),
            input_signal_length=fea[1].to(self.__device),
        )
        cur_hyp, all_hyp = self.__subword_mdl.decoding.ctc_decoder_predictions_tensor(
            logits, decoder_lengths=logits_ln, return_hypotheses=True
        )

        for x in range(logits.shape[0]):
            cur_hyp[x].y_sequence = logits[x][: logits_ln[x]]
            if cur_hyp[x].alignments is None:
                cur_hyp[x].alignments = cur_hyp[x].y_sequence

        del greedy_predicts
        del logits
        del fea

        return cur_hyp

    # def run(self, adc: np.ndarray[int], epd_ts: torch.Tensor = None) -> List[Any]:

    @property
    def output_type(self):
        return self.__output_type

    # def output_type(self):

    @property
    def window_stride(self):
        return self.__subword_mdl.cfg.preprocessor.window_stride

    # def window_stride(self):

    @property
    def asr_mdl_name(self):
        return self.__mdl_name

    # def asr_mdl_name(self):


# class CTCBPEDecoderNemo(ASRDecoder):
