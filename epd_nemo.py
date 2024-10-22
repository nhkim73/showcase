# @name:        epd_nemo.py
# @author:      nam (hoon) kim
# @date:        2023-01-01 22:22:21
# @updated:     2023-05-04 22:09:01
# @description: <description>

import logging
from pathlib import Path
from typing import *

import nemo.collections.asr as nemo_asr
import numpy as np
import tomli
import torch
from lego.asr.frontend.epd import EndPointDetection
from lego.asr.frontend.vad_util import VadUtil
from lego.base.format import *
from lego.base.logger import logger
from lego.base.timestamp import convert_overlap_to_flat
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from tabulate import tabulate

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


__all__ = ["EndPointDetectionNemo"]


class _ConfigSanity(object):
    def __check_adc_cfg_sanity(self, adc_cfg: Dict) -> None:
        if adc_cfg["tgt_spr"] <= 0.0:
            raise ValueError("invalud adc.tgt_spr ... %.2f" % adc_cfg["tgt_spr"])

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

    def __check_vad_cfg_sanity(self, vad_cfg: Dict) -> None:
        for x in vad_cfg.keys():
            # pad_onset or pad_offset could be 0
            if (x == "onset" or x == "offset") and vad_cfg[x] < 0.0:
                raise ValueError("invalid vad.%s ... %.2f" % (x, vad_cfg[x]))
            elif x == "duration":
                for lbl in vad_cfg[x].keys():
                    if any(dur < 0 for dur in vad_cfg[x][lbl]):
                        raise ValueError(
                            "invalid vad.duration ... %s: %s"
                            % (lbl, container_to_str(vad_cfg[x][lbl], "[", "]"))
                        )

            elif (
                isinstance(vad_cfg[x], float) or isinstance(vad_cfg[x], int)
            ) and vad_cfg[x] <= 0.0:
                raise ValueError("invalid vad.%s ... %.2f" % (x, vad_cfg[x]))

        if vad_cfg["win_size_in_sec"] <= vad_cfg["win_rate_in_sec"]:
            raise ValueError(
                "vad.win_size_in_sec should be LARGER than vad.win_rate_in_sec ... (%.2f, %.2f)"
                % (vad_cfg["win_size_in_sec"], vad_cfg["win_rate_in_sec"])
            )

        if not Path(vad_cfg["mdl_name"]).exists():
            raise FileNotFoundError(
                "not found vad.mdl_name ... %s" % vad_cfg["mdl_name"]
            )

    # def __check_vad_cfg_sanity(self, vad_cfg: Dict) -> None:

    def __check_sys_cfg_sanity(self, sys_cfg: Dict) -> None:
        if sys_cfg["use_gpu"] and sys_cfg["cuda_id"] < 0:
            raise ValueError("invalid sys.cuda_id ... %d" % sys_cfg["cuda_id"])

    # def __check_sys_cfg_sanity(self, sys_cfg: Dict) -> None:

    def check(self, adc_cfg: Dict, fea_cfg: Dict, vad_cfg: Dict, sys_cfg: Dict) -> None:
        try:
            self.__check_adc_cfg_sanity(adc_cfg)
            self.__check_fea_cfg_sanity(fea_cfg)
            self.__check_vad_cfg_sanity(vad_cfg)
            self.__check_sys_cfg_sanity(sys_cfg)
        except Exception:
            raise

    # def check(self, adc_cfg: Dict, fea_cfg: Dict, vad_cfg: Dict, sys_cfg: Dict) -> None:

    def check_vad_tgt_lbls(self, mdl_lbls: Dict, tgt_lbls: Dict, bkg_lbl: str) -> None:
        # check duplicate tgt_lbls item
        dup = set([x for x in tgt_lbls if tgt_lbls.count(x) > 1])
        if len(dup):
            raise ValueError(
                "duplicated target labels are found ... %s" % container_to_str(dup)
            )
        # check whether each target lbls in mdl
        for lbl in tgt_lbls:
            if lbl not in mdl_lbls:
                raise NameError(
                    "target label, %s not found in mdl ... mdl lbls %s"
                    % container_to_str(mdl_lbls)
                )
        if bkg_lbl not in mdl_lbls:
            raise NameError(
                "background label, %s not found in mdl ... mdl lbls %s"
                % container_to_str(mdl_lbls)
            )

    # def check_vad_tgt_lbls(self, mdl_lbls: Dict, tgt_lbls: Dict, bkg_lbl: str) -> None:


# class _ConfigSanity(object):


class EndPointDetectionNemo(EndPointDetection):
    def __init__(
        self, adc_cfg: Dict, fea_cfg: Dict, vad_cfg: Dict, sys_cfg: Dict
    ) -> None:
        # check config sanity
        cfg_sanity = _ConfigSanity()
        cfg_sanity.check(adc_cfg, fea_cfg, vad_cfg, sys_cfg)
        logger.info("checked configuration sanity for EndPointDetectionNemo ...")

        # set up adc
        self.__tgt_spr = adc_cfg["tgt_spr"]

        # set up sys
        if sys_cfg["use_gpu"] and torch.cuda.is_available():
            logger.info("use_gpu = true, cuda:%d" % sys_cfg["cuda_id"])
            device = torch.device("cuda:%d" % sys_cfg["cuda_id"])
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
        # set up vad
        self.__split_dur_in_sec = vad_cfg["split_dur_in_sec"]
        self.__framesize_in_sec = vad_cfg["win_size_in_sec"]
        self.__framerate_in_sec = vad_cfg["win_rate_in_sec"]
        self.__framesize = int(self.__framesize_in_sec * self.__tgt_spr)
        self.__framerate = int(self.__framerate_in_sec * self.__tgt_spr)
        self.__onset, self.__offset = VadUtil.compute_vad_onset_offset(
            "absolute", vad_cfg["onset"], vad_cfg["offset"]
        )
        self.__pad_onset = vad_cfg["pad_onset"]
        self.__pad_offset = vad_cfg["pad_offset"]
        self.__dur = vad_cfg["duration"]
        self.__filt_speech_first = vad_cfg["filt_speech_first"]

        trunc_unit = int(self.__framesize_in_sec / self.__framerate_in_sec)
        self.__trunc = int(trunc_unit / 2)
        self.__trunc_l = trunc_unit - self.__trunc

        # load vad mdl (i.e., nemo marblenet) and set it up as evaluation mode
        self.__mdl_name = vad_cfg["mdl_name"]
        self.__vad_mdl = nemo_asr.models.EncDecClassificationModel.restore_from(
            self.__mdl_name
        )
        self.__vad_lbls = self.__vad_mdl.cfg.labels
        tgt_lbls = list(self.__dur.keys())
        try:
            cfg_sanity.check_vad_tgt_lbls(self.__vad_lbls, tgt_lbls, vad_cfg["bkg_lbl"])
        except Exception:
            raise

        self.__tgt_lbls = tgt_lbls
        self.__lbl_ids = [self.__vad_lbls.index(x) for x in self.__tgt_lbls]
        self.__lbl_ids.sort()
        self.__bkg_lbl = vad_cfg["bkg_lbl"]
        self.__vad_mdl = self.__vad_mdl.to(device)
        self.__vad_mdl.eval()

        logger.info("loaded vad mdl ... %s" % vad_cfg["mdl_name"])
        logger.info("target labels ... %s" % container_to_str(self.__tgt_lbls))

    # def __init__(self, adc_cfg: Dict, fea_cfg: Dict, vad_cfg: Dict, sys_cfg: Dict) -> None:

    def __convert_chunk_to_frame(
        self, segm: Tuple[torch.Tensor], apply_norm: Optional[bool] = False
    ) -> Tuple[torch.Tensor]:
        framesize = self.__framesize
        framerate = self.__framerate

        adc, adc_ln, tok, _ = segm

        framesize = min(framesize, adc_ln)
        has_audio = True if adc_ln else False

        frame, frame_ln, frame_tok, num_frame = [], [], [], []
        num_pad_beg = framesize // 2
        num_pad_end = framesize - framesize // 2

        adc = EndPointDetection.normalize(adc) if apply_norm else adc
        pad_beg = torch.zeros(num_pad_beg)
        pad_end = torch.zeros(num_pad_end)
        pad_adc = torch.cat((pad_beg, adc, pad_end))
        adc_ln += framesize

        if has_audio:
            frame_count = torch.div(
                adc_ln - framesize, framerate, rounding_mode="trunc"
            )
            for id in range(frame_count):
                beg_idx = id * framerate
                end_idx = beg_idx + framesize
                frame_adc = pad_adc[beg_idx:end_idx]
                frame.append(frame_adc)

            num_frame.append(frame_count)
            frame_tok.extend([tok] * frame_count)
            frame_ln.extend([framesize] * frame_count)
            frame = torch.stack(frame)
            frame_ln = torch.tensor(frame_ln)
        else:
            frame, frame_ln = None, None

        frame_tok = torch.stack(frame_tok)
        frame_tok_ln = torch.tensor(num_frame)
        return frame, frame_ln, frame_tok, frame_tok_ln

    # def __convert_chunk_to_frame(self, segm: Tuple[torch.Tensor], apply_norm: Optional[bool]=False) -> Tuple[torch.Tensor]:

    def predict_frame_score(
        self, adc: np.ndarray[int], chunk_ts: Tuple[torch.Tensor]
    ) -> List[torch.Tensor]:
        trunc = self.__trunc
        trunc_l = self.__trunc_l
        num_lbl = len(self.__lbl_ids)
        scores = [torch.empty(0) for _ in range(num_lbl)]
        lids = self.__lbl_ids
        for i, x in enumerate(chunk_ts):
            beg_pt = int(x[0] * self.__tgt_spr)
            end_pt = beg_pt + int(x[1] * self.__tgt_spr)

            segm = adc[beg_pt:end_pt].copy()
            segm_adc = AudioSegment(
                segm,
                self.__tgt_spr,
                trim=False,
                trim_ref=np.max,
                trim_top_db=self.__trim_top_db,
                trim_frame_length=self.__trim_frame_len,
                trim_hop_length=self.__trim_shift_len,
            )

            feat = self.__feat.process_segment(segm_adc)
            feat_ln = torch.tensor(feat.shape[0]).long()
            tok = torch.tensor(0).long()  # it is always 0 label2id {'infer':0}
            tok_ln = torch.tensor(1).long()

            frame = self.__convert_chunk_to_frame((feat, feat_ln, tok, tok_ln))
            frame = [x.to(self.__vad_mdl.device) for x in frame]
            compute = [[] for _ in range(len(lids))]

            with autocast():
                log_probs = self.__vad_mdl(
                    input_signal=frame[0], input_signal_length=frame[1]
                )
                probs = torch.softmax(log_probs, dim=-1)

                for k, lid in enumerate(lids):
                    pred = probs[:, lid]
                    if chunk_ts[i][2] == "start":
                        tgt_pred = pred[:-trunc]
                    elif chunk_ts[i][2] == "next":
                        tgt_pred = pred[trunc:-trunc_l]
                    elif chunk_ts[i][2] == "end":
                        tgt_pred = pred[trunc_l:]
                    else:
                        tgt_pred = pred
                    compute[k] += tgt_pred

            # compute should be copied from gpu to cpu mem
            for k in range(num_lbl):
                segm_score = torch.tensor(compute[k]).clone()
                scores[k] = torch.cat((scores[k], segm_score), 0)

            # explict deletion is required ?
            del frame
        return scores

    # def predict_frame_score(self, adc: np.ndarray[int], chunk_ts: Tuple[torch.Tensor]) -> List[torch.Tensor]:

    def run(
        self, adc: np.ndarray[int], chunk_ts: Tuple[torch.Tensor]
    ) -> Tuple[List[List], Dict[int, List[List]]]:
        scores = self.predict_frame_score(adc, chunk_ts)

        epd_segms = []
        for k, score in enumerate(scores):
            id = self.__lbl_ids[k]
            lbl = self.__vad_lbls[id]
            segms = VadUtil.binarization(
                score,
                self.__onset,
                self.__offset,
                self.__pad_onset,
                self.__pad_offset,
                self.__framerate_in_sec,
            )
            segms = VadUtil.filtering(
                segms, self.__dur[lbl][0], self.__dur[lbl][1], self.__filt_speech_first
            )
            for segm in segms:
                epd_segms.append([segm[0].item(), segm[1].item(), lbl])

        if not len(epd_segms):
            return epd_segms

        epd_segms.sort(key=lambda x: x[0])
        # REMARK
        # flat segms (flat segments)
        #   this is a set of segments, which are not overlapped. Even if they are
        #   PARTIALLY overlapped, thery are converted into non-overlapped segments
        #         |----A-----|
        #                |------B------|
        #   ->    |----A---|----B------|
        # overlap segms (overlapped segments)
        #   this is a set of fully overlapped segments in terms of dictionary.
        #   each key stands for the index of flat segms, which holds overlapped
        #   segments.
        #        |------------A----------|
        #          |----B---|   |--C--|
        #   ->  B and C are FULLY overlapped segments of A

        flat_segms, overlap_segms = convert_overlap_to_flat(epd_segms)

        # the last end point of epd could be longer than the end point of chunk ...
        chunk_last_end_ts = chunk_ts[-1][0] + chunk_ts[-1][1]
        flat_segm_last_end_ts = flat_segms[-1][1]
        if flat_segm_last_end_ts > chunk_last_end_ts:
            flat_segms[-1][1] = chunk_last_end_ts

        return (flat_segms, overlap_segms)

    # def run(self, adc: np.ndarray[int], chunk_ts: Tuple[torch.Tensor]) -> Tuple[List[List], Dict[int, List[List]]]:

    @property
    def split_dur_in_sec(self):
        return self.__split_dur_in_sec

    # def split_dur_in_sec(self):

    @property
    def framesize_in_sec(self):
        return self.__framesize_in_sec

    # def framesize_in_sec(self):

    @property
    def vad_mdl_name(self):
        return self.__mdl_name

    # def vad_mdl_name(self):

    @property
    def tgt_lbls(self):
        return self.__tgt_lbls

    # def tgt_lbls(self):

    @property
    def bkg_lbl(self):
        return self.__bkg_lbl

    # def bkg_lbl(self):


# class EndPointDetectionNemo(EndPointDetection):
