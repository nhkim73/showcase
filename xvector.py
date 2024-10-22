# @name:        xvector.py
# @author:      nam (hoon) kim
# @date:        2019-12-10 15:42:46
# @updated:     2020-01-08 10:23:36
# @description: xvector class implementation
#               this depends on kaldi python wrapper (i.e., pykaldi)

from __future__ import absolute_import, division, print_function

import math
import os
import sys

import kaldi.base.io as kio
import numpy as np
from kaldi.cudamatrix import (
    CuMatrix,
    CuSubMatrix,
    CuSubVector,
    CuVector,
    cuda_available,
)
from kaldi.matrix import Matrix, SubMatrix, SubVector, Vector
from kaldi.nnet3 import *
from lego.base.logger import logger

##
# @brief implementation for xvector (a sort of speaker vector)
# @namespace xvector


##
# @brief A class implementation of xvector
class Xvector(object):

    ## @brief Constructor
    # @param self a pointer to obj.
    # @param mdlname speaker encoder model name (*.sencoder or *.dencoder) [string]
    # @param chunk_size default chunk size to estimate speaker or diarization vectors [int]
    # @param min_chunk_size minimum chunk size to estimate spaker or diarization vectors [int]
    # @param use_gpu boolean for indicating the use of gpu
    def __init__(self, mdlname, chunk_size, min_chunk_size, use_gpu=False):

        self.__check_parm_sanity(mdlname, min_chunk_size)
        logger.info(
            "parameters sanity check is done ... (mdl: %s, min_chunk_size: %d)"
            % (mdlname, min_chunk_size)
        )

        ## @private default chunk size to estimate speaker or diarization vectors
        self.__chunk_size = chunk_size
        ## @private minimum chunk size to estimate speaker or diarization vectors
        self.__min_chunk_size = min_chunk_size
        ## @private chunk size obtained from current evidence
        self.__this_chunk_size = chunk_size

        if use_gpu and cuda_available():
            kaldi.cudamatrix.CuDevice.instantiate().select_gpu_id("yes")
            kaldi.cudamatrix.CuDevice.instantiate().allow_multithreading()

        ifs = kio.ifstream.from_file(mdlname)
        ## @private kaldi nnet obj instanciated by pykaldi class Nnet
        self.__nnet = Nnet()
        self.__nnet.read(ifs, True)

        set_batchnorm_test_mode(True, self.__nnet)
        set_dropout_test_mode(True, self.__nnet)
        collapse_model(CollapseModelConfig(), self.__nnet)

        opts = NnetSimpleComputationOptions()
        compiler_config = CachingOptimizingCompilerOptions()
        opts.acoustic_scale = 1.0  # don't touch it !

        ## @private xvector option compiler
        self.__compiler = CachingOptimizingCompiler.new_with_optimize_opts(
            self.__nnet, opts.optimize_config, compiler_config
        )
        ## @private speaker vector dimension
        self.__xvec_dim = self.__nnet.output_dim("output")
        ## @private a buffer for speaker vectors (i.e, xvectors)
        self.__xvec = np.zeros(self.__xvec_dim)
        ## @private a buffer for speaker vector averages
        self.__xvec_avg = np.zeros(self.__xvec_dim)

        logger.info("xvector output dim = %d" % self.__xvec_dim)

    # def __init__(self, chunk_size, min_chunk_size, feat_nframe, feat_dim, use_gpu = False):

    ## @brief it checks parameters sanity to create obj.
    # @param self a pointer to obj.
    # @param mdlname speaker encoder model name (*.sencoder or *.dencoder) [string]
    # @param min_chunk_size default chunk size to estimate speaker or diarization vectors [int]
    def __check_parm_sanity(self, mdlname, min_chunk_size):

        if os.path.isfile(mdlname) == False:
            raise FileExistsError("not found xvector mdl ... %s" % mdlname)
        if min_chunk_size <= 0:
            raise ValueError(
                "min_chunk_size should be positive ... %d" % min_chunk_size
            )

    # def __check_parm_sanity(self, mdlname, min_chunk_size):

    ## @brief it checks parameters sanity to create obj.
    # @param self a pointer to obj.
    # @param feat input features [float matrix]
    # @return speaker | diarization vectors [float vector]
    # impl nnet3-xvector-compute in kaldi
    def convert(self, feat):

        num_rows = feat.num_rows
        feat_dim = feat.num_cols
        xvec_dim = self.__xvec_dim

        # chunk_size actually holds max_chunk_size
        chunk_size = self.__chunk_size
        self.__this_chunk_size = chunk_size

        if num_rows < chunk_size:
            logger.info(
                "max_chunk_size is greater than this chunk size, use this chunk size %d > %d"
                % (self.__this_chunk_size, num_rows)
            )
            self.__this_chunk_size = num_rows
        elif chunk_size == (-1):
            self.__this_chunk_size = num_rows

        for x in range(self.__xvec_dim):
            self.__xvec_avg[x] = 0.0

        tot_wgt = 0.0

        # assuming that pad_input = true
        for chunk_id in range(math.ceil(num_rows / float(self.__this_chunk_size))):
            offset = min(
                self.__this_chunk_size, num_rows - chunk_id * self.__this_chunk_size
            )
            sub_feat = SubMatrix(
                feat, chunk_id * self.__this_chunk_size, offset, 0, feat_dim
            )
            tot_wgt += offset

            if offset < self.__min_chunk_size:
                pad_feat = SubMatrix(np.zeros(shape=(self.__min_chunk_size, feat_dim)))

                l_cxt = int((self.__min_chunk_size - offset) / 2)
                r_cxt = int(self.__min_chunk_size - offset - l_cxt)

                for i in range(l_cxt):
                    pad_feat.row(i).copy_row_from_mat_(sub_feat, 0)

                for i in range(r_cxt):
                    pad_feat.row(self.__min_chunk_size - i - 1).copy_row_from_mat_(
                        sub_feat, offset - 1
                    )
                pad_feat.range(l_cxt, offset, 0, feat_dim)._copy_from_mat_(sub_feat)
                self.__run_nnet_computation(pad_feat)

            else:
                self.__run_nnet_computation(sub_feat)

            self.__xvec_avg = self.__xvec_avg + offset * self.__xvec

        self.__xvec_avg = self.__xvec_avg * (1.0 / tot_wgt)

        return self.__xvec_avg

    # def convert(self, feat):

    ## @brief core implementation of speaker vector extraction
    # @param self a pointer to obj.
    # @param feat input features [float matrix]
    def __run_nnet_computation(self, feat):

        req = ComputationRequest()
        req.need_model_derivative = False
        req.store_component_stats = False
        req.inputs = [IoSpecification().from_interval("input", 0, feat.num_rows)]
        req.outputs = [IoSpecification().from_indexes("output", [Index()], False)]
        computation = self.__compiler.compile(req)
        computer = NnetComputer(NnetComputeOptions(), computation, self.__nnet, Nnet())

        # CUDA-RELATED: if gpu is disable, it works on cpu
        input_feats = CuMatrix.from_matrix(feat)
        computer.accept_input("input", input_feats)
        computer.run()

        # CUDA-RELATED: if gpu is disaself.__xvec_dimble, it works on cpu
        cuda_o = computer.get_output_destructive("output")

        for x in range(self.__xvec_dim):
            self.__xvec[x] = cuda_o._getitem(0, x)

    # def __run_nnet_computation(self, feat):

    ## @brief it returns the dimension of speaker vector
    # @param self a pointer to obj.
    # @return dimension of speaker vector (i.e., xvector dimension)
    def dim(self):
        return self.__xvec_dim

    # def dim(self):


# class Xvector(object):
