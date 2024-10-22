# @name:        plda.py
# @author:      nam (hoon) kim
# @date:        2019-12-10 15:36:39
# @updated:     2020-03-28 09:15:54
# @description: plda calss implementation.
#               this depends on kaldi python wrapper (i.e., pykaldi)

from __future__ import absolute_import, division, print_function

import os

import numpy as np
from kaldi.matrix import _kaldi_matrix  # for sort_svd
from kaldi.matrix import (
    DoubleMatrix,
    DoubleSubMatrix,
    DoubleSubVector,
    DoubleVector,
    Matrix,
    SubMatrix,
    SubVector,
    Vector,
)
from kaldi.matrix.common import MatrixResizeType, MatrixTransposeType
from kaldi.matrix.packed import DoubleSpMatrix, DoubleTpMatrix, SpMatrix, TpMatrix
from lego.base.logger import logger
from lego.io.kaldi import *

##
# @brief implementation for probablistic linear discriminative analysis (plda)
# @namespace plda


## @brief it loads plda resources
# @param filename plda file name
# @return tuple consisting of plda mean, plda transform, plda psi parameters and plda offset
def load_plda(filename):

    if not os.path.isfile(filename):
        raise FileNotFoundError("not found ... %s" % filename)

    with open(filename, "rb") as fin:
        is_binary = fin.read(2).decode()
        assert is_binary == "\0B", "input should be binary"

        is_plda_token = fin.read(7).decode()
        is_plda_token = is_plda_token.strip()
        assert is_plda_token == "<Plda>", "<Plda> token is expected"

        hd = fin.read(3).decode()
        hd = hd.strip()

        if hd == "FV":
            smpl_size = 4
            smpl_type = "float32"
        elif hd == "DV":
            smpl_size = 8
            smpl_type = "float64"
        else:
            raise TypeError("plda mean identifier should be FV or DV")

        assert fin.read(1).decode() == "\4"
        vec_size = np.frombuffer(fin.read(4), dtype="int32", count=1)[0]

        assert vec_size != 0, "invalid mean size ... %d" % vec_size

        ary = fin.read(vec_size * smpl_size)
        plda_mean = DoubleSubVector(np.frombuffer(ary, dtype=smpl_type))

        hd = fin.read(3).decode()
        hd = hd.strip()

        if hd == "FM":
            smpl_size = 4
            smpl_type = "float32"
        elif hd == "DM":
            smpl_size = 8
            smpl_type = "float64"
        else:
            raise TypeError("plda transform identifier should be FM or DM")

        s1, row, s2, col = np.frombuffer(
            fin.read(10), dtype="int8,int32,int8,int32", count=1
        )[0]

        assert row != 0, "invalid row size in plda transform matrix ... %d" % row
        assert col != 0, "invalid col size in plda transform matrix ... %d" % col

        ary = fin.read(row * col * smpl_size)
        vec = np.frombuffer(ary, dtype=smpl_type)
        plda_transform = DoubleSubMatrix(np.reshape(vec, (row, col)))

        hd = fin.read(3).decode()
        hd = hd.strip()

        if hd == "FV":
            smpl_size = 4
            smpl_type = "float32"
        elif hd == "DV":
            smpl_size = 8
            smpl_type = "float64"
        else:
            raise TypeError("plda psi identifier should be FV or DV")

        assert fin.read(1).decode() == "\4"
        vec_size = np.frombuffer(fin.read(4), dtype="int32", count=1)[0]

        assert vec_size != 0, "invalid psi size ... %d" % vec_size

        ary = fin.read(vec_size * smpl_size)
        plda_psi = DoubleSubVector(np.frombuffer(ary, dtype=smpl_type))

        is_plda_token = fin.read(8).decode()
        is_plda_token = is_plda_token.strip()
        assert is_plda_token == "</Plda>", "</Plda> token is expected"

    plda_offset = DoubleVector(plda_mean.dim)
    plda_offset.add_mat_vec_(
        -1.0, plda_transform, MatrixTransposeType.NO_TRANS, plda_mean, 0.0
    )

    return (plda_mean, plda_transform, plda_psi, plda_offset)


# def load_plda(filename):


##
# @brief A class implementation of probabilistic linear discriminative analysis (a.k.a plda)
class Plda(object):

    ## @brief Constructor
    # @param self a pointer to obj.
    # @param plda_name plda resource file name [string]
    def __init__(self, plda_name):

        self.__check_parm_sanity(plda_name)
        logger.info("parameter sanity check is done ...")

        (self.__plda_mean, self.__plda_xfm, self.__plda_psi, self.__plda_offset) = (
            load_plda(plda_name)
        )
        logger.info("load plda mdl ... %s" % plda_name)

        # workspace for transform_spkvec
        ## @private a buffer for plda transformed speaker vector
        self.__plda_xv = DoubleVector(self.__plda_offset.dim)
        ## @private a buffer for square of plda transformed speaker vector
        self.__plda_xvsq = DoubleVector(self.__plda_offset.dim)
        ## @private a buffer for inverse covariance
        self.__inv_cov = DoubleVector(self.__plda_psi.dim)

        # workspace for apply_transform
        ## @private a buffer for transformed plda mean
        self.__xfmed_plda_mean = DoubleVector()
        ## @private a buffer for transformed plda transform
        self.__xfmed_plda_xfm = DoubleMatrix()
        ## @private a buffer for transformed psi parameters
        self.__xfmed_plda_psi = DoubleVector()
        ## @private a buffer for transformed offset
        self.__xfmed_plda_offset = DoubleVector()

    # def __init__(self, plda_name):

    ## @brief it checks parameters sanity to create obj.
    # @param self a pointer to obj.
    # @param plda_name plda resource file name [string]
    def __check_parm_sanity(self, plda_name):
        if os.path.isfile(plda_name) == False:
            raise FileExistsError("not found plda file ... %s" % plda_name)

    # def __check_parm_sanity(self, plda_name):

    ## @brief core implementation of transforming speaker vectors.
    # plda transform is applied to the input speaker vectors
    # @param self a pointer to obj.
    # @param input speaker vector [float vector]
    # @param plda_mean plda mean [float vector]
    # @param plda_xfm plda transform [float vector]
    # @param plda_psi plda psi parameters [float vector]
    # @param plda_offset plda offset [float vector]
    # @param nutt number of occupancy
    # to be used to estimate a speaker vector
    # @return plda-transformed speaker vector (BEWARE: what returns is reference)
    def __transform_spkvec_inner(
        self, in_xv, plda_mean, plda_xfm, plda_psi, plda_offset, nutt=1
    ):

        # apply plda transformation to input spk vector
        # this is impl referring plda::TransforIVector in kaldi

        self.__plda_xv.copy_(plda_offset)
        self.__plda_xv.add_mat_vec_(
            1.0, plda_xfm, MatrixTransposeType.NO_TRANS, DoubleSubVector(in_xv), 1.0
        )

        self.__plda_xvsq.copy_(self.__plda_xv)
        self.__plda_xvsq.apply_pow_(2.0)

        self.__inv_cov.copy_(plda_psi)
        self.__inv_cov.add_(1.0 / float(nutt))
        self.__inv_cov.invert_elements_()

        norm_factor = np.sqrt(plda_mean.dim / np.dot(self.__inv_cov, self.__plda_xvsq))
        self.__plda_xv.scale_(norm_factor)

        # BEWARE: return as reference
        return self.__plda_xv

    # def __transform_spkvec_inner(self, in_xv, plda_mean, plda_xfm, plda_psi, plda_offset, nutt=1):

    ## @brief it applies plda transform for input speaker vector\n
    # plda resources applied is what is loaded from file.
    # see @ref transform_spkvec_with_xfmed_plda as a related method
    # @param self a pointer to obj.
    # @param in_xv input speaker vector [float vector]
    # @param nutt number of occupancy
    # @return plda-transformed speaker vector
    def transform_spkvec(self, in_xv, nutt=1):

        # BEWARE: return as reference
        return self.__transform_spkvec_inner(
            in_xv,
            self.__plda_mean,
            self.__plda_xfm,
            self.__plda_psi,
            self.__plda_offset,
            nutt,
        )

    # def transform_spkvec(self, in_xv, nutt=1):

    ## @brief it applies plda transform for input speaker vector\n
    # plda resources applied is ALREADY TRANSFORMED.
    # see @ref transform_spkvec as a related method
    # @param self a pointer to obj.
    # @param in_xv input speaker vector [float vector]
    # @return plda-transformed speaker vector
    def transform_spkvec_with_xfmed_plda(self, in_xv):

        if (
            self.__xfmed_plda_mean.dim == 0
            or self.__xfmed_plda_xfm.num_rows == 0
            or self.__xfmed_plda_xfm.num_cols == 0
            or self.__xfmed_plda_psi.dim == 0
            or self.__xfmed_plda_offset.dim == 0
        ):
            raise ArithmeticError("not found transformed plda")

        # the SIZE OF self.__plda_xv, self.__plda_svsq and self.__inv_cov
        # are UPDATED to transformed plda parametres in apply_transform.
        # in doing so, allocation can be reduced ...

        # BEWARE: return as reference
        return self.__transform_spkvec_inner(
            in_xv,
            self.__xfmed_plda_mean,
            self.__xfmed_plda_xfm,
            self.__xfmed_plda_psi,
            self.__xfmed_plda_offset,
        )

    # def transform_spkvec_with_xfmed_plda(self, in_xv):

    ## @brief it applies drt(dimension reduction transform) to plda\n
    # @param self a pointer to obj.
    # @param in_xfm input transform (such as pca or lda)
    def apply_transform(self, in_xfm):

        # this method applies in_xfm to plda.
        # by doing so, plda is shifted to target projection.
        # this is impl referring plda::ApplyTransform

        n_row = in_xfm.num_rows
        n_col = in_xfm.num_cols
        dim = self.__plda_mean.dim

        if n_row > dim:
            raise ArithmeticError(
                "it SHOULD be in-xfm row(%d) <= plda mean dim(%d)" % (n_row, dim)
            )
        if n_col != dim:
            raise ArithmeticError(
                "it SHOULD be in-xfm col(%d) == plda mean dim(%d)" % (n_col, dim)
            )

        # update self.__xfmed_plda_mean by applying in_xfm to self.__plda_mean
        mean_updated = DoubleVector(in_xfm.num_rows)
        mean_updated.add_mat_vec_(
            1.0, in_xfm, MatrixTransposeType.NO_TRANS, self.__plda_mean, 0.0
        )
        self.__xfmed_plda_mean.resize_(in_xfm.num_rows)
        self.__xfmed_plda_mean._copy_from_vec_(mean_updated)

        # compute between_var and within_var
        between_var = DoubleSpMatrix(n_col)
        within_var = DoubleSpMatrix(n_col)
        psi_mat = DoubleSpMatrix(n_col)
        xfm_invert = DoubleMatrix(self.__plda_xfm)

        # BEWARE: after transform, dim of __plda_mean is changed.
        dim_updated = self.__xfmed_plda_mean.dim
        between_var_updated = DoubleSpMatrix(dim_updated)
        within_var_updated = DoubleSpMatrix(dim_updated)

        psi_mat.add_diag_vec_(1.0, self.__plda_psi)
        xfm_invert.invert_()
        between_var.add_mat_2_sp_(
            1.0, xfm_invert, MatrixTransposeType.NO_TRANS, psi_mat, 0.0
        )
        within_var.add_mat2_(1.0, xfm_invert, MatrixTransposeType.NO_TRANS, 0.0)

        # update between-var and within-var by appling in_xfm
        between_var_updated.add_mat_2_sp_(
            1.0, in_xfm, MatrixTransposeType.NO_TRANS, between_var, 0.0
        )
        within_var_updated.add_mat_2_sp_(
            1.0, in_xfm, MatrixTransposeType.NO_TRANS, within_var, 0.0
        )

        # update self.__plda_psi and self.__plda_xfm

        # compute prj mat by cholesky (it could be heavy ...)
        csk = DoubleTpMatrix(within_var_updated.num_rows)
        csk.cholesky_(within_var_updated)
        csk.invert_()
        within_var_prj = DoubleMatrix(csk)

        between_var_prj = DoubleSpMatrix(dim_updated)
        between_var_prj.add_mat_2_sp_(
            1.0, within_var_prj, MatrixTransposeType.NO_TRANS, between_var_updated, 0.0
        )

        U = DoubleMatrix(dim_updated, dim_updated)
        s = DoubleVector(dim_updated)
        A = DoubleSpMatrix(between_var_prj.num_cols)

        # eigen decomp. for nonsquare mat
        A.copy_from_sp_(between_var_prj)
        A.tridiagonalize_(U)
        A.qr_(U)
        U.transpose_()
        s.copy_diag_from_mat_(A)

        if s.min() < 0.0:
            raise ArithmeticError("min of eigen vector is negative ... %f" % s.min())

        n_floor = s.apply_floor_(0.0)
        if n_floor:
            logger.info("%d eigenvalue of between-class var is floored to 0")

        # A python api named "sort_svd" might NOT be available.
        # Internally, pykaldi checks the type matching between s and U
        # - i.e., single precision or double precision by calling
        # isinstance. But, they're checked by parents class type.
        # As workaround, a c/c++ primitive wrapped by python is called.
        _kaldi_matrix._sort_double_svd(s, U)

        self.__xfmed_plda_xfm.resize_(dim_updated, dim_updated)
        self.__xfmed_plda_xfm.add_mat_mat_(
            U,
            within_var_prj,
            MatrixTransposeType.TRANS,
            MatrixTransposeType.NO_TRANS,
            1.0,
            0.0,
        )

        self.__xfmed_plda_psi.resize_(dim_updated)
        self.__xfmed_plda_psi._copy_from_vec_(s)

        # update self.__plda_offset
        self.__xfmed_plda_offset.resize_(dim_updated)
        self.__xfmed_plda_offset.add_mat_vec_(
            -1.0,
            self.__xfmed_plda_xfm,
            MatrixTransposeType.NO_TRANS,
            self.__xfmed_plda_mean,
            0.0,
        )

        # update workspace(xv, xvsq, inv_cov) w.r.t transformed plda ...
        self.__plda_xv.resize_(self.__xfmed_plda_offset.dim)
        self.__plda_xvsq.resize_(self.__xfmed_plda_offset.dim)
        self.__inv_cov.resize_(self.__xfmed_plda_psi.dim)

    # def apply_transform(self, in_xfm):

    ## @brief it returns the reference of plda resources\n
    # @param self a pointer to obj.
    # @return tuple consisting of plda mean, plda transform, plda psi parameters and plda offset
    def get_parm(self):
        return (self.__plda_mean, self.__plda_xfm, self.__plda_psi, self.__plda_offset)

    # def get_parm(self):

    ## @brief it returns the reference of transformed plda resources\n
    # @param self a pointer to obj.
    # @return tuple consisting of transformed plda mean, transformed plda transform,
    # transformed plda psi parameters and transformed plda offset
    def get_xfmed_parm(self):
        return (
            self.__xfmed_plda_mean,
            self.__xfmed_plda_xfm,
            self.__xfmed_plda_psi,
            self.__xfmed_plda_offset,
        )

    # def get_xfmed_parm(self):


# class Plda(object):
