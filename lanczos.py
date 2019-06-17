import time
import datetime
import numpy as np
from datetime import datetime, date, time, timedelta
import sys

class Lanczos(object):

    """
    Class for Lanczos filtering. Inspired from 
    NCL's `filwgts_lanczos <http://www.ncl.ucar.edu/Document/Functions/Built-in/filwgts_lanczos.shtml>`_ and `wgt_runave <http://www.ncl.ucar.edu/Document/Functions/Built-in/wgt_runave.shtml>`_ functions.

    :param str filt_type: The type of filter ("lp"=Low Pass, "hp"=High Pass,
     "bp"=Band Pass
    :param int nwts: Number of weights (must be an odd number)
    :param float pca: First cut-off period
    :param float pcb: Second cut-off period (only for band-pass filters)
    :param float delta_t: Time-step

    """

    def __init__(self, filt_type, nwts, pca, pcb=None, delta_t=1):

        """ Initialisation of the filter """

        self.filt_type = filt_type
        self.nwts = nwts
        self.pca = pca
        self.pcb = pcb
        self.delta_t = delta_t

        if self.nwts % 2 == 0:
            raise IOError('Number of weigths must be odd')

        # Because w(n)=w(-n)=0, we would have only nwts-2
        # effective weight. So we add to weights so as to get rid off that
        nwts = self.nwts+2
        weights = np.zeros([nwts])
        nbwgt2 = nwts // 2

        if self.filt_type == 'lp':

            cutoff = float(self.pca)
            cutoff = self.delta_t/cutoff

            weights[nbwgt2] = 2 * cutoff
            k = np.arange(1., nbwgt2)
            sigma = np.sin(np.pi * k / nbwgt2) * nbwgt2 / (np.pi * k)
            firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
            weights[nbwgt2-1:0:-1] = firstfactor * sigma
            weights[nbwgt2+1:-1] = firstfactor * sigma

        elif self.filt_type == 'hp':

            cutoff = float(self.pca)
            cutoff = self.delta_t/cutoff

            weights[nbwgt2] = 1-2 * cutoff
            k = np.arange(1., nbwgt2)
            sigma = np.sin(np.pi * k / nbwgt2) * nbwgt2 / (np.pi * k)
            firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
            weights[nbwgt2-1:0:-1] = -firstfactor * sigma
            weights[nbwgt2+1:-1] = -firstfactor * sigma

        elif self.filt_type == 'bp':

            cutoff1 = np.max([float(self.pca), float(self.pcb)])
            cutoff1 = self.delta_t/cutoff1

            cutoff2 = np.min([float(self.pca), float(self.pcb)])
            cutoff2 = self.delta_t/cutoff2

            weights[nbwgt2] = 2*cutoff2-2*cutoff1
            k = np.arange(1., nbwgt2)
            sigma = np.sin(np.pi * k / nbwgt2) * nbwgt2 / (np.pi * k)
            firstfactor = (np.sin(2.*np.pi*cutoff1*k)/(np.pi*k)) \
                - (np.sin(2.*np.pi*cutoff2*k)/(np.pi*k))
            weights[nbwgt2-1:0:-1] = -firstfactor * sigma
            weights[nbwgt2+1:-1] = -firstfactor * sigma

        else:
            raise IOError('Unknowm filter %s must be "lp", "hp" or "bp"'
                          % filt_type)

        self.wgt = weights

    def wgt_runave(self, data):

        """ Compute the running mean of a ND input array using the filter weights.

        :param numpy.array data: Array to filter 
         out (time must be the first dimension)

        """

        # we retrive the wgt array and initialise the output
        wgt = self.wgt
        output = np.zeros(data.shape)

        nwt = len(wgt)
        nwgt2 = nwt/2
        indw = nwgt2

        if data.ndim > 1:
            shapein = np.array(data.shape)
            shapein = shapein[::-1]
            shapein[-1] = 1
            wgt = np.tile(wgt, shapein)
            wgt = np.transpose(wgt)

        while indw+nwgt2+1 <= data.shape[0]:
            index = np.arange(indw-nwgt2, indw+nwgt2+1)
            output[indw] = np.sum(wgt*data[index], axis=0)
            indw = indw+1

        output[output == 0] = np.nan

        return output
