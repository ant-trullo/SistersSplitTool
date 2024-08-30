"""Given raw spots data, tracked spots and volume spots matrix, this function extracts and organizes information about bursting statistics."""


import numpy as np
from skimage.morphology import label
from PyQt5 import QtWidgets


class ParametersExtraction:
    def __init__(self, raw_sp, sp_tr, sp_vol):

        idx   =  np.unique(sp_tr)[1:]
        pbar  =  ProgressBar(total1=idx.size)
        pbar.show()

        numb_bursts      =  np.zeros(idx.shape)
        numb_off         =  np.zeros(idx.shape)
        bursts_duration  =  np.zeros((idx.size, raw_sp.shape[0]))
        off_duration     =  np.zeros((idx.size, raw_sp.shape[0]))
        bursts_ampl_mx   =  np.zeros((idx.size, raw_sp.shape[0]))
        bursts_ampl_av   =  np.zeros((idx.size, raw_sp.shape[0]))
        bursts_ampl_in   =  np.zeros((idx.size, raw_sp.shape[0]))

        for k in range(idx.size):
            pbar.update_progressbar(k)
            lls             =  ((sp_tr == idx[k]) * raw_sp).sum(2).sum(1)
            sp_vol_k        =  ((sp_tr == idx[k]) * sp_vol).sum(2).sum(1)
            lls_sgn         =  np.sign(lls)
            # lls_sgn_fltrd   =  medfilt(lls_sgn)                               # filter removed!!
            lls_lbl         =  label(lls_sgn)
            numb_bursts[k]  =  lls_lbl.max()
            lls_sil         =  np.abs(1 - np.sign(lls_lbl))
            lls_sil_lbl     =  label(lls_sil)
            if np.where(lls_sil_lbl == 1)[0].size > 0:
                if np.where(lls_sil_lbl == 1)[0][0] == 0:
                    lls_sil_lbl  -=  1
                numb_off[k]  =  lls_sil_lbl.max()

            for jj in range(1, lls_sil_lbl.max() + 1):
                off_duration[k, jj - 1]  =  (lls_sil_lbl == jj).sum()

            for jj in range(1, lls_lbl.max() + 1):
                bursts_duration[k, jj - 1]  =  (lls_lbl == jj).sum()
                bursts_ampl_mx[k, jj - 1]   =  ((lls_lbl == jj) * lls).max()
                bursts_ampl_av[k, jj - 1]   =  ((lls_lbl == jj) * lls).sum() / (sp_vol_k * (lls_lbl == jj)).sum()
                bursts_ampl_in[k, jj - 1]   =  ((lls_lbl == jj) * lls).sum()


        pbar.close()

        jend  =  np.where(bursts_ampl_mx.sum(0) == 0)[0][0]

        bursts_ampl_mx   =  bursts_ampl_mx[:, :jend]
        bursts_ampl_av   =  bursts_ampl_av[:, :jend]
        bursts_ampl_in   =  bursts_ampl_in[:, :jend]
        bursts_duration  =  bursts_duration[:, :jend]
        off_duration     =  off_duration[:, :jend]

# concatenate variables for GUI purposes, In order idx, numb_bursts, bursts_duration, bursts_ampl_mx, bursts_ampl_in, bursts_ampl_av, numb_off, off_duration

        step_info        =  bursts_duration.shape[1]
        statistics_info  =  np.zeros((idx.size, 3 + 5 * step_info))

        statistics_info[:, 0]                                    =  idx
        statistics_info[:, 1]                                    =  numb_bursts
        statistics_info[:, 2:2 + step_info]                      =  bursts_duration
        statistics_info[:, 2 + step_info:2 + 2 * step_info]      =  bursts_ampl_mx
        statistics_info[:, 2 + 2 * step_info:2 + 3 * step_info]  =  bursts_ampl_in
        statistics_info[:, 2 + 3 * step_info:2 + 4 * step_info]  =  bursts_ampl_av
        statistics_info[:, 2 + 4 * step_info]                    =  numb_off
        statistics_info[:, 3 + 4 * step_info:]                   =  off_duration

        self.raw_sp           =  raw_sp
        self.spts_vol         =  sp_vol
        self.numb_off         =  numb_off
        self.off_duration     =  off_duration
        self.numb_bursts      =  numb_bursts
        self.bursts_ampl_mx   =  bursts_ampl_mx
        self.bursts_ampl_av   =  bursts_ampl_av
        self.bursts_ampl_in   =  bursts_ampl_in
        self.bursts_duration  =  bursts_duration
        self.idx              =  idx
        self.statistics_info  =  statistics_info


class ProgressBar(QtWidgets.QWidget):
    """Simple progressbar widget"""
    def __init__(self, parent=None, total1=20):
        super().__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar1  =  QtWidgets.QProgressBar()
        self.progressbar1.setMinimum(1)
        self.progressbar1.setMaximum(total1)

        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar1, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)

    def update_progressbar(self, val1):
        """progress bar updater"""
        self.progressbar1.setValue(val1)
        QtWidgets.qApp.processEvents()
