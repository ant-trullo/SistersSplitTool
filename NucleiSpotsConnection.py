"""Given tracked spots and tracked nuclei, this function generates the false colored video."""

import numpy as np
from skimage.measure import label, regionprops_table
from PyQt5 import QtWidgets


class NucleiSpotsConnection:
    """Only one clss, does all the job"""
    def __init__(self, spots_tracked, nuclei_tracked):

        nuclei_active  =  np.sign(nuclei_tracked).astype(int)
        idx            =  np.unique(spots_tracked)[1:]                                                             # list of all the values in the spots matrix. The zero, which is the first, is removed

        pbar  =  ProgressBar(total1=idx.size)
        pbar.show()

        i         =  0
        rgp_spts  =  regionprops_table(spots_tracked.astype(np.int32), properties=["label", "coords"])   # dictionary with label and coordinate of each non zero pixel
        for k in idx:
            pbar.update_progressbar(i)
#             aa       =  (spots_tracked == k).sum(2).sum(1)
#             t_steps  =  np.where(aa != 0)[0]
            coord_idx  =  np.where(rgp_spts["label"] == k)[0][0]                    # search the dictionary location with the corresponding label, generally is k + 1, but like this is 100% sure
            t_steps    =  np.unique(rgp_spts["coords"][coord_idx][:, 0])            # array with the t coordinate of each pixel of the spot tracked. np.unique to have only once the number of the frame in which the            i  +=  1
            i         +=  1
            for tt in t_steps:
                nuclei_active[tt, :, :]  +=  (nuclei_tracked[tt, :, :] == k)

        pbar.close()

        nuclei_active3c              =   np.zeros((nuclei_tracked.shape[0], nuclei_tracked.shape[1], nuclei_tracked.shape[2], 3))
        nuclei_active3c[:, :, :, 0]  =   (nuclei_active == 2)
        nuclei_active3c[:, :, :, 2]  =   (nuclei_active == 1)
        nuclei_active3c[:, :, :, 1]  =   np.sign(spots_tracked)
        nuclei_active3c              *=  255

        n_active_vector  =  np.zeros(nuclei_active[:, 0, 0].size)
        for t in range(n_active_vector.size):
            l1                  =  label((nuclei_active[t, :, :] == 2) * nuclei_tracked[t, :, :], connectivity=1)
            n_active_vector[t]  =  l1.max()

        self.nuclei_active    =  nuclei_active
        self.nuclei_active3c  =  nuclei_active3c
        self.n_active_vector  =  n_active_vector
        self.popt             =  0
        self.perr             =  0


class ProgressBar(QtWidgets.QWidget):
    """Simple progress bar widget"""
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
        """Progressbar updater"""
        self.progressbar1.setValue(val1)
        QtWidgets.qApp.processEvents()
