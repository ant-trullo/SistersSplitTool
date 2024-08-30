"""This function associates detected spots to tracked nuclei.

Starting from the mask of the detected spots, it searches for the closer nuclei
and gives to the spot the same label of the found nucleus.
"""


import numpy as np
from skimage.morphology import label
from skimage.measure import regionprops
from PyQt5 import QtWidgets

import CloserNucleiFinder


class SpotsConnection:
    """Main class, does all the job."""
    def __init__(self, nuclei_tracked, spots_mask, spt_nuc_maxdist):

        spots_tracked   =  np.zeros(spots_mask.shape)
        t_tot           =  spots_mask.shape[0]
        spots_mask_lbl  =  np.zeros(spots_mask.shape)

        pbar  =  ProgressBar(total1=3 * t_tot)
        pbar.show()

        for t in range(t_tot):
            pbar.update_progressbar(t)
            spots_mask_lbl[t, :, :]  =  label(spots_mask[t, :, :])

        for t in range(t_tot):
            pbar.update_progressbar(t_tot + t)
            a  =  np.unique(spots_mask_lbl[t, :, :])[1:]
            for k in a:
                spots_tracked[t, :, :]  +=  (spots_mask_lbl[t, :, :] == k) * ((spots_mask_lbl[t, :, :] == k) * nuclei_tracked[t, :, :]).max()

        left      =  np.sign(spots_mask) - np.sign(spots_tracked)
        left_lbl  =  np.zeros(left.shape)
        for t in range(t_tot):
            left_lbl[t, :, :]  =  label(left[t, :, :])

        for tt in range(t_tot):
            pbar.update_progressbar(2 * t_tot + tt)
            rgp  =  regionprops(left_lbl[tt, :, :].astype(int))
            for kk in range(len(rgp)):
                m                        =   CloserNucleiFinder.CloserNucleiFinder(nuclei_tracked[tt, :, :], np.array([int(rgp[kk]["Centroid"][0]), int(rgp[kk]["Centroid"][1])]), spt_nuc_maxdist).mx_pt
                spots_tracked[tt, :, :]  +=  (left_lbl[tt, :, :] == rgp[kk]['label']).astype(int) * m

        pbar.close()

        self.spots_tracked  =  spots_tracked
        # self.left_lbl       =  left_lbl


class ProgressBar(QtWidgets.QWidget):
    """Simple progressbar widget."""
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
        """Update progressbar."""
        self.progressbar1.setValue(val1)
        QtWidgets.qApp.processEvents()
