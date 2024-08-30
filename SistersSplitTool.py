"""This is the main widget of the software to track separately sister chromatides.

It works as a post-processing tool of the SegmentTrack software (https://github.com/ant-trullo/SegmentTrack_v4.0)

author: antonio.trullo@igmm.cnrs.fr
"""


import os
import traceback
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from skimage.morphology import label

import HandleSistsCoords3D
import SisterSplit9
import SisterSplit7
import SistersIntsWriter
import AnalysisSaver
import AnalysisLoader
import SistersComprehensive
import TwoSpotsGallery


class SistersTool(QtWidgets.QWidget):
    """Popup tool to track and remove mitotical TS."""
    procStart  =  QtCore.pyqtSignal()

    def __init__(self, analysis_folder):
        QtWidgets.QWidget.__init__(self)

        raw_data              =  AnalysisLoader.RawData(analysis_folder)
        green4D               =  raw_data.green4D
        pix_size              =  raw_data.pix_size
        pix_size_Z            =  raw_data.pix_size_Z
        raw2show              =  np.zeros(raw_data.imarray_red.shape + (3,))
        raw2show[:, :, :, 0]  =  raw_data.imarray_red
        raw2show[:, :, :, 1]  =  raw_data.imarray_green

        nucs_trck  =  np.load(analysis_folder + '/nuclei_tracked.npy')
        spts_trck  =  np.load(analysis_folder + '/spots_tracked.npy')

        sgmt_raw               =  np.zeros(nucs_trck.shape + (3,), dtype=np.uint8)
        sgmt_raw[:, :, :, 2]   =  255 * np.sign(nucs_trck) * (1 - np.sign(spts_trck))
        sgmt_raw[:, :, :, 0]  +=  255 * np.sign(spts_trck)
        sgmt_raw[:, :, :, 1]  +=  255 * np.sign(spts_trck)
        sgmt_raw[:, :, :, 2]  +=  255 * np.sign(spts_trck)

        folder_lbl  =  QtWidgets.QLabel("Gauss Size", self)
        folder_lbl.setText(analysis_folder)

        frame1  =  pg.ImageView(self, name="Frame1")
        frame1.ui.menuBtn.hide()
        frame1.ui.roiBtn.hide()
        frame1.setImage(raw2show, autoRange=True)
        frame1.timeLine.sigPositionChanged.connect(self.update_from_frame1)

        frame2  =  pg.ImageView(self, name="Frame2")
        frame2.ui.menuBtn.hide()
        frame2.ui.roiBtn.hide()
        frame2.setImage(sgmt_raw, autoRange=True)
        frame2.view.setXLink("Frame1")
        frame2.view.setYLink("Frame1")
        frame2.timeLine.sigPositionChanged.connect(self.update_from_frame2)
        frame2.getImageItem().mouseClickEvent  =  self.click

        tab_s  =  QtWidgets.QTabWidget()
        tab2   =  QtWidgets.QWidget()

        frame2_box  =  QtWidgets.QHBoxLayout()
        frame2_box.addWidget(frame2)

        tab2.setLayout(frame2_box)
        tab_s.addTab(tab2, "Segmented")

        frame3  =  pg.ImageView(self, name="Frame3")
        frame3.ui.menuBtn.hide()
        frame3.ui.roiBtn.hide()
        frame3.view.setXLink('Frame2')
        frame3.view.setYLink('Frame2')
        frame3.timeLine.sigPositionChanged.connect(self.update_from_frame3)

        tabs  =  QtWidgets.QTabWidget()
        tab1  =  QtWidgets.QWidget()
        tab3  =  QtWidgets.QWidget()

        frame1_box  =  QtWidgets.QHBoxLayout()
        frame1_box.addWidget(frame1)

        frame3_box  =  QtWidgets.QHBoxLayout()
        frame3_box.addWidget(frame3)

        tab1.setLayout(frame1_box)
        tab3.setLayout(frame3_box)

        tabs.addTab(tab1, "Raw")
        tabs.addTab(tab3, "Sister Track")

        frames_box  =  QtWidgets.QHBoxLayout()
        frames_box.addWidget(tabs)
        frames_box.addWidget(tab_s)

        framepp1  =  pg.PlotWidget(self)

        dist_thr_lbl  =  QtWidgets.QLabel(self)
        dist_thr_lbl.setFixedSize(60, 25)
        dist_thr_lbl.setText("Dist Thr")

        dist_thr_edt  =  QtWidgets.QLineEdit(self)
        dist_thr_edt.textChanged[str].connect(self.dist_thr_var)
        dist_thr_edt.returnPressed.connect(self.track_sists)
        dist_thr_edt.setToolTip("Set the maximum distance between two consecutive positions of a spot (suggested value 15)")
        dist_thr_edt.setFixedSize(45, 25)

        dist_thr_box  =  QtWidgets.QHBoxLayout()
        dist_thr_box.addWidget(dist_thr_lbl)
        dist_thr_box.addWidget(dist_thr_edt)

        put_green_tggl  =  QtWidgets.QCheckBox('Green', self)
        put_green_tggl.stateChanged.connect(self.put_green_enable)
        put_green_tggl.setFixedSize(65, 25)
        put_green_tggl.setStyleSheet("color: green")

        put_red_tggl  =  QtWidgets.QCheckBox('Red', self)
        put_red_tggl.stateChanged.connect(self.put_red_enable)
        put_red_tggl.setFixedSize(65, 25)
        put_red_tggl.setStyleSheet("color: red")

        put_gray_tggl  =  QtWidgets.QCheckBox('Gray', self)
        put_gray_tggl.stateChanged.connect(self.put_gray_enable)
        put_gray_tggl.setFixedSize(65, 25)
        put_gray_tggl.setStyleSheet("color: gray")

        split_spot_tggl  =  QtWidgets.QCheckBox('Split', self)
        split_spot_tggl.stateChanged.connect(self.split_spot_enable)
        split_spot_tggl.setFixedSize(65, 25)

        flip_spots_tggl  =  QtWidgets.QCheckBox('Flip', self)
        flip_spots_tggl.stateChanged.connect(self.flip_spots_enable)
        flip_spots_tggl.setFixedSize(65, 25)

        retrack_onenuc_tggl  =  QtWidgets.QCheckBox('Re-track', self)
        retrack_onenuc_tggl.stateChanged.connect(self.retrack_onenuc_enable)
        retrack_onenuc_tggl.setFixedSize(65, 25)
        # retrack_onenuc_tggl.setStyleSheet("color: green")

        frame_numb_lbl  =  QtWidgets.QLabel(self)
        frame_numb_lbl.setFixedSize(150, 25)
        frame_numb_lbl.setText("Frame 0")

        nuc_tag_lbl  =  QtWidgets.QLabel(self)
        nuc_tag_lbl.setFixedSize(90, 25)
        nuc_tag_lbl.setText("Nuc Tag")

        nuc_tag_edt  =  QtWidgets.QLineEdit(self)
        nuc_tag_edt.textChanged[str].connect(self.nuc_tag_var)
        nuc_tag_edt.returnPressed.connect(self.show_nuc)
        nuc_tag_edt.setToolTip("Select nucleus to inspect by tag")
        nuc_tag_edt.setFixedSize(45, 25)

        nuc_tag_box  =  QtWidgets.QHBoxLayout()
        nuc_tag_box.addWidget(nuc_tag_lbl)
        nuc_tag_box.addWidget(nuc_tag_edt)

        write_sists_results_btn  =  QtWidgets.QPushButton("Write", self)
        write_sists_results_btn.clicked.connect(self.write_sists_results)
        write_sists_results_btn.setToolTip('Calculate the background and write xlsx files')
        write_sists_results_btn.setFixedSize(70, 25)

        save_btn  =  QtWidgets.QPushButton("Save", self)
        save_btn.clicked.connect(self.save_sists)
        save_btn.setToolTip('Save intensity traces')
        save_btn.setFixedSize(70, 25)

        nucs_save_btn  =  QtWidgets.QPushButton("Save Nucs", self)
        nucs_save_btn.clicked.connect(self.save_nucs)
        nucs_save_btn.setToolTip('Save Nuclei average position')
        nucs_save_btn.setFixedSize(70, 25)

        show_gallery_btn  =  QtWidgets.QPushButton("Gallery", self)
        show_gallery_btn.clicked.connect(self.two_spts_gallery)
        show_gallery_btn.setToolTip('Show gallery of the double traces')
        show_gallery_btn.setFixedSize(70, 25)

        commands  =  QtWidgets.QHBoxLayout()
        commands.addLayout(dist_thr_box)
        commands.addLayout(nuc_tag_box)
        commands.addStretch()
        commands.addWidget(put_green_tggl)
        commands.addWidget(put_red_tggl)
        commands.addWidget(flip_spots_tggl)
        commands.addWidget(put_gray_tggl)
        commands.addWidget(split_spot_tggl)
        commands.addWidget(retrack_onenuc_tggl)
        commands.addStretch()
        commands.addWidget(frame_numb_lbl)
        commands.addStretch()
        commands.addWidget(nucs_save_btn)
        commands.addWidget(save_btn)
        commands.addWidget(write_sists_results_btn)
        commands.addWidget(show_gallery_btn)

        frames_plot  =  QtWidgets.QVBoxLayout()
        frames_plot.addWidget(folder_lbl)
        frames_plot.addLayout(frames_box)
        frames_plot.addWidget(framepp1)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(frames_plot)
        layout.addLayout(commands)

        mycmap      =  np.fromfile("mycmap.bin", "uint16").reshape((10000, 3))   # / 255.0
        colors4map  =  []
        for k in range(mycmap.shape[0]):
            colors4map.append(mycmap[k, :])
        colors4map[0]  =  np.array([0, 0, 0])

        if os.path.isfile(analysis_folder + '/sists_coords.npy') and not os.path.isfile(analysis_folder + '/sists_3D_coords.npy'):
            np.save(analysis_folder + '/sists_3D_coords.npy', HandleSistsCoords3D.TranslateSistsCoords(analysis_folder).sists_3D_coords)

        if os.path.isfile(analysis_folder + '/sists_3D_coords.npy'):
            self.sists         =  SistersIntsWriter.LoadSistsTrack(analysis_folder)
            self.spts_left     =  SistersIntsWriter.LabelLeft(spts_trck, self.sists.sists_trck).spts_left
            self.dist_thr      =  SistersIntsWriter.LoadSistsTrackDistThr(analysis_folder).dist_thr
            self.tags_removed  =  SistersIntsWriter.ReadTags2Rm(analysis_folder).tags_removed
            dist_thr_edt.setText(str(self.dist_thr))
            frame3.setImage(self.sists.sists_trck, autoRange=True)
            mycmap          =  pg.ColorMap(np.linspace(0, 1, self.sists.sists_trck.max()), color=colors4map)
            frame3.setColorMap(mycmap)

        self.frame1               =  frame1
        self.frame2               =  frame2
        self.frame3               =  frame3
        self.framepp1             =  framepp1
        self.dist_thr_edt         =  dist_thr_edt
        self.nuc_tag_edt          =  nuc_tag_edt
        self.frame_numb_lbl       =  frame_numb_lbl
        self.put_green_tggl       =  put_green_tggl
        self.put_red_tggl         =  put_red_tggl
        self.put_gray_tggl        =  put_gray_tggl
        self.flip_spots_tggl      =  flip_spots_tggl
        self.split_spot_tggl      =  split_spot_tggl
        self.retrack_onenuc_tggl  =  retrack_onenuc_tggl
        self.software_version     =  "SistersSplitTool"

        self.analysis_folder  =  analysis_folder
        self.raw2show         =  raw2show
        self.nucs_trck        =  nucs_trck
        self.spts_trck        =  spts_trck
        self.sgmt_raw         =  sgmt_raw
        self.spts_idxs        =  np.unique(self.spts_trck[self.spts_trck != 0])  # tags of the tracked spots
        self.green4D          =  green4D
        self.colors4map       =  colors4map
        self.nuc_tag          =  0
        self.pix_size         =  pix_size
        self.pix_size_Z       =  pix_size_Z
        self.mpp_tags2rm      =  Sibs2Remove()
        self.mpp_tags2rm.show()

        if os.path.isfile(analysis_folder + '/sists_3D_coords.npy'):
            [self.mpp_tags2rm.add_tag(uu) for uu in self.tags_removed]

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("Split Sisters")

    def update_from_frame1(self):
        """Keep frames synchronized."""
        self.frame2.setCurrentIndex(self.frame1.currentIndex)
        self.frame_numb_lbl.setText("Frame " + str(self.frame1.currentIndex))

        try:
            self.frame3.setCurrentIndex(self.frame1.currentIndex)
            self.roi_crsr_sgm.setPos([self.frame1.currentIndex, 0], update=True)
        except AttributeError:
            pass

    def update_from_frame2(self):
        """Keep frames synchronized."""
        self.frame1.setCurrentIndex(self.frame2.currentIndex)
        try:
            self.frame3.setCurrentIndex(self.frame2.currentIndex)
            self.roi_crsr_sgm.setPos([self.frame1.currentIndex, 0], update=True)
        except AttributeError:
            pass

    def update_from_frame3(self):
        """Keep frames synchronized."""
        self.frame1.setCurrentIndex(self.frame3.currentIndex)
        self.frame2.setCurrentIndex(self.frame3.currentIndex)
        try:
            self.roi_crsr_sgm.setPos([self.frame1.currentIndex, 0], update=True)
        except AttributeError:
            pass

    def dist_thr_var(self, text):
        """Set distance threshold."""
        self.dist_thr  =  int(text)

    def nuc_tag_var(self, text):
        """Input the tag of the nucleus to inspect."""
        self.nuc_tag  =  int(text)

    def put_green_enable(self):
        """Enable the possibility to assign green color to a spot."""
        if self.put_green_tggl.isChecked():
            self.put_red_tggl.setCheckState(Qt.Unchecked)
            self.put_gray_tggl.setCheckState(Qt.Unchecked)
            self.split_spot_tggl.setCheckState(Qt.Unchecked)
            self.flip_spots_tggl.setCheckState(Qt.Unchecked)
            self.retrack_onenuc_tggl.setCheckState(Qt.Unchecked)

    def put_red_enable(self):
        """Enable the possibility to assign red color to a spot."""
        if self.put_red_tggl.isChecked():
            self.put_gray_tggl.setCheckState(Qt.Unchecked)
            self.put_green_tggl.setCheckState(Qt.Unchecked)
            self.split_spot_tggl.setCheckState(Qt.Unchecked)
            self.flip_spots_tggl.setCheckState(Qt.Unchecked)
            self.retrack_onenuc_tggl.setCheckState(Qt.Unchecked)

    def put_gray_enable(self):
        """Enable the possibility to remove the tag from a spot."""
        if self.put_gray_tggl.isChecked():
            self.put_red_tggl.setCheckState(Qt.Unchecked)
            self.put_green_tggl.setCheckState(Qt.Unchecked)
            self.split_spot_tggl.setCheckState(Qt.Unchecked)
            self.flip_spots_tggl.setCheckState(Qt.Unchecked)
            self.retrack_onenuc_tggl.setCheckState(Qt.Unchecked)

    def split_spot_enable(self):
        """Enable the possibility to split a spots in 2."""
        if self.split_spot_tggl.isChecked():
            self.put_red_tggl.setCheckState(Qt.Unchecked)
            self.put_gray_tggl.setCheckState(Qt.Unchecked)
            self.put_green_tggl.setCheckState(Qt.Unchecked)
            self.flip_spots_tggl.setCheckState(Qt.Unchecked)
            self.retrack_onenuc_tggl.setCheckState(Qt.Unchecked)

    def flip_spots_enable(self):
        """Enable the possibility to split a spots in 2."""
        if self.flip_spots_tggl.isChecked():
            self.split_spot_tggl.setCheckState(Qt.Unchecked)
            self.put_red_tggl.setCheckState(Qt.Unchecked)
            self.put_gray_tggl.setCheckState(Qt.Unchecked)
            self.put_green_tggl.setCheckState(Qt.Unchecked)
            self.retrack_onenuc_tggl.setCheckState(Qt.Unchecked)

    def retrack_onenuc_enable(self):
        """Enable the possibility to track again a couple of sisters."""
        if self.retrack_onenuc_tggl.isChecked():
            self.split_spot_tggl.setCheckState(Qt.Unchecked)
            self.put_red_tggl.setCheckState(Qt.Unchecked)
            self.put_gray_tggl.setCheckState(Qt.Unchecked)
            self.put_green_tggl.setCheckState(Qt.Unchecked)
            self.flip_spots_tggl.setCheckState(Qt.Unchecked)

    def keyPressEvent(self, event):
        """Show 3D spots or add a tag to remove."""
        if event.key() == (Qt.ControlModifier and Qt.Key_A):
            gg              =  HandleSistsCoords3D.coords3D23D(self.sists, self.nuc_tag, self.frame1.currentIndex, self.spts_trck)
            # gg_lbls         =  label(gg, connectivity=1)
            min_cmap        =  np.zeros((3, 3), dtype=np.uint8)
            min_cmap[0, :]  =  np.array([0, 0, 0])
            min_cmap[1, :]  =  np.array([255, 0, 0])
            min_cmap[2, :]  =  np.array([255, 255, 0])
            minicmap        =  pg.ColorMap(np.linspace(0, 1, 3), color=min_cmap)
            w               =  pg.image(gg, title="Frame Number " + str(self.frame1.currentIndex))
            w.setColorMap(minicmap)

        if event.key() == (Qt.ControlModifier and Qt.Key_E):
            self.mpp_tags2rm.add_tag(self.nuc_tag)
            self.mpp_tags2rm.show()

    def click(self, event):
        """Select a nucleus to study its time traces."""
        event.accept()
        pos        =  event.pos()
        modifiers  =  QtWidgets.QApplication.keyboardModifiers()
        ref_crd    =  None

        if modifiers  ==  QtCore.Qt.ShiftModifier:
            if not self.put_red_tggl.isChecked() and not self.put_green_tggl.isChecked() and not self.split_spot_tggl.isChecked() and not self.put_gray_tggl.isChecked() and not self.flip_spots_tggl.isChecked() and not self.retrack_onenuc_tggl.isChecked():
                self.sgm2show  =  np.copy(self.sgmt_raw)
                self.nuc_tag   =  self.nucs_trck[self.frame2.currentIndex, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)]
                self.nuc_tag_edt.setText(str(self.nuc_tag))
                if self.nuc_tag != 0:
                    ref_crd                     =  np.where(self.spts_idxs == self.nuc_tag)[0][0]
                    self.framepp1.clear()
                    self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 0, :], pen='r', symbol='+')
                    self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 1, :], pen='g', symbol='+')
                    self.plot_height            =  max(self.sists.sists_ints[ref_crd, 1, :].max(), self.sists.sists_ints[ref_crd, 0, :].max())
                    self.roi_crsr_sgm           =  pg.LineSegmentROI([[0, 0], [0, self.plot_height]], pen='y')
                    self.framepp1.addItem(self.roi_crsr_sgm)
                    s_tags                      =  (self.spts_trck == self.nuc_tag) * self.sists.sists_trck
                    s_tags                      =  np.unique(s_tags[s_tags != 0])
                    self.sgm2show               =  (self.sgm2show / 3).astype(np.uint8)
                    self.sgm2show[:, :, :, 2]  +=  (165 * (self.nucs_trck == self.nuc_tag) * (1 - np.sign(self.spts_trck))).astype(np.uint8)
                    self.sgm2show[:, :, :, 0]  +=  (165 * HandleSistsCoords3D.coords3D2proj(self.sists.sists_3D_coords, s_tags[0])).astype(np.uint8)
                    try:
                        self.sgm2show[:, :, :, 1]  +=  (165 * HandleSistsCoords3D.coords3D2proj(self.sists.sists_3D_coords, s_tags[1])).astype(np.uint8)
                    except IndexError:
                        pass

                elif self.nuc_tag == 0:
                    self.framepp1.clear()

                cif  =  self.frame2.currentIndex
                self.frame2.setImage(self.sgm2show, autoRange=False, autoLevels=False)
                self.frame2.setCurrentIndex(cif)

            if self.put_red_tggl.isChecked():
                f_idx      =  self.frame2.currentIndex
                if self.nuc_tag != 0 and self.spts_left[f_idx, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)] != 0:
                    ref_crd                       =  np.where(self.spts_idxs == self.nuc_tag)[0][0]
                    bff                           =  HandleSistsCoords3D.AddRed(self.spts_trck, self.nuc_tag, ref_crd, self.spts_left[f_idx], self.sists.sists_trck, self.sists.sists_3D_coords, self.sists.sists_ints, self.green4D, self.analysis_folder, pos, f_idx)
                    self.sists.sists_ints         =  np.copy(bff.sists_ints_fin)
                    self.sists.sists_3D_coords    =  np.copy(bff.sists_3D_coords_fin)
                    self.sists.sists_trck[f_idx]  =  np.copy(bff.sists_trck_fin[f_idx])
                    self.sists.sists_ints         =  np.copy(bff.sists_ints_fin)
                    self.spts_left[f_idx]         =  SistersIntsWriter.LabelLeft(self.spts_trck[f_idx], self.sists.sists_trck[f_idx]).spts_left

                    self.framepp1.clear()
                    self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 0, :], pen='r', symbol='+')
                    self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 1, :], pen='g', symbol='+')
                    self.plot_height   =  max(self.sists.sists_ints[ref_crd, 1, :].max(), self.sists.sists_ints[ref_crd, 0, :].max())
                    self.roi_crsr_sgm  =  pg.LineSegmentROI([[0, 0], [0, self.plot_height]], pen='y')
                    self.framepp1.addItem(self.roi_crsr_sgm)
                    self.sgm2show[f_idx, :, :, 0]  +=  (165 * bff.spt2add).astype(np.uint8)

                    self.frame2.updateImage()
                    self.frame3.updateImage()
                    self.roi_crsr_sgm.setPos([self.frame1.currentIndex, 0], update=True)

            if self.put_green_tggl.isChecked():
                f_idx     =  self.frame2.currentIndex
                # sgm2show  =  np.copy(self.sgmt_raw)

                if self.nuc_tag != 0 and self.spts_left[f_idx, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)] != 0:
                    ref_crd                       =  np.where(self.spts_idxs == self.nuc_tag)[0][0]
                    bff                           =  HandleSistsCoords3D.AddGreen(self.spts_trck, self.nuc_tag, ref_crd, self.spts_left[f_idx], self.sists.sists_trck, self.sists.sists_3D_coords, self.sists.sists_ints, self.green4D, self.analysis_folder, pos, f_idx)
                    self.sists.sists_ints         =  np.copy(bff.sists_ints_fin)
                    self.sists.sists_3D_coords    =  np.copy(bff.sists_3D_coords_fin)
                    self.sists.sists_trck[f_idx]  =  np.copy(bff.sists_trck_fin[f_idx])
                    self.sists.sists_ints         =  np.copy(bff.sists_ints_fin)
                    self.spts_left[f_idx]         =  SistersIntsWriter.LabelLeft(self.spts_trck[f_idx], self.sists.sists_trck[f_idx]).spts_left

                    self.framepp1.clear()
                    self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 0, :], pen='r', symbol='+')
                    self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 1, :], pen='g', symbol='+')
                    self.plot_height   =  max(self.sists.sists_ints[ref_crd, 1, :].max(), self.sists.sists_ints[ref_crd, 0, :].max())
                    self.roi_crsr_sgm  =  pg.LineSegmentROI([[0, 0], [0, self.plot_height]], pen='y')
                    self.framepp1.addItem(self.roi_crsr_sgm)

                    self.framepp1.addItem(self.roi_crsr_sgm)
                    self.sgm2show[f_idx, :, :, 1]  +=  (165 * bff.spt2add).astype(np.uint8)

                    self.frame2.updateImage()
                    self.frame3.updateImage()
                    self.roi_crsr_sgm.setPos([self.frame1.currentIndex, 0], update=True)

            if self.put_gray_tggl.isChecked():
                f_idx      =  self.frame2.currentIndex
                if self.nuc_tag != 0 and self.sists.sists_trck[f_idx, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)] != 0:
                    spt                           =  HandleSistsCoords3D.coords3D2proj_step(self.sists.sists_3D_coords, self.sists.sists_trck[f_idx, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)], f_idx).astype(np.uint8)
                    ref_crd                       =  np.where(self.spts_idxs == self.nuc_tag)[0][0]
                    bff                           =  HandleSistsCoords3D.AddGray(ref_crd, self.sists.sists_trck, self.sists.sists_3D_coords, self.sists.sists_ints, self.green4D, pos, f_idx)
                    self.sists.sists_ints         =  np.copy(bff.sists_ints_fin)
                    self.sists.sists_3D_coords    =  np.copy(bff.sists_3D_coords_fin)
                    self.sists.sists_trck[f_idx]  =  np.copy(bff.sists_trck_fin[f_idx])
                    self.sists.sists_ints         =  np.copy(bff.sists_ints_fin)
                    self.spts_left[f_idx]         =  SistersIntsWriter.LabelLeft(self.spts_trck[f_idx], self.sists.sists_trck[f_idx]).spts_left

                    self.framepp1.clear()
                    self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 0, :], pen='r', symbol='+')
                    self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 1, :], pen='g', symbol='+')
                    self.plot_height   =  max(self.sists.sists_ints[ref_crd, 1, :].max(), self.sists.sists_ints[ref_crd, 0, :].max())
                    self.roi_crsr_sgm  =  pg.LineSegmentROI([[0, 0], [0, self.plot_height]], pen='y')
                    self.framepp1.addItem(self.roi_crsr_sgm)

                    if self.sgm2show[f_idx, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int), 0] == 85:
                        self.sgm2show[f_idx, :, :, 1]  -=  165 * spt
                    elif self.sgm2show[f_idx, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int), 1] == 85:
                        self.sgm2show[f_idx, :, :, 0]  -=  165 * spt
                    self.frame2.updateImage()
                    self.frame3.updateImage()
                    self.roi_crsr_sgm.setPos([self.frame1.currentIndex, 0], update=True)

            if self.flip_spots_tggl.isChecked():
                f_idx       =  self.frame2.currentIndex
                sists_tags  =  np.unique((self.spts_trck[f_idx] == self.nuc_tag) * self.sists.sists_trck[f_idx])[1:]
                if self.nuc_tag != 0 and sists_tags.size > 0:
                    ref_crd                       =  np.where(self.spts_idxs == self.nuc_tag)[0][0]
                    bff                           =  HandleSistsCoords3D.FlipSpots(ref_crd, self.nuc_tag, self.spts_trck, sists_tags,  self.sists.sists_trck, self.sists.sists_3D_coords, self.sists.sists_ints, f_idx)
                    self.sists.sists_ints         =  np.copy(bff.sists_ints_fin)
                    self.sists.sists_3D_coords    =  np.copy(bff.sists_3D_coords_fin)
                    self.sists.sists_trck[f_idx]  =  np.copy(bff.sists_trck_fin[f_idx])
                    self.sists.sists_ints         =  np.copy(bff.sists_ints_fin)

                    self.framepp1.clear()
                    self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 0, :], pen='r', symbol='+')
                    self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 1, :], pen='g', symbol='+')
                    self.plot_height   =  max(self.sists.sists_ints[ref_crd, 1, :].max(), self.sists.sists_ints[ref_crd, 0, :].max())
                    self.roi_crsr_sgm  =  pg.LineSegmentROI([[0, 0], [0, self.plot_height]], pen='y')
                    self.framepp1.addItem(self.roi_crsr_sgm)

                    msk1  =  (bff.spt2ch_tags == 1).astype(np.uint8)
                    if (self.sgm2show[f_idx, :, :, 0] * msk1).max() >= 200:
                        self.sgm2show[f_idx, :, :, 0]  -=  165 * msk1
                        self.sgm2show[f_idx, :, :, 1]  +=  165 * msk1
                    else:
                        self.sgm2show[f_idx, :, :, 1]  -=  165 * msk1
                        self.sgm2show[f_idx, :, :, 0]  +=  165 * msk1

                    if bff.spt2ch_tags.max() == 2:
                        msk2  =  (bff.spt2ch_tags == 2).astype(np.uint8)

                        if (self.sgm2show[f_idx, :, :, 0] * msk2).max() >= 200:
                            self.sgm2show[f_idx, :, :, 0]  -=  165 * msk2
                            self.sgm2show[f_idx, :, :, 1]  +=  165 * msk2
                        else:
                            self.sgm2show[f_idx, :, :, 1]  -=  165 * msk2
                            self.sgm2show[f_idx, :, :, 0]  +=  165 * msk2

                    self.frame2.updateImage()
                    self.frame3.updateImage()
                    self.roi_crsr_sgm.setPos([self.frame1.currentIndex, 0], update=True)

            if self.split_spot_tggl.isChecked():
                f_idx       =  self.frame2.currentIndex
                sists_tags  =  np.unique((self.spts_trck[f_idx] == self.nuc_tag) * self.sists.sists_trck[f_idx])[1:]
                if self.nuc_tag != 0 and sists_tags.size > 0:

                    ref_crd                       =  np.where(self.spts_idxs == self.nuc_tag)[0][0]
                    bff                           =  HandleSistsCoords3D.SplitSpot(ref_crd, self.spts_trck, self.nuc_tag, self.sists.sists_trck, self.sists.sists_3D_coords, self.sists.sists_ints, self.green4D, pos, f_idx, 8, self.pix_size, self.pix_size_Z)
                    self.sists.sists_ints         =  np.copy(bff.sists_ints_fin)
                    self.sists.sists_3D_coords    =  np.copy(bff.sists_3D_coords_fin)
                    self.sists.sists_trck[f_idx]  =  np.copy(bff.sists_trck_fin)[f_idx]
                    self.sists.sists_ints         =  np.copy(bff.sists_ints_fin)

                    prev_tags  =  np.unique(self.sists.sists_trck[f_idx - 1] * (self.spts_trck[f_idx - 1] == self.nuc_tag))[1:]                  # sister's tags in the previous frame
                    if prev_tags.size == 2:
                        spts_red    =  HandleSistsCoords3D.coords3D2proj_step(self.sists.sists_3D_coords, prev_tags[0], f_idx)
                        spts_green  =  HandleSistsCoords3D.coords3D2proj_step(self.sists.sists_3D_coords, prev_tags[1], f_idx)

                        self.sgm2show[f_idx, :, :, 0]  *=  (1 - np.sign(spts_red + spts_green))
                        self.sgm2show[f_idx, :, :, 1]  *=  (1 - np.sign(spts_red + spts_green))
                        self.sgm2show[f_idx, :, :, 0]  +=  250 * spts_red
                        self.sgm2show[f_idx, :, :, 1]  +=  85 * spts_red * (1 - spts_green)
                        self.sgm2show[f_idx, :, :, 1]  +=  250 * spts_green
                        self.sgm2show[f_idx, :, :, 0]  +=  85 * spts_green * (1 - spts_red)

                    elif prev_tags.size == 1:
                        fll_tags  =  np.unique(self.sists.sists_trck[f_idx + 1] * (self.spts_trck[f_idx + 1] == self.nuc_tag))[1:]                  # sister's tags in the following frame
                        if fll_tags.size == 2:
                            spts_red    =  HandleSistsCoords3D.coords3D2proj_step(self.sists.sists_3D_coords, fll_tags[0], f_idx)
                            spts_green  =  HandleSistsCoords3D.coords3D2proj_step(self.sists.sists_3D_coords, fll_tags[1], f_idx)

                            self.sgm2show[f_idx, :, :, 0]  *=  (1 - np.sign(spts_red + spts_green))
                            self.sgm2show[f_idx, :, :, 1]  *=  (1 - np.sign(spts_red + spts_green))
                            self.sgm2show[f_idx, :, :, 0]  +=  250 * spts_red
                            self.sgm2show[f_idx, :, :, 1]  +=  85 * spts_red * (1 - spts_green)
                            self.sgm2show[f_idx, :, :, 1]  +=  250 * spts_green
                            self.sgm2show[f_idx, :, :, 0]  +=  85 * spts_green * (1 - spts_red)

                    self.framepp1.clear()
                    self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 0, :], pen='r', symbol='+')
                    self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 1, :], pen='g', symbol='+')
                    self.plot_height   =  max(self.sists.sists_ints[ref_crd, 1, :].max(), self.sists.sists_ints[ref_crd, 0, :].max())
                    self.roi_crsr_sgm  =  pg.LineSegmentROI([[0, 0], [0, self.plot_height]], pen='y')
                    self.framepp1.addItem(self.roi_crsr_sgm)

                    self.frame2.updateImage()
                    self.frame3.updateImage()
                    self.roi_crsr_sgm.setPos([self.frame1.currentIndex, 0], update=True)

            if self.retrack_onenuc_tggl.isChecked():
                # reload(SisterSplit9)
                nuc_tag   =  self.nucs_trck[self.frame2.currentIndex, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)]
                self.nuc_tag_edt.setText(str(nuc_tag))
                # print(nuc_tag)
                if self.nuc_tag != 0:
                    spots_3D_coords  =  np.load(self.analysis_folder + '/spots_3D_coords.npy')
                    dist_thr         =  InputScalarValue.getNumb(["Dist Thr", "Input the distance threshold for spots distance", "Distance Threshold"])
                    spts_nuc         =  (self.spts_trck == nuc_tag).sum(0)
                    ff               =  np.where(spts_nuc != 0)
                    x_min            =  np.max([0, ff[0].min() - 1])
                    x_max            =  np.min([ff[0].max() + 1, self.spts_trck.shape[1]])
                    y_min            =  np.max([ff[1].min() - 1, 0])
                    y_max            =  np.min([ff[1].max() + 1, self.spts_trck.shape[2]])
                    multip_args      =  list()
                    # pg.image(self.spts_trck[:, x_min:x_max, y_min:y_max])
                    multip_args.append(self.spts_trck[:, x_min:x_max, y_min:y_max])
                    multip_args.append(self.green4D[:, :, x_min:x_max, y_min:y_max])
                    multip_args.append(spots_3D_coords)
                    multip_args.append([x_min, x_max, y_min, y_max])
                    multip_args.append(self.pix_size)
                    multip_args.append(self.pix_size_Z)
                    multip_args.append(dist_thr)
                    SisterSplit7.SisterSplitSingleNucleus(multip_args)

        elif (modifiers & QtCore.Qt.ShiftModifier) and (modifiers & QtCore.Qt.ControlModifier):
            if self.put_gray_tggl.isChecked():
                f_idx    =  self.frame2.currentIndex
                for t_idx in range(f_idx, self.green4D.shape[0]):
                    tags4pos  =  np.unique(self.sists.sists_trck[t_idx] * (self.spts_trck == self.nuc_tag))[1:]
                    for tag4pos in tags4pos:
                        xy4pos  =  np.where(self.sists.sists_trck[t_idx] == tag4pos)
                        pos     =  [xy4pos[0][0], xy4pos[1][0]]

                        if self.nuc_tag != 0:
                            spt                           =  HandleSistsCoords3D.coords3D2proj_step(self.sists.sists_3D_coords, self.sists.sists_trck[t_idx, pos[0], pos[1]], t_idx).astype(np.uint8)
                            ref_crd                       =  np.where(self.spts_idxs == self.nuc_tag)[0][0]
                            bff                           =  HandleSistsCoords3D.AddGray(ref_crd, self.sists.sists_trck, self.sists.sists_3D_coords, self.sists.sists_ints, self.green4D, pos, t_idx)
                            self.sists.sists_ints         =  np.copy(bff.sists_ints_fin)
                            self.sists.sists_3D_coords    =  np.copy(bff.sists_3D_coords_fin)
                            self.sists.sists_trck[t_idx]  =  np.copy(bff.sists_trck_fin[t_idx])
                            self.sists.sists_ints         =  np.copy(bff.sists_ints_fin)
                            self.spts_left[t_idx]         =  SistersIntsWriter.LabelLeft(self.spts_trck[t_idx], self.sists.sists_trck[t_idx]).spts_left

                            if self.sgm2show[t_idx, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int), 0] == 85:
                                self.sgm2show[t_idx, :, :, 1]  -=  165 * spt
                            elif self.sgm2show[t_idx, np.round(pos[0]).astype(int), np.round(pos[1]).astype(int), 1] == 85:
                                self.sgm2show[t_idx, :, :, 0]  -=  165 * spt

                self.framepp1.clear()
                self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 0, :], pen='r', symbol='+')
                self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 1, :], pen='g', symbol='+')
                self.plot_height   =  max(self.sists.sists_ints[ref_crd, 1, :].max(), self.sists.sists_ints[ref_crd, 0, :].max())
                self.roi_crsr_sgm  =  pg.LineSegmentROI([[0, 0], [0, self.plot_height]], pen='y')
                self.framepp1.addItem(self.roi_crsr_sgm)

                self.frame2.updateImage()
                self.frame3.updateImage()
                self.roi_crsr_sgm.setPos([self.frame1.currentIndex, 0], update=True)

            elif self.flip_spots_tggl.isChecked():
                f_idx       =  self.frame2.currentIndex
                sists_tags  =  np.unique((self.spts_trck[f_idx:] == self.nuc_tag) * self.sists.sists_trck[f_idx:])[1:]
                if self.nuc_tag != 0 and sists_tags.size > 0:
                    ref_crd                        =  np.where(self.spts_idxs == self.nuc_tag)[0][0]
                    bff                            =  HandleSistsCoords3D.FlipSpotsFromFrameOn(ref_crd, self.nuc_tag, self.spts_trck, sists_tags,  self.sists.sists_trck, self.sists.sists_3D_coords, self.sists.sists_ints, f_idx)
                    self.sists.sists_ints          =  np.copy(bff.sists_ints_fin)
                    self.sists.sists_3D_coords     =  np.copy(bff.sists_3D_coords_fin)
                    self.sists.sists_trck[f_idx:]  =  np.copy(bff.sists_trck_fin)[f_idx:]
                    self.sists.sists_ints          =  np.copy(bff.sists_ints_fin)

                    for t_idx in range(f_idx, self.sists.sists_trck.shape[0]):
                        msk1  =  (bff.spt2ch_tags[t_idx] == 1).astype(np.uint8)
                        if (self.sgm2show[t_idx, :, :, 0] * msk1).max() >= 200:     # check msk1 is red
                            self.sgm2show[t_idx, :, :, 0]  -=  165 * msk1           # remove red
                            self.sgm2show[t_idx, :, :, 1]  +=  165 * msk1           # add green
                        elif (self.sgm2show[t_idx, :, :, 1] * msk1).max() >= 200:   # check msk1 is green
                            self.sgm2show[t_idx, :, :, 1]  -=  165 * msk1           # remove green
                            self.sgm2show[t_idx, :, :, 0]  +=  165 * msk1           # add red

                        if bff.spt2ch_tags[t_idx].max() == 2:
                            msk2  =  (bff.spt2ch_tags[t_idx] == 2).astype(np.uint8)
                            if (self.sgm2show[t_idx, :, :, 0] * msk2).max() >= 200:  # cheeck msk2 is redd
                                self.sgm2show[t_idx, :, :, 0]  -=  165 * msk2       # remove red
                                self.sgm2show[t_idx, :, :, 1]  +=  165 * msk2       # add green
                            elif (self.sgm2show[t_idx, :, :, 1] * msk2).max() >= 200:  # check msk2 is green
                                self.sgm2show[t_idx, :, :, 1]  -=  165 * msk2      # remove green
                                self.sgm2show[t_idx, :, :, 0]  +=  165 * msk2      # add red

                self.framepp1.clear()
                self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 0, :], pen='r', symbol='+')
                self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 1, :], pen='g', symbol='+')
                self.plot_height   =  max(self.sists.sists_ints[ref_crd, 1, :].max(), self.sists.sists_ints[ref_crd, 0, :].max())
                self.roi_crsr_sgm  =  pg.LineSegmentROI([[0, 0], [0, self.plot_height]], pen='y')
                self.framepp1.addItem(self.roi_crsr_sgm)

                self.frame2.updateImage()
                self.frame3.updateImage()
                self.roi_crsr_sgm.setPos([self.frame1.currentIndex, 0], update=True)

    def track_sists(self):
        """Track sisters separately."""
        self.dist_thr_edt.setStyleSheet("background : red;")
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()
        try:
            if self.dist_thr > 0:
                self.sists      =  SisterSplit7.SisterSplit(self.analysis_folder, self.dist_thr, self.green4D)
            elif self.dist_thr <= 0:
                self.sists      =  SisterSplit9.SisterSplit(self.analysis_folder, self.green4D)
            self.spts_left  =  label(self.spts_trck * (1 - np.sign(self.sists.sists_trck)), connectivity=1)
            self.frame3.setImage(self.sists.sists_trck)
            self.frame1.autoRange()
            mycmap  =  pg.ColorMap(np.linspace(0, 1, self.sists.sists_trck.max()), color=self.colors4map)
            self.frame3.setColorMap(mycmap)
        except Exception:
            traceback.print_exc()
        self.dist_thr_edt.setStyleSheet("background : white;")

    def show_nuc(self):
        """Inspect the nucleus with tag input by the user."""
        if self.nuc_tag in self.spts_idxs:
            self.sgm2show  =  np.copy(self.sgmt_raw)
            ref_crd        =  np.where(self.spts_idxs == self.nuc_tag)[0][0]
            self.framepp1.clear()
            self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 0, :], pen='r', symbol='+')
            self.framepp1.plot(np.arange(self.spts_trck.shape[0]), self.sists.sists_ints[ref_crd, 1, :], pen='g', symbol='+')
            self.plot_height   =  max(self.sists.sists_ints[ref_crd, 1, :].max(), self.sists.sists_ints[ref_crd, 0, :].max())
            self.roi_crsr_sgm  =  pg.LineSegmentROI([[0, 0], [0, self.plot_height]], pen='y')
            self.framepp1.addItem(self.roi_crsr_sgm)

            s_tags                      =  (self.spts_trck == self.nuc_tag) * self.sists.sists_trck
            s_tags                      =  np.unique(s_tags[s_tags != 0])
            print(s_tags)
            self.sgm2show               =  (self.sgm2show / 3).astype(np.uint8)
            self.sgm2show[:, :, :, 2]  +=  (165 * (self.nucs_trck == self.nuc_tag) * (1 - np.sign(self.spts_trck))).astype(np.uint8)
            self.sgm2show[:, :, :, 0]  +=  (165 * HandleSistsCoords3D.coords3D2proj(self.sists.sists_3D_coords, s_tags[0])).astype(np.uint8)
            try:
                self.sgm2show[:, :, :, 1]  +=  (165 * HandleSistsCoords3D.coords3D2proj(self.sists.sists_3D_coords, s_tags[1])).astype(np.uint8)
            except IndexError:
                pass

        elif self.nuc_tag not in self.spts_idxs:
            self.framepp1.clear()
            self.sgm2show  =  self.sgmt_raw

        cif  =  self.frame2.currentIndex
        self.frame2.setImage(self.sgm2show, autoRange=False, autoLevels=False)
        self.frame2.setCurrentIndex(cif)

    def save_sists(self):
        """Save new sisters coordinates."""
        np.save(self.analysis_folder + '/sists_ints.npy', self.sists.sists_ints)
        np.save(self.analysis_folder + '/sists_3D_coords.npy', self.sists.sists_3D_coords)
        if not os.path.isfile(self.analysis_folder + '/AlleleIntensity.xlsx'):
            SistersIntsWriter.WriteMomentaryInfo(self.analysis_folder, self.dist_thr)

    def write_sists_results(self):
        """Save the sister analyis in a xlsx file."""
        self.save_sists()
        # reload(Background4Sisters)
        tags2rm     =  list()
        for l in range(self.mpp_tags2rm.table_widget.rowCount()):
            tags2rm.append((self.mpp_tags2rm.table_widget.item(l, 0).text()))

        tags2rm     =  [int(tag2rm) for tag2rm in tags2rm]
        # sists_bckg  =  Background4Sisters.Background4Sisters(self.sists.sists_3D_coords, self.sists.sists_ints, self.green4D).sists_bkg
        # SistersIntsWriter.SistersIntsWriter(self.analysis_folder, self.sists.sists_ints, sists_bckg, self.sists.sists_3D_coords, self.spts_idxs, tags2rm, self.dist_thr, self.software_version)
        SistersIntsWriter.SistsIntsDist(self.analysis_folder, self.green4D, tags2rm, self.software_version, self.pix_size, self.pix_size_Z, self.dist_thr)
        flag_y_n  =  ChooseYorN.getFlag()
        if flag_y_n == "yes":
            SistersComprehensive.SistersComprehensive(self.analysis_folder)
        AnalysisSaver.NucleiSaver(self.analysis_folder, self.nucs_trck, self.software_version)

    def two_spts_gallery(self):
        """Generate the gallery of all the split time series."""
        TwoSpotsGallery.TwoSpotsGallery(self.analysis_folder)

    def save_nucs(self):
        """Save average nuclei position."""
        AnalysisSaver.NucleiSaver(self.analysis_folder, self.nucs_trck, self.software_version)


class Sibs2Remove(QtWidgets.QWidget):
    """Pop up tool to save the tags of the sibiling to remove."""
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        ksf_h  =  1
        ksf_w  =  1

        table_widget  =  QtWidgets.QTableWidget()
        table_widget.setColumnCount(1)
        table_widget.setHorizontalHeaderLabels(["Tags to Remove"])

        header   =  table_widget.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

        remove_tag_btn  =  QtWidgets.QPushButton("Remove Entry", self)
        remove_tag_btn.setToolTip("Remove tag from the list")
        remove_tag_btn.setFixedSize(int(ksf_h * 190), int(ksf_w * 28))
        remove_tag_btn.clicked.connect(self.remove_tag)

        commands  =  QtWidgets.QVBoxLayout()
        commands.addWidget(remove_tag_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addWidget(table_widget)
        layout.addLayout(commands)

        self.table_widget     =  table_widget
        self.table_numb_raw   =  -1
        self.all_folders      =  []
        self.raw_data_fnames  =  list()

        self.entry  =  0

        self.setLayout(layout)
        self.setGeometry(30, 30, 800, 100)
        self.setWindowTitle('Sisters to remove')

    def add_tag(self, numb):
        """Add a tag to the list in the widget."""
        self.table_numb_raw  +=  1
        self.table_widget.insertRow(self.table_numb_raw)
        self.table_widget.setItem(self.table_numb_raw, 0, QtWidgets.QTableWidgetItem(str(numb)))

    def remove_tag(self):
        """Remove tag."""
        self.table_widget.removeRow(self.table_widget.currentRow())
        self.table_numb_raw  -=  1

    def closewidg(self):
        """Close the tool."""
        self.close()


class ChooseYorN(QtWidgets.QDialog):
    """Choose the spots channel to analyse."""
    def __init__(self, parent=None):
        super(ChooseYorN, self).__init__(parent)

        # choose_lbl  =  QtWidgets.QLabel(dialog_title, self)
        choose_lbl  =  QtWidgets.QLabel("Do you want Spatial Selection?", self)
        choose_lbl.setFixedSize(210, 25)

        choose_y_btn  =  QtWidgets.QPushButton("Yes", self)
        choose_y_btn.setFixedSize(60, 25)
        choose_y_btn.clicked.connect(self.choose_yes)

        choose_n_btn  =  QtWidgets.QPushButton("No", self)
        choose_n_btn.setFixedSize(60, 25)
        choose_n_btn.clicked.connect(self.choose_no)

        choose_box  =  QtWidgets.QHBoxLayout()
        choose_box.addWidget(choose_y_btn)
        choose_box.addStretch()
        choose_box.addWidget(choose_n_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addWidget(choose_lbl)
        layout.addLayout(choose_box)

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, 220, 25)
        # self.setWindowTitle("Spatial Selection?")

    def choose_yes(self):
        """Choose flag yes."""
        self.flag_a_b  =  "yes"
        self.close()

    def choose_no(self):
        """Choose flag no."""
        self.flag_a_b  =  "no"
        self.close()

    def params(self):
        """Function to send choice."""
        return self.flag_a_b

    @staticmethod
    def getFlag(parent=None):
        """Send choice."""
        dialog  =  ChooseYorN(parent)
        result  =  dialog.exec_()
        flag    =  dialog.params()
        return flag


class InputScalarValue(QtWidgets.QDialog):
    def __init__(self, texts, parent=None):
        super().__init__(parent)

        numb_pixels_lbl  =  QtWidgets.QLabel(texts[0], self)
        numb_pixels_lbl.setFixedSize(110, 25)

        numb_pixels_edt = QtWidgets.QLineEdit(self)
        numb_pixels_edt.setToolTip(texts[1])
        numb_pixels_edt.setFixedSize(100, 22)
        numb_pixels_edt.textChanged[str].connect(self.numb_pixels_var)

        input_close_btn  =  QtWidgets.QPushButton("Ok", self)
        input_close_btn.clicked.connect(self.input_close)
        input_close_btn.setToolTip('Input values')
        input_close_btn.setFixedSize(50, 25)

        numb_pixels_lbl_edit_box  =  QtWidgets.QHBoxLayout()
        numb_pixels_lbl_edit_box.addWidget(numb_pixels_lbl)
        numb_pixels_lbl_edit_box.addWidget(numb_pixels_edt)

        input_close_box  =  QtWidgets.QHBoxLayout()
        input_close_box.addStretch()
        input_close_box.addWidget(input_close_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(numb_pixels_lbl_edit_box)
        layout.addLayout(input_close_box)

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, 160, 120)
        self.setWindowTitle(texts[2])

    def numb_pixels_var(self, text):
        self.numb_pixels_value  =  float(text)

    def input_close(self):
        self.close()

    def numb_pixels(self):
        return self.numb_pixels_value

    @staticmethod
    def getNumb(parent=None):
        dialog       =  InputScalarValue(parent)
        result       =  dialog.exec_()
        numb_pixels  =  dialog.numb_pixels()
        return numb_pixels
