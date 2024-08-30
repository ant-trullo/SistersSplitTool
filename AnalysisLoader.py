"""This function loads the previously done analysis."""


import numpy as np
from natsort import natsorted
from PyQt5 import QtWidgets
from aicsimageio import AICSImage

# import MultiLoadCzi5D
import LoadCzi5D
import ServiceWidgets


class RawData:
    """Load and concatenate .czi files."""
    def __init__(self, analysis_folder):

        fnames        =  natsorted(QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", analysis_folder, filter="*.lsm *.czi *.tif *.lif")[0])  # select the raw data files
        im_red_smpl   =  np.load(analysis_folder + '/im_red_smpl.npy')                # read the first and last analyzed frames in red channel mip
        first_frame   =  im_red_smpl[0]                                               # load the first analyzed raw frame
        last_frame    =  im_red_smpl[1]                                                # load the last analyzed raw frame
        crop_vect     =  np.load(analysis_folder + '/crop_vect.npy')                   # load crop corners coordinates
        nucs_spts_ch  =  np.fromfile(analysis_folder + '/nucs_spts_ch.bin', 'uint16')   # load nuclei and spots channel numbers

        err_msg    =  QtWidgets.QMessageBox()
        red_mtx    =  None                              # initialize output nuclei mip matrix
        green_mtx  =  None                              # initialize output spots mip matrix
        green4D    =  None                              # initialize output spots matrix

        pbar  =  ServiceWidgets.ProgressBar(total1=len(fnames))
        pbar.show()
        pbar.update_progressbar1(1)

        for cnt, fname in enumerate(fnames):
            pbar.update_progressbar1(cnt)
            raw_bff            =  LoadCzi5D.LoadCzi5D(fname, nucs_spts_ch)                                        # load the first of the selected files
            raw_bff.red_mtx    =  raw_bff.red_mtx[:, crop_vect[0]:crop_vect[2], crop_vect[1]:crop_vect[3]]        # crop following the saved crop the nuclei channel mip
            raw_bff.green_mtx  =  raw_bff.green_mtx[:, crop_vect[0]:crop_vect[2], crop_vect[1]:crop_vect[3]]      # crop following the saved crop the spots channel mip
            raw_bff.green4D    =  raw_bff.green4D[:, :, crop_vect[0]:crop_vect[2], crop_vect[1]:crop_vect[3]]     # crop following the saved crop the spots channel
            fnd_first          =  np.where(np.sum(raw_bff.red_mtx - first_frame, axis=(1, 2)) == 0)[0]            # check if the first analyzed frame is present in this file, if the first analyzed frame is not present in the file, the for cycle goes on overwriting without storing
            if fnd_first.size != 0:                                                                                   # if it is the case, start to  store the data 
                red_mtx    =  raw_bff.red_mtx[fnd_first[0]:]                                                          # add the raw data loaded in the output nuclei mip matrix
                green_mtx  =  raw_bff.green_mtx[fnd_first[0]:]                                                        # add the raw data loaded in the output spots mip matrix
                green4D    =  raw_bff.green4D[fnd_first[0]:]                                                          # add the raw data loaded in the output spots matrix
                strt_idx   =  cnt + 1                                                                                 # store the fname index
                break                                                                                                 # break the cycle

        if red_mtx is None:
            pbar.close()
            err_msg.setText("The first analyzed frame is not in the selected files")
            err_msg.exec()

        last_flag  =  False
        fnd_lst    =  np.where(np.sum(red_mtx - last_frame, axis=(1, 2)) == 0)[0]
        if fnd_lst.size != 0:
            red_mtx    =  red_mtx[:fnd_lst[0] + 1]
            green_mtx  =  green_mtx[:fnd_lst[0] + 1]
            green4D    =  green4D[:fnd_lst[0] + 1]
            last_flag  =  True
            end_idx    =  strt_idx + 1

        elif fnd_lst.size == 0:
            for cc, ff in enumerate(fnames[strt_idx:]):                                                                 # same as for the previous for cycle, just here we search for the last analyzed frame
                pbar.update_progressbar1(cc + strt_idx)
                raw_bff            =  LoadCzi5D.LoadCzi5D(ff, nucs_spts_ch)
                raw_bff.green_mtx  =  raw_bff.green_mtx[:, crop_vect[0]:crop_vect[2], crop_vect[1]:crop_vect[3]]
                raw_bff.red_mtx    =  raw_bff.red_mtx[:, crop_vect[0]:crop_vect[2], crop_vect[1]:crop_vect[3]]
                raw_bff.green4D    =  raw_bff.green4D[:, :, crop_vect[0]:crop_vect[2], crop_vect[1]:crop_vect[3]]
                fnd_lst            =  np.where(np.sum(raw_bff.red_mtx - last_frame, axis=(1, 2)) == 0)[0]
                if fnd_lst.size == 0:
                    red_mtx    =  np.concatenate((red_mtx, raw_bff.red_mtx), axis=0)
                    green_mtx  =  np.concatenate((green_mtx, raw_bff.green_mtx), axis=0)
                    green4D    =  np.concatenate((green4D, raw_bff.green4D), axis=0)
                elif fnd_lst.size != 0:
                    red_mtx    =  np.concatenate((red_mtx, raw_bff.red_mtx[:fnd_lst[0] + 1]), axis=0)
                    green_mtx  =  np.concatenate((green_mtx, raw_bff.green_mtx[:fnd_lst[0] + 1]), axis=0)
                    green4D    =  np.concatenate((green4D, raw_bff.green4D[:fnd_lst[0] + 1]), axis=0)
                    end_idx    =  cc + strt_idx
                    last_flag  =  True
                    break

        if not last_flag:
            pbar.close()
            err_msg.setText("The last analyzed frame is not in the selected files")
            err_msg.exec()

        img  =  AICSImage(fnames[0])                                                          # read file just for pixel size

        self.imarray_green  =  green_mtx
        self.imarray_red    =  red_mtx
        self.green4D        =  green4D
        self.fnames         =  fnames[strt_idx - 1:end_idx]
        self.pix_size       =  img.physical_pixel_sizes.X
        self.pix_size_Z     =  img.physical_pixel_sizes.Z
        self.crop_vect      =  crop_vect


class SpotsIntsVol:
    """Loads intensity and volume of detected spots."""
    def __init__(self, foldername):

        self.spots_ints     =  np.load(foldername + '/spots_3D_ints.npy')
        self.spots_vol      =  np.load(foldername + '/spots_3D_vol.npy')
        self.spots_tzxy     =  np.load(foldername + '/spots_3D_tzxy.npy')
        self.spots_coords   =  np.load(foldername + '/spots_3D_coords.npy')


class Features:
    """Load spots features."""
    def __init__(self, foldername):

        self.statistics_info  =  np.load(foldername + '/spots_features3D.npy')


class SpotsIntsVolRescue:
    """Load intensity and volume for analysis rescue."""
    def __init__(self):

        self.spots_ints  =  np.load('rescue_spts_ints.npy')
        self.spots_vol   =  np.load('rescue_spts_vol.npy')
        self.spots_tzxy  =  np.load('rescue_spts_tzxy.npy')
