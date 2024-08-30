"""Set of function to work on the 3D coordinate system.

"""


import numpy as np
import multiprocessing
from sklearn.mixture import GaussianMixture
from skimage.measure import label, regionprops_table
# import pyqtgraph as pg

import GaussianScore


def funct4multip(input_jobs):
    """Function for multiprocessing"""
    sists_coords_chop  =  input_jobs[0]
    spots_3D_coords    =  input_jobs[1]

    sists_coords3D  =  list()  # this will contain
    for kk in sists_coords_chop:
        aa  =  (spots_3D_coords[:, 0] == kk[1])
        bb  =  (spots_3D_coords[:, 2] == kk[2])
        cc  =  (spots_3D_coords[:, 3] == kk[3])
        mm  =  np.where(aa * bb * cc)
        for mm_sing in mm[0]:
            sists_coords3D.append(np.append(kk[0], spots_3D_coords[mm_sing]))
    return np.asarray(sists_coords3D)


def coords3D2proj(sists_3D_coords, tag):
    """Function to reconstruct a 2D projected spot by its tag"""
    spt            =  np.zeros((sists_3D_coords[-1, (1, 3, 4)]), dtype=np.uint16)
    sub_3D_coords  =  sists_3D_coords[sists_3D_coords[:, 0] == tag]
    spt[sub_3D_coords[:, 1], sub_3D_coords[:, 3], sub_3D_coords[:, 4]]  =  1
    return spt


def coords3D2proj_step(sists_3D_coords, tag, step):
    """Function to reconstruct a 2D projected spot by its tag in  a single time frame"""
    spt             =  np.zeros((sists_3D_coords[-1, (3, 4)]), dtype=np.uint8)
    sub_3D_coords2  =  sists_3D_coords[sists_3D_coords[:, 0] == tag]
    sub_3D_coords   =  sub_3D_coords2[sub_3D_coords2[:, 1] == step]
    spt[sub_3D_coords[:, 3], sub_3D_coords[:, 4]]  =  1
    return spt


def coords3D23D(sists, nuc_tag, step, spts_trck):
    """Function to reconstruct 3D spot by its tag(s)"""
    s_tags   =  (spts_trck == nuc_tag) * sists.sists_trck
    s_tags   =  np.unique(s_tags[s_tags != 0])
    spt      =  np.zeros((sists.sists_3D_coords[-1, (2, 3, 4)]), dtype=np.uint8)
    clr      =  1
    for tag in s_tags:
        sub_3D_coords2  =  sists.sists_3D_coords[sists.sists_3D_coords[:, 0] == tag]
        sub_3D_coords   =  sub_3D_coords2[sub_3D_coords2[:, 1] == step]
        spt[sub_3D_coords[:, 2], sub_3D_coords[:, 3], sub_3D_coords[:, 4]]  =  clr
        clr            +=  1
    return spt


class TranslateSistsCoords:
    """This function writes 3D sister coordinates starting
    from 3D spots coordinates and 2D sisters coordinates.

    Output is a Nx5 matrix (for each pixel: tag, t, z, x, y).
    """
    def __init__(self, analysis_folder):

        sists_coords     =  np.load(analysis_folder + '/sists_coords.npy')[:-1]
        sists_coords     =  sists_coords[np.argsort(sists_coords[:, 1])]
        spots_3D_coords  =  np.load(analysis_folder + '/spots_3D_coords.npy')
        sizess           =  np.append(0, spots_3D_coords[-1])
        spots_3D_coords  =  spots_3D_coords[:-1]

        cpu_ow  =  multiprocessing.cpu_count()
        if sists_coords.shape[0] > 10 * cpu_ow:
            sists_chops  =  np.array_split(sists_coords, cpu_ow)
            job_args     =  list()
            for k in range(cpu_ow):
                t_max       =  sists_chops[k][:, 1].max()
                t_min       =  sists_chops[k][:, 1].min()
                spots_extr  =  (spots_3D_coords[:, 0] >= t_min) * (spots_3D_coords[:, 0] <= t_max)
                job_args.append([sists_chops[k], spots_3D_coords[spots_extr]])

            pool      =  multiprocessing.Pool()
            results   =  pool.map(funct4multip, job_args)
            pool.close()

            sists_3D_coords  =  results[0]

            for k in range(1, cpu_ow):
                sists_3D_coords  =  np.concatenate([sists_3D_coords, results[k]])
            sists_3D_coords  =  np.asarray(sists_3D_coords)
        else:
            sists_3D_coords  =  funct4multip([sists_coords, spots_3D_coords])

        self.sists_3D_coords  =  np.r_[sists_3D_coords, [sizess]]
        # np.save(analysis_folder + '/sists_3D_coords.npy', sists_3D_coords)


class AddRed:
    """This function aggregates a free spots to a green one"""
    def __init__(self, spts_trck, nuc_tag, ref_crd, s_left, sists_trck, sists_3D_coords, sists_ints, green4D, analysis_folder, pos, f_idx):    # sis_trck is sists.sists_trck

        spots_3D_coords      =  np.load(analysis_folder + '/spots_3D_coords.npy')
        sists_trck_fin       =  np.copy(sists_trck)                                                              # I need to separate input from output: not possible to increase np.array size without reassigning
        sists_3D_coords_fin  =  np.copy(sists_3D_coords)
        sists_ints_fin       =  np.copy(sists_ints)
        sizess               =  np.copy(sists_3D_coords_fin[-1])
        sists_3D_coords_fin  =  sists_3D_coords_fin[:-1]
        ref_spts             =  s_left[np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)]               # spots left reference
        tag2give             =  (spts_trck == nuc_tag) * sists_trck
        tag2give             =  np.unique(tag2give[tag2give != 0])[0]                                            # the first activating is the first labeled sisters and takes the red
        rgp2D                =  regionprops_table((s_left == ref_spts) * 1, properties=["coords"])
        new_coords           =  rgp2D["coords"][0].astype(np.uint16)
        sists_trck_fin[f_idx, new_coords[:, 0], new_coords[:, 1]]  =  tag2give                                   # adding the spot with the proper tag
        spt2add              =  np.zeros(s_left.shape)

        ints2add  =  0
        sub1      =  spots_3D_coords[spots_3D_coords[:, 0] == f_idx]                                             # find 4D coordinates of all the pixels: restrict the field with time
        for kk in new_coords:
            sub2  =  sub1[sub1[:, 2] == kk[0]]                                                                   # restrict with x
            sub3  =  sub2[sub2[:, 3] == kk[1]]                                                                   # restrict with y
            for nn in sub3:
                sists_3D_coords_fin  =  np.append(np.append(tag2give, nn.reshape((1, -1))).reshape((1, -1)), sists_3D_coords_fin, axis=0)    # add with proper tag
                ints2add            +=  green4D[nn[0], nn[1], nn[2], nn[3]]
                spt2add[nn[2], nn[3]]  =  1

        sists_ints_fin[ref_crd, 0, f_idx]  +=  ints2add

        self.sists_3D_coords_fin  =  np.r_[sists_3D_coords_fin, [sizess]]
        self.sists_trck_fin       =  sists_trck_fin
        self.sists_ints_fin       =  sists_ints_fin
        self.spt2add              =  spt2add


class AddGreen:
    """This function aggregates a free spots to a green one"""
    def __init__(self, spts_trck, nuc_tag, ref_crd, s_left, sists_trck, sists_3D_coords, sists_ints, green4D, analysis_folder, pos, f_idx):    # sis_trck is sists.sists_trck

        spots_3D_coords      =  np.load(analysis_folder + '/spots_3D_coords.npy')
        sists_trck_fin       =  np.copy(sists_trck)                                                              # I need to separate input from output: not possible to increase np.array size without reassigning
        sists_3D_coords_fin  =  np.copy(sists_3D_coords)
        sizess               =  np.copy(sists_3D_coords_fin[-1])
        sists_3D_coords_fin  =  sists_3D_coords_fin[:-1]
        sists_ints_fin       =  np.copy(sists_ints)
        ref_spts             =  s_left[np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)]               # spots left reference
        # tag2give             =  np.unique((spts_trck == nuc_tag) * sists_trck)
        tag2give             =  (spts_trck == nuc_tag) * sists_trck
        tag2give             =  np.unique(tag2give[tag2give != 0])                                            # the first activating is the first labeled sisters and takes the red
        if tag2give.size > 1:
            tag2give  =  (tag2give[tag2give != 0])[1]                                                         # the first activating is the first labeled sisters and takes the red
        else:
            tag2give  =  sists_3D_coords[:, 0].max() + 1
        rgp2D                =  regionprops_table((s_left == ref_spts) * 1, properties=["coords"])
        new_coords           =  rgp2D["coords"][0].astype(np.uint16)
        sists_trck_fin[f_idx, new_coords[:, 0], new_coords[:, 1]]  =  tag2give                                   # adding the spot with the proper tag
        spt2add              =  np.zeros(s_left.shape)

        ints2add  =  0
        sub1      =  spots_3D_coords[spots_3D_coords[:, 0] == f_idx]                                             # find 4D coordinates of all the pixels: restrict the field with time
        for kk in new_coords:
            sub2  =  sub1[sub1[:, 2] == kk[0]]                                                                   # restrict with x
            sub3  =  sub2[sub2[:, 3] == kk[1]]                                                                   # restrict with y
            for nn in sub3:
                sists_3D_coords_fin    =  np.append(np.append(tag2give, nn.reshape((1, -1))).reshape((1, -1)), sists_3D_coords_fin, axis=0)    # add with proper tag
                ints2add              +=  green4D[nn[0], nn[1], nn[2], nn[3]]
                spt2add[nn[2], nn[3]]  =  1

        sists_ints_fin[ref_crd, 1, f_idx]  +=  ints2add

        self.sists_3D_coords_fin  =  np.r_[sists_3D_coords_fin, [sizess]]
        self.sists_trck_fin       =  sists_trck_fin
        self.sists_ints_fin       =  sists_ints_fin
        self.spt2add              =  spt2add


class AddGray:
    """This function removes the tag from a tracked sister"""
    def __init__(self, ref_crd, sists_trck, sists_3D_coords, sists_ints, green4D, pos, f_idx):    # sis_trck is sists.sists_trck

        sists_trck_fin       =  np.copy(sists_trck)                                                              # I need to separate input from output: not possible to increase np.array size without reassigning
        sists_3D_coords_fin  =  np.copy(sists_3D_coords)
        sizess               =  np.copy(sists_3D_coords_fin[-1])
        sists_3D_coords_fin  =  sists_3D_coords_fin[:-1]
        sists_ints_fin       =  np.copy(sists_ints)

        tt  =  sists_3D_coords_fin[:, 1] == f_idx                                                                # isolate the coordinate of the spot starting from the t-x-y coordinate oof the click
        xx  =  sists_3D_coords_fin[:, 3] == np.round(pos[0]).astype(int)
        yy  =  sists_3D_coords_fin[:, 4] == np.round(pos[1]).astype(int)
        cc  =  tt * xx * yy                                                                                      # index of the coordinate of the spot(s): in case of two chromatide spots, they are both going to be selected

        if np.sum(np.diff(sists_3D_coords_fin[cc, 0]) ** 2) == 0:                                                # check you have only one label (two different object overlapping)
            ii                   =  sists_3D_coords_fin[:, 0] == sists_3D_coords_fin[cc, 0][0]                   # find all the pixels with the selected label
            rr                   =  ii * tt                                                                      # restrict to the ones in the proper time frame
            coords2rm            =  sists_3D_coords_fin[rr]                                                      # extract the coordinate of the spot in the frame
            sists_3D_coords_fin  =  np.delete(sists_3D_coords_fin, rr, axis=0)                                   # remove it from the sisters

            sists_trck_fin[coords2rm[:, 1], coords2rm[:, 3], coords2rm[:, 4]]  =  0                              # remove from the tracked spots
            ints2rm                                                            =  green4D[coords2rm[:, 1], coords2rm[:, 2], coords2rm[:, 3], coords2rm[:, 4]].sum()  # intensity of the spot to remove
            if sists_ints_fin[ref_crd, 0, f_idx] == ints2rm:                                                     # check if the spot to remove is the red and put 0 to the intensity in the list
                sists_ints_fin[ref_crd, 0, f_idx]  =  0
            elif sists_ints_fin[ref_crd, 1, f_idx] == ints2rm:                                                   # check if the spot to remove is the green and put 0 to the intensity in the list
                sists_ints_fin[ref_crd, 1, f_idx]  =  0

        self.sists_3D_coords_fin  =  np.r_[sists_3D_coords_fin, [sizess]]
        self.sists_trck_fin       =  sists_trck_fin
        self.sists_ints_fin       =  sists_ints_fin


class FlipSpots:
    """This function inverts the tags of one or two sister(s)"""
    def __init__(self, ref_crd, nuc_tag, spts_trck, sists_tags, sists_trck, sists_3D_coords, sists_ints, f_idx):    # sis_trck is sists.sists_trck

        sists_trck_fin       =  np.copy(sists_trck)                                # I need to separate input from output: not possible to increase np.array size without reassigning
        sists_3D_coords_fin  =  np.copy(sists_3D_coords)
        sizess               =  np.copy(sists_3D_coords_fin[-1])
        sists_3D_coords_fin  =  sists_3D_coords_fin[:-1]
        sists_ints_fin       =  np.copy(sists_ints)
        tt                   =  sists_3D_coords_fin[:, 1] == f_idx                  # isolate the coordinate of the spot starting from the t-x-y coordinate of the click
        spt2ch_tags          =  None

        if sists_tags.size == 2:                                                    # if there are 2 sisters
            print("sists_tags.size = " + str(sists_tags.size))
            mm0   =  sists_3D_coords_fin[:, 0] == sists_tags[0]                     # check which are the coordinates to change for each of them
            mm1   =  sists_3D_coords_fin[:, 0] == sists_tags[1]
            sis0  =  tt  * mm0
            sis1  =  tt  * mm1

            sists_3D_coords_fin[sis0, 0]  =  sists_tags[1]                          # exchange labels in the coordinate list
            sists_3D_coords_fin[sis1, 0]  =  sists_tags[0]

            sists_trck_fin[sists_3D_coords_fin[sis0, 1], sists_3D_coords_fin[sis0, 3], sists_3D_coords_fin[sis0, 4]]  =  sists_tags[1]    # update tracked sisters
            sists_trck_fin[sists_3D_coords_fin[sis1, 1], sists_3D_coords_fin[sis1, 3], sists_3D_coords_fin[sis1, 4]]  =  sists_tags[0]
            sists_3D_coords_fin  =  np.r_[sists_3D_coords_fin, [sizess]]
            spt2ch_tags          =  coords3D2proj_step(sists_3D_coords_fin, sists_tags[0], f_idx) * 1 + coords3D2proj_step(sists_3D_coords_fin, sists_tags[1], f_idx) * 2    # in case of two chromatids both of them must have the same tag

        elif sists_tags.size == 1:                                                  # if there is only one spot
            # sbs_tags                      =  np.unique((spts_trck == nuc_tag) * sists_trck)[1:]             # check the tags of the other: both sisters in all their frames
            sbs_tags                      =  (spts_trck == nuc_tag) * sists_trck                            # check the tags of the other: both sisters in all their frames
            sbs_tags                      =  np.unique(sbs_tags[sbs_tags != 0])                             # check the tags of the other: both sisters in all their frames
            tag2put                       =  np.delete(sbs_tags, np.where(sbs_tags == sists_tags[0]))       # remove the tag of the present sister
            mm0                           =  sists_3D_coords_fin[:, 0] == sists_tags[0]                     # identify the index in the coordinate matrix
            sis0                          =  tt * mm0
            sists_3D_coords_fin[sis0, 0]  =  tag2put                                                        # change the sister tag
            sists_trck_fin[sists_3D_coords_fin[sis0, 1], sists_3D_coords_fin[sis0, 3], sists_3D_coords_fin[sis0, 4]]  =  tag2put    # update tracked sisters
            sists_3D_coords_fin           =  np.r_[sists_3D_coords_fin, [sizess]]
            spt2ch_tags                   =  coords3D2proj_step(sists_3D_coords_fin, tag2put, f_idx) * 1    # in case of two chromatids both of them must have the same tag

        sists_ints_fin[ref_crd, :2, f_idx]  =  np.flip(sists_ints_fin[ref_crd, :2, f_idx])                  # update the intensity matrix

        self.sists_3D_coords_fin  =  sists_3D_coords_fin
        self.sists_trck_fin       =  sists_trck_fin
        self.sists_ints_fin       =  sists_ints_fin
        self.spt2ch_tags          =  spt2ch_tags


class FlipSpotsFromFrameOn:
    """This function inverts the tags of one or two sister(s)"""
    def __init__(self, ref_crd, nuc_tag, spts_trck, sists_tags, sists_trck, sists_3D_coords, sists_ints, f_idx):    # sis_trck is sists.sists_trck

        sists_trck_fin       =  np.copy(sists_trck)                                # I need to separate input from output: not possible to increase np.array size without reassigning
        sists_3D_coords_fin  =  np.copy(sists_3D_coords)
        sizess               =  np.copy(sists_3D_coords_fin[-1])
        sists_3D_coords_fin  =  sists_3D_coords_fin[:-1]
        sists_ints_fin       =  np.copy(sists_ints)
        tt                   =  sists_3D_coords_fin[:, 1] >= f_idx                  # isolate the coordinate of the spot starting from the t-x-y coordinate of the click
        spt2chg              =  np.zeros_like(spts_trck)
        spt2ch_tags          =  None

        if sists_tags.size == 2:                                                    # if there are 2 sisters
            mm0   =  sists_3D_coords_fin[:, 0] == sists_tags[0]                     # check which are the coordinates to change for each of them
            mm1   =  sists_3D_coords_fin[:, 0] == sists_tags[1]
            sis0  =  tt  * mm0
            sis1  =  tt  * mm1

            sists_3D_coords_fin[sis0, 0]  =  sists_tags[1]                          # exchange labels in the coordinate list
            sists_3D_coords_fin[sis1, 0]  =  sists_tags[0]

            sists_trck_fin[sists_3D_coords_fin[sis0, 1], sists_3D_coords_fin[sis0, 3], sists_3D_coords_fin[sis0, 4]]  =  sists_tags[1]    # update tracked sisters
            sists_trck_fin[sists_3D_coords_fin[sis1, 1], sists_3D_coords_fin[sis1, 3], sists_3D_coords_fin[sis1, 4]]  =  sists_tags[0]
            spt2chg[sists_3D_coords_fin[sis0, 1], sists_3D_coords_fin[sis0, 3], sists_3D_coords_fin[sis0, 4]]         =  sists_tags[1]    # spots cals for output
            spt2chg[sists_3D_coords_fin[sis1, 1], sists_3D_coords_fin[sis1, 3], sists_3D_coords_fin[sis1, 4]]         =  sists_tags[0]
            sists_3D_coords_fin                                                                                       =  np.r_[sists_3D_coords_fin, [sizess]]
            spt2ch_tags                                                                                               =  coords3D2proj(sists_3D_coords_fin, sists_tags[0]) * 1 + coords3D2proj(sists_3D_coords_fin, sists_tags[1]) * 2    # in case of two chromatids both of them must have the same tag
            spt2ch_tags[:f_idx]                                                                                       =  0

        elif sists_tags.size == 1:                                                  # if there is only one spot
            sbs_tags                      =  np.unique((spts_trck == nuc_tag) * sists_trck)[1:]             # check the tags of the other both sisters in all the ther frames
            tag2put                       =  np.delete(sbs_tags, np.where(sbs_tags == sists_tags[0]))       # remove the tag of the present sister
            mm0                           =  sists_3D_coords_fin[:, 0] == sists_tags[0]                     # identify the index in the coordinate matrix
            sis0                          =  tt  * mm0
            sists_3D_coords_fin[sis0, 0]  =  tag2put                                                        # change the sister tag
            sists_trck_fin[sists_3D_coords_fin[sis0, 1], sists_3D_coords_fin[sis0, 3], sists_3D_coords_fin[sis0, 4]]  =  tag2put    # update tracked sisters
            spt2chg[sists_3D_coords_fin[sis0, 1], sists_3D_coords_fin[sis0, 3], sists_3D_coords_fin[sis0, 4]]         =  tag2put    # spots cals for output
            sists_3D_coords_fin           =  np.r_[sists_3D_coords_fin, [sizess]]
            spt2ch_tags                   =  coords3D2proj(sists_3D_coords_fin, tag2put) * 1                                           # in case of two chromatids both of them must have the same tag

        for k in range(f_idx, sists_trck.shape[0]):
            sists_ints_fin[ref_crd, :2, k]  =  np.flip(sists_ints_fin[ref_crd, :2, k])                  # update the intensity matrix

        self.sists_3D_coords_fin  =  sists_3D_coords_fin
        self.sists_trck_fin       =  sists_trck_fin
        self.sists_ints_fin       =  sists_ints_fin
        self.spt2ch_tags          =  spt2ch_tags


class SplitSpot:
    """This function splits a spot in two"""
    def __init__(self, ref_crd, spts_trck, nuc_tag, sists_trck, sists_3D_coords, sists_ints, green4D, pos, f_idx, num_rep, pix_size, pix_size_Z):

        sists_trck_fin       =  np.copy(sists_trck)                                                              # I need to separate input from output: not possible to increase np.array size without reassigning
        sists_3D_coords_fin  =  np.copy(sists_3D_coords)
        sizess               =  np.copy(sists_3D_coords_fin[-1])
        sists_3D_coords_fin  =  sists_3D_coords_fin[:-1]
        sists_ints_fin       =  np.copy(sists_ints)

        tt  =  sists_3D_coords_fin[:, 1] == f_idx                                                                # isolate the coordinate of the spot starting from the t-x-y coordinate oof the click
        xx  =  sists_3D_coords_fin[:, 3] == np.round(pos[0]).astype(int)
        yy  =  sists_3D_coords_fin[:, 4] == np.round(pos[1]).astype(int)
        cc  =  tt * xx * yy                                                                                      # index of the coordinate of the spot(s): in case of two chromatide spots, they are both going to be selected
        ii  =  sists_3D_coords_fin[:, 0] == sists_3D_coords_fin[cc, 0][0]                                        # find all the pixels with the selected label
        rr  =  ii * tt                                                                                           # restrict to the ones in the proper time frame

        coords2split  =  sists_3D_coords_fin[rr]                                                                 # extract the coordinate of the spot in the frame
        z_min, z_max  =  coords2split[:, 2].min() - 1, coords2split[:, 2].max() + 1                              # bounding box to make segmentation faster
        x_min, x_max  =  coords2split[:, 3].min() - 1, coords2split[:, 3].max() + 1
        y_min, y_max  =  coords2split[:, 4].min() - 1, coords2split[:, 4].max() + 1
        prof          =  green4D[f_idx, z_min:z_max, x_min:x_max, y_min:y_max]                                   # raw data in te bounding box
        spt           =  np.zeros(green4D[f_idx].shape, dtype=np.uint8)                                          # b&w spot initialize
        spt[coords2split[:, 2], coords2split[:, 3], coords2split[:, 4]]  =  1                                    # b&w spot
        spt           =  spt[z_min:z_max, x_min:x_max, y_min:y_max]                                              # b&w spot cropped in the bounding box
        # spt_bff       =  HandleSistsCoords3D.GMM_Segmentation(prof, spt).spts_lbls                               # split the spot in two components

        # num_rep    =  8                                                                                          # spot split has some randomness: repeat the segmentation several times and take the best one
        a_s        =  np.zeros((num_rep))                                                                        # store the value for the quality of the fitting
        spts_bffs  =  np.zeros(spt.shape + (num_rep,), dtype=np.uint32)                                          # store segmentation results
        for rep in range(num_rep):
            spts_bffs[:, :, :, rep]  =  GMM_Segmentation(prof, spt).spts_lbls                                    # spots is segmented
            a_s                      =  GaussianScore.GaussianScore(prof * np.sign(spts_bffs[:, :, :, rep])).r_sqr   # quality of the fitting estimated
        vv       =  np.argmax(a_s)                                                                               # choose the highest quality
        spt_bff  =  spts_bffs[:, :, :, vv]                                                                      # select the best segmentation

        # pg.image(spt_bff)
        spt_spl       =  np.zeros(green4D[0].shape, dtype=np.uint16)
        spt_spl[z_min:z_max, x_min:x_max, y_min:y_max]  =  spt_bff                                               # put results into the original size matrix

        rgp_splt        =  regionprops_table(spt_spl, properties=["label", "centroid", "coords"])               # regionprops of the split spots
        ctrs_new        =  np.zeros((2, 3))
        ctrs_new[0, :]  =  [rgp_splt["centroid-0"][0], rgp_splt["centroid-1"][0], rgp_splt["centroid-2"][0]]    # matrix with their centroids coordinates
        ctrs_new[1, :]  =  [rgp_splt["centroid-0"][1], rgp_splt["centroid-1"][1], rgp_splt["centroid-2"][0]]
        coords_new      =  np.zeros((2, np.max([rgp_splt["coords"][0].shape[0], rgp_splt["coords"][1].shape[0]]), 3), dtype=np.uint32)    # 2xNx3 sib tag, number of pixels, z-x-y coordinates. If a sist has less pixels than the other, the matrix wwill keep zeors
        coords_new[0, :rgp_splt["coords"][0].shape[0], :]  =  rgp_splt["coords"][0]                             # fill matrix with coords
        coords_new[1, :rgp_splt["coords"][1].shape[0], :]  =  rgp_splt["coords"][1]

        prev_tags  =  np.unique(sists_trck[f_idx - 1] * (spts_trck[f_idx - 1] == nuc_tag))[1:]                  # sister's tags in the previous frame
        if prev_tags.size == 2:                                                                                 # check if there are two tags in the previous frame
            sist_ctrs        =  np.zeros((2, 3))                                                                # centroids of the sisters
            tt_id            =  sists_3D_coords_fin[:, 1] == f_idx - 1                                          # isolate coordinates with the time frame
            tag_id0          =  sists_3D_coords_fin[:, 0] == prev_tags[0]                                       # isolate coordinates with the first tag
            tot_id0          =  tt_id * tag_id0                                                                 # indexes satisfying both previous conditions
            sist_coords0     =  sists_3D_coords_fin[tot_id0, :]                                                 # extracting submatrix of sists_3D with the previous conditions
            sist_ctrs[0, :]  =  [sist_coords0[:, 2].mean(), sist_coords0[:, 3].mean(), sist_coords0[:, 4].mean()]   # center of mass in previous frame
            tag_id1          =  sists_3D_coords_fin[:, 0] == prev_tags[1]                                       # as before, but with the second tag
            tot_id1          =  tt_id * tag_id1
            sist_coords1     =  sists_3D_coords_fin[tot_id1, :]
            sist_ctrs[1, :]  =  [sist_coords1[:, 2].mean(), sist_coords1[:, 3].mean(), sist_coords1[:, 4].mean()]

            dists_mtx  =  np.zeros((2, 2))                                                                            # initialization of the distance matrix
            for aa in [0, 1]:                                                                                         # calculate all the possible distances (4)
                for bb in [0, 1]:
                    dists_mtx[aa, bb]  =  (pix_size_Z * (ctrs_new[aa, 0] - sist_ctrs[bb, 0])) ** 2 + (pix_size * (ctrs_new[aa, 1] - sist_ctrs[bb, 1])) ** 2 + (pix_size * (ctrs_new[aa, 2] - sist_ctrs[bb, 2])) ** 2

            dists_min  =  np.where(dists_mtx == dists_mtx.min())                                                    # link the first spot between first and second frame
            coords2ch  =  coords_new[dists_min[0][0]]                                                               # extract first coordinates list to change (first segmented spot)
            coords2ch  =  coords2ch[np.all(coords2ch != 0, axis=1)]                                                 # remove zeros rows, are there just to pad
            for ll in coords2ch:
                idx_tg2ch                            =  (sists_3D_coords_fin[:, 1] == f_idx) * (sists_3D_coords_fin[:, 2] == ll[0]) * (sists_3D_coords_fin[:, 3] == ll[1]) * (sists_3D_coords_fin[:, 4] == ll[2])   # isolate the coordinate
                sists_3D_coords_fin[idx_tg2ch, 0]    =  prev_tags[dists_min[1][0]]
                sists_trck_fin[f_idx, ll[1], ll[2]]  =  prev_tags[dists_min[1][0]]

            coords2ch  =  coords_new[1 - dists_min[0][0]]
            coords2ch  =  coords2ch[np.all(coords2ch != 0, axis=1)]
            for ll in coords2ch:
                idx_tg2ch                            =  (sists_3D_coords_fin[:, 1] == f_idx) * (sists_3D_coords_fin[:, 2] == ll[0]) * (sists_3D_coords_fin[:, 3] == ll[1]) * (sists_3D_coords_fin[:, 4] == ll[2])
                sists_3D_coords_fin[idx_tg2ch, 0]    =  prev_tags[1 - dists_min[1][0]]
                sists_trck_fin[f_idx, ll[1], ll[2]]  =  prev_tags[1 - dists_min[1][0]]

            sub_3D_coords0                     =  sists_3D_coords_fin[sists_3D_coords_fin[:, 0] == prev_tags[0]]        # the smallest tagg goes in the first column: isolate tag
            sub_3D_coords01                    =  sub_3D_coords0[sub_3D_coords0[:, 1] == f_idx]                         # isolate frame number
            sists_ints_fin[ref_crd, 0, f_idx]  =  green4D[f_idx, sub_3D_coords01[:, 2], sub_3D_coords01[:, 3], sub_3D_coords01[:, 4]].sum()  # substitute with the intensity of the just created spot

            sub_3D_coords1                     =  sists_3D_coords_fin[sists_3D_coords_fin[:, 0] == prev_tags[1]]        # second tag, second column
            sub_3D_coords11                    =  sub_3D_coords1[sub_3D_coords1[:, 1] == f_idx]
            sists_ints_fin[ref_crd, 1, f_idx]  =  green4D[f_idx, sub_3D_coords11[:, 2], sub_3D_coords11[:, 3], sub_3D_coords11[:, 4]].sum()  # substitute with the intensity of the just created spot

            self.sists_3D_coords_fin  =  np.r_[sists_3D_coords_fin, [sizess]]
            self.sists_ints_fin       =  sists_ints_fin
            self.sists_trck_fin       =  sists_trck_fin

        else:
            next_tags  =  np.unique(sists_trck[f_idx + 1] * (spts_trck[f_idx + 1] == nuc_tag))[1:]
            if next_tags.size == 2:                                                                                     # check if there are two tags in the previous frame
                sist_ctrs        =  np.zeros((2, 3))                                                                    # centroids of the sisters
                tt_id            =  sists_3D_coords_fin[:, 1] == f_idx + 1                                              # isolate coordinates with the time frame
                tag_id0          =  sists_3D_coords_fin[:, 0] == next_tags[0]                                           # isolate coordinates with the first tag
                tot_id0          =  tt_id * tag_id0                                                                     # indexes satisfying both previous conditions
                sist_coords0     =  sists_3D_coords_fin[tot_id0, :]                                                     # extracting submatrix of sists_3D with the previous conditions
                sist_ctrs[0, :]  =  [sist_coords0[:, 2].mean(), sist_coords0[:, 3].mean(), sist_coords0[:, 4].mean()]   # center of mass in previous frame
                tag_id1          =  sists_3D_coords_fin[:, 0] == next_tags[1]                                           # as before, but with the previous tag
                tot_id1          =  tt_id * tag_id1
                sist_coords1     =  sists_3D_coords_fin[tot_id1, :]
                sist_ctrs[1, :]  =  [sist_coords1[:, 2].mean(), sist_coords1[:, 3].mean(), sist_coords1[:, 4].mean()]

                dists_mtx        =  np.zeros((2, 2))                                                                            # initialization of the distance matrix
                for aa in [0, 1]:                                                                                               # calculate all the possible distances (4)
                    for bb in [0, 1]:
                        dists_mtx[aa, bb]  =  (pix_size_Z * (ctrs_new[aa, 0] - sist_ctrs[bb, 0])) ** 2 +  (pix_size * (ctrs_new[aa, 1] - sist_ctrs[bb, 1])) ** 2 + (pix_size * (ctrs_new[aa, 2] - sist_ctrs[bb, 2])) ** 2

                dists_min  =  np.where(dists_mtx == dists_mtx.min())                                                    # link the first spot between first and second frame
                coords2ch  =  coords_new[dists_min[0][0]]                                                               # extract first coordinates list to change (first segmented spot)
                coords2ch  =  coords2ch[np.all(coords2ch != 0, axis=1)]                                                 # remove zeros rows, are there just to pad
                for ll in coords2ch:
                    idx_tg2ch                            =  (sists_3D_coords_fin[:, 1] == f_idx) * (sists_3D_coords_fin[:, 2] == ll[0]) * (sists_3D_coords_fin[:, 3] == ll[1]) * (sists_3D_coords_fin[:, 4] == ll[2])   # isolate the coordinate
                    sists_3D_coords_fin[idx_tg2ch, 0]    =  next_tags[dists_min[1][0]]
                    sists_trck_fin[f_idx, ll[1], ll[2]]  =  next_tags[dists_min[1][0]]

                coords2ch  =  coords_new[1 - dists_min[0][0]]
                coords2ch  =  coords2ch[np.all(coords2ch != 0, axis=1)]
                for ll in coords2ch:
                    idx_tg2ch                            =  (sists_3D_coords_fin[:, 1] == f_idx) * (sists_3D_coords_fin[:, 2] == ll[0]) * (sists_3D_coords_fin[:, 3] == ll[1]) * (sists_3D_coords_fin[:, 4] == ll[2])
                    sists_3D_coords_fin[idx_tg2ch, 0]    =  next_tags[1 - dists_min[1][0]]
                    sists_trck_fin[f_idx, ll[1], ll[2]]  =  next_tags[1 - dists_min[1][0]]

                sub_3D_coords0                     =  sists_3D_coords_fin[sists_3D_coords_fin[:, 0] == next_tags[0]]        # the smallest tagg goes in the first column: isolate tag
                sub_3D_coords01                    =  sub_3D_coords0[sub_3D_coords0[:, 1] == f_idx]                         # isolate frame number
                sists_ints_fin[ref_crd, 0, f_idx]  =  green4D[f_idx, sub_3D_coords01[:, 2], sub_3D_coords01[:, 3], sub_3D_coords01[:, 4]].sum()  # substitute with the intensity of the just created spot

                sub_3D_coords1                     =  sists_3D_coords_fin[sists_3D_coords_fin[:, 0] == next_tags[1]]        # second tag, second column
                sub_3D_coords11                    =  sub_3D_coords1[sub_3D_coords1[:, 1] == f_idx]
                sists_ints_fin[ref_crd, 1, f_idx]  =  green4D[f_idx, sub_3D_coords11[:, 2], sub_3D_coords11[:, 3], sub_3D_coords11[:, 4]].sum()  # substitute with the intensity of the just created spot

                self.sists_3D_coords_fin  =  np.r_[sists_3D_coords_fin, [sizess]]
                self.sists_ints_fin       =  sists_ints_fin
                self.sists_trck_fin       =  sists_trck_fin


class GMM_Segmentation:
    """Performs 3D spots segmentation using the GMM algorithm"""
    """The segmentation is performed using the GaussianMixtureModel: the 3d Intensity distribution is seen as a 3D histogram
        of coordinate. We generate a big number of 3 random coordinate and then we clusterize these points. From the clusterization
        we extend the surfaces of each subspots inside the starting one.
    """

    def __init__(self, prof, spt):

        prof      =  prof.astype(np.float64)
        prof     *=  100. / prof.max()                                      # 3D raw data profile: the box containing the spot
        z, x, y   =  prof.shape                                             # raw data shape
        data      =  np.zeros((int(prof.sum()) + 50, 3))                    # initialize matrix for fictious data
        idx       =  0
        for xx in range(x):
            for yy in range(y):
                for zz in range(z):
                    if int(np.round(prof[zz, xx, yy])) > 0:
                        x_bff                          =  np.random.rand(int(np.round(prof[zz, xx, yy]))) + xx    # generation of 3D random points: the number depends on the intensity profile, the value on the coordinates
                        y_bff                          =  np.random.rand(int(np.round(prof[zz, xx, yy]))) + yy
                        z_bff                          =  np.random.rand(int(np.round(prof[zz, xx, yy]))) + zz
                        data[idx:idx + x_bff.size, :]  =  np.asarray([z_bff, x_bff, y_bff]).T
                        idx                            =  idx + x_bff.size

        data  =  np.delete(data, np.where(data.sum(1) == 0), axis=0)                                         # remove zeors entry: you have them becaus of the apporximations of int
        gm    =  GaussianMixture(n_components=2, random_state=0).fit(data.reshape((-1, 1)))                  # initialize GMM

        mm     =  gm.fit_predict(data)                                                                       # clustering: give a cluster-label to each 3D point
        data0  =  np.squeeze(data[np.where(mm == 0), :])                                                     # split clusters
        data1  =  np.squeeze(data[np.where(mm == 1), :])

        mask0  =  np.zeros(prof.shape)                                                                       # built the 3D histogram for each label
        for uu in range(data0.shape[0]):
            mask0[int(data0[uu, 0]), int(data0[uu, 1]), int(data0[uu, 2])]  +=  1
        mask1  =  np.zeros(prof.shape)
        for vv in range(data1.shape[0]):
            mask1[int(data1[vv, 0]), int(data1[vv, 1]), int(data1[vv, 2])]  +=  1

        mask                  =  np.zeros(prof.shape)                                        # final mask: each pixel of the box is labeled with the label of the most populated histogram
        mask[mask0 > mask1]   =  1                                                           # of the pixel
        mask[mask0 < mask1]   =  2

        self.spts_lbls  =  (mask * np.sign(spt)).astype(np.uint32)               # mask identify regions of the box: multipling by the spt we segment the initial spot
        self.mask       =  mask
