"""This function tracks separately two spots.

Input is the directory with the bulk analysis.
Starting points are spots associated to a single
nucleus and a distance threshold.
"""

import multiprocessing
import numpy as np
from skimage.measure import regionprops_table, label
from skimage.segmentation import expand_labels
from sklearn.cluster import KMeans
from openpyxl import load_workbook
from scipy.ndimage import binary_erosion, binary_dilation
from PyQt5 import QtWidgets
import pyqtgraph as pg

import HandleSistsCoords3D


# import SisterSplit6
# spots_3D_coords = np.load(analysis_folder + '/spots_3D_coords.npy')
# spts_trk  =  np.load(analysis_folder + '/spots_tracked.npy')
# pix_size, pix_size_Z = raw_data.pix_size, raw_data.pix_size_Z
# import SisterSplit6


# import MultiLoadCzi5D
# fnames = ['/home/atrullo/Desktop/Louise2Spots/Kr_E3_09242021/TestAnalysis3_short/Kr_E3_09242021_i_Out.czi']
# raw_data = MultiLoadCzi5D.MultiLoadCzi5D([fnames, np.array([1, 0])])
# green4d = raw_data.green4D
# analysis_folder  =  '/home/atrullo/Desktop/Louise2Spots/Kr_E3_09242021/TestAnalysis3_short'
# dist_thr = 15


def distance_mtx(info_curnt, info_next, pix_size, pix_size_Z):
    """Function to calculate all the possible distances between 2 groups of points"""
    dists_mtx  =  list()                                                                             # initialization of the final result
    for jj in range(info_curnt.shape[0]):                                                            # since pixel size is different along x-y or z, we need to involve pixel size in the Pitagora theorem
        for kk in range(info_next.shape[0]):
            dists_mtx.append([np.sqrt(((info_curnt[jj, 2] - info_next[kk, 2]) * pix_size_Z) ** 2 + ((info_curnt[jj, 3] - info_next[kk, 3]) * pix_size) ** 2 + ((info_curnt[jj, 4] - info_next[kk, 4]) * pix_size) ** 2)] + info_curnt[jj, :].tolist() + info_next[kk, :].tolist())
    dists_mtx  =  np.asarray(dists_mtx)                  # make a matrix, easier to handle
    return dists_mtx


def coords3D23D_all(spots_3D_coords, step):
    """Function to reconstruct 3D spots in a single frame from coordinate mtx"""
    spt            =  np.zeros(spots_3D_coords[-1, (1, 2, 3)], dtype=np.uint8)
    sub_3D_coords  =  spots_3D_coords[:-1]
    sub_3D_coords  =  sub_3D_coords[sub_3D_coords[:, 0] == step]
    spt[sub_3D_coords[:, 1], sub_3D_coords[:, 2], sub_3D_coords[:, 3]]  =  1
    return spt


def coords3D2proj_r(sists_3D_coords, spt_shape, tag):
    """Function to reconstruct a 2D projected spot by its tag"""
    spt            =  np.zeros(spt_shape)
    sub_3D_coords  =  sists_3D_coords[sists_3D_coords[:, 0] == tag]
    spt[sub_3D_coords[:, 1], sub_3D_coords[:, 3], sub_3D_coords[:, 4]]  =  1
    return spt


# def coords2trck_sists(mini_job):
#     sists_trck_tags_r  =  mini_job[0]
#     sists_3D_coords_r  =  mini_job[1]
#     sists_trck_r       =  np.zeros(sists_3D_coords_r[-1, (1, 3, 4)], dtype=np.uint8)
#     for tag in sists_trck_tags_r:
#         sists_trck_r  +=  np.uint16(tag) * HandleSistsCoords3D.coords3D2proj(sists_3D_coords_r, tag) * (1 - np.sign(sists_trck_r))  # reconstruct sists_trck in 2D from coordinates
#     return sists_trck_r


class SisterSplit:
    """Main function"""
    # def __init__(self, dist_thr, spts_trk, green4d, spots_3D_coords, pix_size, pix_size_Z):
    def __init__(self, analysis_folder, dist_thr, green4d):

        spts_trk          =  np.load(analysis_folder + '/spots_tracked.npy')              # load nuclei-tracked spots
        spots_3D_coords   =  np.load(analysis_folder + '/spots_3D_coords.npy')            # load coordinate of all the pixels of the points
        wb                =  load_workbook(analysis_folder + '/journal.xlsx')
        pix_size          =  wb.active["B16"].value
        pix_size_Z        =  wb.active["B17"].value
        sists_trck        =  np.zeros_like(spts_trk)                                      # initialize the output matrix
        sizess            =  np.append(0, spots_3D_coords[-1])
        sib_idxs          =  np.unique(spts_trk[spts_trk != 0])                           # tags of the nuclei-tracked spots
        sists_ints_dists  =  np.zeros((sib_idxs.size, 3, spts_trk.shape[0]))
        sists_3D_coords   =  np.zeros((0, 5), dtype=np.uint32)

        cpu_ow       =  np.min([multiprocessing.cpu_count(), 16])
        info_traces  =  []                                                                         # check the region explored by the sister spots to reduce the region to work on
        for k in sib_idxs:                                                                         # for each label
            calc      =  np.sign((spts_trk == k).sum(0)) * 1                                       # sum the spots in time
            calc_rgp  =  regionprops_table(calc, properties=["bbox", "centroid"])                  # properties of the region explored
            info_traces.append([k, calc_rgp["centroid-0"][0], calc_rgp["centroid-1"][0], calc_rgp["bbox-0"][0], calc_rgp["bbox-1"][0], calc_rgp["bbox-2"][0], calc_rgp["bbox-3"][0]])
        info_traces  =  np.asarray(info_traces)

        kmeans  =  KMeans(n_clusters=np.min([cpu_ow, info_traces.shape[0]]), random_state=0).fit(info_traces[:, 1:3])      # classifiction of the centroids of the explored region: stack crops will be smaller. One crop for each cpu for multiprocessing

        jobs_args  =  []                                                                 # preparing lists of input ofr the pool
        bb_clstrs  =  []
        for kk in range(cpu_ow):                                                         # for each classifiction
            sib_clstrs    =  np.where(kmeans.labels_ == kk)[0]                           # select the indexes for each cluster
            trcs          =  np.zeros_like(spts_trk)                                     # take the traces
            x_min, y_min  =  spots_3D_coords[-1, 2], spots_3D_coords[-1, 3]              # properly initialize the boundary box
            x_max, y_max  =  0, 0
            for sib_clstr in sib_clstrs:
                trcs  +=  spts_trk == info_traces[sib_clstr, 0]                          # isolate the traces
                x_min  =  int(min(x_min, info_traces[sib_clstr, 3]))                     # search for the corner of the bounding box of the cluster
                x_max  =  int(max(x_max, info_traces[sib_clstr, 5]))
                y_min  =  int(min(y_min, info_traces[sib_clstr, 4]))
                y_max  =  int(max(y_max, info_traces[sib_clstr, 6]))

            bb_clstrs.append([x_min, y_min, x_max, y_max])                               # store the bounding box corners
            spts_trk_clst  =  (trcs * spts_trk)[:, x_min:x_max, y_min:y_max]             # project the selected traces on the intensity and volume matrix
            green4d2send   =  green4d[:, :, x_min:x_max, y_min:y_max]
            jobs_args.append([spts_trk_clst, green4d2send, spots_3D_coords, [x_min, x_max, y_min, y_max], pix_size, pix_size_Z, dist_thr])

        pool     =  multiprocessing.Pool()
        # results  =  pool.map(SisterSplit7.SistersSplitUtility, jobs_args)
        results  =  pool.map(SistersSplitUtility, jobs_args)
        pool.close()

        bb_clstrs      =  np.asarray(bb_clstrs)
        sists_tag_ref  =  0
        for jj2 in range(cpu_ow):                                                       # after the pool reorganize results
            coords2conc         =  np.copy(results[jj2].sists_3D_coords)
            coords2conc[:, 3]  +=  bb_clstrs[jj2, 0]                                    # take into account the change of coordinates due to the crop
            coords2conc[:, 4]  +=  bb_clstrs[jj2, 1]
            coords2conc[:, 0]  +=  sists_tag_ref                                        # tags are updated, in each process tags are from 1 on, when you merge you sum the tags to not repeat them
            sists_3D_coords     =  np.concatenate([sists_3D_coords, coords2conc])       # concatenate results
            sists_tag_ref      +=  2 * (np.unique(jobs_args[jj2][0])[1:]).size          # update tag reference

        sists_3D_coords  =  np.r_[sists_3D_coords, [sizess]]
        sists_trck       =  np.zeros_like(sists_trck)
        sists_trck_tags  =  np.unique(sists_3D_coords[:, 0])[1:]
        for tag in sists_trck_tags:
            sists_trck  +=  np.uint16(tag) * HandleSistsCoords3D.coords3D2proj(sists_3D_coords, tag) * (1 - np.sign(sists_trck))   # reconstruct sists_trck in 2D from coordinates

        for jj3 in range(cpu_ow):
            sib_idxs_bff  =  jobs_args[jj3][0]                                          # organize the matrix of intensity and distances
            sib_idxs_bff  =  np.unique(sib_idxs_bff[sib_idxs_bff != 0])                 # organize the matrix of intensity and distances
            for cnt, sib_idx_bff in enumerate(sib_idxs_bff):                            # keep the order and the index of the sib_idxs
                sib_pos                    =  np.where(sib_idxs == sib_idx_bff)[0]
                sists_ints_dists[sib_pos]  =  results[jj3].sists_ints[cnt]

        self.sists_3D_coords  =  sists_3D_coords
        self.sists_trck       =  sists_trck
        self.sists_ints       =  sists_ints_dists


class SistersSplitUtility:
    """Track and split sisters for multiprocessing"""
    def __init__(self, multip_args):

        spts_trk_r                  =  multip_args[0]
        green4d_r                   =  multip_args[1]
        spots_3D_coords             =  multip_args[2]
        x_min, x_max, y_min, y_max  =  multip_args[3]
        pix_size                    =  multip_args[4]
        pix_size_Z                  =  multip_args[5]
        dist_thr                    =  multip_args[6]

        sib_idxs_r        =  np.unique(spts_trk_r[spts_trk_r != 0])                                                     # list of the spots indexes
        print(sib_idxs_r)
        sists_ints_dists  =  np.zeros((sib_idxs_r.size, 3, spts_trk_r.shape[0]))                                        # initialize the matrix to store intensity and distances
        t_steps           =  spts_trk_r.shape[0]                                                                        # number of time steps
        sists_3D_coords   =  np.zeros((0, 5), dtype=np.uint32)                                                          # initialize matrix for the coordinate: for each pixel there is [label, t, z, x, y]

        for ccc, sib_idx in enumerate(sib_idxs_r):                                                                      # for each tag
            sis_tg1  =  2 * ccc + 1
            sis_tg2  =  2 * ccc + 2
            sibs     =  (spts_trk_r == sib_idx) * 1                                                                     # isolate their common trace
            t_start  =  np.where(np.sum(sibs, axis=(1, 2)) != 0)[0]                                                     # search the frame n which the spot is present

            if t_start.size < 5:                                                                                        # if the spot is active for only 5 frames it is not considered
                t_start  =  t_steps
            else:
                try:                                                                                                        # you can have nuclei that are activated just for few frames far in time: control here
                    while np.sum(np.diff(t_start)[0:3]) > 4:                                                                          # check if it is a single frame appearence (0,1,0)
                        t_start  =  t_start[1:]                                                                             # if it is the case, check the following
                    t_start  =  t_start[0]                                                                                  # take the first frame of the activation
                except IndexError:                                                                                          # when the nucleus activates for some frame far in time each other, just skip the tracking setting the starting point egual to the final point
                    t_start  =  t_steps

            info_sibs      =  []                                                                                        # collect in one shot all the info for tracking
            coords_groups  =  []                                                                                        # coords of pixels must be stored apart in a list
            group_cnt      =  0                                                                                         # list index for the coords

            for ttx in range(t_start, t_steps):
                spfr     =  coords3D23D_all(spots_3D_coords, ttx)                                                       # 3D reconstruction of the spots in a single frame
                # spfr     =  SisterSplit7.coords3D23D_all(spots_3D_coords, ttx)
                spfr     =  spfr[:, x_min:x_max, y_min:y_max]                                                           # cut in x and y following the bounding box previously calculated
                spts3d   =  label(spfr * sibs[ttx])                                                     # select and label in 3D spots for the called nucleus
                if spts3d.sum() > 0:                                                                                    # check the spots is there - it can desappear for a frame or more or still to appear
                    rgp_bff  =  regionprops_table(spts3d, properties=["label", "area", "coords"])                       # regionrprops to check the size of the connected components
                    idxs2rm  =  np.where(rgp_bff["area"] < 2)[0]                                                        # find components with an area smaller than 7 (pretty arbitrary)
                    if idxs2rm.size > 0:
                        for l in idxs2rm:                                                                               # remove those components using coords
                            spts3d[rgp_bff["coords"][l][:, 0], rgp_bff["coords"][l][:, 1], rgp_bff["coords"][l][:, 2]]  =  0
                        # spts3d   =  label(spts3d, connectivity=1)                                                       # relabel
                        spts3d   =  label(spts3d)                                                       # relabel
                        rgp_bff  =  regionprops_table(spts3d, properties=["label", "area", "coords"])                   # recalculate region properties

                    if spts3d.sum() > 0:                                                                                # check the spots is there - it can desappear for a frame or more or still to appear
                        if rgp_bff["area"].min() < rgp_bff["area"].max() / 5:
                            if len(rgp_bff["area"]) == 2:
                                spts3d  =  label(spfr * sibs[ttx])
                            else:
                                it_idx   =  1
                                sml_dil  =  binary_dilation(spts3d == rgp_bff["label"][np.argmin(rgp_bff["area"])], iterations=it_idx) ^ (spts3d == rgp_bff["label"][np.argmin(rgp_bff["area"])])
                                while np.sum(sml_dil * spts3d) == 0:
                                    it_idx  +=  1
                                    sml_dil  =  binary_dilation(spts3d == rgp_bff["label"][np.argmin(rgp_bff["area"])], iterations=it_idx) ^ (spts3d == rgp_bff["label"][np.argmin(rgp_bff["area"])])

                                shll  =  spts3d * sml_dil
                                shll  =  shll[shll != 0]
                                idd   =  np.median(shll)
                                if int(idd) - idd != 0:
                                    idd  =  shll.min()
                                spts3d[spts3d == rgp_bff["label"][np.argmin(rgp_bff["area"])]]  =  idd

                        rgp3d   =  regionprops_table(spts3d, green4d_r[ttx], properties=["label", "area", "centroid", "coords", "intensity_image"])
                        for cnt, k in enumerate(rgp3d["label"]):
                            info_sibs.append([k, ttx, rgp3d["centroid-0"][cnt], rgp3d["centroid-1"][cnt], rgp3d["centroid-2"][cnt], rgp3d["area"][cnt], np.sum(rgp3d["intensity_image"][cnt]), group_cnt])
                            coords_groups.append(rgp3d["coords"][cnt])
                            group_cnt  +=  1

            info_sibs   =  np.asarray(info_sibs)
            if info_sibs.shape[0] == 0:
                t_start  =  t_steps - 1
            else:
                info_start  =  info_sibs[info_sibs[:, 1] == t_start]                                                    # property for the beginning, out of for cycle
                while info_start.size == 0:
                    t_start     +=  1
                    info_start   =  info_sibs[info_sibs[:, 1] == t_start]
                if info_start.shape[0] == 1:                                                                            # check how many there are in the first frame of appearence and organize the output
                    lbl_tm                             =  np.ones_like(coords_groups[int(info_start[0, 7])][:, :2])
                    lbl_tm[:, 1]                       =  t_start
                    lbl_tm[:, 0]                       =  sis_tg1
                    sists_3D_coords                    =  np.concatenate([sists_3D_coords, np.concatenate([lbl_tm, coords_groups[int(info_start[0, 7])]], axis=1)], axis=0)
                    info_curnt                         =  info_start
                    info_curnt[0, 0]                   =  sis_tg1

                elif info_start.shape[0] >= 2:
                    info_start                         =  info_start[info_start[:, 6].argsort()[::-1]]                                     # in case we have more than 2, the two biggest will be selected
                    lbl_tm                             =  np.ones_like(coords_groups[int(info_start[0, 7])][:, :2])
                    lbl_tm[:, 1]                       =  t_start
                    lbl_tm[:, 0]                       =  sis_tg1
                    sists_3D_coords                    =  np.concatenate([sists_3D_coords, np.concatenate([lbl_tm, coords_groups[int(info_start[0, 7])]], axis=1)], axis=0)
                    lbl_tm                             =  np.ones_like(coords_groups[int(info_start[1, 7])][:, :2])
                    lbl_tm[:, 1]                       =  t_start
                    lbl_tm[:, 0]                       =  sis_tg2
                    sists_3D_coords                    =  np.concatenate([sists_3D_coords, np.concatenate([lbl_tm, coords_groups[int(info_start[1, 7])]], axis=1)], axis=0)
                    info_curnt                         =  info_start[:2]
                    info_curnt[0, 0]                   =  sis_tg1
                    info_curnt[1, 0]                   =  sis_tg2
                    sists_ints_dists[ccc, 2, t_start]  =  distance_mtx(info_start[:1, :], info_start[1:2, :], pix_size, pix_size_Z)[0][0]
                    # sists_ints_dists[ccc, 2, t_start]  =  SisterSplit7.distance_mtx(info_start[:1, :], info_start[1:2, :], pix_size, pix_size_Z)[0][0]

            for tt in range(t_start + 1, t_steps):                                                                              # for each frame (OSS: the frame called next is the one of the tt, current for the cycle)
                info_next  =  info_sibs[info_sibs[:, 1] == tt]                                                                  # info of the following frame are taken from the big list previously done
                dists      =  distance_mtx(info_curnt, info_next, pix_size, pix_size_Z)                                         # all the possible distances between the detected points
                # dists      =  SisterSplit7.distance_mtx(info_curnt, info_next, pix_size, pix_size_Z)                            # all the possible distances between the detected points

                if info_curnt.shape[0] == 1 and info_next.shape[0] == 1 and dists[0, 0] <= dist_thr:                            # if you have a single spot in the current and the next frame, just check the distance
                    lbl_tm                        =  np.ones_like(coords_groups[int(info_next[0, 7])][:, :2])
                    lbl_tm[:, 1]                  =  tt
                    new_lbl                       =  info_curnt[0, 0]
                    lbl_tm[:, 0]                  =  new_lbl
                    sists_3D_coords               =  np.concatenate([sists_3D_coords, np.concatenate([lbl_tm, coords_groups[int(info_next[0, 7])]], axis=1)], axis=0)
                    info_curnt                    =  info_next
                    info_curnt[0, 0]              =  new_lbl
                    sists_ints_dists[ccc, 0, tt]  =  info_start[0, 6]

                elif info_curnt.shape[0] == 2 and info_next.shape[0] >= 2 and dists[:, 0].min() <= dist_thr:                  # if there are two spots both in the current and in the following frames
                    info_curnt        =  np.zeros((2, 8))
                    d_min_idx         =  np.argmin(dists[:, 0])
                    lbl_tm            =  np.ones_like(coords_groups[int(dists[d_min_idx, 16])][:, :2])
                    lbl_tm[:, 1]      =  tt
                    lbl_tm[:, 0]      =  dists[d_min_idx, 1]
                    info_curnt[0, :]  =  dists[d_min_idx, 9:18]
                    info_curnt[0, 0]  =  dists[d_min_idx, 1]
                    ctrs1             =  dists[d_min_idx:d_min_idx + 1, 1:6]
                    sists_3D_coords   =  np.concatenate([sists_3D_coords, np.concatenate([lbl_tm, coords_groups[int(dists[d_min_idx, 16])]], axis=1)], axis=0)
                    l2rm1, l2rm2      =  dists[d_min_idx, 1], dists[d_min_idx, 9]
                    dists             =  np.delete(dists, dists[:, 1] == l2rm1, axis=0)
                    dists             =  np.delete(dists, dists[:, 9] == l2rm2, axis=0)
                    if dists[:, 0].min() <= dist_thr:
                        d_min_idx                     =  np.argmin(dists[:, 0])
                        lbl_tm                        =  np.ones_like(coords_groups[int(dists[d_min_idx, 16])][:, :2])
                        lbl_tm[:, 1]                  =  tt
                        lbl_tm[:, 0]                  =  dists[d_min_idx, 1]
                        ctrs2                         =  dists[d_min_idx:d_min_idx + 1, 1:6]
                        sists_3D_coords               =  np.concatenate([sists_3D_coords, np.concatenate([lbl_tm, coords_groups[int(dists[d_min_idx, 16])]], axis=1)], axis=0)
                        info_curnt[1, :]              =  dists[d_min_idx, 9:18]
                        info_curnt[1, 0]              =  dists[d_min_idx, 1]
                        # sists_ints_dists[ccc, 2, tt]  =  SisterSplit7.distance_mtx(ctrs1, ctrs2, pix_size, pix_size_Z)[0][0]
                        sists_ints_dists[ccc, 2, tt]  =  distance_mtx(ctrs1, ctrs2, pix_size, pix_size_Z)[0][0]

                elif info_curnt.shape[0] == 1 and info_next.shape[0] >= 2 and dists[:, 0].min() <= dist_thr:                     # if there is 1 spot in the current and 2 spots in the following, a spot is appearing, so add with the other tag
                    info_curnt        =  np.zeros((2, 8))
                    d_min_idx         =  np.argmin(dists[:, 0])
                    lbl_tm            =  np.ones_like(coords_groups[int(dists[d_min_idx, 16])][:, :2])
                    lbl_tm[:, 1]      =  tt
                    lbl_tm[:, 0]      =  dists[d_min_idx, 1]
                    info_curnt[0, :]  =  dists[d_min_idx, 9:18]
                    info_curnt[0, 0]  =  dists[d_min_idx, 1]
                    sists_3D_coords   =  np.concatenate([sists_3D_coords, np.concatenate([lbl_tm, coords_groups[int(dists[d_min_idx, 16])]], axis=1)], axis=0)
                    dists             =  np.delete(dists, d_min_idx, axis=0)
                    d_min_idx         =  np.argmin(dists[:, 0])
                    info_curnt[1, :]  =  dists[d_min_idx, 9:18]
                    if info_curnt[0, 0]  ==  sis_tg1:
                        info_curnt[1, 0]  =  sis_tg2
                    elif info_curnt[0, 0]  ==  sis_tg2:
                        info_curnt[1, 0]  =  sis_tg1

                    d_min_idx                     =  np.argmin(dists[:, 0])
                    lbl_tm                        =  np.ones_like(coords_groups[int(dists[d_min_idx, 16])][:, :2])
                    lbl_tm[:, 1]                  =  tt
                    lbl_tm[:, 0]                  =  info_curnt[1, 0]   # dists[d_min_idx, 1]
                    # ctrs2                         =  dists[d_min_idx:d_min_idx + 1, 1:6]
                    sists_3D_coords               =  np.concatenate([sists_3D_coords, np.concatenate([lbl_tm, coords_groups[int(dists[d_min_idx, 16])]], axis=1)], axis=0)

                elif info_curnt.shape[0] == 2 and info_next.shape[0] == 1 and dists[:, 0].min() <= dist_thr:                     # if there is 1 spot in the current and 2 spots in the following, a spot is appearing, so add with the other tag
                    info_curnt                    =  np.zeros((1, 8))
                    d_min_idx                     =  np.argmin(dists[:, 0])
                    lbl_tm                        =  np.ones_like(coords_groups[int(dists[d_min_idx, 16])][:, :2])
                    lbl_tm[:, 1]                  =  tt
                    lbl_tm[:, 0]                  =  dists[d_min_idx, 1]
                    info_curnt[0, :]              =  dists[d_min_idx, 9:18]
                    info_curnt[0, 0]              =  dists[d_min_idx, 1]
                    sists_3D_coords               =  np.concatenate([sists_3D_coords, np.concatenate([lbl_tm, coords_groups[int(dists[d_min_idx, 16])]], axis=1)], axis=0)
                    sists_ints_dists[ccc, 2, tt]  =  distance_mtx(dists[d_min_idx:d_min_idx + 1, 1:6], dists[1 - d_min_idx:2 - d_min_idx, 1:6], pix_size, pix_size_Z)[0][0]
                    # sists_ints_dists[ccc, 2, tt]  =  SisterSplit7.distance_mtx(dists[d_min_idx:d_min_idx + 1, 1:6], dists[1 - d_min_idx:2 - d_min_idx, 1:6], pix_size, pix_size_Z)[0][0]

            spts_lft   =  sibs * (1 - coords3D2proj_r(sists_3D_coords, green4d_r[:, 0].shape, sis_tg1)) * (1 - coords3D2proj_r(sists_3D_coords, green4d_r[:, 0].shape, sis_tg2))
            # spts_lft   =  sibs * (1 - SisterSplit7.coords3D2proj_r(sists_3D_coords, green4d_r[:, 0].shape, sis_tg1)) * (1 - SisterSplit7.coords3D2proj_r(sists_3D_coords, green4d_r[:, 0].shape, sis_tg2))
            if np.sum(spts_lft) != 0:
                nozr_frms  =  np.where(np.sum(spts_lft, axis=(1, 2)))[0]
                for nozr_frm in nozr_frms:
                    spfr     =  coords3D23D_all(spots_3D_coords, nozr_frm)
                    # spfr     =  SisterSplit7.coords3D23D_all(spots_3D_coords, nozr_frm)
                    spfr     =  spfr[:, x_min:x_max, y_min:y_max]                                                                               # cut in x and y following the bounding box previously calculated
                    spts3d   =  label(spfr * spts_lft[nozr_frm], connectivity=1)                                                                # select and label in 3D spots for the called nucleus

                    c_sis_tag1_prev  =  sists_3D_coords[sists_3D_coords[:, 1] == nozr_frm - 1]                                                  # select the coords of the spot with sis_tag1 in previouse frame
                    c_sis_tag1_prev  =  c_sis_tag1_prev[c_sis_tag1_prev[:, 0] == sis_tg1]
                    ints_tag1_prev   =  np.sum(green4d_r[nozr_frm - 1, c_sis_tag1_prev[:, 2], c_sis_tag1_prev[:, 3], c_sis_tag1_prev[:, 4]])    # measure its intensity

                    c_sis_tag2_prev  =  sists_3D_coords[sists_3D_coords[:, 1] == nozr_frm - 1]                                                  # same but tag 2
                    c_sis_tag2_prev  =  c_sis_tag2_prev[c_sis_tag2_prev[:, 0] == sis_tg2]
                    ints_tag2_prev   =  np.sum(green4d_r[nozr_frm - 1, c_sis_tag2_prev[:, 2], c_sis_tag2_prev[:, 3], c_sis_tag2_prev[:, 4]])

                    c_sis_tag1_current  =  sists_3D_coords[sists_3D_coords[:, 1] == nozr_frm]                                                       # select the coords of the spot with sis_tag1 in current frame
                    c_sis_tag1_current  =  c_sis_tag1_current[c_sis_tag1_current[:, 0] == sis_tg1]
                    ints_tag1_current   =  np.sum(green4d_r[nozr_frm, c_sis_tag1_current[:, 2], c_sis_tag1_current[:, 3], c_sis_tag1_current[:, 4]]) # measure its intensity
                    ctrs_tg1_current    =  c_sis_tag1_current[:, 2].mean(), c_sis_tag1_current[:, 3].mean(), c_sis_tag1_current[:, 4].mean()         # centroid coordinates

                    c_sis_tag2_current  =  sists_3D_coords[sists_3D_coords[:, 1] == nozr_frm]                                                       # same, but tag 2
                    c_sis_tag2_current  =  c_sis_tag2_current[c_sis_tag2_current[:, 0] == sis_tg2]
                    ints_tag2_current   =  np.sum(green4d_r[nozr_frm, c_sis_tag2_current[:, 2], c_sis_tag2_current[:, 3], c_sis_tag2_current[:, 4]])
                    ctrs_tg2_current    =  c_sis_tag2_current[:, 2].mean(), c_sis_tag2_current[:, 3].mean(), c_sis_tag2_current[:, 4].mean()

                    idxs2wrk  =  np.unique(spts3d[spts3d != 0])
                    for mm in idxs2wrk:
                        spts3d_lbl      =  (spts3d == mm) * 1
                        rgp_spts3d_lbl  =  regionprops_table(spts3d_lbl, properties=["label", "centroid", "coords"])
                        dist_lft1       =  np.sqrt(((rgp_spts3d_lbl["centroid-0"][0] - ctrs_tg1_current[0]) * pix_size_Z) ** 2 + ((rgp_spts3d_lbl["centroid-1"][0] - ctrs_tg1_current[1]) * pix_size) ** 2 + ((rgp_spts3d_lbl["centroid-2"][0] - ctrs_tg1_current[2]) * pix_size) ** 2)
                        dist_lft2       =  np.sqrt(((rgp_spts3d_lbl["centroid-0"][0] - ctrs_tg2_current[0]) * pix_size_Z) ** 2 + ((rgp_spts3d_lbl["centroid-1"][0] - ctrs_tg2_current[1]) * pix_size) ** 2 + ((rgp_spts3d_lbl["centroid-2"][0] - ctrs_tg2_current[2]) * pix_size) ** 2)

                        if dist_lft1 < dist_lft2:
                            if ints_tag1_current < .8 * ints_tag1_prev:
                                lbl_tm           =  np.ones_like(rgp_spts3d_lbl["coords"][0][:, :2])
                                lbl_tm[:, 1]     =  nozr_frm
                                lbl_tm[:, 0]     =  sis_tg1
                                sists_3D_coords  =  np.concatenate([sists_3D_coords, np.concatenate([lbl_tm, rgp_spts3d_lbl["coords"][0]], axis=1)], axis=0)
                        elif dist_lft1 >= dist_lft2:
                            if ints_tag2_current < .8 * ints_tag2_prev:
                                lbl_tm           =  np.ones_like(rgp_spts3d_lbl["coords"][0][:, :2])
                                lbl_tm[:, 1]     =  nozr_frm
                                lbl_tm[:, 0]     =  sis_tg2
                                sists_3D_coords  =  np.concatenate([sists_3D_coords, np.concatenate([lbl_tm, rgp_spts3d_lbl["coords"][0]], axis=1)], axis=0)

        for c, sib_idx in enumerate(sib_idxs_r):                                                                        # for each tag
            sis_tg1  =  2 * c + 1
            sis_tg2  =  2 * c + 2
            for ss in range(t_steps):
                bff_coords                  =  sists_3D_coords[sists_3D_coords[:, 1] == ss]
                bff_coords1                 =  bff_coords[bff_coords[:, 0] == sis_tg1]
                sists_ints_dists[c, 0, ss]  =  np.sum(green4d_r[ss, bff_coords1[:, 2], bff_coords1[:, 3], bff_coords1[:, 4]])
                bff_coords2                 =  bff_coords[bff_coords[:, 0] == sis_tg2]
                sists_ints_dists[c, 1, ss]  =  np.sum(green4d_r[ss, bff_coords2[:, 2], bff_coords2[:, 3], bff_coords2[:, 4]])

        self.sists_ints        =  sists_ints_dists
        self.sists_3D_coords   =  sists_3D_coords


class SisterSplitSingleNucleus:
    """This class splits and tracks only the sisters coming from a single nucleus"""
    def __init__(self, multip_args):

        gg                        =  SistersSplitUtility(multip_args)
        gg.sists_3D_coords[:, 3] +=  multip_args[3][0]                       # take into account the change of coordinates due to the crop
        gg.sists_3D_coords[:, 4] +=  multip_args[3][2]
        # pg.image(coords3D2proj_r(gg.sists_3D_coords, [multip_args[2][-1, 0], multip_args[2][-1, 2], multip_args[2][-1, 3]], 1))
        # pg.image(coords3D2proj_r(gg.sists_3D_coords, [multip_args[2][-1, 0], multip_args[2][-1, 2], multip_args[2][-1, 3]], 2))
        sists_mtx                 =  coords3D2proj_r(gg.sists_3D_coords, [multip_args[2][-1, 0], multip_args[2][-1, 2], multip_args[2][-1, 3]], 1) + 2 * coords3D2proj_r(gg.sists_3D_coords, [multip_args[2][-1, 0], multip_args[2][-1, 2], multip_args[2][-1, 3]], 2)
        min_cmap                  =  np.zeros((3, 3), dtype=np.uint8)
        min_cmap[0, :]            =  np.array([0, 0, 0])
        min_cmap[1, :]            =  np.array([255, 0, 0])
        min_cmap[2, :]            =  np.array([0, 255, 0])
        minicmap                  =  pg.ColorMap(np.linspace(0, 1, 3), color=min_cmap)
        w                         =  pg.image(sists_mtx)
        w.setLevels(0, 2)
        w.setHistogramRange(0, 2)
        w.setColorMap(minicmap)


class ProgressBar(QtWidgets.QWidget):
    """Simple progress bar widget"""
    def __init__(self, parent=None, total1=20):
        super().__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar  =  QtWidgets.QProgressBar()
        self.progressbar.setMinimum(1)
        self.progressbar.setMaximum(total1)

        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)

    def update_progressbar(self, val1):
        """Progress bar updater"""
        self.progressbar.setValue(val1)
        QtWidgets.qApp.processEvents()
