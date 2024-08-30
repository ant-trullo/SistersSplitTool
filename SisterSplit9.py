"""Function to split and track alleles inside nuclei.

Input is the folder with the analysis
"""

import multiprocessing
import numpy as np
from skimage.measure import label, regionprops_table
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from openpyxl import load_workbook
from dask.distributed import Client
import joblib

import HandleSistsCoords3D

# analysis_folder       =  '/home/atrullo/Dropbox/Virginia_Anto/2Spot CRISPR Test file_Late nc14'
# pix_size, pix_size_Z  =  1.3, 4.9


def coords3D23D_all(spots_3D_coords, step):
    """Function to reconstruct 3D spots in a single frame from coordinate mtx"""
    spt            =  np.zeros(spots_3D_coords[-1, (1, 2, 3)], dtype=np.uint8)
    sub_3D_coords  =  spots_3D_coords[:-1]
    sub_3D_coords  =  sub_3D_coords[sub_3D_coords[:, 0] == step]
    spt[sub_3D_coords[:, 1], sub_3D_coords[:, 2], sub_3D_coords[:, 3]]  =  1
    return spt


def detect_outlier(data_1, threshold):
    """Function to identify the outliers"""
    outliers   =  []
    # threshold  =  3
    mean_1     =  np.mean(data_1)
    std_1      =  np.std(data_1)
    for y in data_1:
        z_score  =  (y - mean_1) / std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


class SisterSplit:
    """Main function"""
    # def __init__(self, dist_thr, spts_trk, green4d, spots_3D_coords, pix_size, pix_size_Z):
    def __init__(self, analysis_folder, green4d):

        spts_trk          =  np.load(analysis_folder + '/spots_tracked.npy')              # load nuclei-tracked spots
        spots_3D_coords   =  np.load(analysis_folder + '/spots_3D_coords.npy')            # load coordinate of all the pixels of the points
        nuclei_tracked    =  np.load(analysis_folder + '/nuclei_tracked.npy')             # load tracked nuclei
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
            calc      =  np.sign(calc + (nuclei_tracked == k).sum(0) * 1)
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
            nucs2send      =  nuclei_tracked[:, x_min:x_max, y_min:y_max]
            jobs_args.append([spts_trk_clst, green4d2send, nucs2send, spots_3D_coords, [x_min, x_max, y_min, y_max], pix_size, pix_size_Z])

        # print("pool start")
        # results  =  []
        # for cnt, arg in enumerate(jobs_args):
        #     print(cnt)
        #     results.append(SisterSplitUtility(arg))

        # If you have a remote cluster running Dask
        # client = Client('tcp://scheduler-address:8786')
        # If you want Dask to set itself up on your personal computer
        client  =  Client(processes=False)
        with joblib.parallel_backend('dask'):
            results  =  joblib.Parallel(verbose=100)(joblib.delayed(SisterSplitUtility)(i) for i in jobs_args)
        # print("pool end")

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


class SisterSplitUtility:
    """Track and split sisters for multiprocessing"""
    def __init__(self, multip_args):

        """This tracking algorithm is based on the grouping of the positions taken time frame by time frame
           by each spot. Positions are relative to nucleus centroid: they generally form two clouds of points;
           the idea is to cluster them with a Mixture of Gaussian Model and give an identity to each sister.
           Of course, we need to map back from relative positions to absolute positions."""

        spts_trk_r                  =  multip_args[0]
        green4d_r                   =  multip_args[1]
        nuclei_tracked_r            =  multip_args[2]
        spots_3D_coords             =  multip_args[3]
        x_min, x_max, y_min, y_max  =  multip_args[4]
        pix_size                    =  multip_args[5]
        pix_size_Z                  =  multip_args[6]

        sib_idxs_r        =  np.unique(spts_trk_r[spts_trk_r != 0])                                                     # list of sibiling tags
        sists_ints_dists  =  np.zeros((sib_idxs_r.size, 3, spts_trk_r.shape[0]))                                        # initialize the matrix to store intensity and distances
        sizess            =  np.append(0, spots_3D_coords[-1])                                                          # store info on image stack size
        sists_3D_coords   =  np.zeros((0, 5), dtype=np.uint16)                                                          # initialize sisters coordinates matrix

        for cnt, sib_idx_r in enumerate(sib_idxs_r):                                                                    # for each tag
            sibs          =  (spts_trk_r == sib_idx_r) * 1                                                              # isolate sisters
            nuc           =  (nuclei_tracked_r == sib_idx_r) * 1                                                        # isolate relative nucleus
            ctrs          =  []                                                                                         # initialize centers of the spots to store their positions
            ctrs_rel      =  []                                                                                         # initialize centers relative to nucleus center of spots to store their positions
            coords_store  =  []                                                                                         # initialize list to store coordinates with labels of each spot pixel: (t, z, x, y) for the moment, label will be added after tracking
            for tt in range(sizess[1]):                                                                                 # for each time frame
                rgp_nuc    =  regionprops_table(nuc[tt], properties=["centroid"])                                       # nucleus centroid
                sibs3d     =  label(coords3D23D_all(spots_3D_coords, tt)[:, x_min:x_max, y_min:y_max] * sibs[tt])       # 3d labeling of the sisters in the time frame
                rgp_sib3d  =  regionprops_table(sibs3d, properties=["centroid", "coords"])                              # 3d regionprops (coordinates and centroids)
                for mm in range(len(rgp_sib3d["centroid-0"])):
                    # ctrs.append([tt, rgp_sib3d["centroid-0"][mm] * pix_size_Z, rgp_sib3d["centroid-1"][mm] * pix_size, rgp_sib3d["centroid-2"][mm] * pix_size])
                    ctrs_rel.append([tt, rgp_sib3d["centroid-0"][mm], rgp_sib3d["centroid-1"][mm] - rgp_nuc["centroid-0"][0], rgp_sib3d["centroid-2"][mm] - rgp_nuc["centroid-1"][0]])   # store relative positions of centroids
                    ctrs.append([tt, rgp_sib3d["centroid-0"][mm], rgp_sib3d["centroid-1"][mm], rgp_sib3d["centroid-2"][mm]])                                                             # store absolute positions of centroids
                    coords_bff         =  np.zeros((rgp_sib3d["coords"][mm].shape[0], rgp_sib3d["coords"][mm].shape[1] + 1), dtype=np.uint16)          # buffer coordinate matrix
                    coords_bff[:, 1:]  =  rgp_sib3d["coords"][mm]                                                                                      # add positional coordinate
                    coords_bff[:, 0]   =  tt                                                                                                           # add time coordinate
                    coords_store.append(coords_bff)                                                                                                    # store in the list (no spot identity for the moment)

            ctrs             =  np.asarray(ctrs)
            ctrs_rel         =  np.asarray(ctrs_rel)
            ctrs_rel[:, 2]  +=  np.abs(ctrs_rel[:, 2].min())                                                            # make sure to not have negative values
            ctrs_rel[:, 3]  +=  np.abs(ctrs_rel[:, 3].min())
            ctrs4fit         =  np.copy(ctrs_rel)                                                                       # copy of the ctrs for fit: some will be removed, we need for the final mapping







            # frms_less2spts  =  []                                                                                       # list of all the time frames with spots
            # frms            =  np.unique(ctrs_rel[:, 0])
            # for kk in frms:
            #     jj  =  np.where(ctrs_rel[:, 0] == kk)[0]                                                                # search their positions in the matrix
            #     if jj.size < 2:                                                                                         # if there are less than 2 spots in a given time frame, store the indexes
            #         frms_less2spts.append(jj[0])

            # ctrs_rel  =  np.delete(ctrs_rel, frms_less2spts, axis=0)                                                    # remove centroids of spots alone in the time frame




            out_thr   =  2.5
            outlr_z   =  detect_outlier(ctrs_rel[:, 1], out_thr)                                                        # fake detection generally brings to outliers in coordinate: search for outliers in z
            if len(outlr_z) > 0:
                for ouzz in outlr_z:
                    ctrs_rel  =  np.delete(ctrs_rel, np.where(ctrs_rel[:, 1] == ouzz), axis=0)                          # remove the outliers in z
            outlr_x  =  detect_outlier(ctrs_rel[:, 2], out_thr)                                                         # in the following remove the outliers in x and y
            if len(outlr_x) > 0:
                for ouxx in outlr_x:
                    ctrs_rel  =  np.delete(ctrs_rel, np.where(ctrs_rel[:, 2] == ouxx), axis=0)
            outlr_y  =  detect_outlier(ctrs_rel[:, 3], out_thr)
            if len(outlr_y) > 0:
                for ouyy in outlr_y:
                    ctrs_rel  =  np.delete(ctrs_rel, np.where(ctrs_rel[:, 3] == ouyy), axis=0)

            if ctrs_rel.shape[0] > 1:                                                                                   # fit the coordinates with a sum of 2 Gaussian dfistributuions
                # gm_rel  =  GaussianMixture(n_components=2, random_state=0, covariance_type="spherical").fit(ctrs_rel[:, 1:].reshape((-1, 1)))  # fit with GMM on the clean centroids (covariance_type="full")
                # mm_rel  =  gm_rel.fit_predict(ctrs4fit[:, 1:])                                                          # run the GMM instance on all (non-selected) centroids
                mm_rel  =  KMeans(n_clusters=2, random_state=0).fit(ctrs4fit[:, 1:]).labels_  # classifiction

            else:
                continue                                                                                                # in case after filtering there are no centroids no tracking is possible

            mm_rel_0   =  np.where(mm_rel == 0)[0]                                                                      # isolate clusters group of coordinates
            mm_rel_1   =  np.where(mm_rel == 1)[0]
            ctrs4_0    =  []                                                                                            # list of centroids for sister 1 (first appearing) and sister 2 in relative and absolute coordinates
            ctrs4_1    =  []
            ctrs_0     =  []
            ctrs_1     =  []
            for oo_0 in mm_rel_0:                                                                                       # split centroids for fit and not in the proper categories
                ctrs4_0.append(ctrs4fit[oo_0, :])
                ctrs_0.append(ctrs[oo_0, :])
            for oo_1 in mm_rel_1:
                ctrs4_1.append(ctrs4fit[oo_1, :])
                ctrs_1.append(ctrs[oo_1, :])

            ctrs4_0  =  np.asarray(ctrs4_0)
            ctrs4_1  =  np.asarray(ctrs4_1)
            ctrs_0   =  np.asarray(ctrs_0)
            ctrs_1   =  np.asarray(ctrs_1)

            # out_thr     =  2.5
            outlr_z_40  =  detect_outlier(ctrs4_0[:, 1], out_thr)                                                       # remove again outliers in z, x and y in the subcategories (centroids of spots 1 and spots 2 separately)
            if len(outlr_z_40) > 0:
                for ouzz_40 in outlr_z_40:                                                                              # outliers are identified with respect to z, x and y coordinates as before
                    idz_bff  =  np.where(ctrs4_0[:, 1] == ouzz_40)
                    ctrs4_0  =  np.delete(ctrs4_0, idz_bff, axis=0)
                    ctrs_0   =  np.delete(ctrs_0, idz_bff, axis=0)
            outlr_x_40  =  detect_outlier(ctrs4_0[:, 2], out_thr)
            if len(outlr_x_40) > 0:
                for ouxx_40 in outlr_x_40:
                    idx_bff  =  np.where(ctrs4_0[:, 2] == ouxx_40)
                    ctrs4_0  =  np.delete(ctrs4_0, idx_bff, axis=0)
                    ctrs_0   =  np.delete(ctrs_0, idx_bff, axis=0)
            outlr_y_40  =  detect_outlier(ctrs4_0[:, 3], out_thr)
            if len(outlr_y_40) > 0:
                for ouyy_40 in outlr_y_40:
                    idy_bff  =  np.where(ctrs4_0[:, 3] == ouyy_40)
                    ctrs4_0  =  np.delete(ctrs4_0, idy_bff, axis=0)
                    ctrs_0   =  np.delete(ctrs_0, idy_bff, axis=0)

            outlr_z_41  =  detect_outlier(ctrs4_1[:, 1], out_thr)
            if len(outlr_z_41) > 0:
                for ouzz_41 in outlr_z_41:
                    idz_bff  =  np.where(ctrs4_1[:, 1] == ouzz_41)
                    ctrs4_1  =  np.delete(ctrs4_1, idz_bff, axis=0)
                    ctrs_1   =  np.delete(ctrs_1, idz_bff, axis=0)
            outlr_x_41  =  detect_outlier(ctrs4_1[:, 2], out_thr)
            if len(outlr_x_41) > 0:
                for ouxx_41 in outlr_x_41:
                    idx_bff  =  np.where(ctrs4_0[:, 2] == ouxx_41)
                    ctrs4_1  =  np.delete(ctrs4_1, idx_bff, axis=0)
                    ctrs_1   =  np.delete(ctrs_1, idx_bff, axis=0)
            outlr_y_41  =  detect_outlier(ctrs4_1[:, 3], out_thr)
            if len(outlr_y_41) > 0:
                for ouyy_41 in outlr_y_40:
                    idy_bff  =  np.where(ctrs4_1[:, 3] == ouyy_41)
                    ctrs4_1  =  np.delete(ctrs4_1, idy_bff, axis=0)
                    ctrs_1   =  np.delete(ctrs_1, idy_bff, axis=0)

            # mtx_ctrs  =  np.zeros_like(spts_trk_r)
            # for cc_0 in ctrs_0:
            #     mtx_ctrs[np.round(cc_0[0]).astype(int), np.round(cc_0[2]).astype(int), np.round(cc_0[3]).astype(int)]  =  2
            # for cc_1 in ctrs_1:
            #     mtx_ctrs[np.round(cc_1[0]).astype(int), np.round(cc_1[2]).astype(int), np.round(cc_1[3]).astype(int)]  =  1

            if ctrs_0[:, 0].min() <= ctrs_1[:, 0].min():                            # the first appearing spot is red, so check it (red and green are the colors assigned for visualization in the GUI)
                ctrs_red    =  np.copy(ctrs_0)
                ctrs_green  =  np.copy(ctrs_1)
            else:
                ctrs_red    =  np.copy(ctrs_1)
                ctrs_green  =  np.copy(ctrs_0)

            frms_red  =  np.unique(ctrs_red[:, 0])
            for frm_red in frms_red:
                sng_frame_red  =  ctrs_red[ctrs_red[:, 0] == frm_red]
                if sng_frame_red.shape[0] > 1:
                    dists_red  =  np.zeros((sng_frame_red.shape[0]))
                    for uu in range(sng_frame_red.shape[0]):
                        dists_red[uu]  =  ((ctrs_red[:, 1].mean() - sng_frame_red[uu, 1]) * pix_size_Z) ** 2 + ((ctrs_red[:, 2].mean() - sng_frame_red[uu, 2]) * pix_size) ** 2 + ((ctrs_red[:, 3].mean() - sng_frame_red[uu, 3]) * pix_size) ** 2
                    ll           =  np.argmin(dists_red)
                    idxs2rm_red  =  list(np.arange(sng_frame_red.shape[0]))
                    idxs2rm_red.remove(ll)
                    for aa_red in idxs2rm_red:
                        rm_row_idx_red  =  np.where(((ctrs_red == sng_frame_red[aa_red]) * 1).sum(1) == 4)[0][0]
                        ctrs_red        =  np.delete(ctrs_red, rm_row_idx_red, axis=0)

            frms_green  =  np.unique(ctrs_green[:, 0])
            for frm_green in frms_green:
                sng_frame_green  =  ctrs_green[ctrs_green[:, 0] == frm_green]
                if sng_frame_green.shape[0] > 1:
                    dists_green  =  np.zeros((sng_frame_green.shape[0]))
                    for uu in range(sng_frame_green.shape[0]):
                        dists_green[uu]  =  ((ctrs_green[:, 1].mean() - sng_frame_green[uu, 1]) * pix_size_Z) ** 2 + ((ctrs_green[:, 2].mean() - sng_frame_green[uu, 2]) * pix_size) ** 2 + ((ctrs_green[:, 3].mean() - sng_frame_green[uu, 3]) * pix_size) ** 2
                    ll           =  np.argmin(dists_green)
                    idxs2rm_green  =  list(np.arange(sng_frame_green.shape[0]))
                    idxs2rm_green.remove(ll)
                    for aa_green in idxs2rm_green:
                        rm_row_idx_green  =  np.where(((ctrs_green == sng_frame_green[aa_green]) * 1).sum(1) == 4)[0][0]
                        ctrs_green        =  np.delete(ctrs_green, rm_row_idx_green, axis=0)

            tag1  =  2 * cnt + 1                                                # sister tags are odd for red and even for green, assigned in increasing order
            tag2  =  2 * cnt + 2

            for uu_red in ctrs_red:                                                                                                                      # once centroids are tracked, we need to associate the list of the coordinate of each pixel of the corresponding spot
                ll_bff_red                  =  np.where(np.sum(np.equal(ctrs, uu_red) * 1, axis=1) == 4)[0][0]                                           # since coordinates where stored in a container with the same index that relative centroids have in ctrs matrix, search centroid by centroid the coordinate in the coord_store matrix
                coords_lbl_tzxy_bff         =  np.zeros((coords_store[ll_bff_red].shape[0], coords_store[ll_bff_red].shape[1] + 1), dtype=np.uint16)     # initialize a matrix with proper size to write in label, t, z, x, y
                coords_lbl_tzxy_bff[:, 1:]  =  coords_store[ll_bff_red]                                                                                  # introduce coordinates
                coords_lbl_tzxy_bff[:, 0]   =  tag1                                                                                                      # introduce tag
                sists_3D_coords             =  np.concatenate((sists_3D_coords, coords_lbl_tzxy_bff))                                                    # concatenate

            for uu_green in ctrs_green:                                                                                                                  # same as before
                ll_bff_green                =  np.where(np.sum(np.equal(ctrs, uu_green) * 1, axis=1) == 4)[0][0]
                coords_lbl_tzxy_bff         =  np.zeros((coords_store[ll_bff_green].shape[0], coords_store[ll_bff_green].shape[1] + 1), dtype=np.uint16)
                coords_lbl_tzxy_bff[:, 1:]  =  coords_store[ll_bff_green]
                coords_lbl_tzxy_bff[:, 0]   =  tag2
                sists_3D_coords             =  np.concatenate((sists_3D_coords, coords_lbl_tzxy_bff))

        # sists_3D_coords  =  np.r_[sists_3D_coords, [sizess]]

        for c, sib_idx in enumerate(sib_idxs_r):                                                                        # for each tag
            sis_tg1  =  2 * c + 1
            sis_tg2  =  2 * c + 2
            for ss in range(sizess[1]):
                bff_coords                  =  sists_3D_coords[sists_3D_coords[:, 1] == ss]                                 # select the coordinates of the pixels of each spots
                bff_coords1                 =  bff_coords[bff_coords[:, 0] == sis_tg1]
                sists_ints_dists[c, 0, ss]  =  np.sum(green4d_r[ss, bff_coords1[:, 2], bff_coords1[:, 3], bff_coords1[:, 4]])       # use the coordinates to get the intensity values
                bff_coords2                 =  bff_coords[bff_coords[:, 0] == sis_tg2]
                sists_ints_dists[c, 1, ss]  =  np.sum(green4d_r[ss, bff_coords2[:, 2], bff_coords2[:, 3], bff_coords2[:, 4]])
                if bff_coords1.sum() != 0 and bff_coords2.sum() != 0:                                                               # for all the frames with 2 spots, calculate the distance
                    sists_ints_dists[c, 2, ss]  =  np.sqrt(((bff_coords1[:, 2].mean() - bff_coords2[:, 2].mean()) * pix_size_Z) ** 2 + ((bff_coords1[:, 3].mean() - bff_coords2[:, 3].mean()) * pix_size) ** 2 + ((bff_coords1[:, 4].mean() - bff_coords2[:, 4].mean()) * pix_size) ** 2)

        self.sists_ints        =  sists_ints_dists
        self.sists_3D_coords   =  sists_3D_coords
