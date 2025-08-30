import math
import os
import os.path

import nibabel
import nibabel as nib
import numpy as np
import torch
import torch.nn


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag = test_flag
        if test_flag:
            self.seqtypes = ["voided", "mask"]
        else:
            #self.seqtypes = ["diseased", "mask", "healthy"]
            self.seqtypes = ["healthyvoided", "healthy", "t1n"]

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        self.mask_vis = []
        for root, dirs, files in os.walk(self.directory):
            dirs_sorted = sorted(dirs)
            for dir_id in dirs_sorted:
                datapoint = dict()
                sli_dict = dict()
                for ro, di, fi in os.walk(root + "/" + str(dir_id)):
                    fi_sorted = sorted(fi)
                    for f in fi_sorted:
                        seqtype = f.split("-")[-1].split(".")[0]
                        #print('seqtype:', seqtype)
                        datapoint[seqtype] = os.path.join(root, dir_id, f)
                        if seqtype == "mask":
                            slice_range = []
                            mask_to_define_rand = np.array(
                                nibabel.load(datapoint["mask"]).dataobj
                            )
                            if test_flag:
                                mask_to_define_rand = np.pad(
                                    mask_to_define_rand, ((0, 0), (0, 0), (34, 35))
                                )
                                mask_to_define_rand = mask_to_define_rand[8:-8, 8:-8, :]
                            for i in range(0, 224):
                                mask_slice = mask_to_define_rand[:, :, i]
                                if np.sum(mask_slice) != 0:
                                    slice_range.append(i)

                    # assert (
                    #     set(datapoint.keys()) == self.seqtypes_set
                    # ), f"datapoint is incomplete, keys are {datapoint.keys()}"
                    self.database.append(datapoint)
                    self.mask_vis.append(slice_range)

            break

    def __getitem__(self, x):
        filedict = self.database[x]
        slicedict = self.mask_vis[x]

        #print("input files: ", filedict)
        #print("slice dict:", slicedict)

        out_single = []

        if self.test_flag:
            for seqtype in self.seqtypes:
                if seqtype == "voided":
                    nib_img = np.array(nibabel.load(filedict[seqtype]).dataobj).astype(
                        np.float32
                    )
                    path = filedict[seqtype]
                    t1_numpy_pad = np.pad(nib_img, ((0, 0), (0, 0), (34, 35)))
                    t1_numpy_crop = t1_numpy_pad[8:-8, 8:-8, :] # crop-pad to (224, 224, 224)
                    t1_clipped = np.clip(
                        t1_numpy_crop,
                        np.quantile(t1_numpy_crop, 0.001),
                        np.quantile(t1_numpy_crop, 0.999),
                    )
                    t1_normalized = (t1_clipped - np.min(t1_clipped)) / (
                        np.max(t1_clipped) - np.min(t1_clipped)
                    )
                    img_preprocessed = torch.tensor(t1_normalized)
                elif seqtype == "mask":
                    nib_img = np.array(nibabel.load(filedict[seqtype]).dataobj).astype(
                        np.float32
                    )
                    path = filedict[seqtype]
                    mask_numpy_pad = np.pad(nib_img, ((0, 0), (0, 0), (34, 35)))
                    mask_numpy_crop = mask_numpy_pad[8:-8, 8:-8, :]
                    img_preprocessed = torch.tensor(mask_numpy_crop)
                else:
                    print("unknown seqtype")

                out_single.append(img_preprocessed)

            out_single = torch.stack(out_single)

            image = out_single[0:2, ...]
            path = filedict[seqtype]

            return (image, path, slicedict)

        else:
            for seqtype in self.seqtypes:
                nib_img = np.array(nibabel.load(filedict[seqtype]).dataobj).astype(
                    np.float32
                )
                path = filedict[seqtype]
                img_preprocessed = torch.tensor(nib_img)

                out_single.append(img_preprocessed)

            out_single = torch.stack(out_single) # "diseased", "mask", "healthy"

            image = out_single[0:2, ...] # "diseased", "mask" 
            label = out_single[2, ...] # "healthy"
            label = label.unsqueeze(0)
            path = filedict[seqtype]

            return (image, label, path, slicedict)

    def __len__(self):
        return len(self.database)
