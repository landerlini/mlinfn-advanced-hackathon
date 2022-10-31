# Copyright (C) 2021 A. Retico for the AIM-COVID19-WG of developers.
#
# This file is part of Analysis_PIPELINE_LungQuant.
#
# upsilon_analysis is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np 
import nibabel as nib
from scipy.ndimage import label

def cc2d(lung_mask0):
    lung_mask = lung_mask0.copy()

    # Let us create a binary mask.
    binary_mask = np.round(lung_mask).astype('int')

    # Now, we perform region labelling. This way, every connected component
    # will have their own colour value.
    labelled_mask, num_labels = label(binary_mask)
    lab_area = []
    lab_mask = []
#    label_data = np.zeros((num_labels+1), dtype = 'float32')
    # Let us now remove all the too small regions.
    for lab in range(num_labels+1):
        current_area = np.sum(lung_mask[labelled_mask == lab])
        lab_area.append([current_area, lab])
    print(lab_area)
    lab_area = np.array(lab_area)
    sorted_lab = lab_area[lab_area[:, 0].argsort()] 
#    sorted_lab = np.sort(np.array(lab_area), axis = -1)
    print(sorted_lab)
    label_def = sorted_lab[-2:]
    print(label_def)
    final_mask = np.zeros(lung_mask.shape)
    final_mask[labelled_mask == int(label_def[0][1])] = 1
    final_mask[labelled_mask == int(label_def[1][1])] = 1

    return np.round(final_mask)
