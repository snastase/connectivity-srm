import json
import numpy as np
from gifti_io import read_gifti, write_gifti

hemi = 'rh'

# Load in HCP-MMP1 parcellation on fsaverage6
mmp = read_gifti(f'data/MMP_fsaverage6.{hemi}.gii')[0]

# Load in ROI labels
with open('MMP_ROIs.json') as f:
    rois = json.load(f)

roi_colors = {'EAC': rois['EAC']['A1'],
              'AAC': rois['AAC']['STSdp'],
              'PCC': rois['PCC']['POS2'],
              'TPOJ': rois['TPOJ']['TPOJ1']}
    
# Create separate masks
masks = {}
for roi in rois:
    mask = np.zeros(mmp.shape)
    for area in rois[roi]:
        mask[mmp == rois[roi][area]] = 1
        
    write_gifti(mask,
            f'data/MMP_{roi}_fsaverage6.{hemi}.gii',
            f'data/MMP_fsaverage6.{hemi}.gii')
    
    masks[roi] = mask.astype(bool)
    n_voxels = np.sum(mask)
    np.save(f'data/{roi}_mask_{hemi}.npy', mask)
    print(f"Created {hemi} {roi} mask containing "
          f"{n_voxels} voxels")
    
# Create single parcellation map
mask_map = np.zeros(mmp.shape)
for i, mask_name in enumerate(masks):
    mask_map[masks[mask_name]] = i + 1
    
write_gifti(mask_map,
            f'data/MMP_ROIs_fsaverage6.{hemi}.gii',
            f'data/MMP_fsaverage6.{hemi}.gii')