import numpy as np
from scipy.ndimage import zoom
from pietorch import blend_dst_numpy
import cv2

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def resize_batch(imgs, target_size):
    sx = imgs.shape[1]
    sy = imgs.shape[2]
    return zoom(imgs, (1, float(target_size[0]) / sx, float(target_size[1] / sy), 1), order=0)

def create_interp_mask(img_shape, patch_center, patch_width, type):

    mask_i = np.zeros(img_shape)

    coor_min_x = patch_center[0] - int(patch_width[0] / 2)
    coor_max_x = patch_center[0] + int(patch_width[0] / 2)
    coor_min_y = patch_center[1] - int(patch_width[1] / 2)
    coor_max_y = patch_center[1] + int(patch_width[1] / 2)

    # clip coordinates to within image dims
    coor_min_x = np.clip(coor_min_x, 0, img_shape[0]).astype(np.int)
    coor_max_x = np.clip(coor_max_x, 0, img_shape[0]).astype(np.int)
    coor_min_y = np.clip(coor_min_y, 0, img_shape[1]).astype(np.int)
    coor_max_y = np.clip(coor_max_y, 0, img_shape[1]).astype(np.int)

    if type == 'rectangular':
        mask_i[coor_min_x:coor_max_x,coor_min_y:coor_max_y,0] = 1.
        patch_shape = np.array([coor_max_x-coor_min_x,coor_max_y-coor_min_y])

    elif type == 'circular':
        width = np.min(np.array([(coor_max_x-coor_min_x)/2, (coor_max_y-coor_min_y)/2])).astype(np.int)
        mask_i = cv2.circle(mask_i, patch_center.astype(np.int), width, 1., -1)
        patch_shape = np.array([width, width])

    elif type == 'elliptic':
        angle = np.random.randint(0,360)
        width = (int((coor_max_x-coor_min_x)/2), int((coor_max_y-coor_min_y)/2))
        mask_i = cv2.ellipse(mask_i, patch_center.astype(np.int), width, angle, 0., 360,1.,thickness = -1)
        patch_shape = np.array([coor_max_x - coor_min_x, coor_max_y - coor_min_y])

    return mask_i, patch_shape

class BatchProvider():
    """
    Class for accessing mini batches of training, testing and validation data
    """

    def __init__(self, X, y, indices, add_dummy_dimension=False, **kwargs):

        self.X = X
        self.y = y
        self.indices = indices
        self.unused_indices = indices.copy()

        self.num_labels_per_subject = kwargs.get('num_labels_per_subject', 1)
        if self.num_labels_per_subject > 1:
            self.annotator_range = kwargs.get('annotator_range', range(self.num_labels_per_subject))

        self.resize_to = kwargs.get('resize_to', None)
        self.segmentation = kwargs.get('segmentation', None)

        self.do_augmentations = kwargs.get('do_augmentations', False)
        self.augmentation_options = kwargs.get('augmentation_options', None)
        self.do_poisson_blend = kwargs.get('do_poisson_blend', False)
        self.PII_options = kwargs.get('PII_options', None)
        if self.PII_options and 'version' in self.PII_options:
            if self.PII_options['version'] == 'old':
                self.next_batch = self.next_batch_old_PII
        self.rescale_range = kwargs.get('rescale_range', None)
        self.rescale_rgb = kwargs.get('rescale_rgb', None)
        self.normalise_images = True if not self.rescale_range else False  # normalise if not rescale

    def next_batch(self, batch_size):
        """
        Function for getting a single random batch. 
        """

        if len(self.unused_indices) < batch_size:
            self.unused_indices = self.indices

        batch_indices = np.random.choice(self.unused_indices, batch_size, replace=False)
        self.unused_indices = np.setdiff1d(self.unused_indices, batch_indices)

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(batch_indices)

        if self.do_poisson_blend:
            X_batch = []
            y_batch = []
            patch_centre_batch = []
            patch_width_batch = []

            for i in batch_indices:
                target = self.X[i]
                lab1 = self.y[i]
                target, lab1 = self._post_process(target, lab1)

                # create random anomaly
                dims = np.array(np.shape(target))
                core = self.PII_options['core_percent'] * dims  # width of core region
                offset = (1 - self.PII_options['core_percent']) * dims / 2  # offset to center core

                min_width = np.round(0.05 * dims[1])
                max_width = np.round(0.2 * dims[1])  # make sure it is less than offset

                images_fused = []
                # simulate multiple anomalies, adapt if necessary
                anomalies = np.random.randint(10, 20)
                for a in range(anomalies):
                    img = self.X[np.random.randint(len(batch_indices))]

                    # get patch of source image
                    center_dim1 = np.random.randint(offset[0], offset[0] + core[0], size=dims[-1])
                    center_dim2 = np.random.randint(offset[1], offset[1] + core[1], size=dims[-1])
                    patch_center = np.stack([center_dim1, center_dim2],-1)
                    patch_width = np.random.randint(min_width, max_width, size=2)

                    source_mask, shape = create_interp_mask(img.shape, patch_center[0], patch_width, 'rectangular')
                    source = np.expand_dims(np.reshape(img[np.where(source_mask==1)], shape),-1)

                    # get corner coords for source patch in target
                    target_center_dim1 = np.random.randint(patch_width[0], core[0] - patch_width[0], size=dims[-1])
                    target_center_dim2 = np.random.randint(patch_width[1], core[1] - patch_width[1], size=dims[-1])
                    corner = np.stack((target_center_dim1,target_center_dim2), 1)

                    # simulate multiple raters
                    raters = np.random.randint(1,10)
                    for r in range(raters):
                        # location of anomaly in source patch - mu in middle of patch
                        loc = np.random.normal(loc=np.array([shape[1]/2,shape[0]/2]), scale=np.array([shape[1]/2,shape[0]/2])/6, size=2).astype(np.int)
                        # width of anomaly in source patch
                        width_var = np.random.normal(loc=np.array([shape[1]/2,shape[0]/2]), scale=1, size=2).astype(np.int)

                        mask, _ = create_interp_mask(source.shape, loc, width_var, self.PII_options['style'])

                        strength = np.random.randint(10, 20)
                        patchex = blend_dst_numpy(target[:,:,0], source[:,:,0]*strength, mask[:,:,0], corner[0], True)

                        images_fused.append(patchex)


                label=abs(np.mean(np.array(images_fused), 0) - target[:, :, 0])*10
                label_var=np.std(np.array(images_fused), 0)
                PII_sample=np.mean(np.array(images_fused), 0)

                if self.segmentation:
                    PII_sample *= self.segmentation[i, :, :, 0]
                    label *= self.segmentation[i, :, :, 0]
                    label_var *= self.segmentation[i, :, :, 0]

                X_batch.extend([PII_sample])
                y_batch.extend([label])
                patch_centre_batch.extend([corner])
                patch_width_batch.extend([patch_width])

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            patch_centre_batch = np.array(patch_centre_batch)
            patch_width_batch = np.array(patch_width_batch)

        else:
            X_batch = self.X[batch_indices, ...]
            y_batch = self.y[batch_indices, ...]

        if self.num_labels_per_subject > 1:
            y_batch = self._select_random_label(y_batch, self.annotator_range)

        if not self.do_poisson_blend:
            return X_batch, y_batch

        return np.expand_dims(X_batch,-1), np.expand_dims(y_batch,-1), (patch_centre_batch, patch_width_batch)
