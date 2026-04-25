from os.path import join

_BASE_DATA_PATH = "data"

dataset_config = {
    'nwpu': {
        'path': join(_BASE_DATA_PATH, 'nwpu-45'),
        'resize': (256, 256),
        'crop': None,
        'flip': False,
        'normalize': ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        # 'class_order': [
        #     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
        #     22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
        #     42, 43, 44
        # ]
        # 'class_order':[
        #     0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
        #     30,31,32,33,34,35,36,37,38,39,40,41,42,43,44
        # ]
        # Use the next 3 lines to use MNIST with a 3x32x32 input
        # 'extend_channel': 3,
        # 'pad': 2,
        # 'normalize': ((0.1,), (0.2752,))    # values including padding
    },
    'aid': {
        'path': join(_BASE_DATA_PATH, 'aid-30'),
        'resize': (256, 256),
        'crop': None,
        'flip': False,
        'normalize': ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        'class_order': [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
            22, 23, 24, 25, 26, 27, 28, 29
        ]
        # Use the next 3 lines to use MNIST with a 3x32x32 input
        # 'extend_channel': 3,
        # 'pad': 2,
        # 'normalize': ((0.1,), (0.2752,))    # values including padding
    },
    'ucmerced': {
        'path': join(_BASE_DATA_PATH, 'ucmerced21'),
        'resize': (256, 256),
        'rotate': True,
        'crop': None,
        'flip': True,
        'normalize': ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        # 'class_order': [
        #     0, 1, 2, 3, 4, 5, 6, 15, 13, 11, 18, 9, 20, 16, 8, 14, 17, 10, 7, 12, 19
        # ]
        'class_order': [
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
        ]
        # 'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # Use the next 3 lines to use MNIST with a 3x32x32 input
        # 'extend_channel': 3,
        # 'pad': 2,
        # 'normalize': ((0.1,), (0.2752,))    # values including padding
    }
}

# Add missing keys:
for dset in dataset_config.keys():
    for k in ['resize', 'pad', 'crop', 'normalize', 'class_order', 'extend_channel']:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if 'flip' not in dataset_config[dset].keys():
        dataset_config[dset]['flip'] = False
