DATASET = "chest_xray_3"

IMG_SIZE = 64
N_INPUT_CHANNELS = 1
NUM_CLASSES=2
ALL_ARCHITECTURES = ['resnet20', 'resnet32', 'resnet44', 'resnet56']

# CCT
EMBEDDING_DIM = [128, 256]
N_CONV_LAYERS_LIST=[1, 2, 3]
KERNEL_SIZE_LIST=[3, 5, 7]

pooling_kernel_size=3
pooling_stride=2
pooling_padding=1


NUM_LAYERS_LIST = [2, 4, 6, 7]
NUM_HEADS_LIST=[2, 2, 4, 4]
MLP_RATIO_LIST=[1, 1, 2, 2]

