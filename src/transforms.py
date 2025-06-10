import albumentations as A
from albumentations.pytorch import ToTensorV2


def preprocess(preprocess_config):
    return A.Compose([
        A.Resize(
            height =preprocess_config['side_size'],
            width =preprocess_config['side_size'],
        ),
        A.Normalize(
            mean=preprocess_config['mean'],
            std=preprocess_config['std']
        ),
        ToTensorV2()
    ])