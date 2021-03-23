from torch.utils.data import Dataset, DataLoader

"""*******************************************************************************"""
"dataset.py에 대한 정보"
" numpy 기반의 stack 버퍼를 받은 다음, feed transform을 통해서 네티워크의 feed 데이터 생성"
" 여기서 transform을 거치게 되어 패치 단위의 데이터가 나오게 됨."
"""*******************************************************************************"""


class Supervised_dataset(Dataset):
    """supervised dataset v1"""
    """ 먼저 이 class는 이미 numpy로 만들어진 객체를 input으로 받는다."""
    """ 만든 가장 큰 이유는 transform을 하기 위해서다."""

    def __init__(self, input, target, train=True, transform=None):
        """
        Args:
            input, target : N, H, W, C
            input : color + features
            GT : ref color
        """
        self.input_for_network = input
        self.GT_for_network = target
        self.transform = transform
        self.is_train = train

    def __len__(self):
        return self.input_for_network.shape[0]

    def __getitem__(self, idx):
        input = self.input_for_network[idx]  # color + features + diff
        GT = self.GT_for_network[idx]

        sample = {'input': input, 'target': GT}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Supervised_dataset_with_design_v1(Dataset):
    """
    input : input, design and GT stack buffers
    output : patches of input, design and GT
    feature #1 : v1에 맞춰 따로 design을 full res numpy 단계에서 만들어 놓은 것을 기반으로 함.
    """

    def __init__(self, input, design, target, train=True, transform=None):
        """
        Args:
            input, target : N, H, W, C
            input : color + features
            GT : ref color
        """
        self.input_for_network = input
        self.design_for_netwrok = design
        self.GT_for_network = target

        self.transform = transform
        self.is_train = train


    def __len__(self):
        return self.input_for_network.shape[0]

    def __getitem__(self, idx):
        input = self.input_for_network[idx]  # color + features + diff
        design = self.design_for_netwrok[idx]
        GT = self.GT_for_network[idx]

        sample = {'input': input, 'design': design, 'target': GT}

        if self.transform:
            sample = self.transform(sample)

        return sample


