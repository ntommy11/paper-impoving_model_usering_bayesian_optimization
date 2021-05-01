from itertools import accumulate
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import json
import numpy as np
import random

#for custom dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader

#for Bayesian Optimization
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float
# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 초기 하이퍼파라미터
learning_rate = 0.001
training_epochs = 100
batch_size = 32

filename= 'cnn_data017ver1.json' # 전처리된 학습 데이터 파일 이름.

#trajectory custom dataset (이동패턴 커스텀 데이터셋 정의)
class Trajectory_Dataset(Dataset):
    def __init__(self,filename):
        with open(filename) as json_file:
            cnn_data = json.load(json_file)
        train_set = cnn_data['train']
        self.x_data = []
        self.y_data = []
        for x,y in train_set:
            self.x_data.append(x)
            self.y_data.append(y)
        self.x_data = torch.FloatTensor(self.x_data)
        self.y_data = torch.LongTensor(self.y_data)
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        x = self.x_data[idx].view([1,24,24])
        y = self.y_data[idx]
        return x,y


# 데이터 나누기!! -> train, valid, test
def split_dataset(
    dataset: Dataset, lengths: List[int], deterministic_partitions: bool = False
) -> List[Dataset]:
    """
    Split a dataset either randomly or deterministically.

    Args:
        dataset: the dataset to split
        lengths: the lengths of each partition
        deterministic_partitions: deterministic_partitions: whether to partition
            data in a deterministic fashion

    Returns:
        List[Dataset]: split datasets
    """
    if deterministic_partitions:
        indices = list(range(sum(lengths)))
    else:
        indices = torch.randperm(sum(lengths)).tolist()
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(accumulate(lengths), lengths)
    ]

def get_partition_data_loaders(
    train_valid_set: Dataset,
    test_set: Dataset,
    downsample_pct: float = 0.5,
    train_pct: float = 0.8,
    batch_size: int = 128,
    num_workers: int = 0,
    deterministic_partitions: bool = False,
    downsample_pct_test: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Helper function for partitioning training data into training and validation sets,
        downsampling data, and initializing DataLoaders for each partition.

    Args:
        train_valid_set: torch.dataset
        downsample_pct: the proportion of the dataset to use for training, and
            validation
        train_pct: the proportion of the downsampled data to use for training
        batch_size: how many samples per batch to load
        num_workers: number of workers (subprocesses) for loading data
        deterministic_partitions: whether to partition data in a deterministic
            fashion
        downsample_pct_test: the proportion of the dataset to use for test, default
            to be equal to downsample_pct

    Returns:
        DataLoader: training data
        DataLoader: validation data
        DataLoader: test data
    """
    # Partition into training/validation
    # pyre-ignore [6]
    downsampled_num_examples = int(downsample_pct * len(train_valid_set))
    n_train_examples = int(train_pct * downsampled_num_examples)
    n_valid_examples = downsampled_num_examples - n_train_examples
    train_set, valid_set, _ = split_dataset(
        dataset=train_valid_set,
        lengths=[
            n_train_examples,
            n_valid_examples,
            len(train_valid_set) - downsampled_num_examples,  # pyre-ignore [6]
        ],
        deterministic_partitions=deterministic_partitions,
    )
    if downsample_pct_test is None:
        downsample_pct_test = downsample_pct
    # pyre-ignore [6]
    downsampled_num_test_examples = int(downsample_pct_test * len(test_set))
    test_set, _ = split_dataset(
        test_set,
        lengths=[
            downsampled_num_test_examples,
            len(test_set) - downsampled_num_test_examples,  # pyre-ignore [6]
        ],
        deterministic_partitions=deterministic_partitions,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader, test_loader


# 특정 데이터셋 파일을 읽고 데이터로더 만들기
def load_trajectory(
    filename: str = "'cnn_data017ver1.json'", # 파일명 예시
    downsample_pct: float = 1.0,
    train_pct: float = 0.8,
    data_path: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    deterministic_partitions: bool = False,
    downsample_pct_test: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    dataset = Trajectory_Dataset(filename)
    
    train_valid_len = int(0.7*dataset.__len__())
    test_len = dataset.__len__() - train_valid_len
    train_valid_set, test_set = random_split(dataset, [train_valid_len, test_len])
    return get_partition_data_loaders(
        train_valid_set=train_valid_set,
        test_set=test_set,
        downsample_pct=downsample_pct,
        train_pct=train_pct,
        batch_size=batch_size,
        num_workers=num_workers,
        deterministic_partitions=deterministic_partitions,
        downsample_pct_test=downsample_pct_test,
    )


train_loader, valid_loader, test_loader = load_trajectory(filename)
print('train_loader len:',train_loader.dataset.__len__())
print('valid_loader len:',valid_loader.dataset.__len__())
print('test_loader len:',test_loader.dataset.__len__())


#model 정의하기
class CNN(torch.nn.Module):
    def __init__(self,dropout1_ratio,dropout2_ratio):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 24, 24, 1)
        #    Conv     -> (?, 24, 24, 32)
        #    Pool     -> (?, 12, 12, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.dropout1 = torch.nn.Dropout(p=dropout1_ratio)

        # 두번째층
        # ImgIn shape=(?, 12, 12, 32)
        #    Conv      ->(?, 12, 12, 64)
        #    Pool      ->(?, 6, 6, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        
        self.dropout2 = torch.nn.Dropout(p=dropout2_ratio)
        
        # 전결합층 6x6x64 inputs -> 24*24 outputs
        self.fc = torch.nn.Linear(6 * 6 * 64, 24*24, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.dropout1(out)
        out = self.layer2(out)
        out = self.dropout2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out

# 학습 
def train(
    net: torch.nn.Module,
    train_loader: DataLoader,
    parameters: Dict[str, float],
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """
    Train CNN on provided data set.

    Args:
        net: initialized neural network
        train_loader: DataLoader containing training set
        parameters: dictionary containing parameters to be passed to the optimizer.
            - lr: default (0.001)
            - momentum: default (0.0)
            - weight_decay: default (0.0)
            - num_epochs: default (1)
        dtype: torch dtype
        device: torch device
    Returns:
        nn.Module: trained CNN.
    """
    # Initialize network
    net.to(dtype=dtype, device=device)  # pyre-ignore [28]
    net.train()
    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    optimizer = optim.SGD(
        net.parameters(),
        lr=parameters.get("lr", 0.001),
        momentum=parameters.get("momentum", 0.0),
        weight_decay=parameters.get("weight_decay", 0.0),
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(parameters.get("step_size", 30)),
        gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
    )
    num_epochs = parameters.get("num_epochs", 100)

    # Train Network
    # pyre-fixme[6]: Expected `int` for 1st param but got `float`.
    total_batch = len(train_loader.dataset)
    for epoch in range(num_epochs):
        avg_cost = 0
        for inputs, labels in train_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            avg_cost += loss/total_batch
        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
    return net

# 평가
def evaluate(
    net: nn.Module, 
    data_loader: DataLoader, 
    dtype: torch.dtype, 
    device: torch.device
) -> float:
    """
    Compute classification accuracy on provided dataset.

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device
    Returns:
        float: classification accuracy
    """
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

# 학습+평가를 합쳐서 Bayesian Optimizer에 넣어줄 것이다.
def train_evaluate(parameterization):
    net = CNN(dropout1_ratio=0.2,dropout2_ratio=0.2).to(device)
    net = train(net=net, train_loader=train_loader, parameters=parameterization, dtype=dtype, device=device)
    return evaluate(
        net=net,
        data_loader=valid_loader,
        dtype=dtype,
        device=device,
    )

#without any hyper-parameters tuning
parameters=[
    {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
    {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
    {"name": "num_epochs", "fixed": 50}
]

# 대조군 하이퍼파라미터로 실험
# 최적화 전후 성능을 비교하기 위함 
train_evaluate({"num_epochs":100}) # 대조군 기본 값


# BAYESIAN OPTIMIZE 실행, 최고의 하이퍼파라미터를 구한다.
best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
    ],
    evaluation_function=train_evaluate,
    objective_name='accuracy',
)

# 얻은 값들로 다시 학습 시도 
train_evaluate(best_parameters)