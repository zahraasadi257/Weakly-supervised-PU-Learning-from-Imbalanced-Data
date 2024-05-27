from collections import OrderedDict
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import settings


num_classes = 10


p_num = 1000
sn_num = 500
u_num = 45000

pv_num = 100
snv_num = 100
uv_num = 10

u_cut = 45000


pi_prime = 0.5
rho_prime =0.5

non_pu_fraction = 0.5


u_per = 0.7


cls_training_epochs = 50
convex_epochs = 50

p_batch_size = 256
sn_batch_size = 256
u_batch_size = 512

learning_rate_ppe = 1e-4
learning_rate_cls = 1e-4
weight_decay = 1e-5

milestones = [80, 120]
lr_d = 0.1

non_negative = True
nn_threshold = 0
nn_rate = 1



#### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Hyperparameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

pi = 0.1
true_rho = 0.1
rho = 0.1
positive_classes = [1]
negative_classes = [0,2,3,4,5,6,7,8,9]
neg_ps =[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

#Dataset-Positive Class-Oversampling/None-Part\Model-#SN Classes-#SN Samples
test_name1= "CIFAR10-1-O-PU*-1-500"
test_name2 = 'CIFAR10-1-O-PUbN-1-500'
test_name3 = 'CIFAR10-1-O-PUplusPN-1-500'
test_name4 = 'CIFAR10-1-O-PUPN-1-500'

oversampling = True

pu_prob_est = True #Always True

ppe_load_name = None #First time running PU*
# ppe_load_name = '/content/drive/MyDrive/PUbiasedN/weights/CIFAR10_7_O_PU-1-500' #Other times

#  #Dataset-Positive class-Oversampling/None-Part\Model-#SN classes-#SN samples
# ppe_save_name = None
ppe_save_name = '/content/drive/MyDrive/PUbiasedN/weights/CIFAR10_1_O_PU-1-500'

#######################################################################################################################################


# <<< PUplusPN >>>
balanced = True
PUplusPN = True
iwpn = True
PN_base = False
partial_n = False
adjust_p = False
adjust_sn = False

# <<< PU*PN >>>
# balanced = True
# PUplusPN = False
# iwpn = False
# PN_base = True
# partial_n = False
# adjust_p = False
# adjust_sn = False

#<<< PU*PubN >>>
# balanced = False
# PUplusPN = False
# iwpn = False
# PN_base = False
# partial_n = True
# adjust_p = True
# adjust_sn = True



use_true_post = False
hard_label = False
pu_then_pn = False
pu = False
pnu = False

random_seed = 0

settings.test_batch_size = 1024
settings.validation_interval = 100


params = OrderedDict([
    ('num_classes', num_classes),
    ('\test_name1',test_name1),
    ('\test_name2',test_name2),
    ('\test_name3',test_name3),
    ('\test_name4',test_name4),
    ('\np_num', p_num),
    ('sn_num', sn_num),
    ('u_num', u_num),
    ('\npv_num', pv_num),
    ('snv_num', snv_num),
    ('uv_num', uv_num),
    ('\nu_cut', u_cut),
    ('\oversampling',oversampling),
    ('\npi', pi),
    ('rho', rho),
    ('pi_prime', pi_prime),
    ('rho_prime', rho_prime),
    ('true_rho', true_rho),
    ('\npositive_classes', positive_classes),
    ('negative_classes', negative_classes),
    ('neg_ps', neg_ps),
    ('\nnon_pu_fraction', non_pu_fraction),
    ('balanced', balanced),
    ('\nu_per', u_per),
    ('adjust_p', adjust_p),
    ('adjust_sn', adjust_sn),
    ('\ncls_training_epochs', cls_training_epochs),
    ('convex_epochs', convex_epochs),
    ('\np_batch_size', p_batch_size),
    ('sn_batch_size', sn_batch_size),
    ('u_batch_size', u_batch_size),
    ('\nlearning_rate_cls', learning_rate_cls),
    ('learning_rate_ppe', learning_rate_ppe),
    ('weight_decay', weight_decay),
    ('milestones', milestones),
    ('lr_d', lr_d),
    ('\nnon_negative', non_negative),
    ('nn_threshold', nn_threshold),
    ('nn_rate', nn_rate),
    ('\npu_prob_est', pu_prob_est),
    ('use_true_post', use_true_post),
    ('\npartial_n', partial_n),
    ('hard_label', hard_label),
    ('\niwpn', iwpn),
    ('pu_then_pn', pu_then_pn),
    ('PUplusPN', PUplusPN),
    ('PN_base', PN_base),
    ('pu', pu),
    ('pnu', pnu),
    ('\nrandom_seed', random_seed),
    ('\nppe_save_name', ppe_save_name),
    ('ppe_load_name', ppe_load_name),
])


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Load and transform data
cifar10 = torchvision.datasets.CIFAR10(
    './data/CIFAR10', train=True, download=True, transform=transform)

cifar10_test = torchvision.datasets.CIFAR10(
    './data/CIFAR10', train=False, download=True, transform=transform)


train_data = torch.zeros(cifar10.train_data.shape)
train_data = train_data.permute(0, 3, 1, 2)
# must use one dimensional vector
train_labels = torch.tensor(cifar10.train_labels)

for i, (image, _) in enumerate(cifar10):
    train_data[i] = image

test_data = torch.zeros(cifar10_test.test_data.shape)
test_data = test_data.permute(0, 3, 1, 2)
test_labels = torch.tensor(cifar10_test.test_labels)

for i, (image, _) in enumerate(cifar10_test):
    test_data[i] = image

# Net = PreActResNet18

class Net(nn.Module):
    def __init__(self, num_classes=1):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3, 96, 3, padding=1)
        self.conv2=nn.Conv2d(96, 96, 3, padding=1)
        self.conv3=nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4=nn.Conv2d(96, 192, 3, padding=1)
        self.conv5=nn.Conv2d(192, 192, 3, padding=1)
        self.conv6=nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7=nn.Conv2d(192, 192, 3, padding=1)
        self.conv8=nn.Conv2d(192, 192, 1)
        self.conv9=nn.Conv2d(192, 10, 1)
        self.b1=nn.BatchNorm2d(96)
        self.b2=nn.BatchNorm2d(96)
        self.b3=nn.BatchNorm2d(96)
        self.b4=nn.BatchNorm2d(192)
        self.b5=nn.BatchNorm2d(192)
        self.b6=nn.BatchNorm2d(192)
        self.b7=nn.BatchNorm2d(192)
        self.b8=nn.BatchNorm2d(192)
        self.b9=nn.BatchNorm2d(10)
        self.fc1=nn.Linear(10*8*8, 1000)
        self.fc2=nn.Linear(1000, 1000)
        self.fc3=nn.Linear(1000, 1)
        self.af = F.relu

    def forward(self, x):
        h = self.conv1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.conv2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.conv3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.conv4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.conv5(h)
        h = self.b5(h)
        h = self.af(h)
        h = self.conv6(h)
        h = self.b6(h)
        h = self.af(h)
        h = self.conv7(h)
        h = self.b7(h)
        h = self.af(h)
        h = self.conv8(h)
        h = self.b8(h)
        h = self.af(h)
        h = self.conv9(h)
        h = self.b9(h)
        h = self.af(h)
        h = h.contiguous().view(-1, 8*8*10)
        h = self.fc1(h)
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        h = self.fc3(h)
        return h
