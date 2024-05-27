from collections import OrderedDict
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
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

p_batch_size = 125
sn_batch_size = 125
u_batch_size = 250

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
positive_classes = [0]
negative_classes = [1,2,3,4,5,6,7,8,9]
neg_ps =[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

#Dataset-Positive Class-Oversampling/None-Part\Model-#SN Classes-#SN Samples
test_name1= "mnist-0-O-PU*-1-500"
test_name2 = 'mnist-0-O-PUbN-1-500'
test_name3 = 'mnist-0-O-PUplusPN-1-500'
test_name4 = 'mnist-0-O-PUPN-1-500'

oversampling = True

pu_prob_est = True #Always True

ppe_load_name = None #First time running PU*
# ppe_load_name = '/content/drive/MyDrive/PUbiasedN/weights/mnist_7_O_PU-1-500' #Other times

#  #Dataset-Positive class-Oversampling/None-Part\Model-#SN classes-#SN samples
# ppe_save_name = None
ppe_save_name = '/content/drive/MyDrive/PUbiasedN/weights/mnist_1_O_PU-1-500'

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

settings.test_batch_size = 500
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


# torchvision.datasets.MNIST outputs a set of PIL images
# We transform them to tensors
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load and transform data
mnist = torchvision.datasets.MNIST(
    './data/MNIST', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(
    './data/MNIST', train=False, download=True, transform=transform)


train_data = torch.zeros(mnist.train_data.size())

for i, (image, _) in enumerate(mnist):
    train_data[i] = image

train_data = train_data.unsqueeze(1)
train_labels = mnist.train_labels

test_data = torch.zeros(mnist_test.test_data.size())

for i, (image, _) in enumerate(mnist_test):
    test_data[i] = image

test_data = test_data.unsqueeze(1)
test_labels = mnist_test.test_labels



class Net(nn.Module):

    def __init__(self, num_classes=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, 1)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, 5, 1)
        self.bn2 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(4*4*10, 40)
        self.fc2 = nn.Linear(40, num_classes)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*10)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
