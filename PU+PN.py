import argparse
import importlib
import random
import yaml
from tqdm import tqdm
import numpy as np

import torch
import torch.utils.data
from tensorboardX import SummaryWriter

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

import training
import settings
from utils import save_checkpoint, load_checkpoint
from newsgroups.cbs import generate_cbs_features

PATH = '/content/drive/MyDrive/PUbiasedN/model.pth'

parser = argparse.ArgumentParser(description='Main File')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Name of dataset: mnist, cifar10 or newsgroups')

parser.add_argument('--random-seed', type=int, default=None)
parser.add_argument('--params-path', type=str, default=None)
parser.add_argument('--ppe-save-path', type=str, default=None)
parser.add_argument('--ppe-load-path', type=str, default=None)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


prepare_data = importlib.import_module(f'{args.dataset}.pu_biased_n')
params = prepare_data.params
Net = prepare_data.Net
if args.dataset == 'newsgroups':
    NetCBS = prepare_data.NetCBS
train_data_orig = prepare_data.train_data
test_data_orig = prepare_data.test_data
train_labels_orig = prepare_data.train_labels
test_labels = prepare_data.test_labels


if args.params_path is not None:
    with open(args.params_path) as f:
        params_file = yaml.load(f)
    for key in params_file:
        params[key] = params_file[key]


if args.random_seed is not None:
    params['\nrandom_seed'] = args.random_seed

if args.ppe_save_path is not None:
    params['\nppe_save_name'] = args.ppe_save_path

if args.ppe_load_path is not None:
    params['ppe_load_name'] = args.ppe_load_path


num_classes = params['num_classes']
test_name1 = params['\test_name1']
test_name2 = params['\test_name2']
test_name3 = params['\test_name3']
test_name4 = params['\test_name4']
p_num = params['\np_num']
n_num = params.get('n_num', 0)
sn_num = params['sn_num']
u_num = params['u_num']

pv_num = params['\npv_num']
nv_num = params.get('nv_num', 0)
snv_num = params['snv_num']
uv_num = params['uv_num']

u_cut = params['\nu_cut']

pi = params['\npi']
rho = params['rho']
true_rho = params.get('true_rho', rho)
pi_prime = params['pi_prime']
rho_prime = params['rho_prime']

oversampling = params['\oversampling']

positive_classes = params['\npositive_classes']
negative_classes = params.get('negative_classes', None)
neg_ps = params['neg_ps']

non_pu_fraction = params['\nnon_pu_fraction']  # gamma
balanced = params['balanced']

u_per = params['\nu_per']  # tau
adjust_p = params['adjust_p']
adjust_sn = params['adjust_sn']

cls_training_epochs = params['\ncls_training_epochs']
convex_epochs = params['convex_epochs']

p_batch_size = params['\np_batch_size']
n_batch_size = params.get('n_batch_size', 0)
sn_batch_size = params['sn_batch_size']
u_batch_size = params['u_batch_size']

learning_rate_cls = params['\nlearning_rate_cls']
weight_decay = params['weight_decay']

if 'learning_rate_ppe' in params:
    learning_rate_ppe = params['learning_rate_ppe']
else:
    learning_rate_ppe = learning_rate_cls

milestones = params.get('milestones', [1000])
milestones_ppe = params.get('milestones_ppe', milestones)
lr_d = params.get('lr_d', 1)

non_negative = params['\nnon_negative']
nn_threshold = params['nn_threshold']  # beta
nn_rate = params['nn_rate']

cbs_feature = params.get('\ncbs_feature', False)
cbs_feature_later = params.get('cbs_feature_later', False)
cbs_alpha = params.get('cbs_alpha', 10)
cbs_beta = params.get('cbs_beta', 4)
n_select_features = params.get('n_select_features', 0)
svm = params.get('svm', False)
svm_C = params.get('svm_C', 1)

pu_prob_est = params['\npu_prob_est']
use_true_post = params['use_true_post']

partial_n = params['\npartial_n']  # PUbN
hard_label = params['hard_label']
PN_base = params.get('PN_base', False)
pn_then_pu = params.get('pn_then_pu', False)
pu_then_pn = params.get('pu_then_pn', False)  # PU -> PN

PUplusPN = params.get('PUplusPN', False)
iwpn = params['\niwpn']
pu = params['pu']
pnu = params['pnu']
unbiased_pn = params.get('unbiased_pn', False)

random_seed = params['\nrandom_seed']

ppe_save_name = params.get('\nppe_save_name', None)
ppe_load_name = params.get('ppe_load_name', None)

log_dir = 'logs/MNIST'
visualize = False

priors = params.get('\npriors', None)
if priors is None:
    priors = [1/num_classes for _ in range(num_classes)]


settings.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


for key, value in params.items():
    print('{}: {}'.format(key, value))
print('\nvalidation_interval', settings.validation_interval)
print('', flush=True)


def posteriors(labels):
    posteriors = torch.zeros(labels.size())
    for i in range(num_classes):
        if i in positive_classes:
            posteriors[labels == i] = 1
        else:
            posteriors[labels == i] = neg_ps[i] * rho * 1/priors[i]
    return posteriors.unsqueeze(1)


def pick_p_data(data, labels, n, u_num, sn_num, pi, rho, pn_then_pu, pu_then_pn, pu, pnu, iwpn, partial_n, pu_prob_est, oversampling):
    p_idxs = np.zeros_like(labels)
    for i in range(num_classes):
        if i in positive_classes:
            p_idxs[(labels == i).numpy().astype(bool)] = 1
    p_idxs = np.argwhere(p_idxs == 1).reshape(-1)
    selected_p = np.random.choice(p_idxs, n, replace=False)

    orginal_x = data[selected_p]
    orginal_y = labels[selected_p]
    if oversampling:
        x_pn = data[selected_p]
        y_pn = labels[selected_p]
        for i in tqdm(range(3)):
                # generated_x = torch.empty((1, 3, 32, 32))
                generated_x = []
                for x,y in zip(orginal_x, orginal_y):
                    # x = x + np.random.normal(0, 0.01, size=(1,28,28))
                    x = x + np.random.normal(0, 0.01, size=(3,32,32))
                    generated_x.append(x)
                    # generated_x = torch.cat((generated_x,x.unsqueeze(dim=0)), dim=0)
                    y_pn = torch.cat((y_pn, y.unsqueeze(dim=0)), dim=0)
                generated_x = torch.stack(generated_x, dim=0)
                x_pn = torch.cat((x_pn,generated_x), dim=0)

        if pn_then_pu or pu:
            x_pu = data[selected_p]
            y_pu = labels[selected_p]
            for i in tqdm(range(int(((u_num*(1-pi)-u_num*pi)/n)-1))):
                generated_x = []
                for x,y in zip(orginal_x, orginal_y):
                    # x = x + np.random.normal(0, 0.01, size=(1,28,28))
                    x = x + np.random.normal(0, 0.01, size=(3,32,32))
                    generated_x.append(x)
                    y_pu = torch.cat((y_pu, y.unsqueeze(dim=0)), dim=0)
                generated_x = torch.stack(generated_x, dim=0)
                x_pu = torch.cat((x_pu,generated_x), dim=0)
            return x_pn, posteriors(y_pn), x_pu, posteriors(y_pu)

        if pu_then_pn or pu_prob_est:
            x_pux = data[selected_p]
            y_pux = labels[selected_p]
            N = u_num * (1-pi-rho)
            y = N - u_num*pi - u_num*rho
            for i in tqdm(range(int((y*(pi/(pi+rho)))/n)-1)):
                # generated_x = torch.empty((1, 3, 32, 32))
                generated_x = []
                for x,y in zip(orginal_x, orginal_y):
                    # x = x + np.random.normal(0, 0.01, size=(1,28,28))
                    x = x + np.random.normal(0, 0.01, size=(3,32,32))
                    generated_x.append(x)
                    y_pux = torch.cat((y_pux, y.unsqueeze(dim=0)), dim=0)
                generated_x = torch.stack(generated_x, dim=0)
                x_pux = torch.cat((x_pux,generated_x), dim=0)
            return orginal_x, posteriors(orginal_y), x_pux, posteriors(y_pux)

        if pnu:
            x_pnu = data[selected_p]
            y_pnu = labels[selected_p]
            N = u_num*(1-pi) + sn_num
            for i in tqdm(range(int((N-(u_num*pi+n))/n))):
                # generated_x = torch.empty((1, 3, 32, 32))
                generated_x = []
                for x,y in zip(orginal_x, orginal_y):
                    # x = x + np.random.normal(0, 0.01, size=(1,28,28))
                    x = x + np.random.normal(0, 0.01, size=(3,32,32))
                    generated_x.append(x)
                    y_pnu = torch.cat((y_pnu, y.unsqueeze(dim=0)), dim=0)
                generated_x = torch.stack(generated_x, dim=0)
                x_pnu = torch.cat((x_pnu,generated_x), dim=0)
            return orginal_x, posteriors(orginal_y), x_pnu, posteriors(y_pnu)
    else:
        return orginal_x, posteriors(orginal_y), orginal_x, posteriors(orginal_y)


def pick_n_data(data, labels, n):
    n_idxs = np.zeros_like(labels)
    for i in range(num_classes):
        if negative_classes is None:
            if i not in positive_classes:
                n_idxs[(labels == i).numpy().astype(bool)] = 1
        else:
            if i in negative_classes:
                n_idxs[(labels == i).numpy().astype(bool)] = 1
    n_idxs = np.argwhere(n_idxs == 1).reshape(-1)
    selected_n = np.random.choice(n_idxs, n, replace=False)
    return data[selected_n], labels[selected_n]


def pick_sn_data(data, labels, n,  p_num, u_num, pi, rho, pn_then_pu, pu_then_pn, partial_n, pu_prob_est, oversampling):
    neg_nums = np.random.multinomial(n, neg_ps)
    print('numbers in each subclass', neg_nums)
    selected_sn = []
    for i in range(num_classes):
        if neg_nums[i] != 0:
            idxs = np.argwhere(labels == i).reshape(-1)
            selected = np.random.choice(idxs, neg_nums[i], replace=False)
            selected_sn.extend(selected)
    selected_sn = np.array(selected_sn)
    orginal_x = data[selected_sn]
    orginal_y = labels[selected_sn]
    x_sn = data[selected_sn]
    y_sn = labels[selected_sn]
    if oversampling:
      for i in tqdm(range(int((p_num*4-n)/n))):
          generated_x = []
          for x,y in zip(orginal_x, orginal_y):
              # x = x + np.random.normal(0, 0.01, size=(1,28,28))
              x = x + np.random.normal(0, 0.01, size=(3,32,32))
              generated_x.append(x)
              y_sn = torch.cat((y_sn, y.unsqueeze(dim=0)), dim=0)
          generated_x = torch.stack(generated_x, dim=0)
          x_sn = torch.cat((x_sn,generated_x), dim=0)
      if pu_then_pn or pu_prob_est:
          x_snx = data[selected_sn]
          y_snx = labels[selected_sn]
          N = u_num * (1-pi-rho)
          y = N - u_num*pi - u_num*rho
          for i in tqdm(range(int((y*(rho/(pi+rho)))/n)-1)):
              generated_x = []
              for x,y in zip(orginal_x, orginal_y):
                  # x = x + np.random.normal(0, 0.01, size=(1,28,28))
                  x = x + np.random.normal(0, 0.01, size=(3,32,32))
                  generated_x.append(x)
                  y_snx = torch.cat((y_snx, y.unsqueeze(dim=0)), dim=0)
              generated_x = torch.stack(generated_x, dim=0)
              x_snx = torch.cat((x_snx,generated_x), dim=0)
          return orginal_x, posteriors(orginal_y), x_snx, posteriors(y_snx),   
      return x_sn, posteriors(y_sn), orginal_x, posteriors(orginal_y)
    else:
        return x_sn, posteriors(y_sn), orginal_x, posteriors(orginal_y)



def pick_u_data(data, labels, n):
    if negative_classes is None:
        selected_u = np.random.choice(len(data), n, replace=False)
    else:
        u_idxs = np.zeros_like(labels)
        for i in range(num_classes):
            if i in positive_classes or i in negative_classes:
                u_idxs[(labels == i).numpy().astype(bool)] = 1
        u_idxs = np.argwhere(u_idxs == 1).reshape(-1)
        selected_u = np.random.choice(u_idxs, n, replace=False)
    return data[selected_u], posteriors(labels[selected_u])


t_labels = torch.zeros(test_labels.size())

for i in range(num_classes):
    if i in positive_classes:
        t_labels[test_labels == i] = 1
    elif negative_classes is None or i in negative_classes:
        t_labels[test_labels == i] = -1
    else:
        t_labels[test_labels == i] = 0


t_labels_pu_star = torch.zeros(test_labels.size())

for i in range(num_classes):
    if i in positive_classes:
        t_labels_pu_star[test_labels == i] = 1
    if i in negative_classes:
        if (neg_ps[i] * rho * 1/priors[i]) == 1:
            t_labels_pu_star[test_labels == i] = 1
        else:
            t_labels_pu_star[test_labels == i] = -1


def pick_p_data_val(data, labels, n):
    p_idxs = np.zeros_like(labels)
    for i in range(num_classes):
        if i in positive_classes:
            p_idxs[(labels == i).numpy().astype(bool)] = 1
    p_idxs = np.argwhere(p_idxs == 1).reshape(-1)
    selected_p = np.random.choice(p_idxs, n, replace=False)
    return data[selected_p], posteriors(labels[selected_p])


def pick_sn_data_val(data, labels, n):
    neg_nums = np.random.multinomial(n, neg_ps)
    print('numbers in each subclass', neg_nums)
    selected_sn = []
    for i in range(num_classes):
        if neg_nums[i] != 0:
            idxs = np.argwhere(labels == i).reshape(-1)
            selected = np.random.choice(idxs, neg_nums[i], replace=False)
            selected_sn.extend(selected)
    return data[selected_sn], posteriors(labels[selected_sn])



np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True


idxs = np.random.permutation(len(train_data_orig))

valid_data = train_data_orig[idxs][u_cut:]
valid_labels = train_labels_orig[idxs][u_cut:]
train_data = train_data_orig[idxs][:u_cut]
train_labels = train_labels_orig[idxs][:u_cut]


u_data, u_pos = pick_u_data(train_data, train_labels, u_num)
p_data, p_pos, pu_data, pu_pos = pick_p_data(train_data, train_labels, p_num, u_num, sn_num, pi, rho, pn_then_pu, pu_then_pn, pu, pnu, iwpn, partial_n, pu_prob_est, oversampling)
sn_data, sn_pos, snu_data, snu_pos = pick_sn_data(train_data, train_labels, sn_num,  p_num, u_num, pi, rho, pn_then_pu, pu_then_pn, partial_n, pu_prob_est, oversampling)

print("u_data = ",u_data.size())
print("p_data = ",p_data.size(), "pu_data = ",pu_data.size())
print("sn_data = ",sn_data.size(), "snu_data = ",snu_data.size())

uv_data, uv_pos = pick_u_data(valid_data, valid_labels, uv_num)
pv_data, pv_pos = pick_p_data_val(valid_data, valid_labels, pv_num)
snv_data, snv_pos = pick_sn_data_val(valid_data, valid_labels, snv_num)

if cbs_feature:
    cbs_features = generate_cbs_features(
        p_data, sn_data, u_data,
        pv_data, snv_data, uv_data,
        test_data_orig, n_select_features=n_select_features,
        alpha=cbs_alpha, beta=cbs_beta)
    p_data, sn_data, u_data, pv_data, snv_data, uv_data, test_data = \
        [torch.tensor(data) for data in cbs_features]
    Net = NetCBS
else:
    test_data = test_data_orig



u_set = torch.utils.data.TensorDataset(u_data, u_pos)
u_validation = uv_data, uv_pos

p_set = torch.utils.data.TensorDataset(p_data, p_pos)
p_validation = pv_data, pv_pos

sn_set = torch.utils.data.TensorDataset(sn_data, sn_pos)
sn_validation = snv_data, snv_pos

pu_set = torch.utils.data.TensorDataset(pu_data, pu_pos)
snu_set = torch.utils.data.TensorDataset(snu_data, snu_pos)



if not cbs_feature:
    # Not considering cbs feature here
    n_set = torch.utils.data.TensorDataset(
        *pick_n_data(train_data, train_labels, n_num))
    n_validation = pick_n_data(valid_data, valid_labels, nv_num)


test_posteriors = posteriors(test_labels)
test_idxs = np.argwhere(t_labels != 0).reshape(-1)

test_set = torch.utils.data.TensorDataset(
    test_data[test_idxs],
    t_labels.unsqueeze(1).float()[test_idxs],
    test_posteriors[test_idxs])


test_posteriors_pu_star = t_labels_pu_star
test_idxs2 = np.argwhere(t_labels_pu_star != 0).reshape(-1)

test_set_pu_star = torch.utils.data.TensorDataset(
    test_data[test_idxs2],
    t_labels_pu_star.unsqueeze(1).float()[test_idxs2],
    test_posteriors_pu_star[test_idxs2])


if svm:
    clf = LinearSVC(max_iter=5000)
    labels = np.concatenate(
        [np.ones(p_data.shape[0]), -np.ones(sn_data.shape[0])])
    clf.fit(np.concatenate([p_data, sn_data]), labels)
    print('Accuracy: {:.2f}'.format(
          clf.score(test_data, t_labels.numpy())*100))
    print('F1-score: {:.2f}'.format(
          f1_score(t_labels.numpy(), clf.predict(test_data))*100))


if pu_prob_est and ppe_load_name is None:
    print('')
    print(" ===> PU* <=== ")
    model = Net().cuda() if args.cuda else Net()
    ppe = training.PUClassifier3(
            model, pi=pi, pi_prime=0.5*(pi/(pi+rho)), rho=rho, rho_prime=0.5*(rho/(pi+rho)),  oversampling=oversampling,
            lr=learning_rate_ppe, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            milestones=milestones, lr_d=lr_d, prob_est=True, test_name=test_name1)
    ppe.train(pu_set, snu_set, u_set, test_set_pu_star,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)
    if ppe_save_name is not None:
        save_checkpoint(ppe.model, cls_training_epochs, ppe_save_name)
    ppe_model = ppe.model


if ppe_load_name is not None:
    ppe_model = Net().cuda() if args.cuda else Net()
    ppe_model = load_checkpoint(ppe_model, ppe_load_name)



#<<< PU*PubN >>>
# balanced = False
# PUplusPN = False
# iwpn = False
# PN_base = False
# partial_n = True
# adjust_p = True
# adjust_sn = True


if (partial_n or (iwpn and (adjust_p or adjust_sn))) and not use_true_post:
    pu_setx = torch.utils.data.TensorDataset(
        pu_set.tensors[0], torch.sigmoid(
            training.Training()
            .feed_in_batches(ppe_model, pu_set.tensors[0])).cpu())
    snu_setx = torch.utils.data.TensorDataset(
        snu_set.tensors[0], torch.sigmoid(
            training.Training()
            .feed_in_batches(ppe_model, snu_set.tensors[0])).cpu())
    u_setx = torch.utils.data.TensorDataset(
        u_set.tensors[0], torch.sigmoid(
            training.Training()
            .feed_in_batches(ppe_model, u_set.tensors[0])).cpu())
    p_validationx = p_validation[0], torch.sigmoid(
        training.Training()
        .feed_in_batches(ppe_model, p_validation[0])).cpu()
    sn_validationx = sn_validation[0], torch.sigmoid(
        training.Training()
        .feed_in_batches(ppe_model, sn_validation[0])).cpu()
    u_validationx = u_validation[0], torch.sigmoid(
        training.Training()
        .feed_in_batches(ppe_model, u_validation[0])).cpu()


# eta
sep_value = np.percentile(
    u_set.tensors[1].numpy().reshape(-1), int((1-pi-true_rho)*u_per*100))
print('\nsep_value =', sep_value)


if cbs_feature_later:
    cbs_features = generate_cbs_features(
        p_data.numpy(), sn_data.numpy(), u_data.numpy(),
        pv_data.numpy(), snv_data.numpy(), uv_data.numpy(),
        test_data.numpy(), n_select_features=n_select_features,
        alpha=cbs_alpha, beta=cbs_beta)
    p_data, sn_data, u_data, pv_data, snv_data, uv_data, test_data = \
        [torch.tensor(data) for data in cbs_features]
    Net = NetCBS

u_setx = torch.utils.data.TensorDataset(u_data, u_setx.tensors[1])
u_validationx = uv_data, u_validationx[1]

p_setx = torch.utils.data.TensorDataset(pu_data, pu_setx.tensors[1])
p_validationx = pv_data, p_validationx[1]

sn_setx = torch.utils.data.TensorDataset(snu_data, snu_setx.tensors[1])
sn_validationx = snv_data, sn_validationx[1]

test_setx = torch.utils.data.TensorDataset(
    test_data[test_idxs], test_set.tensors[1], test_set.tensors[2])


if partial_n:
    print('')
    print(" ===> PubN <=== ")
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUbNClassifier(
            model, balanced=balanced, pi=pi, pi_prime=0.5*(pi/(pi+rho)), rho=rho,
            rho_prime=0.5*(rho/(pi+rho)),  oversampling=oversampling,
            sep_value=sep_value,
            adjust_p=adjust_p, adjust_sn=adjust_sn, hard_label=hard_label,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d, test_name=test_name2)
    cls.train(p_setx, sn_setx, u_setx, test_setx,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validationx, sn_validationx, u_validationx,
              cls_training_epochs, convex_epochs=convex_epochs)



# <<< PUplusPN >>>
balanced = True
PUplusPN = True
iwpn = True
PN_base = False
partial_n = False
adjust_p = False
adjust_sn = False



if PUplusPN:
    ux = u_set.tensors[0].cpu()
    fux_prob = torch.sigmoid(training.Training().feed_in_batches(ppe_model, u_set.tensors[0])).type(settings.dtype).cpu()
    mask = fux_prob <= 0.01
    ux = ux[mask.squeeze()]
    print('Number of used unlabeled samples: ',len(ux))
    sn_data_new = torch.cat((ux, sn_data), 0)
    p_data_new = p_data
    for i in tqdm(range(int((len(sn_data_new)-len(p_data))/len(p_data)))):
          generated_x = []
          for x in p_data:
              # x = x + np.random.normal(0, 0.01, size=(1,28,28))
              x = x + np.random.normal(0, 0.01, size=(3,32,32))
              generated_x.append(x)
              # y_p = torch.cat((y_sn, y.unsqueeze(dim=0)), dim=0)
          generated_x = torch.stack(generated_x, dim=0)
          p_data_new = torch.cat((p_data_new,generated_x), dim=0)
    
    p_set_new = torch.utils.data.TensorDataset(p_data_new)
    sn_set_new = torch.utils.data.TensorDataset(sn_data_new)
    print("Number of Oversampled P and SN data",len(p_set_new),len(sn_set_new))

if iwpn:
    print('')
    print(" ===> PuplusPN (PU*PN+Unknown-N) <=== ")
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNClassifier(
            model, pi=pi/(pi+rho),
            adjust_p=adjust_p, adjust_n=adjust_sn,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d, test_name=test_name3)
    cls.train(p_set_new, sn_set_new, test_set, p_batch_size, sn_batch_size,
              p_validation, sn_validation,
              cls_training_epochs, convex_epochs=convex_epochs)




# <<< PU*PN >>>
# balanced = True
# PUplusPN = False
# iwpn = False
# PN_base = True
# partial_n = False
# adjust_p = False
# adjust_sn = False


if PN_base:
    sn_data_new = sn_data
    p_data_new = p_data
    for i in tqdm(range(3)):
        generated_x = []
        for x in p_data:
            # x = x + np.random.normal(0, 0.01, size=(1,28,28))
            x = x + np.random.normal(0, 0.01, size=(3,32,32))
            generated_x.append(x)
        generated_x = torch.stack(generated_x, dim=0)
        p_data_new = torch.cat((p_data_new,generated_x), dim=0)
    for i in tqdm(range(int((len(p_data_new)-len(sn_data))/len(sn_data)))):
        # generated_x = torch.empty((1, 3, 32, 32))
        generated_x = []
        for x in sn_data:
            # x = x + np.random.normal(0, 0.01, size=(1,28,28))
            x = x + np.random.normal(0, 0.01, size=(3,32,32))
            generated_x.append(x)
        generated_x = torch.stack(generated_x, dim=0)
        sn_data_new = torch.cat((sn_data_new,generated_x), dim=0)
    p_set_new = torch.utils.data.TensorDataset(p_data_new)
    sn_set_new = torch.utils.data.TensorDataset(sn_data_new)
    print("Number of Oversampled P and SN data",len(p_set_new),len(sn_set_new))

    print('')
    print(" ===> PU*PN <=== ")
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNClassifier(
            model, pi=pi/(pi+rho), pu_model=ppe_model,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d, test_name=test_name4)
    cls.train(p_set_new, sn_set_new, test_set, p_batch_size, sn_batch_size,
               p_validation, sn_validation,
               cls_training_epochs, convex_epochs=convex_epochs)


if pn_then_pu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNClassifier(
            model, pi=pi/(pi+rho),
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d)
    cls.train(p_set, sn_set, test_set, p_batch_size, sn_batch_size,
              p_validation, sn_validation,
              cls_training_epochs, convex_epochs=convex_epochs)
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls2 = training.PUClassifier(
            model, pn_model=cls.model, pi=pi, balanced=balanced,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls2.train(p_set, u_set, test_set, p_batch_size, u_batch_size,
               p_validation, u_validation,
               cls_training_epochs, convex_epochs=convex_epochs)

if pu_then_pn:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUClassifier3(
            model, pi=pi, rho=rho,
            lr=learning_rate_ppe, weight_decay=weight_decay,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate,
            milestones=milestones, lr_d=lr_d)
    cls.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

    print('')
    model = Net().cuda() if args.cuda else Net()
    cls2 = training.PNClassifier(
            model, pi=pi/(pi+rho), pu_model=cls.model,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d)
    cls2.train(p_set, sn_set, test_set, p_batch_size, sn_batch_size,
               p_validation, sn_validation,
               cls_training_epochs, convex_epochs=convex_epochs)

if pu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUClassifier(
            model, pi=pi, balanced=balanced,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, u_set, test_set, p_batch_size, u_batch_size,
              p_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if pnu:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNUClassifier(
            model, pi=pi,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            pn_fraction=non_pu_fraction,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, sn_set, u_set, test_set,
              p_batch_size, sn_batch_size, u_batch_size,
              p_validation, sn_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

if unbiased_pn:
    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PNClassifier(
            model, pi=pi,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d)
    cls.train(p_set, sn_set, test_set, p_batch_size, sn_batch_size,
              p_validation, sn_validation,
              cls_training_epochs, convex_epochs=convex_epochs)


n_embedding_points = 500

if visualize:

    indx = np.random.choice(test_data.size(0), size=n_embedding_points)

    embedding_data = test_data[indx]
    embedding_labels = t_labels.numpy().copy()
    # Negative data that are not sampled
    embedding_labels[test_posteriors.numpy().flatten() < 1/2] = 0
    embedding_labels = embedding_labels[indx]
    features = cls.last_layer_activation(embedding_data)
    writer = SummaryWriter(log_dir=log_dir)
    # writer.add_embedding(embedding_data.view(n_embedding_points, -1),
    #                      metadata=embedding_labels,
    #                      tag='Input', global_step=0)
    writer.add_embedding(features, metadata=embedding_labels,
                         tag='PUbN Features',
                         global_step=cls_training_epochs)
