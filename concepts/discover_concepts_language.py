from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from concepts import stat_ap

DATA_DIR = './Data/'


def get_convert_matrix(verb_class_num=117, obj_class_num=80):
    import pickle
    import numpy as np
    verb_to_HO_matrix = np.zeros((600, verb_class_num), np.float32)
    hoi_to_vb = pickle.load(open(DATA_DIR + '/Data/' + '/hoi_to_vb.pkl', 'rb'))
    for k, v in hoi_to_vb.items():
        verb_to_HO_matrix[k][v] = 1
    verb_to_HO_matrix = np.transpose(verb_to_HO_matrix)

    obj_to_HO_matrix = np.zeros((600, obj_class_num), np.float32)
    hoi_to_obj = pickle.load(open(DATA_DIR + '/Data/' + '/hoi_to_obj.pkl', 'rb'))
    for k, v in hoi_to_obj.items():
        obj_to_HO_matrix[k][v] = 1
    obj_to_HO_matrix = np.transpose(obj_to_HO_matrix)
    return verb_to_HO_matrix, obj_to_HO_matrix

class ConceptModel(torch.nn.Module):

    def __init__(self, input_dim, out_dim,  concept_emb=None):
        super(ConceptModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.Linear(1024, out_dim),
        )
        # self.fc = torch.nn.Linear(input_dim, out_dim)
        self.concept_emb =  concept_emb

    def forward(self, inputs):
        return self.fc(inputs)


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)


DATA_DIR1 = './misc/'
# Load data
features_verb = np.load(DATA_DIR1 + "./verb_embedding.npy")
features_obj = np.load(DATA_DIR1 + "./obj_embedding.npy")
features_obj = np.expand_dims(features_obj, axis=0)
features_obj = np.tile(features_obj, [117, 1, 1])
features_obj = features_obj.reshape([-1, 300])
features_verb = np.expand_dims(features_verb, axis=1)
features_verb = np.tile(features_verb, [1, 80, 1])
features_verb = features_verb.reshape([-1, 300])
verb_label = np.eye(117)
verb_label = np.expand_dims(verb_label, axis=1)
verb_label = np.tile(verb_label, [1, 80, 1])
verb_label = np.reshape(verb_label, [-1, 117])

obj_label = np.eye(80)
obj_label = np.expand_dims(obj_label, axis=0)
obj_label = np.tile(obj_label, [117, 1, 1])
obj_label = np.reshape(obj_label, [-1, 80])

features = np.concatenate([features_verb, features_obj], axis=1)

IS_HOI_PREDICTION=False

# Model and optimizer
model = ConceptModel(input_dim=features.shape[1], out_dim=600 if IS_HOI_PREDICTION else 117)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix(117, 80)
fix_verb_obj_matrix = np.asarray(np.matmul(verb_to_HO_matrix, obj_to_HO_matrix.transpose()) > 0., np.float32)

idx_train = np.zeros(117*80, np.float32)
for i in range(600):
    v = np.argwhere(verb_to_HO_matrix[:, i])
    o = np.argwhere(obj_to_HO_matrix[:, i])
    idx_train[v*80+o] = 1.

idx_val = np.asarray(idx_train - 1., np.bool)
idx_train = np.asarray(idx_train, np.bool)

# import ipdb;ipdb.set_trace()
adj = np.matmul(np.concatenate([verb_label, obj_label], axis=-1),
                np.concatenate([verb_label, obj_label], axis=-1).transpose()) >= 1.
adj = np.asarray(adj, np.float32)
adj = adj / np.sum(adj, axis=-1)

if IS_HOI_PREDICTION:
    labels = (np.matmul(verb_label, verb_to_HO_matrix) + np.matmul(obj_label, obj_to_HO_matrix) >= 1.).astype(np.float32)
else:
    labels = verb_label.astype(np.float32)
idx_train = idx_train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.cuda:
    model.to(device)
    features = torch.from_numpy(features).to(device)
    adj = torch.from_numpy(adj).to(device)
    labels = torch.from_numpy(labels).to(device)
    idx_train = torch.from_numpy(idx_train).to(device)
    idx_val = torch.from_numpy(idx_val).to(device)


def train(epoch):
    t = time.time()
    optimizer.zero_grad()
    # output, support = model((features, adj))
    output = model(features)
    # import ipdb;ipdb.set_trace()
    loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train])
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
    model.eval()
    # output, support = model((features, adj))
    output = model(features)
    # import ipdb;ipdb.set_trace()
    preds = torch.sigmoid(output) * labels

    if IS_HOI_PREDICTION:
        v_hoi = torch.from_numpy(verb_to_HO_matrix.transpose()).to(device)
        preds = torch.matmul(preds, v_hoi) / torch.matmul(torch.ones_like(preds), v_hoi)
    preds = torch.sum(preds, dim=-1)
    preds = preds.squeeze(dim=-1)
    ap_new, ap_all, ap_all_fix, ap_all_know = stat_ap(preds.detach().cpu().numpy().reshape([117, 80]), False)
    return ap_new, ap_all, ap_all_fix, ap_all_know


# Train model
t_total = time.time()
import tqdm
from tqdm import tqdm
with tqdm(range(args.epochs)) as t:
    for i in t:
        ap_new, ap_all, ap_all_fix, ap_all_know = train(i)
        t.set_postfix({'ap_new': '{:.4}'.format(ap_new), 'ap_all_know': '{:.4}'.format(ap_all_know)})
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
