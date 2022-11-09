import torch
from torch_geometric.nn import global_add_pool
from util.genotypes import NA_PRIMITIVES,  SC_PRIMITIVES, LA_PRIMITIVES
from util.operations import *
from torch.autograd import Variable
import numpy as np
import util.data_utils as data_utils



class NaMixedOp(nn.Module):

  def __init__(self, in_dim, out_dim, with_linear):
    super(NaMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.with_linear = with_linear

    for primitive in NA_PRIMITIVES:
      op = NA_OPS[primitive](in_dim, out_dim)
      self._ops.append(op)

      if with_linear:
        self._ops_linear = nn.ModuleList()
        op_linear = torch.nn.Linear(in_dim, out_dim)
        self._ops_linear.append(op_linear)

  def forward(self, x, weights, edge_index, ):
    mixed_res = []
    if self.with_linear:
      for w, op, linear in zip(weights, self._ops, self._ops_linear):
        mixed_res.append(w * F.elu(op(x, edge_index)+linear(x)))
    else:
      for w, op in zip(weights, self._ops):
        mixed_res.append(w * F.elu(op(x, edge_index)))
    return sum(mixed_res)

class ScMixedOp(nn.Module):

  def __init__(self):
    super(ScMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in SC_PRIMITIVES:
      op = SC_OPS[primitive]()
      self._ops.append(op)

  def forward(self, x, weights):
    mixed_res = []
    for w, op in zip(weights, self._ops):
      mixed_res.append(w * op(x))
    return sum(mixed_res)

class LaMixedOp(nn.Module):

  def __init__(self, hidden_size, num_layers=None):
    super(LaMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in LA_PRIMITIVES:
      op = LA_OPS[primitive](hidden_size, num_layers)
      self._ops.append(op)

  def forward(self, x, weights):
    mixed_res = []
    for w, op in zip(weights, self._ops):
      mixed_res.append(w * F.relu(op(x)))
    return sum(mixed_res)

class NodeClassificationOutputModule(nn.Module):
    def __init__(self, node_embedding_dim, num_classes):
        super(NodeClassificationOutputModule, self).__init__()
        self.linear = nn.Linear(node_embedding_dim, num_classes)

    def forward(self, inputs):
        x = self.linear(inputs)
        return x


class GraphClassificationOutputModule(nn.Module):
    def __init__(self, node_embedding_dim, hidden_dim, num_classes):
        super(GraphClassificationOutputModule, self).__init__()
        self.linear1 = nn.Linear(node_embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, batch):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.linear2(x)
        return x


class LinkPredictionOutputModule(nn.Module):
    def __init__(self, node_embedding_dim):
        super(LinkPredictionOutputModule, self).__init__()
        self.linear_a = nn.Linear(node_embedding_dim, node_embedding_dim)
        # self.linear_b = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.linear = nn.Linear(2 * node_embedding_dim, 1)

    def forward(self, inputs, pos_edge_index, neg_edge_index):
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        node_a = torch.index_select(inputs, 0, total_edge_index[0]) # 按行取多行正
        node_a = self.linear_a(node_a)
        node_b = torch.index_select(inputs, 0, total_edge_index[1]) # 按行取多行负
        node_b = self.linear_a(node_b)
        x = torch.cat((node_a, node_b), 1)
        x = self.linear(x)
        x = torch.clamp(torch.sigmoid(x), min=1e-8, max=1 - 1e-8)
        return x

class MTLAGL(nn.Module):
    '''
    implement auto multi-task graph learning
    which is the combination of three cells, node aggregator, skip connection, and layer aggregator
    '''
    def __init__(self, tasks, in_dim, node_embedding_dim, num_gc_outputs, num_nc_outputs, hidden_size, num_layers=3, dropout=0.5, epsilon=0.0, with_conv_linear=False, args=None):
        super(MTLAGL, self).__init__()
        self.name = "MTLAGL"
        self.in_dim = in_dim
        self.node_embedding_dim = node_embedding_dim
        self.num_gc_outputs = num_gc_outputs
        self.num_nc_outputs = num_nc_outputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epsilon = epsilon
        self.explore_num = 0
        self.with_linear = with_conv_linear
        self.args = args
        self.tasks = tasks
        self.weight_pareto = args.weight_pareto

        # node aggregator op
        if self.weight_pareto:
            self.lin1 = nn.Linear((len(self.tasks) + in_dim), hidden_size)
        else:
            self.lin1 = nn.Linear(in_dim, hidden_size)
        self.layer1 = NaMixedOp(hidden_size, hidden_size, self.with_linear)
        self.layer2 = NaMixedOp(hidden_size, hidden_size, self.with_linear)
        self.layer3 = NaMixedOp(hidden_size, hidden_size, self.with_linear)

        # skip op
        self.layer4 = ScMixedOp()
        self.layer5 = ScMixedOp()
        if not self.args.fix_last:
            self.layer6 = ScMixedOp()

        # layer aggregator op
        self.layer7 = LaMixedOp(hidden_size, num_layers)

        if "gc" in self.tasks:
            self.gc_output_layer = GraphClassificationOutputModule(node_embedding_dim, node_embedding_dim,
                                                                   num_gc_outputs)
        if "nc" in self.tasks:
            self.nc_output_layer = NodeClassificationOutputModule(node_embedding_dim, num_nc_outputs)
        if "lp" in self.tasks:
            self.lp_output_layer = LinkPredictionOutputModule(node_embedding_dim)

        self._initialize_alphas()

    def forward(self, data, weight=None):
        x, edge_index = data.x, data.edge_index
        if weight != None:
            b = x.shape[0]
            a = weight.repeat(b, 1)
            x = torch.cat((x, a), dim=1)

        self.na_weights = F.softmax(self.na_alphas, dim=-1)
        self.sc_weights = F.softmax(self.sc_alphas, dim=-1)
        self.la_weights = F.softmax(self.la_alphas, dim=-1)

        # generate weights by softmax
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self.layer1(x, self.na_weights[0], edge_index)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.layer2(x1, self.na_weights[1], edge_index)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x3 = self.layer3(x2, self.na_weights[2], edge_index)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        if self.args.fix_last:
            x4 = (x3, self.layer4(x1, self.sc_weights[0]), self.layer5(x2, self.sc_weights[1]))
        else:
            x4 = (self.layer4(x1, self.sc_weights[0]), self.layer5(x2, self.sc_weights[1]),
                  self.layer6(x3, self.sc_weights[2]))

        x5 = self.layer7(x4, self.la_weights[0])
        x5 = F.dropout(x5, p=self.dropout, training=self.training)

        gc_output = self.gc_output_layer(x5, data.batch) if "gc" in self.tasks else None
        nc_output = self.nc_output_layer(x5) if "nc" in self.tasks else None
        lp_output = self.lp_output_layer(x5, data.pos_edge_index, data.neg_edge_index) if "lp" in self.tasks else None
        return gc_output, nc_output, lp_output

    def _loss(self, data):
        gc_train_logit, nc_train_logit, lp_train_logit = self(data)
        # Evaluate Loss and Accuracy
        # GC
        gc_loss = nc_loss = lp_loss = 0
        if "gc" in self.tasks:
            gc_loss = F.cross_entropy(gc_train_logit, data.y)
        # NC
        if "nc" in self.tasks:
            node_labels = data.node_y.argmax(1)
            train_mask = data.train_mask.squeeze()
            nc_loss = F.cross_entropy(nc_train_logit[train_mask == 1], node_labels[train_mask == 1])
        # LP
        if "lp" in self.tasks:
            train_link_labels = data_utils.get_link_labels(data.pos_edge_index, data.neg_edge_index)
            lp_loss = F.binary_cross_entropy_with_logits(lp_train_logit.squeeze(), train_link_labels)
        loss = gc_loss + nc_loss + lp_loss

        return loss

    def _losspareto(self, data, weight):
        losses = {}
        # Forward pass
        # Dict to tensor
        ww = []
        for k in weight.items():
            ww.append(k[1])
        ww = torch.tensor(ww).cuda()

        gc_train_logit, nc_train_logit, lp_train_logit = self(data, ww)
        # Evaluate Loss and Accuracy
        for i, t in enumerate(self.args.tasks):
            if "gc" == t:
                losses[t] = F.cross_entropy(gc_train_logit, data.y)
            # NC
            if "nc" == t:
                node_labels = data.node_y.argmax(1)
                train_mask = data.train_mask.squeeze()
                losses[t] = F.cross_entropy(nc_train_logit[train_mask == 1], node_labels[train_mask == 1])
            # LP
            if "lp" == t:
                train_link_labels = data_utils.get_link_labels(data.pos_edge_index,
                                                               data.neg_edge_index)
                losses[t] = F.binary_cross_entropy_with_logits(lp_train_logit.squeeze(), train_link_labels)
            if i > 0:
                loss = loss + weight[t] * losses[t]
            else:
                loss = weight[t] * losses[t]

        lossess = []
        for k in losses.items():
            lossess.append(k[1])
        lossess = torch.tensor(lossess).cuda()

        cossim = torch.nn.functional.cosine_similarity(lossess, ww, dim=0)
        loss -= self.args.lamda * cossim

        return loss

    def new(self):
        model_new = MTLAGL(self.tasks, self.in_dim, self.node_embedding_dim, self.num_gc_outputs, self.num_nc_outputs, self.hidden_size, self.num_layers, self.dropout, self.epsilon, self.with_conv_linear, self.args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_alphas(self):
        # k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_na_ops = len(NA_PRIMITIVES)
        num_sc_ops = len(SC_PRIMITIVES)
        num_la_ops = len(LA_PRIMITIVES)
        # self.na_alphas = Variable(1e-3*torch.randn(3, num_na_ops).cuda(), requires_grad=True)
        self.na_alphas = Variable(1e-3 * torch.randn(3, num_na_ops), requires_grad=True)
        if self.args.fix_last:
            # self.sc_alphas = Variable(1e-3*torch.randn(2, num_sc_ops).cuda(), requires_grad=True)
            self.sc_alphas = Variable(1e-3 * torch.randn(2, num_sc_ops), requires_grad=True)
        else:
            # self.sc_alphas = Variable(1e-3*torch.randn(3, num_sc_ops).cuda(), requires_grad=True)
            self.sc_alphas = Variable(1e-3 * torch.randn(3, num_sc_ops), requires_grad=True)

        # self.la_alphas = Variable(1e-3*torch.randn(1, num_la_ops).cuda(), requires_grad=True)
        self.la_alphas = Variable(1e-3 * torch.randn(1, num_la_ops), requires_grad=True)
        self._arch_parameters = [
            self.na_alphas,
            self.sc_alphas,
            self.la_alphas,
        ]

    def genotype(self):

        def _parse(na_weights, sc_weights, la_weights):
            gene = []
            na_indices = torch.argmax(na_weights, dim=-1)
            for k in na_indices:
                gene.append(NA_PRIMITIVES[k])
            # sc_indices = sc_weights.argmax(dim=-1)
            sc_indices = torch.argmax(sc_weights, dim=-1)
            for k in sc_indices:
                gene.append(SC_PRIMITIVES[k])
            # la_indices = la_weights.argmax(dim=-1)
            la_indices = torch.argmax(la_weights, dim=-1)
            for k in la_indices:
                gene.append(LA_PRIMITIVES[k])
            return '||'.join(gene)

        gene = _parse(F.softmax(self.na_alphas, dim=-1).data.cpu(), F.softmax(self.sc_alphas, dim=-1).data.cpu(),
                      F.softmax(self.la_alphas, dim=-1).data.cpu())

        return gene

    def sample_arch(self):

        num_na_ops = len(NA_PRIMITIVES)
        num_sc_ops = len(SC_PRIMITIVES)
        num_la_ops = len(LA_PRIMITIVES)

        gene = []
        for i in range(3):
            op = np.random.choice(NA_PRIMITIVES, 1)[0]
            gene.append(op)
        for i in range(2):
            op = np.random.choice(SC_PRIMITIVES, 1)[0]
            gene.append(op)
        op = np.random.choice(LA_PRIMITIVES, 1)[0]
        gene.append(op)
        return '||'.join(gene)

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(na_weights, sc_weights, la_weights):
            gene = []
            na_indices = torch.argmax(na_weights, dim=-1)
            for k in na_indices:
                gene.append(NA_PRIMITIVES[k])
            # sc_indices = sc_weights.argmax(dim=-1)
            sc_indices = torch.argmax(sc_weights, dim=-1)
            for k in sc_indices:
                gene.append(SC_PRIMITIVES[k])
            # la_indices = la_weights.argmax(dim=-1)
            la_indices = torch.argmax(la_weights, dim=-1)
            for k in la_indices:
                gene.append(LA_PRIMITIVES[k])
            return '||'.join(gene)

        gene = _parse(F.softmax(self.na_alphas, dim=-1).data.cpu(), F.softmax(self.sc_alphas, dim=-1).data.cpu(),
                      F.softmax(self.la_alphas, dim=-1).data.cpu())

        return gene

    def get_weights_from_arch(self, arch):
        arch_ops = arch.split('||')
        # print('arch=%s' % arch)
        num_na_ops = len(NA_PRIMITIVES)
        num_sc_ops = len(SC_PRIMITIVES)
        num_la_ops = len(LA_PRIMITIVES)

        # na_alphas = Variable(torch.zeros(3, num_na_ops).cuda(), requires_grad=True)
        # sc_alphas = Variable(torch.zeros(2, num_sc_ops).cuda(), requires_grad=True)
        # la_alphas = Variable(torch.zeros(1, num_la_ops).cuda(), requires_grad=True)
        na_alphas = Variable(torch.zeros(3, num_na_ops), requires_grad=True)
        sc_alphas = Variable(torch.zeros(2, num_sc_ops), requires_grad=True)
        la_alphas = Variable(torch.zeros(1, num_la_ops), requires_grad=True)

        for i in range(3):
            ind = NA_PRIMITIVES.index(arch_ops[i])
            na_alphas[i][ind] = 1

        for i in range(3, 5):
            ind = SC_PRIMITIVES.index(arch_ops[i])
            sc_alphas[i - 3][ind] = 1

        ind = LA_PRIMITIVES.index(arch_ops[5])
        la_alphas[0][ind] = 1

        arch_parameters = [na_alphas, sc_alphas, la_alphas]
        return arch_parameters

    def set_model_weights(self, weights):
        self.na_weights = weights[0]
        self.sc_weights = weights[1]
        self.la_weights = weights[2]
        # self._arch_parameters = [self.na_alphas, self.sc_alphas, self.la_alphas]
