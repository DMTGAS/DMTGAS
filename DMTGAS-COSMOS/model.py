import torch
from torch_geometric.nn import global_add_pool
from util.genotypes import NA_PRIMITIVES,  SC_PRIMITIVES, LA_PRIMITIVES
from util.operations import *
from torch.autograd import Variable
import numpy as np
import util.data_utils as data_utils

def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")

class NaOp(nn.Module):
  def __init__(self, primitive, in_dim, out_dim, act, with_linear=False):
    super(NaOp, self).__init__()

    self._op = NA_OPS[primitive](in_dim, out_dim)
    self.op_linear = nn.Linear(in_dim, out_dim)
    self.act = act_map(act)
    self.with_linear = with_linear

  def forward(self, x, edge_index):
    if self.with_linear:
      return self.act(self._op(x, edge_index)+self.op_linear(x))
    else:
      return self.act(self._op(x, edge_index))

# class NaMLPOp(nn.Module):
#     def __init__(self, primitive, in_dim, out_dim, act):
#         super(NaMLPOp, self).__init__()
#         self._op = NA_MLP_OPS[primitive](in_dim, out_dim)
#         self.act = act_map(act)
#
#     def forward(self, x, edge_index):
#         return self.act(self._op(x, edge_index))

class ScOp(nn.Module):
    def __init__(self, primitive):
        super(ScOp, self).__init__()
        self._op = SC_OPS[primitive]()

    def forward(self, x):
        return self._op(x)

class LaOp(nn.Module):
    def __init__(self, primitive, hidden_size, act, num_layers=None):
        super(LaOp, self).__init__()
        self._op = LA_OPS[primitive](hidden_size, num_layers)
        self.act = act_map(act)

    def forward(self, x):
        return self.act(self._op(x))

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
    def __init__(self, tasks, genotype, in_dim, node_embedding_dim, num_gc_outputs, num_nc_outputs, hidden_size, num_layers=3, in_dropout=0.5, out_dropout=0.5, act='relu', args=None):
        super(MTLAGL, self).__init__()
        self.name = "MTLAGL"
        self.in_dim = in_dim
        self.genotype = genotype
        self.node_embedding_dim = node_embedding_dim
        self.num_gc_outputs = num_gc_outputs
        self.num_nc_outputs = num_nc_outputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self.args = args
        self.tasks = tasks
        self.weight_pareto = args.weight_pareto
        ops = genotype.split('||')

        # node aggregator op
        if self.weight_pareto:
            self.lin1 = nn.Linear((len(self.tasks) + in_dim), hidden_size)
        else:
            self.lin1 = nn.Linear(in_dim, hidden_size)
        self.gnn_layers = nn.ModuleList(
            [NaOp(ops[i], hidden_size, hidden_size, act, with_linear=args.with_linear) for i in range(num_layers)])

        # skip op
        if self.args.fix_last:
            if self.num_layers > 1:
                self.sc_layers = nn.ModuleList([ScOp(ops[i + num_layers]) for i in range(num_layers - 1)])
            else:
                self.sc_layers = nn.ModuleList([ScOp(ops[num_layers])])
        else:
            # no output conditions.
            skip_op = ops[num_layers:2 * num_layers]
            if skip_op == ['none'] * num_layers:
                skip_op[-1] = 'skip'
                print('skip_op:', skip_op)
            self.sc_layers = nn.ModuleList([ScOp(skip_op[i]) for i in range(num_layers)])

        # layer aggregator op
        self.layer6 = LaOp(ops[-1], hidden_size, 'linear', num_layers)

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
        # generate weights by softmax
        x = self.lin1(x)
        x = F.dropout(x, p=self.in_dropout, training=self.training)
        js = []
        for i in range(self.num_layers):
            x = self.gnn_layers[i](x, edge_index)
            if self.args.with_layernorm:
                layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
                x = layer_norm(x)
            x = F.dropout(x, p=self.in_dropout, training=self.training)
            if i == self.num_layers - 1 and self.args.fix_last:
                js.append(x)
            else:
                js.append(self.sc_layers[i](x))
        x5 = self.layer6(js)
        x5 = F.dropout(x5, p=self.out_dropout, training=self.training)

        gc_output = self.gc_output_layer(x5, data.batch) if "gc" in self.tasks else None
        nc_output = self.nc_output_layer(x5) if "nc" in self.tasks else None
        lp_output = self.lp_output_layer(x5, data.pos_edge_index, data.neg_edge_index) if "lp" in self.tasks else None
        return gc_output, nc_output, lp_output

    def _loss(self, data, log_vars):
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

        if self.weight_pareto:
            gc_precision = torch.exp(-log_vars["gc"]) if "gc" in self.tasks else 0
            nc_precision = torch.exp(-log_vars["nc"]) if "nc" in self.tasks else 0
            lp_precision = torch.exp(-log_vars["lp"]) if "lp" in self.tasks else 0
            loss = torch.sum(gc_precision * gc_loss + log_vars["gc"], -1) + \
                   torch.sum(nc_precision * nc_loss + log_vars["nc"], -1) + \
                   torch.sum(lp_precision * lp_loss + log_vars["lp"], -1)
        else:
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