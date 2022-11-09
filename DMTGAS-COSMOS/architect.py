import copy

import torch
import numpy as np
from torch.autograd import Variable
from util.MGDA_utils import gradient_normalizers, MinNormSolver
import util.data_utils as data_utils

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):

    def __init__(self, model, args):
        self.args = args
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, data_train, eta, network_optimizer):
        if self.args.weight_pareto:
            a = {}
        else:
            loss = self.model._loss(data_train) #train loss
            theta = _concat(self.model.parameters()).data# w
            try:
                moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
            except:
                moment = torch.zeros_like(theta)
            dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta#gradient, L2 norm
            unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta)) # one-step update, get w' for Eq.7 in the paper
        return unrolled_model

    def _compute_unrolled_model_pareto(self, data_train, eta, network_optimizer, weight_train):
        if self.args.weight_pareto:
            a = {}
        else:
            loss = self.model._losspareto(data_train, weight_train) #train loss
            theta = _concat(self.model.parameters()).data# w
            try:
                moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
            except:
                moment = torch.zeros_like(theta)
            dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta#gradient, L2 norm
            unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta)) # one-step update, get w' for Eq.7 in the paper
        return unrolled_model

    def step(self, data_train, data_val, eta, network_optimizer, unrolled, weight_val):
        device = torch.device('cuda:%d' % self.args.gpu if torch.cuda.is_available() else 'cpu')
        data_val = data_val.to(device)
        data_train = data_train.to(device)

        if unrolled:

            weight_train = copy.deepcopy(weight_val)
            self.optimizer.zero_grad()
            self._backward_step_unrolled(data_train, data_val, eta, network_optimizer, weight_val, weight_train)
        else:
            self.optimizer.zero_grad()
            self._backward_step(data_val, weight_val) #valid
        self.optimizer.step()

    def step_MGDA(self, data_train, data_val, eta, network_optimizer, unrolled):

        device = torch.device('cuda:%d' % self.args.gpu if torch.cuda.is_available() else 'cpu')
        data_val = data_val.to(device)
        data_train = data_train.to(device)
        grads = {}
        losses = {}
        weight_val = {}
        gc_loss = nc_loss = lp_loss = 0
        # obtain and store the gradient value
        for t in self.args.tasks:
            self.optimizer.zero_grad()
            # forward pass
            gc_train_logit, nc_train_logit, lp_train_logit = self.model(data_val)
            if t == "gc":
                gc_loss = F.cross_entropy(gc_train_logit, data_val.y)
                gc_loss.backward()
                losses[t] = gc_loss
            if t == "nc":
                node_labels = data_val.node_y.argmax(1)
                train_mask = data_val.train_mask.squeeze()
                nc_loss = F.cross_entropy(nc_train_logit[train_mask == 1], node_labels[train_mask == 1])
                nc_loss.backward()
                losses[t] = nc_loss
            if t == "lp":
                train_link_labels = data_utils.get_link_labels(data_val.pos_edge_index,
                                                               data_val.neg_edge_index)
                lp_loss = F.binary_cross_entropy_with_logits(lp_train_logit.squeeze(), train_link_labels)
                lp_loss.backward()
                losses[t] = nc_loss
            grads[t] = []
            # can use scalable method proposed in the MOO-MTL paper for large scale problem
            # but we keep use the gradient of all parameters in this experiment
            # for name, param in model.named_parameters():  # 查看可优化的参数有哪些
            #     if param.requires_grad:
            #         print(name)
            for idx, param in enumerate(self.model.arch_parameters()):
                if param.grad is not None:  # arch parameters have 3 layers
                    grads[t].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
        gn = gradient_normalizers(grads, losses, self.args.normalization_type)
        for t in self.args.tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]
        # Frank-Wolfe iteration to compute scales.
        # print([grads[t] for t in self.args.tasks] )
        sol, min_norm = MinNormSolver.find_min_norm_element_FW([grads[t] for t in self.args.tasks])
        for i, t in enumerate(self.args.tasks):
            weight_val[t] = float(sol[i])
        print('\nweight_val:')
        print(weight_val)

        if unrolled:
            grads = {}
            losses = {}
            weight_train = {}
            gc_loss = nc_loss = lp_loss = 0
            # obtain and store the gradient value
            for t in self.args.tasks:
                self.optimizer.zero_grad()
                # forward pass
                gc_train_logit, nc_train_logit, lp_train_logit = self.model(data_train)
                if t == "gc":
                    gc_loss = F.cross_entropy(gc_train_logit, data_train.y)
                    gc_loss.backward()
                    losses[t] = gc_loss
                if t == "nc":
                    node_labels = data_train.node_y.argmax(1)
                    train_mask = data_train.train_mask.squeeze()
                    nc_loss = F.cross_entropy(nc_train_logit[train_mask == 1], node_labels[train_mask == 1])
                    nc_loss.backward()
                    losses[t] = nc_loss
                if t == "lp":
                    train_link_labels = data_utils.get_link_labels(data_train.pos_edge_index,
                                                                   data_train.neg_edge_index)
                    lp_loss = F.binary_cross_entropy_with_logits(lp_train_logit.squeeze(), train_link_labels)
                    lp_loss.backward()
                    losses[t] = nc_loss
                grads[t] = []
                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment
                # for name, param in model.named_parameters():  # 查看可优化的参数有哪些
                #     if param.requires_grad:
                #         print(name)
                for idx, param in enumerate(self.model.parameters()):
                    if param.grad is not None:
                        grads[t].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
            gn = gradient_normalizers(grads, losses, self.args.normalization_type)
            for t in self.args.tasks:
                for gr_i in range(len(grads[t])):
                    grads[t][gr_i] = grads[t][gr_i] / gn[t]
            # Frank-Wolfe iteration to compute scales.
            # print([grads[t] for t in args.tasks] )
            sol, min_norm = MinNormSolver.find_min_norm_element_FW([grads[t] for t in self.args.tasks])
            for i, t in enumerate(self.args.tasks):
                weight_train[t] = float(sol[i])
            self.optimizer.zero_grad()
            self._backward_step_unrolled(data_train, data_val, eta, network_optimizer, weight_val, weight_train)
        else:
            self.optimizer.zero_grad()
            self._backward_step(data_val, weight_val)  # valid
        self.optimizer.step()

    def _backward_step(self, data_val, weight_val):
        device = torch.device('cuda:%d' % self.args.gpu if torch.cuda.is_available() else 'cpu')
        data_val = data_val.to(device)
        if self.args.weight_pareto:
            loss = self.model._losspareto(data_val, weight_val) #valid
        else:
            loss = self.model._loss(data_val) #valid
        loss.backward()

    def _backward_step_unrolled(self, data_train, data_val, eta, network_optimizer, weight_val, weight_train):
        if self.weight_pareto:
            unrolled_model = self._compute_unrolled_model_pareto(data_train, eta, network_optimizer, weight_train)
            unrolled_loss = unrolled_model._losspareto(data_val, weight_val)  # validation loss

            unrolled_loss.backward()  # one-step update for w?
            dalpha = [v.grad for v in unrolled_model.arch_parameters()]  # L_vali w.r.t alpha
            vector = [v.grad.data for v in
                      unrolled_model.parameters()]  # gradient, L_train w.r.t w, double check the model construction
            implicit_grads = self._hessian_vector_product_pareto(vector, data_train, weight_train)
        else:
            unrolled_model = self._compute_unrolled_model(data_train, eta, network_optimizer)
            unrolled_loss = unrolled_model._loss(data_val) # validation loss

            unrolled_loss.backward() # one-step update for w?
            dalpha = [v.grad for v in unrolled_model.arch_parameters()] #L_vali w.r.t alpha
            vector = [v.grad.data for v in unrolled_model.parameters()] # gradient, L_train w.r.t w, double check the model construction
            implicit_grads = self._hessian_vector_product(vector, data_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        #update alpha, which is the ultimate goal of this func, also the goal of the second-order darts
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, data_train, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v) # R * d(L_val/w', i.e., get w^+
        loss = self.model._loss(data_train) # train loss
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters()) # d(L_train)/d_alpha, w^+

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v) # get w^-, need to subtract 2 * R since it has add R
        loss = self.model._loss(data_train)# train loss
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())# d(L_train)/d_alpha, w^-

        #reset to the orignial w, always using the self.model, i.e., the original model
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

    def _hessian_vector_product_pareto(self, vector, data_train, weight_train, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v) # R * d(L_val/w', i.e., get w^+
        loss = self.model._losspareto(data_train, weight_train) # train loss
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters()) # d(L_train)/d_alpha, w^+

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v) # get w^-, need to subtract 2 * R since it has add R
        loss = self.model._losspareto(data_train, weight_train)# train loss
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())# d(L_train)/d_alpha, w^-

        #reset to the orignial w, always using the self.model, i.e., the original model
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
