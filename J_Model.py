import numpy as np
import torch

import J_utils
from J_Layers import FCLayer
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PCModel(object):
    def __init__(self, nodes, mu_dt, act_fn, use_bias=False, kaiming_init=False):
        self.nodes = nodes
        self.mu_dt = mu_dt
        self.mu_dt_old = None

        self.softmax = torch.nn.Softmax(dim=1)

        self.norm_act = False

        self.n_nodes = len(nodes) # For traversing activations
        self.n_layers = len(nodes) - 1 # For traversing weights

        self.layers = []
        for l in range(self.n_layers):
            #_act_fn = J_utils.Linear() if (l == self.n_layers - 1) else act_fn
            _act_fn = J_utils.Linear() if (l == 0) else act_fn
            #_act_fn = act_fn

            layer = FCLayer(
                in_size=nodes[l],
                out_size=nodes[l + 1],
                act_fn=_act_fn,
                use_bias=use_bias,
                kaiming_init=kaiming_init,
            )
            layer.to(DEVICE)
            self.layers.append(layer)

    def set_input(self, feature):
        #self.mus[0] = feature.clone()
        self.mus[-1] = feature.clone().to(DEVICE)

    def set_target(self, target):
        #self.mus[-1] = target.clone()
        self.mus[0] = target.clone().to(DEVICE)
    
    def forward(self, val):
        for layer in self.layers:
            val = layer.forward(val)
        return val

    '''
    # Training Part
    '''

    def propagate_mu(self):
        for l in range(1, self.n_nodes):
            self.mus[l] = torch.zeros((self.mus[0].shape[0], self.nodes[l])).to(DEVICE)
        #for l in range(1, self.n_layers):
        #    self.mus[l] = self.layers[l - 1].forward(self.mus[l - 1])

    def train_updates(self, n_iters, fixed_preds=False, print_log=False):
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        for itr in range(n_iters):
            for l in range(1, self.n_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.errs[n] + self.mu_dt*(self.mus[n] - self.preds[n] - self.errs[n])
            
            if print_log:
                print(self.mus[0])


    def update_grads(self):
        for l in range(self.n_layers):
            self.layers[l].update_gradient(self.errs[l + 1])

    def train_batch_supervised(self, img_batch, label_batch, n_iters, fixed_preds=False, mu_dt=None, print_log=False):
        if mu_dt is not None:
            self.mu_dt_old = self.mu_dt #save old mu_dt
            self.mu_dt = mu_dt

        self.reset()
        #self.set_input(img_batch)
        #self.propagate_mu()
        #self.set_target(label_batch)
        self.set_target(label_batch)
        self.propagate_mu()
        self.set_input(img_batch)
        self.train_updates(n_iters, fixed_preds=fixed_preds, print_log=print_log)
        self.update_grads()

        if mu_dt is not None:
            self.mu_dt = self.mu_dt_old

    '''
    # Testing Part
    '''
    def test_updates(self, n_iters, fixed_preds=False, print_log=False):     
        self.errs[0] = torch.zeros_like(self.mus[0])
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        
        #last_preds = self.preds[-1].clone()
        #last_preds = self.softmax(self.mus[0].clone())
        last_preds = self.mus[0].clone()
        conv_times = [None for _ in range(last_preds.shape[0])]

        for itr in range(n_iters):
            self.mus[0] = self.mus[0] + self.mu_dt*(self.layers[0].backward(self.errs[1]))
            for l in range(1, self.n_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            self.errs[0] = self.errs[0] + self.mu_dt*(self.mus[0] - self.errs[0])
            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.errs[n] + self.mu_dt*(self.mus[n] - self.preds[n] - self.errs[n])
            
            #conv_times = self.convergence(last_preds=last_preds, curr_preds=self.preds[-1].clone(), 
            #                              conv_times=conv_times, itr=itr)
            #last_preds = self.preds[-1].clone()
            #conv_times = self.convergence(last_preds=last_preds, curr_preds=self.softmax(self.mus[0].clone()), 
            #                              conv_times=conv_times, itr=itr)
            #last_preds = self.softmax(self.mus[0].clone())
            conv_times = self.convergence(last_preds=last_preds, curr_preds=self.mus[0].clone(), 
                                          conv_times=conv_times, itr=itr)
            last_preds = self.mus[0].clone()

            if print_log:
                print(last_preds)

            if all([x is not None for x in conv_times]): #every batch has converged
                break
        
        return conv_times

    def test_batch_supervised(self, img_batch, label_batch, n_iters, fixed_preds=False, mu_dt=None, 
                              tol=1e-4, norm="L2", print_log=False):
        if mu_dt is not None:
            self.mu_dt_old = self.mu_dt
            self.mu_dt = mu_dt
        
        self.tol = tol
        if norm == "L2":
            self.norm = lambda x: torch.sqrt(torch.sum(x**2))
        elif norm == "L1":
            self.norm = lambda x: torch.sum(torch.abs(x))
        elif norm == "Max":
            self.norm = lambda x: torch.max(torch.abs(x))
        elif norm == "Activity":
            self.norm_act = True
            self.norm = lambda x: 0 if torch.any(x >= 1).item() else 100 
            
        self.reset()
        #self.set_input(img_batch)
        #self.propagate_mu()
        #self.set_target(torch.full_like(label_batch, 0.5))
        self.set_target(torch.full_like(label_batch, 0.5))
        self.propagate_mu()
        self.set_input(img_batch)
        conv_times = self.test_updates(n_iters, fixed_preds=fixed_preds, print_log=print_log)

        if mu_dt is not None:
            self.mu_dt = self.mu_dt_old
        #return self.preds[-1], conv_times
        return self.mus[0], conv_times
        #return self.softmax(self.mus[0]), conv_times
        #return self.forward(img_batch)
    
    def test_batch_supervised_errors(self, img_batch, label_batch, n_iters, fixed_preds=False, mu_dt=None, 
                              tol=1e-4, norm="L2", print_log=False):
        if mu_dt is not None:
            self.mu_dt_old = self.mu_dt
            self.mu_dt = mu_dt
        
        self.tol = tol
        if norm == "L2":
            self.norm = lambda x: torch.sqrt(torch.sum(x**2))
        elif norm == "L1":
            self.norm = lambda x: torch.sum(torch.abs(x))
        elif norm == "Max":
            self.norm = lambda x: torch.max(torch.abs(x))
        elif norm == "Activity":
            self.norm_act = True
            self.norm = lambda x: 0 if torch.any(x >= 1).item() else 100 
            
        self.reset()
        #self.set_input(img_batch)
        #self.propagate_mu()
        #self.set_target(torch.full_like(label_batch, 0.5))
        self.set_target(torch.full_like(label_batch, 0.5))
        self.propagate_mu()
        self.set_input(img_batch)
        #if return_error:
        #    conv_times, errors = self.test_updates_errors(n_iters, fixed_preds=fixed_preds, print_log=print_log, return_error=return_error)
        #else:
        conv_times = self.test_updates_errors(n_iters, fixed_preds=fixed_preds, print_log=print_log)

        if mu_dt is not None:
            self.mu_dt = self.mu_dt_old
        #return self.preds[-1], conv_times
        #if return_error:
        #    return self.mus[0], conv_times, errors
        #else:
        return self.mus[0], conv_times
        #return self.softmax(self.mus[0]), conv_times
        #return self.forward(img_batch)
    
    def test_updates_errors(self, n_iters, fixed_preds=False, print_log=False):     
        self.errs[0] = torch.zeros_like(self.mus[0])
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        
        #last_preds = self.preds[-1].clone()
        #last_preds = self.softmax(self.mus[0].clone())
        last_preds = self.mus[0].clone()
        errors = []
        conv_times = [None for _ in range(last_preds.shape[0])]

        for itr in range(n_iters):
            self.mus[0] = self.mus[0] + self.mu_dt*(self.layers[0].backward(self.errs[1]))
            for l in range(1, self.n_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            self.errs[0] = self.errs[0] + self.mu_dt*(self.mus[0] - self.errs[0])
            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.errs[n] + self.mu_dt*(self.mus[n] - self.preds[n] - self.errs[n])
            
            #conv_times = self.convergence(last_preds=last_preds, curr_preds=self.preds[-1].clone(), 
            #                              conv_times=conv_times, itr=itr)
            #last_preds = self.preds[-1].clone()
            #conv_times = self.convergence(last_preds=last_preds, curr_preds=self.softmax(self.mus[0].clone()), 
            #                              conv_times=conv_times, itr=itr)
            #last_preds = self.softmax(self.mus[0].clone())

            err = torch.cat([self.errs[n] for n in range(1, self.n_nodes)], dim=1)
            #err = torch.sum(err*2, axis=1)
            #err = self.norm(err)
            errors.append(err)

            if len(errors) >= 2:
                conv_times = self.convergence(last_preds=errors[-2], curr_preds=errors[-1], 
                                          conv_times=conv_times, itr=itr)
                errors.pop(0)
            #last_preds = self.mus[0].clone()

            if print_log:
                print(last_preds)

            if all([x is not None for x in conv_times]): #every batch has converged
                break
        
        return conv_times#, torch.stack(errors, dim=0).numpy()
    


    def test_batch_supervised_phase_space(self, img_batch, label_batch, n_iters, fixed_preds=False, mu_dt=None, error_dt=None, 
                              tol=1e-4, norm="L2", print_log=False, norm_errors=True, activities_index=0):
        if mu_dt is not None:
            self.mu_dt_old = self.mu_dt
            self.mu_dt = mu_dt
        if error_dt is not None:
            self.error_dt = error_dt
        else:
            self.error_dt = self.mu_dt
        
        self.tol = tol
        if norm == "L2":
            self.norm = lambda x: torch.sqrt(torch.sum(x**2))
        elif norm == "L1":
            self.norm = lambda x: torch.sum(torch.abs(x))
        elif norm == "Max":
            self.norm = lambda x: torch.max(torch.abs(x))
        elif norm == "Activity":
            self.norm_act = True
            self.norm = lambda x: 0 if torch.any(x >= 1).item() else 100 
        elif norm is None:
            self.norm_act = True
            self.norm = lambda x: 0 if torch.any(x >= 1).item() else 100
            
        self.reset()
        self.set_target(torch.full_like(label_batch, 0.5))
        self.propagate_mu()
        self.set_input(img_batch)
        conv_times, phase_plots = self.test_updates_phase_space(n_iters, norm=norm, fixed_preds=fixed_preds, 
                                                                print_log=print_log, norm_errors=norm_errors,
                                                                activities_index=activities_index)

        if mu_dt is not None:
            self.mu_dt = self.mu_dt_old
        if error_dt is not None:
            self.error_dt = None

        return self.mus[0], conv_times, phase_plots


    def test_updates_phase_space(self, n_iters, norm, fixed_preds=False, print_log=False, norm_errors=True, activities_index=0):     
        self.errs[0] = torch.zeros_like(self.mus[0])
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        
        last_preds = self.mus[0].clone()
        phase_space = []
        errors = []
        conv_times = [None for _ in range(last_preds.shape[0])]

        for itr in range(n_iters):
            self.mus[0] = self.mus[0] + self.mu_dt*(self.layers[0].backward(self.errs[1]))
            for l in range(1, self.n_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            self.errs[0] = self.errs[0] + self.error_dt*(self.mus[0] - self.errs[0])
            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.errs[n] + self.error_dt*(self.mus[n] - self.preds[n] - self.errs[n])

            if norm_errors:
                err = torch.cat([self.errs[n] for n in range(1, self.n_nodes)], dim=1)
                errors.append(err)
    
                if len(errors) >= 2:
                    conv_times = self.convergence(last_preds=errors[-2], curr_preds=errors[-1], 
                                              conv_times=conv_times, itr=itr)
                    errors.pop(0) #No need to save everything 
                #add to phase space list
                if norm == "L1":
                    phase_space.append(torch.sum(torch.abs(err), axis=1).cpu().numpy())
                elif norm == "L2":
                    phase_space.append(torch.sum(torch.pow(err, 2), axis=1).cpu().numpy())
                elif norm == "Max":
                    phase_space.append(torch.max(torch.abs(err), axis=1).cpu().numpy())
                elif norm == "Activity":
                    phase_space.append(torch.max(torch.abs(err), axis=1).numpy())
                elif norm is None:
                    phase_space.append(torch.max(torch.abs(err), axis=1).numpy())
            else:
                conv_times = self.convergence(last_preds=last_preds, curr_preds=self.mus[0].clone(), 
                                          conv_times=conv_times, itr=itr)
                last_preds = self.mus[0].clone()
                #add to phase space list
                if norm == "L1":
                    phase_space.append(torch.sum(torch.abs(self.mus[activities_index].clone()), axis=1).cpu().numpy())
                elif norm == "L2":
                    phase_space.append(torch.sum(torch.pow(self.mus[activities_index].clone(), 2), axis=1).cpu().numpy())
                elif norm == "Max":
                    phase_space.append(torch.max(torch.abs(self.mus[activities_index].clone()), axis=1).cpu().numpy())
                elif norm == "Activity":
                    phase_space.append(torch.sum(torch.pow(self.mus[activities_index].clone(), 2), axis=1).cpu().numpy())
                    #phase_space.append(torch.max(torch.abs(self.mus[activities_index].clone())).item())
                elif norm is None:
                    phase_space.append(self.mus[activities_index].clone().squeeze().cpu().numpy())
            if print_log:
                print(last_preds)
            
            if all([x is not None for x in conv_times]): #every batch has converged
                break
        
        return conv_times, np.array(phase_space)
    '''
    # Misc.
    # Mainly used to reset the net
    '''
    def convergence(self, last_preds, curr_preds, conv_times, itr):
        for idx, time in enumerate(conv_times):
            if time is None: #has not converged yet, check convergence
                if self.norm_act:
                    if self.norm(curr_preds[idx]) < self.tol:
                       #print("Converged, should be at least one time in output")
                        conv_times[idx] = (itr+1)*self.mu_dt 
                else:
                    diff = curr_preds[idx] - last_preds[idx]
                    #print(f"{self.norm(diff)}")
                    if self.norm(diff).item() < self.tol: #converged!
                        #print("Converged, should be at least one time in output")
                        conv_times[idx] = (itr+1)*self.mu_dt
        return conv_times


    def reset(self):
        self.preds = [[] for _ in range(self.n_nodes)]
        self.errs = [[] for _ in range(self.n_nodes)]
        self.mus = [[] for _ in range(self.n_nodes)]

    def reset_mus(self, batch_size, init_std):
        for l in range(self.n_layers):
            self.mus[l] = J_utils.set_tensor(
                torch.empty(batch_size, self.layers[l].in_size).normal_(mean=0, std=init_std)
            )
    

    def save_weights(self, path="weights\\"):
        for idx, layer in enumerate(self.layers):
            with open(f'{path}weights_layer_{idx}.pickle', 'wb') as file:
                pickle.dump(layer.weights, file, protocol=pickle.HIGHEST_PROTOCOL)


    def load_weights(self, path="weights\\"):
        for idx, layer in enumerate(self.layers):
            with open(f'{path}weights_layer_{idx}.pickle', 'rb') as file:
                layer.weights =pickle.load(file)

    @property
    def params(self):
        return self.layers
