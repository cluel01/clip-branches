import torch
import numpy as np
from .abstractclassifier import AbstractClassifier
from .boxnet import BoxNet,BoxNetBranches
from .utils import numpy_to_torch,WeightedBCELoss

class BoxNetClassifier(AbstractClassifier):
    def __init__(self, nboxes=2, dim=2, tau=1, alpha=1, 
                 device='cpu', loss_fn=None, l1_reg=1e-3, l2_reg=1e-3,
                 training_epochs=1000, learning_rate=1e-1, patience_early_stopping=100, 
                 alpha_tau_decay_step=1, prediction_threshold=0.5,box_pruning_step=100, box_pruning_threshold=1,
                 verbose=False, verbosity=10,num_threads=10, random_state=42):
        
        if loss_fn is None:
            loss_fn = torch.nn.BCELoss()
        
        if patience_early_stopping is None:
            patience_early_stopping = torch.inf
        
        self.nboxes = nboxes
        self.dim = dim
        self.device = device
        self.loss_fn = loss_fn
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.learning_rate = learning_rate
        self.tau = tau
        self.alpha = alpha
        self.training_epochs = training_epochs
        self.patience_early_stopping = patience_early_stopping
        self.alpha_tau_decay_step = alpha_tau_decay_step
        self.prediction_threshold = prediction_threshold
        self.verbose = verbose
        self.verbosity = verbosity
        self.random_state = random_state
        self.scheduler = None
        self.num_threads = num_threads
        self.box_pruning_step = box_pruning_step
        self.box_pruning_threshold = box_pruning_threshold

        
        self.setup_classifier()

    def setup_classifier(self):
        self.model = BoxNet(
            nboxes=self.nboxes,
            dim = self.dim,
            tau=self.tau, 
            alpha=self.alpha,
            random_state=self.random_state)

        if self.device != "cpu":
            self.model.block = self.model.block.to(self.device)
        
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, 
                patience=30, min_lr=1e-5)
        
    def fit(self, X, y):
        torch.set_num_threads(self.num_threads)
        #self.dim = X.shape[1]
        self.setup_classifier()
        
        X = numpy_to_torch(X, self.device)
        y = numpy_to_torch(y, self.device)

        self._train(X,y)
        
        

    def _train(self,X,y):
        best_model = None
        best_loss = np.inf
        
        self.model.train()
        for epoch in range(self.training_epochs):
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            z,probs = self.model(X)
            
            # Compute loss
            J = self.loss_fn(z, y)
            J += self.l1_reg * torch.norm(torch.abs(self.model.block.length), p=1)
            J += self.l2_reg * torch.sum(torch.pow(self.model.block.length, 2))
            
            loss = J.detach().item()
            
            # Backward pass
            J.backward()  
            
            # Update weights
            self.optimizer.step()
            
            # Decay alpha and tau
            if self.alpha_tau_decay_step > 0 and ((epoch+1) % self.alpha_tau_decay_step) == 0:
                self.model.block.tau = max(self.model.block.tau * 0.95, 0.02)
                self.model.alpha = max(self.model.alpha * 0.95, 0.02)

            #Box pruning
            if self.box_pruning_step > 0  and ((epoch+1) / self.box_pruning_step) == 1: #and ((epoch+1) % self.box_pruning_step) == 0: #
                mask = self.model.block.box_mask.clone()
                box_importance = self.calc_box_importance(probs,y)
                filter_mask = box_importance >= self.box_pruning_threshold
                self.model.block.box_mask[mask] = filter_mask

                #Filter out boxes
                if len(X.shape) == 3:
                    X = X[:,filter_mask,:]
                
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step(loss)
                
            if loss < best_loss:
                best_loss = loss
                #best_model = self.model.state_dict()
                _patience = self.patience_early_stopping # Reset patience
            else:
                _patience -= 1 # Reduce patience
            
            if not _patience or loss == 0.:
                break
                
            # Print progress
            if self.verbose:
                if (((epoch+1) % self.verbosity) == 0) or (epoch == self.training_epochs-1):
                    print("Epoch: ", epoch+1)
                    print("Loss: ", loss)

        #self.model.load_state_dict(best_model)

        #Check for box importance
        _,probs = self.model(X)
        self.box_importance = self.calc_box_importance(probs,y).cpu().numpy()
        if self.verbose:
            print(f"Included true positives: {self.box_importance};Total positives: {y.sum().cpu().numpy()};")

    def predict(self, X):
        y_pred = []
        self.model.eval()
        with torch.inference_mode():
            X = numpy_to_torch(X, self.device)
            probs,_ = self.model(X)
            y_pred = (probs > self.prediction_threshold).long()

        return y_pred.cpu()

    def set_hyperparameters(self, hyperparameters:dict):
        return BoxNetClassifier(**hyperparameters)
    
    def get_num_boxes(self):
        return torch.sum(torch.abs(self.model.block.length) > 1e-3).item()/(self.dim)
    

    def get_boxes(self):
        #return the boxes based on the mins and lenghts + subsets
        min_list,max_list,fidx_list = [],[],[]
        block = self.model.block
        for i in range(self.model.block.nboxes):
            mins  = block.mins[i].detach().cpu().numpy()
            length = block.length[i].detach().cpu().numpy()
            maxs = mins + length
            
            fidx = list(range(self.dim))
            min_list.append(mins)
            max_list.append(maxs)
            fidx_list.append(fidx)
            
        return np.array(min_list),np.array(max_list),np.array(fidx_list)
    
    def calc_box_importance(self,probs,y):
        probs = probs[y.bool()]

        # probs = probs[probs.max(1)[0] > self.prediction_threshold]
        # y_pred = torch.bincount(torch.argmax(probs,dim=1))
        # box_importance = torch.zeros(probs.shape[1],device=self.device)
        # box_importance[:len(y_pred)] = y_pred

        y_pred = (probs > self.prediction_threshold).long()
        box_importance = torch.sum(y_pred,dim=0)
        return box_importance

    
class BoxNetBranchesClassifier(BoxNetClassifier):
    def __init__(self, D,feature_subsets=None,nboxes=2, dim=2, tau=1, alpha=1, 
                 device='cpu', loss_fn=None, l1_reg=0, l2_reg=0,
                 training_epochs=100, learning_rate=1e-1, patience_early_stopping=50, 
                 alpha_tau_decay_step=1, prediction_threshold=0.5,box_pruning_step=100, box_pruning_threshold=1,
                 verbose=False, verbosity=10, random_state=42,num_threads=10):
        self.D = D      
        
        if feature_subsets is not None:
            dim = feature_subsets.shape[1]
            nboxes = feature_subsets.shape[0]
        self.feature_subsets = feature_subsets

        super().__init__(nboxes, dim, tau, alpha, device, loss_fn, l1_reg, l2_reg,training_epochs=training_epochs, box_pruning_step=box_pruning_step,
                          box_pruning_threshold=box_pruning_threshold,learning_rate=learning_rate, patience_early_stopping=patience_early_stopping, 
                          alpha_tau_decay_step=alpha_tau_decay_step,prediction_threshold=prediction_threshold, verbose=verbose, verbosity=verbosity, 
                          random_state=random_state,num_threads=num_threads)


        

    def setup_classifier(self):
        #self.model.reset_parameters()
        self.model = BoxNetBranches(
            D=self.D,
            nboxes=self.nboxes,
            dim = self.dim,
            tau=self.tau, 
            alpha=self.alpha,
            feature_subsets=self.feature_subsets,
            random_state=self.random_state,
            ).to(self.device)
        
        self.feature_subsets = self.model.feature_subsets
        
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, 
                patience=30, min_lr=1e-5)

        
    def fit(self, X, y):
        torch.set_num_threads(self.num_threads)
        
        self.setup_classifier()

        X = numpy_to_torch(X, self.device)
        y = numpy_to_torch(y, self.device)

        X = X[:, self.feature_subsets]  

        super()._train(X,y)

    
    def predict(self, X):
        feature_subsets = self.feature_subsets
        mask = self.model.block.box_mask.detach().cpu()
        feature_subsets = feature_subsets[mask]
        X = X[:, feature_subsets]
        return super().predict(X)
    
    def get_subsets(self):
        return self.model.feature_subsets
    
    def get_boxes(self):
        #return the boxes based on the mins and lenghts + subsets
        min_list,max_list,fidx_list = [],[],[]
        block = self.model.block
        for i in range(self.model.block.nboxes):
            mins  = block.mins[i].detach().cpu().numpy()
            length = block.length[i].detach().cpu().numpy()
            maxs = mins + np.abs(length)
            
            fidx = self.feature_subsets[i].tolist()
            min_list.append(mins)
            max_list.append(maxs)
            fidx_list.append(fidx)
            
        return np.array(min_list),np.array(max_list),np.array(fidx_list)

    