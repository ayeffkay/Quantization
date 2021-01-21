import torch
from catalyst.runners.supervised import SupervisedRunner
from sklearn.metrics import r2_score, mean_squared_error

import set_seed


class CustomRunner(object):
    def __init__(self, model=None, device=torch.device('cpu'), 
                 input_key='features', input_target_key='targets', 
                 train=False, evaluate=False, loaders=None, 
                 optimizer=None, criterion=None, num_epochs=1):
        
        self.model = model
        self.runner = SupervisedRunner(model=model, device=device, 
                                       input_key=input_key, 
                                       input_target_key=input_target_key)
        self.logs = None
        if train:
            self.train(optimizer, criterion, loaders, num_epochs)
        if evaluate:
            self.logs = self.test(loaders['test'])
            
            
    def train(self, optimizer, criterion, loaders, num_epochs):
        loaders = {'train': loaders['train'], 
                   'valid': loaders['valid']}
        self.runner.train(model=self.model, 
                          optimizer=optimizer, 
                          criterion=criterion, 
                          loaders=loaders, 
                          logdir='./logs', 
                          num_epochs=num_epochs,
                          verbose=True, 
                          load_best_on_end=True, 
                          initial_seed=set_seed.SEED, 
                          valid_loader='valid')
    
    
    def test(self, loader):
        targets = torch.cat([b['targets'].view(-1) for b in loader]).cpu().numpy()
        outputs = torch.cat([o['logits'].view(-1) for o in self.runner.predict_loader(loader=loader)]).cpu().numpy()
        
        r2 = r2_score(targets, outputs)
        mse = mean_squared_error(targets, outputs)
        
        return {'mse': mse, 'r^2': r2}
