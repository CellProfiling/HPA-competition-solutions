from tensorboardX import SummaryWriter
from fastai.sgdr import *

class TensorboardLogger(Callback):
    def __init__(self, model, md, log_name, metrics_names=[], path=None, histogram_freq=100):
        super().__init__()
        self.model = model
        self.md = md
        self.metrics_names = ["validation_loss"]
        self.metrics_names += metrics_names
        self.histogram_freq = histogram_freq
        
        path = path or os.path.join(md.path, "logs")
        self.log_dir = os.path.join(path, log_name)
        
    def on_train_begin(self):
        self.iteration = 0
        self.epoch = 0
        self.writer = SummaryWriter(log_dir=self.log_dir)
    def on_batch_begin(self): pass
    def on_phase_begin(self): pass
    def on_epoch_end(self, metrics):
        self.epoch += 1
        
        for val, name in zip(metrics, self.metrics_names):
            self.writer.add_scalar(name, val, self.iteration) 
        
        
        for name, emb in self.model.named_children():
            if isinstance(emb, nn.Embedding):
                self.writer.add_embedding(list(emb.parameters())[0], global_step=self.iteration, tag=name)
                
    def on_phase_end(self): pass
    def on_batch_end(self, loss):
        self.iteration += 1
        self.writer.add_scalar("loss", loss, self.iteration)
        
        if self.iteration%self.histogram_freq==0:
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param, self.iteration)
    def on_train_end(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dummy_input = tuple(next(iter(self.md.trn_dl))[:-1])
                self.writer.add_graph(self.model, dummy_input)
        except Exception as e:
            print("Unable to create graph.")
            print(e)
        self.writer.close()