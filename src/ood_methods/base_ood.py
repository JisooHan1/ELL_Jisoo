class BaseOOD:
    def __init__(self, model):
        self.model = model
        self.penultimate_layer = None
        self.register_hooks()

    # hooks
    def hook_function(self):
        def hook(_model, _input, output):
            self.penultimate_layer = output
        return hook

    def register_hooks(self):
        self.model.avgpool.register_forward_hook(self.hook_function())

    # apply method
    def apply_method(self, id_loader):
        pass

    # compute ood score
    def ood_score(self, dataloader):
        pass
