class BaseOOD:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.penultimate_layer = None
        self.register_hooks()

    # hooks
    def hook_function(self):
        def hook(_model, _input, output):
            self.penultimate_layer = output.flatten(1)  # (batch x channel x 1 x 1) -> (batch x channel)
        return hook

    def register_hooks(self):
        self.model.avgpool.register_forward_hook(self.hook_function())  # for pytorch imported resnet (for debugging)
        # self.model.GAP.register_forward_hook(self.hook_function())  # for manual resnet (batch x channel x 1 x 1)

    # apply method
    def apply_method(self, id_loader):
        pass

    # compute ood score
    def ood_score(self, dataloader):
        pass
