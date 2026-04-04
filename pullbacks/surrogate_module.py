from torch import nn


class SurrogateModule(nn.Module):
    def __init__(self, *args, temperature=1.0, standard_backward=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.standard_backward = standard_backward

    def extra_repr(self):
        ret = super().extra_repr()
        if ret:
            ret += ", "
        return f"{ret}standard_backward={self.standard_backward}, temperature={self.temperature}"

    @classmethod
    def replace_class_with_surrogate(
        cls, module, temperature=1.0, standard_backward=False
    ):
        module.__class__ = cls
        module.temperature = temperature
        module.standard_backward = standard_backward
