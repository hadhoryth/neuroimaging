import sys
sys.path.insert(0, '/Users/XT/Documents/GitHub/neat-python/')
from neat.genes import BaseGene, DefaultConnectionGene
from neat.genome import DefaultGenome, DefaultGenomeConfig
from neat.attributes import FloatAttribute, StringAttribute, BaseAttribute

# TODO implement mutation for bias and weigths as tensors


class TensorAttribute(BaseAttribute):
    pass


class TensorGene(BaseGene):
    """
        For now weights are not mutating
        Aggregation is handling inside tensor neural_net class
    """
    _gene_attributes = [FloatAttribute('bias'),
                        StringAttribute('activation', options='sigmoid')]

    def __init__(self, key):
        assert isinstance(
            key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)
        self.weights = []

    def distance(self, other, config):
        d = abs(self.bias - other.bias)  # + abs(self.response - other.response)
        if self.activation != other.activation:
            d += 1.0
        return d * config.compatibility_weight_coefficient

    @classmethod
    def get_config_params(cls):
        params = []
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(
                cls, '__gene_attributes__'))
            warnings.warn(
                "Class '{!s}' {!r} needs '_gene_attributes' not '__gene_attributes__'".format(
                    cls.__name__, cls),
                DeprecationWarning)
        for a in cls._gene_attributes:
            if hasattr(a, 'get_config_params'):
                params += a.get_config_params()
        return params

    def init_attributes(self, config):
        for a in self._gene_attributes:
            if hasattr(a, 'name'):
                setattr(self, a.name, a.init_value(config))


class TensorGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = TensorGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)
