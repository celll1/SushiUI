"""Track leaf accumulation before mutable weights can leave their device."""
import torch


class ParameterUpdateFence:
    def __init__(self, parameters):
        self.parameters = tuple(parameters)
        self.reset()

    def reset(self):
        self.task = None
        self.expected = None
        self.accumulated = set()
        self.nodes = None

    def capture_forward_nodes(self):
        # Device rebinding recreates AccumulateGrad; retain the original forward's nodes.
        self.nodes = tuple(torch.autograd.graph.get_gradient_edge(p).node for p in self.parameters)

    def _snapshot(self):
        task = torch._C._current_graph_task_id()
        if task < 0:
            raise RuntimeError("Mutable update fence requires an active backward graph")
        if self.task is not None and self.task != task:
            raise RuntimeError("Mutable update fence was not released before the next backward")
        if self.expected is None:
            self.task = task
            # This is also the engine query used by Torch's register_multi_grad_hook.
            nodes = self.nodes or tuple(torch.autograd.graph.get_gradient_edge(p).node for p in self.parameters)
            self.expected = {id(p) for p, node in zip(self.parameters, nodes)
                             if torch._C._will_engine_execute_node(node)}

    def observe_accumulation(self, parameter):
        self._snapshot()
        self.accumulated.add(id(parameter))

    def backward_complete(self):
        self._snapshot()

    def ready(self):
        return (self.expected is not None and self.expected <= self.accumulated
                and all(p.grad is None for p in self.parameters if id(p) in self.expected))
