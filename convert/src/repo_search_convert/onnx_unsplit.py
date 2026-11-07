"""Rewrite every MatMul -> Split of the ONNX graph into one MatMul per part.

The graph fuses its projections — q/k/v, gate/up and the per-layer embedding
projection — into one MatMul whose output a last-axis Split cuts apart,
sometimes through ops that act on each column group alone (a grouped
RMSNorm, a per-channel scale). On CUDA that Split is a strided copy of every
token's activations. Slicing the weight's columns instead, and running the
in-between ops on each part, computes the same values without the copy.
"""

import collections

import numpy as np
import onnx
from onnx import helper, numpy_helper

_SEPARABLE = ("Reshape", "SimplifiedLayerNormalization", "Mul")


def _axis(node: onnx.NodeProto) -> int:
    return next((a.i for a in node.attribute if a.name == "axis"), 0)


class _Graph:
    def __init__(self, model: onnx.ModelProto):
        self.g = model.graph
        self.init = {i.name: i for i in self.g.initializer}

    def array(self, name: str) -> np.ndarray:
        return numpy_helper.to_array(self.init[name])

    def constant(self, name: str, value: np.ndarray) -> str:
        if name in self.init:
            raise RuntimeError(f"initializer {name} already exists")
        tensor = numpy_helper.from_array(np.ascontiguousarray(value), name)
        self.init[name] = tensor
        self.g.initializer.append(tensor)
        return name

    def producers(self) -> dict[str, onnx.NodeProto]:
        return {o: n for n in self.g.node for o in n.output}

    def consumers(self) -> dict[str, int]:
        count = collections.Counter(x for n in self.g.node for x in n.input)
        count.update(o.name for o in self.g.output)
        return count

    def is_last_axis_split(self, node: onnx.NodeProto) -> bool:
        return (
            node.op_type == "Split"
            and _axis(node) == -1
            and len(node.input) == 2
            and node.input[1] in self.init
        )


def _fold_nested_splits(graph: _Graph) -> int:
    """Folds a last-axis Split whose output only feeds another last-axis
    Split into the outer one."""
    folded = 0
    while True:
        users = collections.defaultdict(list)
        for n in graph.g.node:
            for x in n.input:
                users[x].append(n)
        hit = None
        for outer in graph.g.node:
            if not graph.is_last_axis_split(outer):
                continue
            for j, out in enumerate(outer.output):
                inner = users[out]
                if len(inner) == 1 and graph.is_last_axis_split(inner[0]):
                    hit = (outer, j, inner[0])
                    break
            if hit:
                break
        if hit is None:
            return folded
        outer, j, inner = hit
        sizes = [int(v) for v in graph.array(outer.input[1])]
        parts = [int(v) for v in graph.array(inner.input[1])]
        if sum(parts) != sizes[j]:
            raise RuntimeError(f"{inner.name}: parts {parts} do not sum to {sizes[j]}")
        outer.input[1] = graph.constant(
            f"{outer.name}/folded{folded}/split",
            np.array(sizes[:j] + parts + sizes[j + 1 :], dtype=np.int64),
        )
        outputs = list(outer.output[:j]) + list(inner.output) + list(outer.output[j + 1 :])
        del outer.output[:]
        outer.output.extend(outputs)
        graph.g.node.remove(inner)
        folded += 1


def _plan(graph: _Graph, chain: list[onnx.NodeProto], width: int, sizes: list[int]) -> list:
    """How each op between the MatMul and the Split runs on one part.

    The chain is column-separable when every op either reshapes between the
    full width and column groups that no part boundary cuts, normalizes one
    group at a time, or scales per channel.
    """
    plan, group = [], None
    for n in chain:
        if n.op_type == "Reshape":
            shape = [int(v) for v in graph.array(n.input[1])]
            if shape[-1] == width and all(v in (0, -1) for v in shape[:-1]):
                plan.append(("ungroup", shape))
                group = None
            elif len(shape) == 3 and shape[:2] == [0, -1] and width % shape[2] == 0:
                plan.append(("rows", shape))
                group = shape[2]
            elif len(shape) == 4 and shape[:2] == [0, 0] and shape[2] * shape[3] == width:
                plan.append(("groups", shape))
                group = shape[3]
            else:
                raise RuntimeError(f"{n.name}: reshape {shape} is not column-separable")
            if group is not None and any(p % group for p in sizes):
                raise RuntimeError(f"{n.name}: split {sizes} cuts a {group}-column group")
        elif n.op_type == "SimplifiedLayerNormalization":
            if group is None or _axis(n) != -1 or graph.array(n.input[1]).shape != (group,):
                raise RuntimeError(f"{n.name}: normalizes across part boundaries")
            plan.append(("norm", None))
        elif n.op_type == "Mul":
            if n.input[1] not in graph.init:
                raise RuntimeError(f"{n.name}: scale is not a constant")
            scale = graph.array(n.input[1])
            if scale.shape == (width,) and group is None:
                plan.append(("scale_slice", scale))
            elif group is not None and scale.shape == (group,):
                plan.append(("scale", scale))
            else:
                raise RuntimeError(f"{n.name}: scale {scale.shape} is not per channel")
        else:
            raise RuntimeError(f"{n.name}: {n.op_type} is not column-separable")
    return plan


def unsplit(model: onnx.ModelProto) -> dict:
    """Rewrites `model` in place; fails if any Split is left."""
    graph = _Graph(model)
    folded = _fold_nested_splits(graph)
    producers, consumers = graph.producers(), graph.consumers()
    new_nodes, dead, rewritten = [], [], 0
    for split in list(graph.g.node):
        if split.op_type != "Split":
            continue
        if not graph.is_last_axis_split(split):
            raise RuntimeError(f"{split.name}: not a last-axis Split with constant sizes")
        chain, x = [], split.input[0]
        while True:
            n = producers[x]
            if consumers[x] != 1:
                raise RuntimeError(f"{n.name}: output {x} has other consumers")
            if n.op_type == "MatMul" and n.input[1] in graph.init:
                matmul = n
                break
            if n.op_type not in _SEPARABLE:
                raise RuntimeError(f"{split.name}: reached {n.op_type} {n.name} before a MatMul")
            chain.append(n)
            x = n.input[0]
        chain.reverse()
        weight = graph.array(matmul.input[1])
        width = weight.shape[-1]
        sizes = [int(v) for v in graph.array(split.input[1])]
        if sum(sizes) != width:
            raise RuntimeError(f"{split.name}: sizes {sizes} do not sum to {width}")
        plan = _plan(graph, chain, width, sizes)
        start = 0
        for j, part in enumerate(sizes):
            tag = f"{matmul.name}/part{j}"
            end = start + part
            last = split.output[j]
            cur = last if not chain else f"{tag}/MatMul/output_0"
            new_nodes.append(
                helper.make_node(
                    "MatMul",
                    [matmul.input[0], graph.constant(f"{matmul.input[1]}/part{j}", weight[:, start:end])],
                    [cur],
                    name=f"{tag}/MatMul",
                )
            )
            for k, ((kind, data), n) in enumerate(zip(plan, chain)):
                out = last if k == len(chain) - 1 else f"{tag}/{k}/output_0"
                name = f"{tag}/{k}/{n.op_type}"
                if kind == "ungroup":
                    shape = graph.constant(f"{name}/shape", np.array(data[:-1] + [part], dtype=np.int64))
                    node = helper.make_node("Reshape", [cur, shape], [out], name=name)
                elif kind == "rows":
                    node = helper.make_node("Reshape", [cur, n.input[1]], [out], name=name)
                elif kind == "groups":
                    shape = graph.constant(
                        f"{name}/shape", np.array([0, 0, part // data[3], data[3]], dtype=np.int64)
                    )
                    node = helper.make_node("Reshape", [cur, shape], [out], name=name)
                elif kind == "norm":
                    node = helper.make_node(n.op_type, [cur, n.input[1]], [out], name=name, domain=n.domain)
                    node.attribute.extend(n.attribute)
                elif kind == "scale_slice":
                    scale = graph.constant(f"{n.input[1]}/part{j}", data[start:end])
                    node = helper.make_node("Mul", [cur, scale], [out], name=name)
                else:
                    node = helper.make_node("Mul", [cur, n.input[1]], [out], name=name)
                new_nodes.append(node)
                cur = out
            start = end
        dead.extend([split, matmul, *chain])
        rewritten += 1
    for n in dead:
        graph.g.node.remove(n)
    graph.g.node.extend(new_nodes)
    _drop_unused_initializers(graph.g)
    _sort_topologically(graph.g)
    left = sum(n.op_type == "Split" for n in graph.g.node)
    if left:
        raise RuntimeError(f"{left} Split nodes remain")
    return {"folded": folded, "rewritten": rewritten, "nodes": len(graph.g.node)}


def _drop_unused_initializers(g: onnx.GraphProto) -> None:
    used = {x for n in g.node for x in n.input}
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)


def _sort_topologically(g: onnx.GraphProto) -> None:
    ready = {i.name for i in g.input} | {i.name for i in g.initializer} | {""}
    order, rest = [], list(g.node)
    while rest:
        waiting = []
        for n in rest:
            if all(x in ready for x in n.input):
                order.append(n)
                ready.update(n.output)
            else:
                waiting.append(n)
        if len(waiting) == len(rest):
            raise RuntimeError("the rewritten graph has a cycle or a dangling input")
        rest = waiting
    del g.node[:]
    g.node.extend(order)
