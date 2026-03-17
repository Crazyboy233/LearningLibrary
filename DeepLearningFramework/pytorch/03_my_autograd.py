class Value:
    def __init__(self, data, op='', children=()):
        self.data = data # 实际数值
        self.grad = 0.0 # 梯度
        self.op = op    # 加法/乘法/别的操作
        self.children = children # 记录依赖
        self._backward = lambda: None # 反向函数

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, op='+', children=(self, other))

        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1

        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, op='*', children=(self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out
    
    def backward(self):
        # 拓扑排序，确保依赖先反后反
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

a = Value(2.0)
b = Value(3.0)
c = a * b + a + 5
c.backward()

print(a.grad, b.grad)
