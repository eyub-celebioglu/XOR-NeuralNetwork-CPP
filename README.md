# XOR-NeuralNetwork-CPP

A small multi-layer perceptron written from scratch in C++ (standard library only) and trained with backpropagation to learn XOR.

I wrote it to understand what happens inside a neural network without a framework doing the work.

## Network

- architecture `2 → 4 → 1`, fully connected
- `tanh` activation, weights and biases initialised uniformly in [-1, 1]
- plain gradient descent on the four XOR samples, 10,000 epochs, learning rate 0.1
- the average error is printed every 1,000 epochs

XOR is the classic example of a problem that is not linearly separable, so it needs at least one hidden layer:

```
x1  x2 | y
 0   0 | 0
 0   1 | 1
 1   0 | 1
 1   1 | 0
```

## Notes

I started with a sigmoid activation and switched to `tanh`. Because it is centred on zero and has larger gradients, training converged more reliably on this problem.

Backpropagation uses the usual deltas:

- output layer: `δ = (target − output) · f'(z)`
- hidden layers: `δ = (Σ δ_next · w) · f'(z)`
- update: `w ← w + η · δ · input`

## Build and run

```bash
g++ -std=c++11 -O2 neural-net.cpp -o neural-net
./neural-net
```

## Result

```
input 0 0 → 0.0003   (expected 0)
input 0 1 → 0.9991   (expected 1)
input 1 0 → 0.9991   (expected 1)
input 1 1 → -0.0006  (expected 0)
```

## Possible next steps

- parallelise the per-neuron computations with OpenMP
- mini-batch training and a real dataset (e.g. MNIST)
