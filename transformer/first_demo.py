import torch
import numpy as np
import matplotlib.pyplot as plt
from pyitcast.transformer_utils import Batch

from pyitcast.transformer_utils import get_std_opt
from pyitcast.transformer_utils import LabelSmoothing
from pyitcast.transformer_utils import SimpleLossCompute
from transformer import *

from pyitcast.transformer_utils import run_epoch

from pyitcast.transformer_utils import greedy_decode


def data_generator(V, batch, num_batch):
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10), dtype='int64'))

        data[:, 0] = 1

        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        yield Batch(source, target)


V = 10
batch_size = 20
num_batch = 30

# res = data_generator(V, batch_size, num_batch)
# print(res)


model = make_model(V, V, N=2)

model_optimizer = get_std_opt(model)

criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

loss = SimpleLossCompute(model.generator, criterion, model_optimizer)


def run(model, loss, epochs=5):
    for epoch in range(epochs):
        model.train()
        run_epoch(data_generator(V, 10, 20), model, loss)
        model.eval()
        run_epoch(data_generator(V, 10, 5), model, loss)

    model.eval()

    source = Variable(torch.LongTensor([[1, 3, 2, 5, 4, 6, 7, 8, 9]]))
    source_mask = Variable(torch.ones(1, 1, 9))

    result = greedy_decode(model, source, source_mask, max_len=9, start_symbol=1)
    print(result)


run(model, loss)
pass
