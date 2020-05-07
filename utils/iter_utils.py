import math


def batchItems(items, size, maxBatches=None):
    assert size > 0
    maxBatches = maxBatches or math.inf
    batch = []
    batchCounter = 0
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batchCounter += 1
            batch = []
            if batchCounter >= maxBatches:
                return
    if len(batch):
        yield batch