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


def unzip(iter_iters, *defaults):
    endOfIter = object()
    iter_iters = iter(iter_iters)
    row = next(iter_iters, endOfIter)

    if row == endOfIter:
        if len(defaults):
            return defaults
        else:
            raise Exception('endOfIter')

    lists = [[item] for item in row]
    expectedRowlen = len(lists)
    for row in iter_iters:
        for i, item in enumerate(row):
            lists[i].append(item)
        if (i + 1) != expectedRowlen:
            raise Exception('All rows must have equal length')
    return lists


if __name__ == '__main__':
    def unzip_test():
        list2d = [
            [1, 2, 3],
            [11, 22, 33],
            [111, 222, 333],
            [1111, 2222, 3333]
        ]
        l1, l2, l3 = unzip(list2d)
        assert l1 == [1, 11, 111, 1111]
        assert l2 == [2, 22, 222, 2222]
        assert l3 == [3, 33, 333, 3333]

        l1, l2, l3 = unzip([], [], [], [])
        assert l1 == []
        assert l2 == []
        assert l3 == []


    unzip_test()
