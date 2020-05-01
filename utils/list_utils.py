def splitList(items, numOfParts):
    assert 0 < numOfParts <= len(items)
    partLen = len(items) // numOfParts
    for currentPart in range(numOfParts):
        partStart = currentPart * partLen
        yield items[partStart:partStart + partLen]


if __name__ == '__main__':
    def splitList_test():
        l = [1, 2, 3, 4]
        r = splitList(l, 4)
        assert [*r] == [[1], [2], [3], [4]]

        l = [1, 2, 3, 4]
        r = splitList(l, 1)
        assert [*r] == [[1, 2, 3, 4]]

        l = [1, 2, 3, 4]
        r = splitList(l, 2)
        assert [*r] == [[1, 2], [3, 4]]

        l = [1, 2, 3, 4]
        r = splitList(l, 3)
        assert [*r] == [[1], [2], [3]]


    splitList_test()
