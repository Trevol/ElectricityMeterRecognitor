from time import sleep

from tqdm import tqdm


def main():
    # https://stackoverflow.com/questions/37506645/can-i-add-message-to-the-tqdm-progressbar
    items = [(i, str(i)) for i in range(10)]
    pbar = tqdm(items, 'test-test test')
    for item in pbar:
        pbar.set_description(str(item))
        sleep(.3)


def main():
    # https://stackoverflow.com/questions/37506645/can-i-add-message-to-the-tqdm-progressbar
    n = 10
    items = ((i, str(i)) for i in range(n))
    pbar = tqdm(items, 'test-test test', total=n)
    for item in pbar:
        # pbar.set_description(str(item))
        sleep(.3)


main()
