import numpy as np

# n 表示數據大小
def create_dataset(n):
    dataset = np.random.random((n, 2))
    return list(dataset)


if __name__ == '__main__':
    print(create_dataset(5))
