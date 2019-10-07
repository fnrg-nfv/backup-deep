import random

def uniform(s: int = 0, d: int = 100, n: int = 100):
    """
    Uniform sampler
    :param s: range start
    :param d: range destination
    :param n: number of samples
    :return: list of samples in increasing order
    """
    sample_result = []
    for i in range(n):
        sample_result.append(random.uniform(s, d))
    sample_result.sort()
    return sample_result

# test
def main():
    # uniform
    print(uniform(0, 100, 100))

if __name__ == '__main__':
    main()