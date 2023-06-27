# A decorator used for calculate fuction's cost time
def cost_time(func):
    from time import time
    def inner(*args):
        start_time = time()
        ret = func(*args)
        end_time = time()
        spent_time = end_time - start_time
        print('%s running time: %fs' % (func.__name__, spent_time))
        return ret
    return inner