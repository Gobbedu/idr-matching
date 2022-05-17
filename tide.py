import time

def timestamp():
    return time.time()

tt = 0

def time_spent(function):
    def wrapper(*args, **kwargs):
        ini = timestamp()
        ret = function(*args, **kwargs)
        fin = timestamp()
        global tt
        tt = fin - ini
        return ret
    return wrapper
