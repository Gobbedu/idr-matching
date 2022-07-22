import time

import file_handling
import gen_graph

# time functions
tt = 0

def timer(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = function(*args, **kwargs)
        print("time: %.2f" % (time.time() - start))
        return ret
    return wrapper


base_dir = './data/2-seg/'

@timer  # funobj = timer(fun) -> funobj()
def graph_test():
    exception_counter = 0
    exception_files = []

    filelist = file_handling.get_list_of_files(base_dir)
    print("%d images found" % len(filelist))

    for img_path in filelist:
        try:
            print("current image: %s" % img_path)
            gen_graph.graph_routine(img_path)
        except Exception as ex:
            print("exception %s on image: %s" % (ex, img_path))
            exception_counter += 1
            exception_files.append(img_path)
    print("%d/%d images failed: %s" % (exception_counter, len(filelist), exception_files))

graph_test()
