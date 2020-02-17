import csv


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, "w")
        self.logger = csv.writer(self.log_file, delimiter="\t")

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, "r") as input_file:
        value = float(input_file.read().rstrip("\n\r"))

    return value


def calculate_accuracy(predict, target, top_n=1):
    assert len(target) == len(predict)
    n_samples = len(target)
    correct = 0
    _, topk = predict.topk(top_n, dim=1)

    # to do
    # more effective
    for index, item in enumerate(target.data):
        t = topk.cpu().data[index]
        if (item in t) == True:
            correct += 1

    return 1.0 * correct / n_samples


def calculate_accuracy(predict, target, top_n=1):
    assert len(target) == len(predict)
    n_samples = len(target)
    correct = 0
    _, topk = predict.topk(top_n, dim=1)

    # to do
    # more effective
    for index, item in enumerate(target.data):
        t = topk.cpu().data[index]
        if (item in t) == True:
            correct += 1

    return 1.0 * correct / n_samples
