class BestMetricRecorder(object):

    def __init__(self, init_value, metric_name):
        self.best_value = init_value
        self.latest_value = init_value
        self.metric_name = metric_name
        self.best_metric = None
        self.latest_metric = None

    def __call__(self):
        return self.best_value

    def value(self):
        return self.best_value

    def metric(self):
        return self.best_metric

    def improved(self, results):
        assert (self.metric_name in results)
        if self.compare(results[self.metric_name], self.best_value) > 0:
            self.best_value = results[self.metric_name]
            self.best_metric = results
            return True
        self.latest_value = results[self.metric_name]
        self.latest_metric = results
        return False

    def compare(self, current_value, best_value):
        """You must implement this method.
        if `current_value` better than `best_value` return 1
        elif `current_value` worse than `best_value` return -1
        else return 0
        """
        raise NotImplementedError