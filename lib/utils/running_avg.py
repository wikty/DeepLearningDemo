class RunningAvg():
    """Running average of a quantity or a list/tuple/dict of quantities.

    Example:
    ```
    # average for numbers
    loss_avg = RunningAvg()
    loss_avg.step(2)
    loss_avg.step(4)
    loss_avg()  # return 3.0

    # average for a dict of numbers
    loss_avg = RunningAvg()
    loss_avg.step({'loss': 1, 'accuracy': 2})
    loss_avg.update({'loss': 0, 'accuracy': 4})
    loss_avg()  # return {'loss': 0.5, 'accuracy': 3.0}
    ```
    """

    def __init__(self):
        self.data = None
        self.steps = 0

    def _v(self, val):
        try:
            val = float(val)
        except Exception as e:
            msg = 'Running average quantity must be able to convert to float.'
            raise Exception(msg)
        return val

    def process(self, val):
        assert isinstance(val, (int, float, list, tuple, dict))
        if isinstance(val, (int, float)):
            val = self._v(val)
        elif isinstance(val, (tuple, list)):
            val = [self._v(v) for v in val]
        elif isinstance(val, dict):
            val = {k:self._v(v) for k, v in val.items()}
        return val

    def reset(self):
        self.data = None
        self.steps = 0

    def step(self, val):
        if self.data is None:
            self.data = self.process(val)
        elif isinstance(self.data, float):
            assert isinstance(val, (int, float))
            self.data += self.process(val)
        elif isinstance(self.data, list):
            assert isinstance(val, (tuple, list))
            assert len(val) == len(self.data)
            val = self.process(val)
            for i in range(len(val)):
                self.data[i] += val[i]
        elif isinstance(self.data, dict):
            assert isinstance(val, dict)
            assert len(val) == len(self.data)
            val = self.process(val)
            for key in val:
                self.data[key] += val[key]
        self.steps += 1

    def __call__(self):
        msg = "RunningAvg step is zero."
        assert self.steps != 0
        if isinstance(self.data, float):
            return self.data / self.steps
        elif isinstance(self.data, list):
            return [v/self.steps for v in self.data]
        else:
            return {k:(v/self.steps) for k, v in self.data.items()}

