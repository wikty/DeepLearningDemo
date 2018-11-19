import csv
from io import StringIO

from tabulate import tabulate


class Table(object):

    def __init__(self):
        self.data = {}
        self.num_rows = 0

    def __len__(self):
        """Return a tuple of (num_rows, num_columns)."""
        return self.num_rows, len(self.data)

    def __iter__(self):
        for i in range(self.num_rows):
            yield self.row(i)

    @property
    def shape(self):
        return self.__len__()

    def row_count(self):
        return self.num_rows

    def column_count(self):
        return len(self.data)

    def clear(self):
        self.data = {}
        self.num_rows = 0

    def row(self, i):
        """
        Args:
            i (int): the index of the row. the index syntax is same
            with the python built-in list.
        """
        row = {}
        for key in self.data:
            try:
                row[key] = self.data[key][i]
            except IndexError as e:
                raise IndexError('The index out of the range of table.')
        return row

    def column(self, header):
        return [v for v in self.data.get(header, [])]

    def mean(self, headers=[]):
        assert set(headers).issubset(set(self.data.keys()))
        result = {k:0.0 for k in headers}
        for row in self:
            for header in headers:
                result[header] += row[header]
        for header in result:
            result[header] /= float(self.num_rows)
        return result

    def max(self, header):
        assert header in self.data
        max_value = None
        max_row = None
        for row in self:
            if (max_value is None) or (max_value < row[header]):
                max_value = row[header]
                max_row = row 
        return max_row

    def min(self, header):
        assert header in self.data
        min_value = None
        min_row = None
        for row in self:
            if (min_value is None) or (min_value > row[header]):
                min_value = row[header]
                min_row = row 
        return min_row

    def filter(self, callback, **kwargs):
        """Filter the rows of table and return a generator for the 
        return value of the callback function.
        
        Args:
            callback (function): the interface of function is `callback(
            row_dict, **kwargs) -> the value to generator`
        """
        for i in range(self.num_rows):
            row = self.row(i)
            rtn = callback(row, **kwargs)
            if rtn is not None:
                yield rtn

    def insert(self, row, i):
        """insert a new row on the index `i`.
        Args:
            row (dict|list|tuple): the data of the new row.
            i (int): the index position for the new row. the index 
                syntax is same with the python built-in list.
        """
        assert isinstance(row, (list, tuple, dict))
        if self.data:
            assert len(self.data) == len(row)

        if isinstance(row, (list, tuple)):
            keys, values = range(len(row)), row
        else:
            keys, values = row.keys(), row.values()
        
        for key, value in zip(keys, values):
            if key not in self.data:
                self.data[key] = [value]
            else:
                try:
                    self.data[key].insert(i, value)
                except IndexError as e:
                    raise IndexError('The index out of the range of table.')
        self.num_rows += 1

    def prepend(self, row):
        """prepend a new row."""
        self.insert(row, 0)

    def append(self, row):
        """append a new row."""
        self.insert(row, self.num_rows)

    def extend(self, rows):
        """
        Args:
            rows (list): a list of dict.
        """
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert isinstance(rows[0], dict)
        for row in rows:
            self.append(row)

    def insert_column(self, header, values):
        """
        Args:
            header (str): the header name for the new column. if 
                the header is already in table, will update it.
            values (list|tuple): the element of list is the value
                for each row in the new column.
        """
        assert isinstance(values, (list, tuple))
        if self.data:
            assert len(self.data[self.data.keys()[0]]) == len(values)
        self.data[header] = [v for v in values]

    def tabulate(self, fmt='pipe'):
        txt = tabulate(self.data, 
                       headers='keys', tablefmt=fmt)
        return txt

    def csv(self):
        rows = [{} for i in range(self.num_rows)]
        for key in self.data:
            for i, value in enumerate(self.data[key]):
                rows[i][key] = value
        f = StringIO(newline=None)
        writer = csv.DictWriter(f, fieldnames=self.data.keys())
        writer.writeheader()
        writer.writerows(rows)
        txt = f.getvalue()
        f.close()
        return txt