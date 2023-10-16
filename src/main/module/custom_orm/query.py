from sqlalchemy import Table, and_, or_


def _convert(table, query_list) -> list:
    res = []
    for query in query_list:
        res.append(query(table))
    return res


class Query:
    def __call__(self, table: Table):
        pass


class And(Query):
    def __init__(self, *query_list):
        self.query_list = query_list

    def __call__(self, table):
        return and_(*_convert(table, self.query_list))


class Or(Query):
    def __init__(self, *query_list):
        self.query_list = query_list

    def __call__(self, table):
        return or_(*_convert(table, self.query_list))


class Between(Query):
    def __init__(self, name, lte, gte):
        self.column_name = name
        self.duration = (lte, gte)

    def __call__(self, table):
        return table.columns.__getattr__(self.column_name).between(*self.duration)


class Column(Query):
    def __init__(self, name, value):
        self.column_name = name
        self.value = value

    def __call__(self, table):
        return table.columns.__getattr__(self.column_name) == self.value
