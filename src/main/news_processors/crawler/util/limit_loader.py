import yaml

limit_file = 'resources/config/crawler/limit.yml'


class LimitLoader:
    @staticmethod
    def get_limit() -> dict:
        with open(limit_file, 'r', encoding='UTF-8') as limit_yml:
            limit = yaml.safe_load(limit_yml)
        return limit

    @staticmethod
    def update_limit(limit: dict):
        with open(limit_file, 'w', encoding='UTF-8') as limit_yml:
            yaml.dump(limit, limit_yml, default_flow_style=False)
