from dataclasses import fields
from collections.abc import Mapping

class BaseConfig(Mapping):
    def __iter__(self):
        for f in fields(self):
            yield f.name

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return len(fields(self))

    def get(self, key, default): 
        return getattr(self, key, default)