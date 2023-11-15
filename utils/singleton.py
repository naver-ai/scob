"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license

Define singleton metaclass
"""

from typing import Dict


class Singleton(type):
    _instances: Dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def deallocate_instance(cls):
        try:
            del Singleton._instances[cls]
        except KeyError:
            pass
