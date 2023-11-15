"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

import ujson


class JsonlReader:
    def __init__(self, jsonl_file_path, is_pytest=False):
        self.jsonl_file_path = jsonl_file_path
        self.offsets = [0]
        self.jsonl_size = 0

        with open(self.jsonl_file_path, "r", encoding="utf-8") as f:
            while f.readline():
                self.offsets.append(f.tell())
                self.jsonl_size += 1

    def read_jsonl(self, idx):
        with open(self.jsonl_file_path, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            json_data = ujson.loads(line)
        return json_data
