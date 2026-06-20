#!/usr/bin/env python3
"""从 jsonl 文件中提取 correct=false 的条目"""

import json
import argparse
from pathlib import Path


def filter_incorrect(input_file: str, output_file: str | None = None) -> int:
    input_path = Path(input_file)
    if output_file is None:
        output_path = input_path.with_stem(f"{input_path.stem}_incorrect")
    else:
        output_path = Path(output_file)

    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("correct") is False:
                fout.write(line + "\n")
                count += 1

    print(f"已提取 {count} 条 correct=false 的条目")
    print(f"输出文件: {output_path}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 jsonl 中提取 correct=false 的条目")
    parser.add_argument("input", help="输入 jsonl 文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径 (默认: input_incorrect.jsonl)")
    args = parser.parse_args()

    filter_incorrect(args.input, args.output)
