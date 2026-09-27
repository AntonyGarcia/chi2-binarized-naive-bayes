"""
Parse the output of a comparison script (implementations/*/*.py) captured with
capture_test_pvalues.py: k=10 AUC [95% CI] of each model section, and the rows of every
DeLong / McNemar section with their full-precision p-values from the JSON record.
"""

import json
import re

RULE = re.compile(r"^\s*=+\s*$")
DASHES = re.compile(r"^\s*-{10,}\s*$")


def _auc_at_k10(body):
    for line in body.splitlines():
        m = re.match(r"^\s+10\s+.*?(\d\.\d+) \[(\d\.\d+), (\d\.\d+)\]\s*$", line)
        if m:
            return tuple(float(v) for v in m.groups())
    return None


def parse(txt_path, json_path):
    text = open(txt_path, encoding="utf-8", errors="replace").read()
    lines = text.splitlines()

    # Model sections: a title line enclosed by two rules, followed by a k table.
    aucs = {}
    for i in range(1, len(lines) - 1):
        if RULE.match(lines[i - 1]) and RULE.match(lines[i + 1]) and lines[i].strip():
            title = lines[i].strip()
            end = next((j for j in range(i + 2, len(lines)) if RULE.match(lines[j])), len(lines))
            auc = _auc_at_k10("\n".join(lines[i + 2:end]))
            if auc and "Test" not in title:
                aucs.setdefault(title, auc)

    # Test sections, in printed order.
    sections = []
    i = 0
    while i < len(lines):
        m = re.search(r"(DeLong|McNemar) Test: .*?vs (.+?)\s*\(k=10\)", lines[i])
        if m:
            kind, ref = m.group(1), m.group(2).strip()
            j = i + 1
            while j < len(lines) and not DASHES.match(lines[j]):
                j += 1
            rows = []
            for line in lines[j + 1:]:
                if not line.strip():
                    break
                name = re.split(r"\s{2,}", line.strip())[0]
                tokens = line.strip()[len(name):].split()
                p_str = tokens[4] if kind == "DeLong" else tokens[3]
                better = float(tokens[2]) > 0 if kind == "DeLong" else int(tokens[0]) > int(tokens[1])
                rows.append({"name": name, "p_str": p_str, "better": better})
            sections.append({"test": kind, "reference": ref, "rows": rows})
            i = j
        i += 1

    # Attach full-precision p-values: calls are recorded in the order they are made.
    calls = json.load(open(json_path, encoding="utf-8"))
    used = [False] * len(calls)
    for sec in sections:
        for row in sec["rows"]:
            decimals = len(row["p_str"].split(".")[1])
            target = float(row["p_str"])
            for k, call in enumerate(calls):
                if used[k] or call["test"] != sec["test"]:
                    continue
                if round(call["p_value"], decimals) == round(target, decimals):
                    used[k] = True
                    row["p"] = call["p_value"]
                    break
            else:
                raise RuntimeError(f"{txt_path}: no recorded p-value for {sec['test']} {row['name']} {row['p_str']}")
    return aucs, sections


if __name__ == "__main__":
    import sys

    aucs, sections = parse(sys.argv[1], sys.argv[2])
    for title, auc in aucs.items():
        print(f"AUC  {title:<45} {auc[0]:.3f} [{auc[1]:.3f}, {auc[2]:.3f}]")
    for sec in sections:
        print(f"-- {sec['test']} vs {sec['reference']}")
        for row in sec["rows"]:
            print(f"   {row['name']:<45} p={row['p']:.6g}  better={row['better']}")
