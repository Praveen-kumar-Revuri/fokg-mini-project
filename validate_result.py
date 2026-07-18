"""Validate result.ttl against the mini-project requirements."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from rdflib import Graph, RDF


def validate(result_file: str, test_file: str) -> bool:
    test_graph = Graph()
    test_graph.parse(test_file, format="nt")
    test_uris = {str(s) for s in test_graph.subjects(RDF.type, RDF.Statement)}

    result_uris: set[str] = set()
    scores: list[float] = []
    bad_format = 0

    with open(result_file, encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue

            if "hasTruthValue" not in line or "XMLSchema#double" not in line:
                bad_format += 1
                continue

            uri = line.split(">")[0].strip("<")
            result_uris.add(uri)

            match = re.search(r'"([0-9.eE+-]+)"', line)
            if match:
                scores.append(float(match.group(1)))

    missing = test_uris - result_uris
    extra = result_uris - test_uris
    ok = (
        not missing
        and not extra
        and not bad_format
        and scores
        and min(scores) >= 0.0
        and max(scores) <= 1.0
    )

    print(f"Test facts:        {len(test_uris)}")
    print(f"Result lines:      {len(result_uris)}")
    print(f"Missing from result: {len(missing)}")
    print(f"Extra in result:     {len(extra)}")
    print(f"Bad format lines:    {bad_format}")
    if scores:
        print(f"Score range:         [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"Overall:             {'PASS' if ok else 'FAIL'}")

    if missing:
        print(f"Example missing: {next(iter(missing))}")
    return ok


if __name__ == "__main__":
    result_path = sys.argv[1] if len(sys.argv) > 1 else "result.ttl"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "data/KG-2022-test.nt.txt"
    success = validate(result_path, test_path)
    sys.exit(0 if success else 1)
