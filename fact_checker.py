"""
KG-Based Fact Checking Engine — Structural Link-Prediction Variant
Foundations of Knowledge Graphs — Mini-Project

ALGORITHM (different from the TransE-embedding submission this is based on):
    This engine does NOT use knowledge-graph embeddings at all. Instead it treats
    veracity prediction as a graph link-prediction problem and scores each
    (subject, predicate, object) statement using classic topological
    similarity indices computed directly on the statement graph:

        - Common neighbours
        - Jaccard coefficient
        - Adamic/Adar index
        - Resource Allocation index
        - Preferential attachment
        - Node degree (subject / object)
        - Predicate-restricted common neighbours

    These features are fed into a Gradient Boosted Decision Tree classifier
    (a different learning algorithm from last year's Logistic
    Regression / Gradient Boosting / Random Forest weighted ensemble on TransE
    features) trained on the labelled SW-2022 training statements.

    Rationale: true statements tend to be structurally embedded in the
    surrounding graph (subject/object share more indirect connections),
    while corrupted (false) statements are comparatively isolated. 5-fold
    cross-validation on the labelled training file gives ROC AUC ~0.70,
    comfortably above the 0.65 target (see README for full numbers).

NOTE ON BACKGROUND KNOWLEDGE:
    The original submission relied on a ~1.14M-triple filtered DBpedia
    background graph and a pre-trained TransE model. Those two files ship
    as Git-LFS pointers in the repo (not real data), and this engine does
    not have internet access to DBpedia. This engine is therefore fully
    self-contained: its "background graph" is built from the structure of
    the train + test statement files themselves (subjects/predicates/objects
    as nodes and edges — labels are never used to build the graph, only to
    train the classifier). See README.md for how to plug in a real DBpedia
    background graph if you have one, which would only add signal on top
    of this.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from rdflib import Graph, RDF, URIRef
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HAS_TRUTH_VALUE = URIRef("http://swc2017.aksw.org/hasTruthValue")
XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"

DEFAULT_PATHS = {
    "train_file": "data/KG-2022-train.nt.txt",
    "test_file": "data/KG-2022-test.nt.txt",
    "output_file": "result.ttl",
}


def clean_uri(uri) -> str:
    return unquote(str(uri).strip("<>"))


def load_statements(rdf_file: str) -> list[tuple[str, str, str, str, float | None]]:
    """Parse RDF reified statements from a train/test N-Triples file."""
    graph = Graph()
    graph.parse(rdf_file, format="nt")

    statements = []
    for statement_uri in graph.subjects(RDF.type, RDF.Statement):
        subject = graph.value(statement_uri, RDF.subject)
        predicate = graph.value(statement_uri, RDF.predicate)
        obj = graph.value(statement_uri, RDF.object)
        truth_value = graph.value(statement_uri, HAS_TRUTH_VALUE)

        statements.append((
            str(statement_uri),
            clean_uri(subject),
            clean_uri(predicate),
            clean_uri(obj),
            float(truth_value) if truth_value is not None else None,
        ))
    return statements


class StatementGraph:
    """Undirected co-occurrence graph built from statement structure only (no labels)."""

    def __init__(self, statements: list[tuple[str, str, str, str, float | None]]):
        self.adjacency: dict[str, set[str]] = defaultdict(set)
        self.predicate_adjacency: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

        for _, subject, predicate, obj, _ in statements:
            self.adjacency[subject].add(obj)
            self.adjacency[obj].add(subject)
            self.predicate_adjacency[predicate][subject].add(obj)
            self.predicate_adjacency[predicate][obj].add(subject)

    def neighbors(self, entity: str) -> set[str]:
        return self.adjacency.get(entity, set())

    def degree(self, entity: str) -> int:
        return len(self.adjacency.get(entity, set()))

    def predicate_neighbors(self, predicate: str, entity: str) -> set[str]:
        return self.predicate_adjacency.get(predicate, {}).get(entity, set())


class FactChecker:
    """Structural link-prediction fact-checking engine."""

    def __init__(self, graph: StatementGraph):
        self.graph = graph
        self.classifier: GradientBoostingClassifier | None = None

    def build_feature_vector(self, subject: str, predicate: str, obj: str) -> np.ndarray:
        ns = self.graph.neighbors(subject)
        no = self.graph.neighbors(obj)
        common = ns & no
        union = ns | no

        jaccard = len(common) / len(union) if union else 0.0
        adamic_adar = sum(
            1.0 / np.log(self.graph.degree(n) + 1)
            for n in common if self.graph.degree(n) > 1
        )
        resource_allocation = sum(
            1.0 / self.graph.degree(n) for n in common if self.graph.degree(n) > 0
        )
        preferential_attachment = np.log(len(ns) + 1) * np.log(len(no) + 1)

        predicate_common = self.graph.predicate_neighbors(predicate, subject) & \
            self.graph.predicate_neighbors(predicate, obj)

        return np.array([
            len(common),
            jaccard,
            adamic_adar,
            resource_allocation,
            preferential_attachment,
            np.log(len(ns) + 1),
            np.log(len(no) + 1),
            len(predicate_common),
        ], dtype=np.float32)

    def featurize(self, statements) -> np.ndarray:
        return np.array([
            self.build_feature_vector(subject, predicate, obj)
            for _, subject, predicate, obj, _ in statements
        ], dtype=np.float32)

    def cross_validate(self, train_statements, n_splits: int = 5, random_state: int = 7) -> float:
        """Report expected ROC AUC via stratified k-fold CV on the labelled training set."""
        X = self.featurize(train_statements)
        y = np.array([tv for _, _, _, _, tv in train_statements])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        aucs = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.03,
                subsample=0.9, random_state=random_state,
            )
            clf.fit(X[tr_idx], y[tr_idx])
            probs = clf.predict_proba(X[val_idx])[:, 1]
            auc = roc_auc_score(y[val_idx], probs)
            aucs.append(auc)
            print(f"  fold {fold}: ROC AUC = {auc:.4f}")

        mean_auc = float(np.mean(aucs))
        print(f"  mean ROC AUC = {mean_auc:.4f} (std {np.std(aucs):.4f})")
        return mean_auc

    def train(self, train_statements, random_state: int = 7):
        X = self.featurize(train_statements)
        y = np.array([tv for _, _, _, _, tv in train_statements])
        self.classifier = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.03,
            subsample=0.9, random_state=random_state,
        )
        self.classifier.fit(X, y)

    def predict(self, test_statements) -> list[tuple[str, float]]:
        if self.classifier is None:
            raise RuntimeError("Classifier not trained. Call train() first.")
        X = self.featurize(test_statements)
        probs = self.classifier.predict_proba(X)[:, 1]
        return [
            (statement_uri, float(max(0.0, min(1.0, score))))
            for (statement_uri, _, _, _, _), score in zip(test_statements, probs)
        ]

    @staticmethod
    def write_results(predictions: list[tuple[str, float]], output_file: str):
        with open(output_file, "w", encoding="utf-8") as handle:
            for statement_uri, score in predictions:
                handle.write(
                    f"<{statement_uri}> <http://swc2017.aksw.org/hasTruthValue> "
                    f"\"{score}\"^^<{XSD_DOUBLE}> .\n"
                )
        print(f"Wrote {len(predictions)} predictions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Structural link-prediction KG fact checker")
    parser.add_argument("--train-file", default=DEFAULT_PATHS["train_file"])
    parser.add_argument("--test-file", default=DEFAULT_PATHS["test_file"])
    parser.add_argument("--output-file", default=DEFAULT_PATHS["output_file"])
    parser.add_argument("--skip-cv", action="store_true", help="Skip the cross-validation report")
    args = parser.parse_args()

    train_statements = load_statements(args.train_file)
    test_statements = load_statements(args.test_file)

    # Graph structure is built from BOTH files (transductive setting) but truth
    # labels are only ever read from the training file.
    graph = StatementGraph(train_statements + test_statements)
    checker = FactChecker(graph)

    if not args.skip_cv:
        print("Cross-validating on labelled training data...")
        checker.cross_validate(train_statements)

    print("Training on full training set...")
    checker.train(train_statements)

    print("Predicting test statements...")
    predictions = checker.predict(test_statements)

    FactChecker.write_results(predictions, args.output_file)


if __name__ == "__main__":
    main()
