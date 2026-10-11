# AmbigDocs V2 external-validation protocol

Status: offline preparation only. No model/API run has been performed.

## Source and observed schema

The official test split is `test.json` from the public
[yoonsanglee/AmbigDocs dataset](https://huggingface.co/datasets/yoonsanglee/AmbigDocs),
linked by the [upstream repository](https://github.com/lilys012/AmbigDocs).
The inspected split contains **7,220** rows. The dataset card reports Apache-2.0.

Each row has exactly `qid`, `ambiguous_entity`, `question`, and `documents`.
Each document has exactly `pid`, `title`, `text`, and `answer`. The inspected
document-count distribution is: 2 documents (3,650 rows), 3 (1,916), 4 (982),
5 (407), 6 (165), 7 (73), 8 (23), 9 (3), and 10 (1). `qid` is unique.

## Frozen 100-case sample

`benchmarks/ambigdocs_test_sample_100.json` records the official `test` split,
population 7,220, and seed 42. Sampling is deterministic:

1. sort all qids numerically;
2. select 100 qids with `random.Random(42).sample`;
3. sort the selected qids for the manifest.

There is no content-based replacement or cherry-picking. The manifest stores
the exact qids.

## Mechanical mapping

For each document, the adapter creates one `Evidence`:

- `evidence_id`: `E1`, `E2`, ... in source document order;
- `source_id`: `ambigdocs:{qid}:{pid}`;
- `location`: document `title`, preserving disambiguated entity identity;
- `content`: document `text`.

For each document entity, it creates one `FactNeed`:

- `subject`: document `title`;
- `dimension`: the original AmbigDocs `question`, preserved exactly;
- `value_type`: `string`;
- no scope constraint.

The original question reaches `SemanticCompiler` through the serialized
`FactNeed.dimension`; the compiler otherwise receives only FactNeeds and
Evidence. The Direct baseline receives the same question and the same
FactNeeds, so both systems receive equivalent semantic task information. This
protocol-wiring correction was made before any live AmbigDocs model run.

Gold is a separate mechanical association target `(title, answer, evidence_id,
pid)`. It does not invent Claim modality, cardinality, scope, authority, or
summary state. The title is the dataset's entity identity, and `pid` is kept
for provenance checks. This is **entity-label binding**, not entity discovery:
the disambiguated title is already exposed through `Evidence.location` and
`FactNeed.subject`, so both systems receive the entity label to bind.

`E1`/`E2` are only compact transport aliases. Before scoring, the adapter
resolves `evidence_ref -> Evidence -> source_id -> pid`; canonical identity is
always the AmbigDocs `pid`, never the internal alias.

## Metrics

The Direct baseline output is frozen as exactly:

```json
{"associations":[{"entity":"...","answer":"...","evidence_ref":"E1"}]}
```

ContextCanon Claims and Direct associations are both projected to the same
canonical `(entity/title, normalized answer, pid)` representation before scoring.

The offline scorer accepts `success` plus those canonical associations. It
reports completed-only metrics and end-to-end metrics over all attempted cases;
failed compilation/schema cases contribute no associations and therefore count
as incorrect in the end-to-end denominator. Schema/compilation success is always
reported separately.

- **compilation success rate**: successful semantic compilation / selected cases;
- **entity-label binding accuracy**: predicted `(entity, pid)` pairs that equal
  Gold / all Gold entity-document pairs (recall-style);
- **provenance attribution accuracy**: correct predicted entity-document pairs /
  all predicted pairs (precision-style);
- **normalized exact answer association**: exact `(entity, normalized answer,
  pid)` tuples / all Gold tuples;
- **answer token recall / F1**: token overlap after the frozen answer
  normalization, matched within the same entity and pid.

Answer normalization is fixed before any live run: lowercase, remove
punctuation, remove the articles `a`, `an`, and `the`, then normalize
whitespace. It does not parse numbers, dates, aliases, authority, or entity
similarity.

This is a ContextCanon-specific adaptation of AmbigDocs, not a reproduction of
the official AmbigDocs leaderboard evaluation. It measures multi-entity
evidence binding, answer/entity association, provenance binding, and structured
compilation robustness. It does not prove entity discovery because the
disambiguated titles are provided to both systems.

These targets do not measure V2
modality, cardinality, scope, source authority, or CLEAR/UNRESOLVED/INCOMPLETE;
AmbigDocs does not provide mechanically sufficient labels for those dimensions,
and it does not provide general governance accuracy Gold.

## Direct baseline contract

The future Direct one-call baseline uses the same question, FactNeeds, Evidence
refs, source/location metadata, and document text. It does not receive a
separate `ambiguous_entity` field. It returns only the frozen association schema
above. It receives no document `answer`, no Gold labels, and no dataset-derived
evaluation fields. No live runner is included in this phase.

## Files and validation

- `experiments/v2_ambigdocs.py`: loader, mapping, sample loader, Direct prompt,
  and deterministic scorer;
- `benchmarks/ambigdocs_test_sample_100.json`: frozen qid manifest;
- `tests/test_v2_ambigdocs.py`: offline mapping, leakage, determinism, and scorer tests.

Unsupported or intentionally unscored fields: modality, cardinality, temporal
or geographic scope, source authority, answer truth beyond the dataset's
provided answer strings, V2 summary semantics, and general governance accuracy.
