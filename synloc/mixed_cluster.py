"""Public-schema mixed-type encoding and parameter-only clusterCov sampling.

The supplied schema must contain only independently established public codes and
bounds. Neither validation nor reviewed=True constitutes disclosure clearance.
"""

import copy
import json
import re
import zipfile

import numpy as np
import pandas as pd

from .clusterCovSampler import ClusterCovSampler
from .dists import clusterCov


_VERSION = 1
_MAX_METADATA = 1_000_000
_MAX_COLUMNS = 512
_MAX_COMPONENTS = 1000
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_FIELDS = {"version", "metadata", "means", "covariances", "weights"}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate metadata key.")
        result[key] = value
    return result


def _number(value):
    return type(value) in (int, float) and abs(value) <= 2**53 and np.isfinite(value)


def _validate_schema(schema):
    """Validate and copy the complete public decoding contract, fail closed."""
    if not isinstance(schema, dict) or set(schema) != {"version", "columns", "constraints"}:
        raise ValueError("Schema must contain exactly version, columns and constraints.")
    if type(schema["version"]) is not int or schema["version"] != _VERSION:
        raise ValueError("Unsupported mixed schema version.")
    columns = schema["columns"]
    constraints = schema["constraints"]
    if not isinstance(columns, list) or not 1 <= len(columns) <= _MAX_COLUMNS or not isinstance(constraints, list) or len(constraints) > _MAX_COLUMNS:
        raise ValueError("Invalid schema size.")
    names, specs, width = set(), {}, 0
    for spec in columns:
        if not isinstance(spec, dict) or "name" not in spec or "kind" not in spec:
            raise ValueError("Invalid column specification.")
        name, kind = spec["name"], spec["kind"]
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in names:
            raise ValueError("Column names must be unique simple identifiers.")
        names.add(name)
        specs[name] = spec
        if kind == "category":
            if set(spec) != {"name", "kind", "categories", "labels"}:
                raise ValueError("Invalid category specification.")
            categories = spec["categories"]
            if (not isinstance(categories, list) or not 1 <= len(categories) <= _MAX_COLUMNS
                    or any(type(x) is not int or abs(x) > 2**53 for x in categories)
                    or len(set(categories)) != len(categories)):
                raise ValueError("Categories must be unique public integer codes.")
            labels = spec["labels"]
            if (not isinstance(labels, list) or len(labels) != len(categories)
                    or any(not isinstance(label, str) or not label.strip() or len(label) > 200 for label in labels)):
                raise ValueError("Category labels must describe every public code.")
            width += len(categories)
        elif kind in ("integer", "continuous"):
            if set(spec) != {"name", "kind", "min", "max"}:
                raise ValueError("Invalid numeric specification.")
            lo, hi = spec["min"], spec["max"]
            if any(x is not None and not _number(x) for x in (lo, hi)):
                raise ValueError("Bounds must be finite public numbers or null.")
            if lo is not None and hi is not None and lo > hi:
                raise ValueError("Invalid numeric bounds.")
            if kind == "integer" and (lo is None or hi is None or any(float(x) != int(x) or abs(x) > 2**53 for x in (lo, hi))):
                raise ValueError("Integer bounds must be finite exact integers.")
            width += 1
        else:
            raise ValueError("Unknown column kind or transform.")
    if width > _MAX_COLUMNS:
        raise ValueError("Encoded dimension exceeds the limit.")
    seen = set()
    for constraint in constraints:
        if not isinstance(constraint, dict) or set(constraint) != {"kind", "left", "right"} or constraint["kind"] != "less_equal":
            raise ValueError("Unknown or invalid constraint.")
        left, right = constraint["left"], constraint["right"]
        if not isinstance(left, str) or not isinstance(right, str) or left not in specs or right not in specs or left == right:
            raise ValueError("Constraint references invalid columns.")
        a, b = specs[left], specs[right]
        if a["kind"] != "integer" or b["kind"] != "integer" or a["min"] > b["min"]:
            raise ValueError("Constraints require compatible bounded integer columns.")
        if left in seen:
            raise ValueError("A constraint left column may appear only once.")
        seen.add(left)
    # Disallow chains: constraint order cannot silently determine the result.
    if any(c["right"] in seen for c in constraints):
        raise ValueError("Chained constraints are not supported.")
    result = copy.deepcopy(schema)
    encoded = json.dumps(result, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_METADATA:
        raise ValueError("Metadata exceeds the limit.")
    return result, width


def _encoded_names(width):
    return [f"score_{i}" for i in range(width)]


def _encode(data, schema, width):
    names = [spec["name"] for spec in schema["columns"]]
    if not isinstance(data, pd.DataFrame) or list(data.columns) != names or len(data) == 0:
        raise ValueError("Data must contain exactly the schema columns in order and at least one row.")
    try:
        values = data.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Data must contain numeric source codes and values.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite source values are not supported.")
    encoded = np.empty((len(data), width), dtype=float)
    offset = 0
    for i, spec in enumerate(schema["columns"]):
        value = values[:, i]
        if spec["kind"] == "category":
            categories = np.asarray(spec["categories"], dtype=float)
            block = value[:, None] == categories
            if not block.any(axis=1).all():
                raise ValueError(f"Source category outside public dictionary: {spec['name']}.")
            encoded[:, offset:offset + len(categories)] = block
            offset += len(categories)
        else:
            lo, hi = spec["min"], spec["max"]
            if (lo is not None and np.any(value < lo)) or (hi is not None and np.any(value > hi)):
                raise ValueError(f"Source value outside public bounds: {spec['name']}.")
            if spec["kind"] == "integer" and np.any(value != np.rint(value)):
                raise ValueError(f"Source value is not an integer: {spec['name']}.")
            encoded[:, offset] = value
            offset += 1
    for c in schema["constraints"]:
        if np.any(values[:, names.index(c["left"])] > values[:, names.index(c["right"])]):
            raise ValueError("Source violates a public cross-variable constraint.")
    return pd.DataFrame(encoded, columns=_encoded_names(width))


def _inspect_arrays(zf, items):
    """Bound NPY headers and declared sizes before NumPy allocates arrays."""
    headers = {}
    for item in items:
        with zf.open(item) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("Unsupported NPY header version.")
            size = dtype.itemsize
            for dimension in shape:
                size *= dimension
            if size > _MAX_ARCHIVE_BYTES or size != item.file_size - stream.tell():
                raise ValueError("Invalid declared array size.")
            headers[item.filename[:-4]] = (shape, dtype)
    shape, _ = headers["means"]
    if len(shape) != 2 or not 1 <= shape[0] <= _MAX_COMPONENTS or not 1 <= shape[1] <= _MAX_COLUMNS:
        raise ValueError("Invalid mixed parameter dimensions.")
    k, p = shape
    for name, expected in (("means", (k, p)), ("covariances", (k, p, p)), ("weights", (k,))):
        actual, dtype = headers[name]
        if actual != expected or dtype.kind not in "fiu" or dtype.itemsize > 8:
            raise ValueError("Invalid mixed parameter array.")
    shape, dtype = headers["metadata"]
    if shape != () or dtype.kind != "U" or dtype.itemsize > 4 * _MAX_METADATA:
        raise ValueError("Invalid metadata array.")
    shape, dtype = headers["version"]
    if shape != () or dtype.kind not in "iu" or dtype.itemsize > 8:
        raise ValueError("Invalid version array.")


class MixedClusterCovSampler:
    """Decode draws from a validated, self-contained mixed parameter archive.

    Use :meth:`load` to read an NPZ. ``raw_scores`` exposes the joint latent
    draws and ``decode`` optionally reports counts of each intervention.
    """

    def __init__(self, means, covariances, weights, schema):
        self._schema, width = _validate_schema(schema)
        means_array = np.asarray(means)
        if means_array.ndim != 2 or means_array.shape[0] > _MAX_COMPONENTS or means_array.shape[1] != width:
            raise ValueError("Invalid mixed parameter dimensions.")
        self._model = ClusterCovSampler(means, covariances, weights, _encoded_names(width))
        self.columns = tuple(s["name"] for s in self._schema["columns"])

    @property
    def schema(self):
        """Return a copy of the independently supplied public schema."""
        return copy.deepcopy(self._schema)

    @classmethod
    def load(cls, path):
        """Load only the bounded version-1 NPZ format, without pickle."""
        try:
            with open(path, "rb") as source:
                with zipfile.ZipFile(source) as zf:
                    items = zf.infolist()
                    if (len(items) != len(_FIELDS) or {x.filename for x in items} != {f + ".npy" for f in _FIELDS}
                            or sum(x.file_size for x in items) > _MAX_ARCHIVE_BYTES
                            or any(x.flag_bits & 1 for x in items)):
                        raise ValueError("Invalid or oversized mixed archive.")
                    if next(x.file_size for x in items if x.filename == "metadata.npy") > 4 * _MAX_METADATA + 1024:
                        raise ValueError("Metadata exceeds the limit.")
                    _inspect_arrays(zf, items)
                source.seek(0)
                with np.load(source, allow_pickle=False) as archive:
                    if len(archive.files) != len(_FIELDS) or set(archive.files) != _FIELDS:
                        raise ValueError("Invalid mixed archive fields.")
                    version = archive["version"]
                    metadata = archive["metadata"]
                    if version.shape != () or version.dtype.kind not in "iu" or int(version) != _VERSION:
                        raise ValueError("Unsupported mixed archive version.")
                    if metadata.shape != () or metadata.dtype.kind != "U":
                        raise ValueError("Metadata must be a scalar JSON string.")
                    text = str(metadata.item())
                    if len(text.encode("utf-8")) > _MAX_METADATA:
                        raise ValueError("Metadata exceeds the limit.")
                    schema = json.loads(text, object_pairs_hook=_object, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON value.")))
                    return cls(archive["means"], archive["covariances"], archive["weights"], schema)
        except (zipfile.BadZipFile, json.JSONDecodeError, OSError, EOFError) as exc:
            raise ValueError("Invalid mixed model archive.") from exc

    def raw_scores(self, n, random_state=None):
        """Draw encoded Gaussian scores in original feature units."""
        return self._model.sample(n, random_state)

    def decode(self, raw_scores, *, return_diagnostics=False):
        """Decode scores, optionally returning per-rule record counts.

        Argmax ties use first dictionary entry; integers use NumPy ties-to-even
        rounding, then public bounds. Cross-constraints lower the left value.
        """
        if not isinstance(raw_scores, pd.DataFrame) or list(raw_scores.columns) != list(self._model.columns):
            raise ValueError("Raw scores must have the exact encoded column names and order.")
        try:
            values = raw_scores.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Raw scores must be numeric.") from exc
        if not np.isfinite(values).all():
            raise ValueError("Raw scores must be finite.")
        result, raw_numeric, diagnostics, offset = {}, {}, {"rows": len(values), "columns": {}, "constraints": {}}, 0
        for spec in self._schema["columns"]:
            name, kind = spec["name"], spec["kind"]
            if kind == "category":
                cats = np.asarray(spec["categories"], dtype=np.int64)
                block = values[:, offset:offset + len(cats)]
                chosen = np.argmax(block, axis=1)
                decoded = cats[chosen]
                valid = np.logical_or(block == 0, block == 1).all(axis=1) & (block.sum(axis=1) == 1)
                ties = (block == block.max(axis=1, keepdims=True)).sum(axis=1) > 1
                diagnostics["columns"][name] = {
                    "invalid_one_hot_before": int((~valid).sum()),
                    "invalid_category_after": int((~np.isin(decoded, cats)).sum()),
                    "ties": int(ties.sum()),
                    "decoded": len(values),
                }
                offset += len(cats)
            else:
                raw = values[:, offset]
                raw_numeric[name] = raw
                lo, hi = spec["min"], spec["max"]
                rounded = np.rint(raw) if kind == "integer" else raw.copy()
                decoded = np.clip(rounded, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
                outside = np.zeros(len(raw), dtype=bool)
                if lo is not None:
                    outside |= raw < lo
                if hi is not None:
                    outside |= raw > hi
                diagnostics["columns"][name] = {
                    "outside_bounds_before": int(outside.sum()),
                    "outside_bounds_after": int(((decoded < lo if lo is not None else np.zeros(len(raw), dtype=bool)) | (decoded > hi if hi is not None else np.zeros(len(raw), dtype=bool))).sum()),
                    "noninteger_before": int((np.rint(raw) != raw).sum()) if kind == "integer" else 0,
                    "noninteger_after": int((np.rint(decoded) != decoded).sum()) if kind == "integer" else 0,
                    "rounded": int((rounded != raw).sum()),
                    "clipped": int((decoded != rounded).sum()),
                }
                if kind == "integer":
                    decoded = decoded.astype(np.int64)
                offset += 1
            result[name] = decoded
        frame = pd.DataFrame(result, index=raw_scores.index)
        for c in self._schema["constraints"]:
            left, right = c["left"], c["right"]
            changed = frame[left] > frame[right]
            raw_changed = raw_numeric[left] > raw_numeric[right]
            frame[left] = np.minimum(frame[left], frame[right])
            diagnostics["constraints"][f"{left}<={right}"] = {
                "raw_violations_before": int(raw_changed.sum()),
                "violations_before": int(changed.sum()),
                "changed": int(changed.sum()),
                "violations_after": int((frame[left] > frame[right]).sum()),
            }
        return (frame, diagnostics) if return_diagnostics else frame

    def sample(self, n, random_state=None, *, return_diagnostics=False):
        """Draw and decode; same seed gives the same direct and loaded result."""
        return self.decode(self.raw_scores(n, random_state), return_diagnostics=return_diagnostics)

    def _export(self, path):
        with open(path, "wb") as destination:
            np.savez_compressed(destination, version=np.asarray(_VERSION, dtype=np.int64),
                                metadata=np.asarray(json.dumps(self._schema, allow_nan=False, separators=(",", ":"))),
                                means=self._model.means_, covariances=self._model.covariances_, weights=self._model.weights_)


class MixedClusterCovModel:
    """Fit clusterCov to one-hot categories under a caller-supplied public schema.

    Rows must be persons, not identifiers. Bounds/dictionaries must be fixed
    independently of the fitted data, reviewed as public, and supplied explicitly.
    Numeric scaling affects clustering only. The source and internal clearance
    diagnostics must not be included in a released archive.
    """

    def __init__(self, data, schema, n_clusters=8, size_min=None, normalize=True):
        self._schema, width = _validate_schema(schema)
        encoded = _encode(data, self._schema, width)
        self._fit = clusterCov(encoded, n_clusters=n_clusters, size_min=size_min, normalize=normalize, clipping=False)
        self._sampler = None

    def fit_model(self):
        """Fit the encoded mixture and retain internal contributor diagnostics."""
        self._sampler = None
        self._fit.fit_model()
        self._sampler = MixedClusterCovSampler(self._fit.means_, self._fit.covariances_, self._fit.weights_, self._schema)
        return self

    def _require_sampler(self):
        if self._sampler is None:
            raise RuntimeError("Call fit_model() before sampling, reporting or exporting.")
        return self._sampler

    def clearance_report(self):
        """Return confidential internal diagnostics, not privacy approval."""
        self._require_sampler()
        return self._fit.clearance_report()

    def export_model(self, path, *, reviewed=False):
        """Export only after research acknowledgement and contributor gate.

        Acknowledgement does not imply regulatory or privacy approval.
        """
        sampler = self._require_sampler()
        if self._fit.clearance_report()["required_contributor_check"]["status"] == "failed":
            raise ValueError("Export blocked by failed required contributor checks.")
        if reviewed is not True:
            raise ValueError("Export requires reviewed=True to acknowledge unresolved privacy review.")
        sampler._export(path)

    def raw_scores(self, n, random_state=None):
        return self._require_sampler().raw_scores(n, random_state)

    def decode(self, raw_scores, *, return_diagnostics=False):
        return self._require_sampler().decode(raw_scores, return_diagnostics=return_diagnostics)

    def sample(self, n, random_state=None, *, return_diagnostics=False):
        return self._require_sampler().sample(n, random_state, return_diagnostics=return_diagnostics)
