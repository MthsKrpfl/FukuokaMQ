"""
Reads an MQ challenge or toy file, plain or .bz2, into an Instance: header,
degree-2 monomial basis in the requested order, polynomials, and the known
solution when an answer file is given.

Requires Sage.
"""
from __future__ import annotations
import bz2
import re
from pathlib import Path

from mq_instance import Instance
from hashing import sha256_file


# any format deviation, reported with its line number
class ParseError(Exception):
    """Format deviation -- report with line number, no silent fallback."""

# read a plain or .bz2 file as text
def _read_text(path: str | Path) -> str:
    path = Path(path)
    if path.suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    return path.read_text(encoding="utf-8")

_RE_FIELD = re.compile(r"Galois\s+Field\s*:\s*(.+)", re.IGNORECASE)
_RE_NVARS = re.compile(r"Number of variables.*?:\s*(\d+)", re.IGNORECASE)
_RE_NPOLY = re.compile(r"Number of (?:polynomials|equations).*?:\s*(\d+)", re.IGNORECASE)


# tell a path apart from an already read text blob
def _looks_like_path(s: str) -> bool:
    if "\n" in s or len(s) > 4096:
        return False
    try:
        return Path(s).exists()
    except OSError:
        return False


# q, n and m from the file header
def parse_header(text_or_path) -> dict:
    if isinstance(text_or_path, Path) or (
        isinstance(text_or_path, str) and _looks_like_path(text_or_path)
    ):
        text = _read_text(text_or_path)
    else:
        text = str(text_or_path)

    mfield = _RE_FIELD.search(text)
    mn = _RE_NVARS.search(text)
    mm = _RE_NPOLY.search(text)
    if not (mfield and mn and mm):
        raise ParseError(f"incomplete header (Field/n/m): {text_or_path!r}")

    field_raw = mfield.group(1).strip()
    n = int(mn.group(1))
    m = int(mm.group(1))

    modulus_str = None
    fr = field_raw.replace(" ", "")
    if fr.startswith("GF(2)[x]") or "GF(2^8)" in fr or "GF(256)" in fr:
        q = 256
        if "/" in field_raw:
            modulus_str = field_raw.split("/", 1)[1].strip()
    else:
        mq = re.match(r"GF\((\d+)\)", fr)
        if not mq:
            raise ParseError(f"unknown field: {field_raw!r}")
        q = int(mq.group(1))
    return {"q": q, "n": n, "m": m, "modulus_str": modulus_str, "field_raw": field_raw}

# the sage field for q, GF(2^8) needs the modulus from the header
def make_field(q: int, modulus_str: str | None):
    from sage.all import GF, PolynomialRing, sage_eval

    if q == 2:
        K = GF(2)
        return K, (lambda tok: K(int(tok)))
    if q == 256:
        Rx = PolynomialRing(GF(2), "x")
        if not modulus_str:
            raise ParseError("GF(256) without modulus in header")
        modulus = sage_eval(modulus_str.replace("^", "**"), locals={"x": Rx.gen()})
        K = GF(2 ** 8, name="z", modulus=Rx(modulus))

        def conv_hex(tok: str):
            val = int(tok, 16)
            try:
                return K.from_integer(val)
            except AttributeError:
                return K.fetch_int(val)
        return K, conv_hex
    K = GF(q)
    return K, (lambda tok: K(int(tok)))


# degree <= 2 monomial basis, ordered like the coefficients in the file
def monomials_degree_le2(ring, order_mode: str = "degrevlex_desc"):
    gens = list(ring.gens())
    n = len(gens)
    deg2 = [gens[i] * gens[j] for i in range(n) for j in range(i, n)]
    deg1 = list(gens)
    one = ring.one()

    if order_mode == "degrevlex_desc":
        allm = sorted(deg2 + deg1, reverse=True) + [one]
    elif order_mode == "degrevlex_asc":
        allm = [one] + sorted(deg2 + deg1)
    elif order_mode == "nested":
        allm = deg2 + deg1 + [one]
    elif order_mode == "nested_rev":
        allm = list(reversed(deg2)) + list(reversed(deg1)) + [one]
    else:
        raise ValueError(f"unknown order_mode: {order_mode}")
    return allm


# how many coefficients one polynomial must carry
def expected_ncols(n: int) -> int:
    return n * (n + 1) // 2 + n + 1


# challenge type from the file name
def _typ_from_name(instance_id: str) -> int | None:
    m = re.search(r"type[-_]?(\d+)", instance_id, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"challenge-(\d+)-", instance_id, re.IGNORECASE)
    return int(m.group(1)) if m else None


# full parse: header, monomials, polynomials, optional solution
def parse_instance(path: str, solution_tokens: list[str] | None = None,
                   order_mode: str = "degrevlex_desc") -> Instance:
    from sage.all import PolynomialRing

    path = Path(path)
    text = _read_text(path)
    hdr = parse_header(text)
    q, n, m = hdr["q"], hdr["n"], hdr["m"]

    K, conv = make_field(q, hdr["modulus_str"])
    ring = PolynomialRing(K, n, names=[f"x{i}" for i in range(n)], order="degrevlex")
    mons = monomials_degree_le2(ring, order_mode)
    ncols = expected_ncols(n)
    if len(mons) != ncols:
        raise ParseError(f"internal monomial error: {len(mons)} != expected {ncols}")

    lines = text.splitlines()
    start = next((i + 1 for i, ln in enumerate(lines) if set(ln.strip()) == {"*"}), None)
    if start is None:
        raise ParseError(f"{path.name}: '***' separator line not found")

    polys = []
    for lineno, raw in enumerate(lines[start:], start=start + 1):
        s = raw.strip().rstrip(";").strip()
        if not s:
            continue
        toks = s.split()
        if len(toks) != ncols:
            raise ParseError(
                f"{path.name}:{lineno}: {len(toks)} coefficients, expected {ncols} "
                f"(n={n}); monomial count mismatch.")
        coeffs = [conv(t) for t in toks]
        poly = sum((c * mon for c, mon in zip(coeffs, mons)), ring.zero())
        polys.append(poly)
        if len(polys) == m:
            break

    if len(polys) != m:
        raise ParseError(f"{path.name}: read {len(polys)} polynomials, expected {m}")

    solution = None
    if solution_tokens:
        if len(solution_tokens) != n:
            raise ParseError(
                f"{path.name}: solution has {len(solution_tokens)} values, expected {n}")
        solution = tuple(conv(t) for t in solution_tokens)

    return Instance(
        instance_id=path.name.replace(".bz2", ""),
        typ=_typ_from_name(path.name) or -1,
        q=q, n=n, m=m, d=n - m,
        ring=ring, polys=polys, solution=solution,
        source_path=str(path), sha256=sha256_file(path),
    )


# gate: the known solution must be a root of every polynomial
def assert_solution_is_root(inst: Instance) -> None:
    if inst.solution is None:
        return
    for idx, p in enumerate(inst.polys):
        val = p(*inst.solution)
        if val != 0:
            raise AssertionError(
                f"{inst.instance_id}: polynomial {idx} evaluates to {val} (!=0) at the "
                f"solution -> parser/format/monomial order wrong")
