from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
SENTENCE_RE = re.compile(r"[^。！？；]+[。！？；]+|[^。！？；]+$")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class LuxunDocument:
    source_id: str
    title: str
    book: str
    date: str
    license: str
    paragraphs: list[str]


def extract_tei_document(path: str | Path) -> LuxunDocument:
    """Extract metadata and body paragraphs from one TextGrid TEI XML file."""
    xml_path = Path(path)
    root = ET.parse(xml_path).getroot()

    title = _first_text(root, ".//tei:titleStmt/tei:title") or _first_text(
        root, ".//tei:sourceDesc//tei:title[@level='a']"
    )
    book = _first_text(root, ".//tei:sourceDesc//tei:title[@level='m']")
    date = _first_text(root, ".//tei:sourceDesc//tei:date")
    licence = root.find(".//tei:publicationStmt//tei:licence", TEI_NS)
    licence_value = ""
    if licence is not None:
        licence_value = licence.attrib.get("target", "") or normalize_text("".join(licence.itertext()))

    paragraphs: list[str] = []
    for para in root.findall(".//tei:text/tei:body//tei:p", TEI_NS):
        text = normalize_text(_element_text_without(para, skip_tags={"note"}))
        if text:
            paragraphs.append(text)

    return LuxunDocument(
        source_id=_source_id(xml_path, root),
        title=title or xml_path.stem,
        book=book,
        date=date,
        license=licence_value,
        paragraphs=paragraphs,
    )


def normalize_text(text: str) -> str:
    """Normalize whitespace while keeping original Chinese punctuation."""
    return SPACE_RE.sub("", text.strip())


def segment_paragraphs(
    paragraphs: Iterable[str],
    min_chars: int = 30,
    max_chars: int = 160,
) -> list[str]:
    """Split TEI body paragraphs into training-sized sentence segments."""
    if min_chars <= 0 or max_chars < min_chars:
        raise ValueError("Require 0 < min_chars <= max_chars")

    units: list[str] = []
    for paragraph in paragraphs:
        for sentence in SENTENCE_RE.findall(normalize_text(paragraph)):
            sentence = sentence.strip()
            if sentence:
                units.extend(_split_overlong(sentence, max_chars))

    segments: list[str] = []
    buffer = ""
    for unit in units:
        if len(unit) >= min_chars and not buffer:
            segments.append(unit)
            continue

        candidate = buffer + unit
        if len(candidate) < min_chars:
            buffer = candidate
            continue

        if len(candidate) <= max_chars:
            segments.append(candidate)
        else:
            segments.extend(part for part in _split_overlong(candidate, max_chars) if len(part) >= min_chars)
        buffer = ""

    if len(buffer) >= min_chars:
        segments.append(buffer)
    return segments


def build_segment_records(
    documents: Iterable[LuxunDocument],
    min_chars: int = 30,
    max_chars: int = 160,
) -> list[dict]:
    """Build stable JSONL-ready segment records from parsed documents."""
    records: list[dict] = []
    next_id = 1
    for doc in documents:
        for target in segment_paragraphs(doc.paragraphs, min_chars=min_chars, max_chars=max_chars):
            records.append(
                {
                    "id": f"luxun_{next_id:06d}",
                    "source_id": doc.source_id,
                    "title": doc.title,
                    "book": doc.book,
                    "date": doc.date,
                    "license": doc.license,
                    "target": target,
                    "length": len(target),
                }
            )
            next_id += 1
    return records


def build_raw_records(documents: Iterable[LuxunDocument]) -> list[dict]:
    """Build article-level records for inspection and traceability."""
    records: list[dict] = []
    for doc in documents:
        records.append(
            {
                "source_id": doc.source_id,
                "title": doc.title,
                "book": doc.book,
                "date": doc.date,
                "license": doc.license,
                "paragraphs": doc.paragraphs,
                "text": "\n".join(doc.paragraphs),
            }
        )
    return records


def load_textgrid_documents(input_dir: str | Path) -> list[LuxunDocument]:
    xml_paths = sorted(Path(input_dir).rglob("*.xml"))
    return [extract_tei_document(path) for path in xml_paths]


def prepare_textgrid_corpus(
    input_dir: str | Path,
    output_dir: str | Path,
    min_chars: int = 30,
    max_chars: int = 160,
) -> tuple[Path, Path, int, int]:
    """Parse TextGrid XML files and write raw article + segment JSONL files."""
    documents = load_textgrid_documents(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "luxun_raw.jsonl"
    segment_path = out_dir / "luxun_segments.jsonl"
    write_jsonl(build_raw_records(documents), raw_path)
    segments = build_segment_records(documents, min_chars=min_chars, max_chars=max_chars)
    write_jsonl(segments, segment_path)
    return raw_path, segment_path, len(documents), len(segments)


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Lu Xun TextGrid TEI XML into JSONL segments.")
    parser.add_argument(
        "--input_dir",
        default="data/raw/luxun/textgrid/luxun_textgrid_full_xml",
        help="Directory containing TextGrid TEI XML files.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/luxun_style",
        help="Directory for luxun_raw.jsonl and luxun_segments.jsonl.",
    )
    parser.add_argument("--min_chars", type=int, default=30, help="Minimum segment length in characters.")
    parser.add_argument("--max_chars", type=int, default=160, help="Maximum segment length in characters.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raw_path, segment_path, n_docs, n_segments = prepare_textgrid_corpus(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    print(f"Parsed {n_docs} XML documents.")
    print(f"Wrote article records: {raw_path}")
    print(f"Wrote segment records: {segment_path} ({n_segments} segments)")


def _first_text(root: ET.Element, xpath: str) -> str:
    element = root.find(xpath, TEI_NS)
    if element is None:
        return ""
    return normalize_text("".join(element.itertext()))


def _element_text_without(element: ET.Element, skip_tags: set[str]) -> str:
    chunks: list[str] = []

    def visit(node: ET.Element) -> None:
        if node.text:
            chunks.append(node.text)
        for child in node:
            if _local_name(child.tag) not in skip_tags:
                visit(child)
            if child.tail:
                chunks.append(child.tail)

    visit(element)
    return "".join(chunks)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _source_id(path: Path, root: ET.Element) -> str:
    stem = path.stem
    if stem.startswith("textgrid_"):
        return f"textgrid:{stem.removeprefix('textgrid_')}"

    idno = _first_text(root, ".//tei:publicationStmt/tei:idno")
    return idno or stem


def _split_overlong(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    return [text[start : start + max_chars] for start in range(0, len(text), max_chars)]
