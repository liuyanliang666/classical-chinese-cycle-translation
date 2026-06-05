from pathlib import Path

from ccnlp.luxun_textgrid import (
    build_segment_records,
    extract_tei_document,
    segment_paragraphs,
)


SAMPLE_TEI = """<?xml version='1.0' encoding='UTF-8'?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>祝福</title>
      </titleStmt>
      <publicationStmt>
        <idno>l_l_00016</idno>
        <availability>
          <licence target="http://creativecommons.org/licenses/by-nc/3.0/deed.en_US">
            Distributed under a Creative Commons Attribution-NonCommercial 3.0 Unported License
          </licence>
        </availability>
      </publicationStmt>
      <sourceDesc>
        <bibl>
          <title level="a">祝福</title>
          <title level="m">彷徨</title>
          <date>1924</date>
        </bibl>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text>
    <front><head>祝福</head></front>
    <body>
      <div>
        <p>旧历的年底毕竟最像年底，村镇上不必说，就在天空中也显出将到新年的气象来。灰白色的沉重的晚云中间时时发出闪光，接着一声钝响，是送灶的爆竹。</p>
        <p>我就站住，豫备她来讨钱。</p>
        <p>“你回来了？”她先这样问。<note>编辑注应被忽略</note></p>
      </div>
    </body>
  </text>
</TEI>
"""


def test_extract_tei_document_reads_metadata_and_body_only(tmp_path: Path):
    xml_path = tmp_path / "textgrid_4b00b.0.xml"
    xml_path.write_text(SAMPLE_TEI, encoding="utf-8")

    doc = extract_tei_document(xml_path)

    assert doc.source_id == "textgrid:4b00b.0"
    assert doc.title == "祝福"
    assert doc.book == "彷徨"
    assert doc.date == "1924"
    assert doc.license == "http://creativecommons.org/licenses/by-nc/3.0/deed.en_US"
    assert doc.paragraphs[0].startswith("旧历的年底毕竟最像年底")
    assert "祝福祝福" not in "".join(doc.paragraphs)
    assert "编辑注应被忽略" not in "".join(doc.paragraphs)


def test_segment_paragraphs_filters_lengths_and_merges_short_sentences():
    paragraphs = [
        "我就站住，豫备她来讨钱。",
        "“你回来了？”她先这样问。她那没有精采的眼睛忽然发光了。",
        "旧历的年底毕竟最像年底，村镇上不必说，就在天空中也显出将到新年的气象来。灰白色的沉重的晚云中间时时发出闪光，接着一声钝响，是送灶的爆竹。",
    ]

    segments = segment_paragraphs(paragraphs, min_chars=30, max_chars=80)

    assert segments[0] == "我就站住，豫备她来讨钱。“你回来了？”她先这样问。她那没有精采的眼睛忽然发光了。"
    assert all(30 <= len(segment) <= 80 for segment in segments)
    assert "旧历的年底毕竟最像年底，村镇上不必说，就在天空中也显出将到新年的气象来。" in segments


def test_build_segment_records_uses_stable_ids_and_metadata(tmp_path: Path):
    xml_path = tmp_path / "textgrid_4b00b.0.xml"
    xml_path.write_text(SAMPLE_TEI, encoding="utf-8")
    doc = extract_tei_document(xml_path)

    records = build_segment_records([doc], min_chars=30, max_chars=100)

    assert records[0]["id"] == "luxun_000001"
    assert records[0]["source_id"] == "textgrid:4b00b.0"
    assert records[0]["title"] == "祝福"
    assert records[0]["book"] == "彷徨"
    assert records[0]["target"]
