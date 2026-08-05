"""Command-line interface for page-aware source preparation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..provenance.prepare import PreparationError, prepare_source


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--chapter-id", required=True)
    parser.add_argument("--ocr-engine", required=True)
    parser.add_argument("--source-pdf-locator", required=True)
    parser.add_argument("--source-pdf-sha256", required=True)
    parser.add_argument("--printed-page-start", required=True, type=int)
    parser.add_argument("--source-pdf-page-start", required=True, type=int)
    parser.add_argument("--page-count", required=True, type=int)
    parser.add_argument("--page-map", type=Path)
    parser.add_argument("--generation-timestamp")
    args = parser.parse_args(argv)
    try:
        prepare_source(
            input_path=args.input,
            output_path=args.output,
            document_id=args.document_id,
            chapter_id=args.chapter_id,
            ocr_engine=args.ocr_engine,
            source_pdf_locator=args.source_pdf_locator,
            source_pdf_sha256=args.source_pdf_sha256,
            printed_page_start=args.printed_page_start,
            source_pdf_page_start=args.source_pdf_page_start,
            page_count=args.page_count,
            page_map_path=args.page_map,
            generation_timestamp=args.generation_timestamp,
        )
    except PreparationError as error:
        parser.error(str(error))
    print((args.output / "manifest.json").as_posix())


if __name__ == "__main__":
    main(sys.argv[1:])
