# Document Ingestion

ID: doc-02

## Definition

Ingestion converts source files into normalized documents with stable IDs and metadata.
A content hash detects changes without depending on file modification times.

## Trade-offs

Strict normalization improves reproducibility, but aggressive cleanup can remove tables, headings, or other useful structure.

## Example

An ingestion job stores the source path, document ID, title, and SHA-256 content hash for every Markdown file.

## Limitations

Parsing rules are format-specific, and scanned documents require a separate OCR step.
