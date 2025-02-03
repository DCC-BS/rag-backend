from data.docling_loader import DoclingLoader
from utils.config import load_config
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
)

load_config()
path = "data/EL/1 Anmeldebogen/13 Erwerbseinkommen/13.1 Einkommen/13.1.2  hypothetisches EK/Anrechnung des hyp. Einkommens - Schulung 2012.pptx"
try:
    loader = DoclingLoader(file_path=path,organization="Test")
    docs = loader.lazy_load()
    print(next(docs))
    print(next(docs))
except Exception as e:
    print(e)


print("Second approach")
print("---"* 100)

accelerator_device = AcceleratorDevice.CUDA
accelerator_options = AcceleratorOptions(num_threads=16)
converter = DocumentConverter(
    allowed_formats=[
        InputFormat.PDF,
        InputFormat.DOCX,
        InputFormat.PPTX,
        InputFormat.HTML,
        # InputFormat.XLSX,
    ],
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(
                do_table_structure=True,
                table_structure_options=TableStructureOptions(
                    mode=TableFormerMode.ACCURATE
                ),
                accelerator_options = accelerator_options
            )
        )
    },
)

result = converter.convert(path)
print(result.document.export_to_markdown())