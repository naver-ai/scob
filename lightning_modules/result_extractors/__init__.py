"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

from lightning_modules.result_extractors.transformer_decoder import (
    TransformerDecDonutKIEResultExtractor,
    TransformerDecOCRReadResultExtractor,
    TransformerDecTableParsingResultExtractor,
    TransformerDecTextReadResultExtractor,
)
from utils.constants import DecoderTypes, Tasks

extractor_dict = {
    (
        DecoderTypes.TRANSFORMER,
        Tasks.OCR_READ_TEXTINSTANCEPADDING,
    ): TransformerDecOCRReadResultExtractor,
    (
        DecoderTypes.TRANSFORMER,
        Tasks.DONUT_KIE,
    ): TransformerDecDonutKIEResultExtractor,
    (
        DecoderTypes.TRANSFORMER,
        Tasks.OCR_READ,
    ): TransformerDecOCRReadResultExtractor,
    (
        DecoderTypes.TRANSFORMER,
        Tasks.TEXT_READ,
    ): TransformerDecTextReadResultExtractor,
    (
        DecoderTypes.TRANSFORMER,
        Tasks.OCR_READ_2HEAD,
    ): TransformerDecOCRReadResultExtractor,
    (
        DecoderTypes.TRANSFORMER,
        Tasks.TABLE_PARSING,
    ): TransformerDecTableParsingResultExtractor,
    (
        DecoderTypes.TRANSFORMER,
        Tasks.OTOR,
    ): TransformerDecOCRReadResultExtractor,
    (
        DecoderTypes.TRANSFORMER,
        Tasks.OTOR_ORACLE,
    ): TransformerDecOCRReadResultExtractor,
}


def get_result_extractors(dataset_items):
    extractors = {}
    for dataset_item in dataset_items:
        dataset_name = dataset_item.name
        for task in dataset_item.tasks:
            task_name = task.name
            decoder_name = task.decoder

            decoder_type = decoder_name.split("__")[0]
            extractor_class = extractor_dict[(decoder_type, task_name)]

            dtd_key = (dataset_name, task_name, decoder_name)
            extractors[dtd_key] = extractor_class(dataset_item)
    return extractors
