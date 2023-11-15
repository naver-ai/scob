"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""


class Tasks:
    # base head tasks
    OCR_READ = "ocr_read"
    TEXT_READ = "text_read"
    DONUT_KIE = "donut_kie"
    OCR_READ_TEXTINSTANCEPADDING = "ocr_read_TextInstancePadding"
    TABLE_PARSING = "table_parsing"
    OTOR = "otor"
    OTOR_ORACLE = "otor_oracle"

    # 2head tasks
    OCR_READ_2HEAD = "ocr_read_2head"


class DecoderTypes:
    TRANSFORMER = "transformer_decoder"


class HeadTypes:
    TWO_HEAD = "2head"
    BASE = "base"


class Seperators:
    DTD = "||"


class Phase:
    PRETRAINING = "pretraining"
    FINETUNING = "finetuning"


# TODO: Use this in preprocess as well
SPE_TOKENS = {
    (Tasks.OCR_READ, "start"): "[START_TEXT_BLOCK]",
    (Tasks.OCR_READ, "end"): "[END_TEXT_BLOCK]",
}

AVAILABLE_TASKS = {
    DecoderTypes.TRANSFORMER: {
        Tasks.OCR_READ,
        Tasks.TEXT_READ,
        Tasks.DONUT_KIE,
        Tasks.OCR_READ_2HEAD,
        Tasks.OCR_READ_TEXTINSTANCEPADDING,
        Tasks.TABLE_PARSING,
        Tasks.OTOR,
        Tasks.OTOR_ORACLE,
    },
}


COMMON_SPECIAL_TOKENS = [
    "[START_PROMPT]",
    "[END_PROMPT]",
    "[dataset]",
    "[DONTCARE]",
    "[END]",
    "[DIV]",
    "[START_OCR_READ]",
    "[START_OCR_READ_TextInstance_PADDING]",
    "[START_BOX]",
    "[END_BOX]",
    "[START_TEXT_READ]",
    "[START_TEXT]",
    "[END_TEXT]",
    "[START_OCR_READ_2HEAD]",
    "[START_TEXT_BLOCK]",
    "[END_TEXT_BLOCK]",
    "[EMPTY_TEXT_IMAGE]",
    "[CHAR_PAD]",
]
