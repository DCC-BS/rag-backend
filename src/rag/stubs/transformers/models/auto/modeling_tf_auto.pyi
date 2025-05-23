"""
This type stub file was generated by pyright.
"""

from .auto_factory import _BaseAutoModelClass

"""Auto Model class."""
logger = ...
TF_MODEL_MAPPING_NAMES = ...
TF_MODEL_FOR_PRETRAINING_MAPPING_NAMES = ...
TF_MODEL_WITH_LM_HEAD_MAPPING_NAMES = ...
TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = ...
TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = ...
TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = ...
TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES = ...
TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = ...
TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = ...
TF_MODEL_FOR_MASKED_LM_MAPPING_NAMES = ...
TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = ...
TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = ...
TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = ...
TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = ...
TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = ...
TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = ...
TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = ...
TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = ...
TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = ...
TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = ...
TF_MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = ...
TF_MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES = ...
TF_MODEL_MAPPING = ...
TF_MODEL_FOR_PRETRAINING_MAPPING = ...
TF_MODEL_WITH_LM_HEAD_MAPPING = ...
TF_MODEL_FOR_CAUSAL_LM_MAPPING = ...
TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING = ...
TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = ...
TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING = ...
TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING = ...
TF_MODEL_FOR_VISION_2_SEQ_MAPPING = ...
TF_MODEL_FOR_MASKED_LM_MAPPING = ...
TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = ...
TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = ...
TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = ...
TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING = ...
TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = ...
TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = ...
TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = ...
TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = ...
TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = ...
TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = ...
TF_MODEL_FOR_MASK_GENERATION_MAPPING = ...
TF_MODEL_FOR_TEXT_ENCODING_MAPPING = ...

class TFAutoModelForMaskGeneration(_BaseAutoModelClass):
    _model_mapping = ...

class TFAutoModelForTextEncoding(_BaseAutoModelClass):
    _model_mapping = ...

class TFAutoModel(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModel = ...

class TFAutoModelForAudioClassification(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForAudioClassification = ...

class TFAutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForPreTraining = ...

class _TFAutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = ...

_TFAutoModelWithLMHead = ...

class TFAutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForCausalLM = ...

class TFAutoModelForMaskedImageModeling(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForMaskedImageModeling = ...

class TFAutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForImageClassification = ...

class TFAutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForZeroShotImageClassification = ...

class TFAutoModelForSemanticSegmentation(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForSemanticSegmentation = ...

class TFAutoModelForVision2Seq(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForVision2Seq = ...

class TFAutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForMaskedLM = ...

class TFAutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForSeq2SeqLM = ...

class TFAutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForSequenceClassification = ...

class TFAutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForQuestionAnswering = ...

class TFAutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForDocumentQuestionAnswering = ...

class TFAutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForTableQuestionAnswering = ...

class TFAutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForTokenClassification = ...

class TFAutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForMultipleChoice = ...

class TFAutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForNextSentencePrediction = ...

class TFAutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = ...

TFAutoModelForSpeechSeq2Seq = ...

class TFAutoModelWithLMHead(_TFAutoModelWithLMHead):
    @classmethod
    def from_config(cls, config): ...
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs): ...
