"""When adding a new embedder:

- Create the class in `malid/embedders/`.
- Add the import to this file. (Make sure all slow imports are deferred until the embedder is actually initialized.)
- Add the embedder to the embedder registry below.
- Run tests on CPU and GPU: `pytest tests/test_embeddings.py && pytest --gpu tests/test_embeddings.py`
"""


from typing import Type, Union
from malid.embedders.base_embedder import BaseEmbedder, BaseFineTunedEmbedder
from malid.embedders.unirep import (
    UnirepEmbedder,
    UnirepFineTunedEmbedder,
    UnirepEmbedderCDR3Only,
    UnirepFineTunedEmbedderCDR3Only,
)
from malid.embedders.biotransformers import (
    Esm2Embedder,
    Esm2FineTunedEmbedder,
    Esm2EmbedderCDR3Only,
    Esm2FineTunedEmbedderCDR3Only,
)
from malid.embedders.ablang import (
    AbLangEmbeddder,
)

# List of embedder class types
# Insert new embedders here
_EMBEDDERS = [
    UnirepEmbedder,
    UnirepFineTunedEmbedder,
    UnirepEmbedderCDR3Only,
    UnirepFineTunedEmbedderCDR3Only,
    Esm2Embedder,
    Esm2FineTunedEmbedder,
    Esm2EmbedderCDR3Only,
    Esm2FineTunedEmbedderCDR3Only,
    AbLangEmbeddder,
]

# Re-index by name
_EMBEDDERS_DICT = {
    embedder_class_type.name: embedder_class_type for embedder_class_type in _EMBEDDERS
}


def get_embedder_by_name(
    name: str,
) -> Union[Type[BaseEmbedder], Type[BaseFineTunedEmbedder]]:
    return _EMBEDDERS_DICT[name]
