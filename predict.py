# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
import logging

DEFAULT_MODEL = "mlabonne/phixtral-2x2_8"
CACHE_DIR = "pretrained_weights"
TOKENIZER_PATH = './tokenizer'

# To download tensorizer weights instead of load them from a local source, `REMOTE_PATH_TO_TENSORIZER_WEIGHTS` to their URL
REMOTE_PATH_TO_TENSORIZER_WEIGHTS = None
PATH_TO_TENSORIZER_WEIGHTS = REMOTE_PATH_TO_TENSORIZER_WEIGHTS if REMOTE_PATH_TO_TENSORIZER_WEIGHTS else "./tensorized_models/phixtral.tensors"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        
        print("Loading pipeline...")
        torch.set_default_device("cuda")
        self.model = self.load_tensorizer(weights)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH
        )
        print("setup took: ", time.time() - start)
        
    def load_tensorizer(self, weights):
        st = time.time()
        logger.info(f'deserializing weights from {weights}')
        config = AutoConfig.from_pretrained(DEFAULT_MODEL)

        model = no_init_or_tensor(
            lambda: AutoModelForCausalLM.from_pretrained(
                None, config=config, state_dict=OrderedDict()
            )
        )
        des = TensorDeserializer(weights, plaid_mode=True)
        des.load_into_module(model)
        logging.info(f'weights loaded in {time.time() - st}')
        return model

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        max_length: int = Input(
            description="Max length", ge=0, le=2048, default=200
        ),
    ) -> str:
        """Run a single prediction on the model"""     
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        outputs = self.model.generate(**inputs, max_length=max_length)
        result = self.tokenizer.batch_decode(outputs)[0]

        return result
