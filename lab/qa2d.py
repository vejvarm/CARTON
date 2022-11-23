# MODEL SOURCE: https://huggingface.co/domenicrosati/QA2D-t5-small?text=which+province+is+halifax+in.+nova+scotia
# USAGE GUIDE: https://huggingface.co/docs/transformers/main/en/model_doc/t5#transformers.T5ForConditionalGeneration
# TRAINING DATASET: https://huggingface.co/datasets/domenicrosati/QA2D
import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from helpers import setup_logger

LOGGER = setup_logger(__name__, loglevel=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained("domenicrosati/QA2D-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("domenicrosati/QA2D-t5-small")


def infer_one(qa_string: str):
    input_ids = tokenizer(qa_string, return_tensors="pt").input_ids
    LOGGER.debug(f"input_ids in infer_one: ({input_ids.shape}) {input_ids}")

    outputs = model.generate(input_ids)
    LOGGER.debug(f"outputs in infer_one: ({outputs.shape}) {outputs}")

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    infer_one("Which administrative territory is the country of origin of Frank and Jesse ? United States of America")
    infer_one("In what manner did Ghena Dimitrova die ? natural causes")
    infer_one("What had that one as a cause of death ? Rita Levi-Montalcini, Chris Marker, The Rev")
    infer_one("What situation did James W. Horne die in ? natural causes")
    infer_one("Which cemetery is the resting place of Sophie Grinberg-Vinaver ? Père Lachaise Cemetery")  # the resting place of Sophie Grinberg-Vinaver is Père Lach
    infer_one("Which cemetery is the resting place of Q15987602 ? Q311")  # the resting place of Q15987602 is Q311