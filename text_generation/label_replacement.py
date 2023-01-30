import re
from unidecode import unidecode

from action_executor.actions import ESActionOperator
from constants import RepresentEntityLabelAs
from args import parse_and_get_args
args = parse_and_get_args()


class LabelReplacer:
    placeholder_names = ["Mary", "Anna", "Emma", "Elizabeth", "Minnie", "Margaret", "Ida", "Alice", "Bertha", "Sarah",
                         "Annie", "Clara", "Ella", "Florence", "Cora", "Martha", "Laura", "Nellie", "Grace", "Carrie",
                         "Maude", "Mabel", "Bessie", "Jennie", "Gertrude", "Julia", "Hattie", "Edith", "Mattie", "Rose",
                         "Catherine", "Lillian", "Ada", "Lillie", "Helen", "Jessie", "Louise", "Ethel", "Lula",
                         "Myrtle", "Eva", "Frances", "Lena", "Lucy", "Edna", "Maggie", "Pearl", "Daisy", "Fannie",
                         "Josephine", "Dora", "Rosa", "Katherine", "Agnes", "Marie", "Nora", "May", "Mamie", "Blanche",
                         "Stella", "Ellen", "Nancy", "Effie", "Sallie", "Nettie", "Della", "Lizzie", "Flora", "Susie",
                         "Maud", "Mae", "Etta", "Harriet", "Sadie", "Caroline", "Katie", "Lydia", "Elsie", "Kate",
                         "Susan", "Mollie", "Alma", "Addie", "Georgia", "Eliza", "Lulu", "Nannie", "Lottie", "Amanda",
                         "Belle", "Charlotte", "Rebecca", "Ruth", "Viola", "Olive", "Amelia", "Hannah", "Jane",
                         "Virginia"]

    def __init__(self, operator: ESActionOperator):
        self.op = operator

    def labs2subs(self, utterance: str, entities: list[str], labels_as: RepresentEntityLabelAs) -> tuple[str, dict]:
        if labels_as not in RepresentEntityLabelAs:
            raise NotImplementedError(f'Chosen RepresentEntityLabeAs Enum ({labels_as}) is not supported.')

        if labels_as == RepresentEntityLabelAs.LABEL:
            return unidecode(utterance), dict()

        if labels_as == RepresentEntityLabelAs.GROUP:
            return self._replace_labels_by_groups(utterance, entities)

        return self._replace_labels_in_utterance(utterance, entities, labels_as)

    def _replace_labels_by_groups(self, utterance: str, entities: list[str]) -> tuple[str, dict]:
        utterance, inverse_map = self._replace_labels_in_utterance(utterance,
                                                                   entities,
                                                                   RepresentEntityLabelAs.ENTITY_ID)

        group_inverse_map = dict()
        pattern = r"Q\d+(?:\s?(?:,|and)\s?Q\d+)*"
        matches = re.findall(pattern, utterance, flags=re.IGNORECASE)
        for j, match in enumerate(matches):
            repl = f"group{j}"
            utterance = utterance.replace(match, repl)
            group_inverse_map[repl] = self.subs2labs(match, inverse_map)

        return utterance, group_inverse_map

    def _replace_labels_in_utterance(self, utterance: str, entities: list[str],
                                     labels_as: RepresentEntityLabelAs) -> tuple[str, dict]:
        # Use a dictionary to map the values of the labels_as parameter to the appropriate replacement string
        label_replacements = {
            RepresentEntityLabelAs.ENTITY_ID: lambda e, i: e.lower(),
            RepresentEntityLabelAs.PLACEHOLDER: lambda e, i: f"entity{i}",
            RepresentEntityLabelAs.PLACEHOLDER_NAMES: lambda e, i: self.placeholder_names[i],
        }

        if labels_as not in label_replacements.keys():
            raise NotImplementedError(f'Chosen RepresentEntityLabeAs Enum ({labels_as}) is not supported.')

        utterance = unidecode(utterance)
        inverse_map = dict()

        for idx, ent in enumerate(entities):
            label = self.op.get_entity_label(ent)
            replacement = label_replacements[labels_as](ent, idx)
            utterance = utterance.replace(label, replacement)
            inverse_map[replacement] = label

        return utterance, inverse_map

    @staticmethod
    def subs2labs(utterance: str, inverse_map: dict[str:str]):
        for eid, lab in inverse_map.items():
            utterance = re.sub(eid, lab, utterance, flags=re.IGNORECASE)

        return utterance