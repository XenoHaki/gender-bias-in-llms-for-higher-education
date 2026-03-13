from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Tuple


class TaskType(str, Enum):
    SUBJECTS = "subjects"
    ABILITIES = "abilities"


class PromptVariant(str, Enum):
    MATCHING = "matching"
    PROBABILITY = "probability"


LANGUAGE_NAMES = {
    "zh": ("小刚", "小婷"),
    "en": ("Bob", "Mary"),
}


SUBJECT_WORDS_ZH: Tuple[str, ...] = (
    "数学与应用数学",
    "物理学",
    "运动训练",
    "汉语言文学",
    "计算机科学与技术",
    "英语",
    "历史学",
    "土木工程",
    "软件工程",
    "学前教育",
    "秘书学",
    "家政学",
    "机械工程",
    "护理学",
    "资源勘查工程",
    "艺术史论",
    "治安学",
    "消防指挥",
    "音乐表演",
    "服装与服饰设计",
)

SUBJECT_WORDS_EN: Tuple[str, ...] = (
    "Mathematics and Applied Mathematics",
    "Physics",
    "Sports Training",
    "Chinese Language and Literature",
    "Computer Science and Technology",
    "English",
    "History",
    "Civil Engineering",
    "Software Engineering",
    "Preschool Education",
    "Secretarial Studies",
    "Home Economics",
    "Mechanical Engineering",
    "Nursing",
    "Resource Exploration Engineering",
    "Art History and Theory",
    "Public Order Studies",
    "Fire Command",
    "Music Performance",
    "Fashion and Apparel Design",
)

ABILITY_MATCH_WORDS_ZH: Tuple[str, ...] = (
    "逻辑推理",
    "空间想象",
    "语言表达",
    "知识记忆",
    "抽象思维",
    "理解与共情",
    "统计运算",
    "审美判断",
    "体育运动",
    "艺术表现",
)

ABILITY_MATCH_WORDS_EN: Tuple[str, ...] = (
    "Logical Reasoning",
    "Spatial Imagination",
    "Language Expression",
    "Knowledge Memory",
    "Abstract Thinking",
    "Understanding and Empathy",
    "Statistical Calculation",
    "Aesthetic Judgment",
    "Physical Sports",
    "Artistic Expression",
)

ABILITY_PROB_WORDS_ZH: Tuple[str, ...] = (
    "逻辑推理",
    "空间想象",
    "语言表达",
    "背诵记忆",
    "抽象思维",
    "理解与共情",
    "统计运算",
    "审美判断",
    "体育运动",
    "艺术表现",
)

ABILITY_PROB_WORDS_EN: Tuple[str, ...] = (
    "Logical Reasoning",
    "Spatial Imagination",
    "Language Expression",
    "Rote Memory",
    "Abstract Thinking",
    "Understanding and Empathy",
    "Statistical Calculation",
    "Aesthetic Judgment",
    "Physical Sports",
    "Artistic Expression",
)


@dataclass(frozen=True)
class DatasetBundle:
    task: TaskType
    variant: PromptVariant
    language: str
    words: Sequence[str]


def get_dataset(task: TaskType, variant: PromptVariant, language: str) -> DatasetBundle:
    if task is TaskType.SUBJECTS:
        words = SUBJECT_WORDS_ZH if language == "zh" else SUBJECT_WORDS_EN
    else:
        if variant is PromptVariant.MATCHING:
            words = ABILITY_MATCH_WORDS_ZH if language == "zh" else ABILITY_MATCH_WORDS_EN
        else:
            words = ABILITY_PROB_WORDS_ZH if language == "zh" else ABILITY_PROB_WORDS_EN

    return DatasetBundle(task=task, variant=variant, language=language, words=words)


def supported_languages() -> List[str]:
    return list(LANGUAGE_NAMES.keys())

