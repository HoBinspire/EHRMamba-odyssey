"""Tokenizer module."""

import glob
import json
import os
import re
from itertools import chain
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BatchEncoding, PreTrainedTokenizerFast


class ConceptTokenizer:
    """Tokenizer for event concepts using HuggingFace Library.

    Parameters
    ----------
    pad_token: str
        Padding token.
    mask_token: str
        Mask token.
    start_token: str
        Sequence Start token. Models can use class token instead.
    end_token: str
        Sequence End token.
    class_token: str
        Class token.
    reg_token: str
        Registry token.
    unknown_token: str
        Unknown token.
    data_dir: str
        Directory containing the data.
    time_tokens: List[str]
        List of time-related special tokens.
    tokenizer_object: Optional[Tokenizer]
        Tokenizer object.
    tokenizer: Optional[PreTrainedTokenizerFast]
        Tokenizer object.
    padding_side: str
        Padding side.

    Attributes
    ----------
    pad_token: str
        Padding token.
    mask_token: str
        Mask token.
    start_token: str
        Sequence Start token.
    end_token: str
        Sequence End token.
    class_token: str
        Class token.
    reg_token: str
        Registry token.
    unknown_token: str
        Unknown token.
    task_tokens: List[str]
        List of task-specific tokens.
    tasks: List[str]
        List of task names.
    task2token: Dict[str, str]
        Dictionary mapping task names to tokens.
    special_tokens: List[str]
        Special tokens including padding, mask, start, end, class, registry tokens.
    tokenizer: PreTrainedTokenizerFast
        HuggingFace fast tokenizer object.
    tokenizer_object: Tokenizer
        Tokenizer object from tokenizers library.
    tokenizer_vocab: Dict[str, int]
        Vocabulary mapping tokens to indices.
    token_type_vocab: Dict[str, Any]
        Vocabulary for token types.
    data_dir: str
        Directory containing data files.

    """

    def __init__(
        self,
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        start_token: str = "[VS]",
        end_token: str = "[VE]",
        class_token: str = "[CLS]",
        reg_token: str = "[REG]",
        unknown_token: str = "[UNK]",
        data_dir: str = "data_files", # 数据目录
        time_tokens: List[str] = [f"[W_{i}]" for i in range(0, 4)]  # ["[W_0]", "[W_1]", "[W_2]", "[W_3]","[M_0]", "[M_1]", "[M_2]", "[M_3]", "[M_4]", "[M_5]", "[M_6]", "[M_7]", "[M_8]", "[M_9]", "[M_10]", "[M_11]", "[M_12]","[LT]"]
        + [f"[M_{i}]" for i in range(0, 13)]
        + ["[LT]"],
        tokenizer_object: Optional[Tokenizer] = None,  # 分词器对象（原始架构）
        tokenizer: Optional[PreTrainedTokenizerFast] = None,  # 快速分词器对象（对 tokenizer_obj进一步封装）
        padding_side: str = "right",
    ) -> None:

        # 维护特殊 token 
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.start_token = start_token
        self.end_token = end_token
        self.class_token = class_token
        self.reg_token = reg_token
        self.unknown_token = unknown_token
        self.padding_side = padding_side

        self.special_tokens = [
            pad_token,
            unknown_token,
            mask_token,
            start_token,
            end_token,
            class_token,
            reg_token,
        ]

        # 维护 任务类型 & 对应token，映射字典
        self.task_tokens = ["[MOR_1M]", "[LOS_1W]", "[REA_1M]"] + [
            f"[C{i}]" for i in range(0, 5)
        ]
        self.tasks = ["mortality_1month", "los_1week", "readmission_1month"] + [
            f"c{i}" for i in range(5)
        ]
        self.task2token = self.create_task_to_token_dict()

        # 维护 时间 token（加入 特殊 token 列表）
        if time_tokens is not None:
            self.special_tokens += time_tokens

        # 赋值分词器（可能为空）
        self.tokenizer_object = tokenizer_object
        self.tokenizer = tokenizer

        self.tokenizer_vocab: Dict[str, int] = {}  # token 到 索引的词汇表 
        self.token_type_vocab: Dict[str, Any] = {}  # token 类型词汇表
        self.data_dir = data_dir

        self.special_token_ids: List[int] = []  # 特殊 token id 列表
        self.first_token_index: Optional[int] = None  # 首尾 token 索引（可以标记词汇表的边界）
        self.last_token_index: Optional[int] = None

    def fit_on_vocab(self, with_tasks: bool = True) -> None:
        """(需要本地文件支持)
        初始化并配置医学事件分词器，完成以下核心功能：
        1. 从本地加载医学概念词汇表（诊断/药物/手术等）和特殊令牌
        2. 构建 完整的词汇表映射（token→index）
        3. 初始化 WordPiece分词器及其预处理配置
        4. 设置关键索引（首尾/特殊令牌）和标签映射基础
        5. 验证分词器与自定义词汇表的一致性
        
        Args:
            with_tasks: 是否包含任务tokens, 默认 True

        Returns:
            None
        """
        # 构建所有可能的医学概念字典 token_type_vocab：special_tokens，医学概念token, task_tokens
        self.token_type_vocab["special_tokens"] = self.special_tokens  # token 类型词汇表中加入 特殊 token list

        vocab_json_files = glob.glob(os.path.join(self.data_dir, "*vocab.json"))  # 从 data_dir 目录下 *vocab.json 文件加载 医学概念词汇表（按类别存储）
        for file in vocab_json_files:
            with open(file, "r") as vocab_file:
                vocab = json.load(vocab_file)
                vocab_type = file.split("/")[-1].split(".")[0]
                self.token_type_vocab[vocab_type] = vocab

        if with_tasks:
            self.token_type_vocab["task_tokens"] = self.task_tokens

        # 展平 token_type_vocab 中所有 token 为 list, 为每个token分配一个唯一int索引
        tokens = list(chain.from_iterable(list(self.token_type_vocab.values())))
        self.tokenizer_vocab = {token: i for i, token in enumerate(tokens)}

        # 更新 special tokens 列表
        if with_tasks:
            self.special_tokens += self.task_tokens

        # 初始化 WordPiece 分词器（主分词器）
        self.tokenizer_object = Tokenizer(
            models.WordPiece(
                vocab=self.tokenizer_vocab,
                unk_token=self.unknown_token,
                max_input_chars_per_word=1000, # 单个 token 允许最大字符数
            ),
        )
        self.tokenizer_object.pre_tokenizer = pre_tokenizers.WhitespaceSplit()  # 预分词器：以空格对文本做简单分割（再由主分词器分割子词）
        self.tokenizer = self.create_tokenizer(self.tokenizer_object)  # 最终的分词器，对 tokenizer_obj 进一步封装和配置

        # 从词汇表中 获取 首尾，特殊 tokens 的 id
        self.first_token_index = self.get_first_token_index()
        self.last_token_index = self.get_last_token_index()
        self.special_token_ids = self.get_special_token_ids()

        # 初始化每个 token 到 label 的映射（是其自身），后续可扩展为具体的标签（如"医学编码" -> "对应的名称"）
        self.token_to_label_dict = {
            token: token for token in self.tokenizer_vocab.keys()
        }

        # 确保 HuggingFace 分词器内部词汇表 与 自定义的 tokenizer_vocab 完全一致（防止训练/推理偏差）
        assert self.tokenizer_vocab == self.tokenizer.get_vocab(), (
            "Tokenizer vocabulary does not match original"
        )

    def load_token_labels(self, codes_dir: str) -> Dict[str, str]:
        """（需要本地文件支持）
        加载并合并医学概念编码与名称的映射关系文件​​，
        并更新分词器的 token_to_label_dict 字典，使其包含更详细的医学概念标签信息 → (主要作用)

        Args:
            codes_dir(str): json 映射文件的目录

        Returns:
            Dict[str, str]: 一个包含 医学概念编码 到 名称映射关系的字典
        """
        merged_dict = {}

        # 读取 codes_dir 中所有 .json 文件，每个文件中包含 医学概念编码到名称 的映射
        for filename in os.listdir(codes_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(codes_dir, filename)
                with open(filepath, "r") as file:
                    data = json.load(file)
                    merged_dict.update(data)

        # 赋值 token_to_label_dict 字典
        for token, label in merged_dict.items():
            self.token_to_label_dict[token] = label

        return merged_dict

    def create_tokenizer(
        self,
        tokenizer_obj: Tokenizer,
    ) -> PreTrainedTokenizerFast:
        """将 Tokenizer 对象，封装成 PreTrainedTokenizerFast 对象（高级的分词器, 提供了更多功能和优化）.

        Parameters
        ----------
        tokenizer_obj: Tokenizer
            Tokenizer object.

        Returns
        -------
        PreTrainedTokenizerFast
            Tokenizer object.

        """
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            bos_token=self.start_token,
            eos_token=self.end_token,
            unk_token=self.unknown_token,
            pad_token=self.pad_token,
            cls_token=self.class_token,
            mask_token=self.mask_token,
            padding_side=self.padding_side,
        )
        return self.tokenizer

    def __call__(
        self,
        batch: Union[str, List[str]],
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        truncation: bool = True,
        padding: str = "max_length",  # 填充策略
        max_length: int = 2048,
    ) -> BatchEncoding:
        """封装分词器的调用逻辑：
            1. 将输入的医学文本或事件序列（单个字符串或字符串列表）转换为 Token ID 序列；
            2. 格式化输出： 包含 input_ids、attention_mask 等字段的 BatchEncoding 对象
            3. 自动将结果转换为 PyTorch 张量（return_tensors="pt"）

        Args:
            batch (Union[str, List[str]]): 输入文本（单条或批量）
            return_attention_mask (bool): 是否生成注意力掩码，标识有效 Token 位置。（默认有效）
            return_token_type_ids (bool): 是否生成 Token 类型 ID（区分句子A/B，用于BERT类模型）
            truncation (bool): 是否截断超过 max_length 的序列（默认截断）
            padding (str): 填充策略（"max_length"/"longest"/False）
            max_length (int): 序列最大长度（截断/填充依据）

        Returns:
            BatchEncoding: 类字典结构的对象，包含以下字段 input_ids(torch.Tensor); attention_mask(torch.Tensor); token_type_ids(torch.Tensor)
        """
        return self.tokenizer(
            batch,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            return_tensors="pt",
        )

    def encode(self, concept_tokens: str) -> List[int]:
        """轻量级的编码函数，返回输入 str 的 token 列表，比 __call__ 更加轻量

        Args:
            concept_tokens(str)

        Returns
            List[int]: 对应的 token id 列表
        """
        return self.tokenizer_object.encode(concept_tokens).ids  # 这里调用 tokenizer_object 是考虑到 不需要生成(attention_mask、token_type_ids 等额外开销，更加轻量化)

    def decode(self, concept_ids: List[int]) -> str:
        """将 id 列表解码为 token 字符串.

        Args:
            concept_ids(List[int])

        Returns:
            str: 解码结果
        """
        return self.tokenizer_object.decode(concept_ids)

    def decode_to_labels(self, concept_input: Union[List[str], List[int]]) -> List[str]:
        """将医学概念序列（Token 或 ID 列表）解码为​ 人类可以理解的标签序列

        Args:
            concept_input (Union[List[str], List[int]])

        Returns:
            List[str]: 解码后的 人类可以理解的 标签序列列表

        """
        if isinstance(concept_input[0], int): # 如果是 id 列表，先解码为 token
            concept_input = [self.id_to_token(token_id) for token_id in concept_input]

        decoded_sequence = []
        for item in concept_input:
            match = re.match(r"^(.*?)(_\d)$", item)
            if match: # 匹配带后缀的 token
                base_part, suffix = match.groups()
                replaced_item = (  # 对前缀用字典作映射，后缀直接拼接
                    self.token_to_label_dict.get(base_part, base_part) + suffix
                )
            else:
                replaced_item = self.token_to_label_dict.get(item, item) # （第二个参数是 不存在 key 时候的 默认返回值）
            decoded_sequence.append(replaced_item)

        return decoded_sequence

    def token_to_id(self, token: str) -> int:
        """获取 token 对应的 id. （用预定义的 tokenizer_object 成员方法）

        Parameters
            token(str)

        Returns
            int: Token id.

        """
        return self.tokenizer_object.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        """返回 id 对应的 token

        Args:
            token_id (int)

        Returns:
            str
        """
        return self.tokenizer_object.id_to_token(token_id)

    def token_to_label(self, token: str) -> str:
        """直接调用 decode_to_label函数，将单个 token 映射为 人类可以理解的 label.

        Args:
            token (str)

        Returns:
            str
        """
        return self.decode_to_labels([token])[0]

    def get_all_token_indexes(self, with_special_tokens: bool = True) -> Set[int]:
        """返回所有可能的 token id 集合

        Args:
        with_special_tokens(bool): 是否包含 特殊token, 默认包含

        Returns:
            Set[int]
        """
        all_token_ids = set(self.tokenizer_vocab.values())
        special_token_ids = set(self.get_special_token_ids())

        return (
            all_token_ids if with_special_tokens else all_token_ids - special_token_ids
        )

    def get_first_token_index(self) -> int:
        """获取词汇表中 id 最小的 token 的 id

        Args:
            None

        Returns
            int: 词汇表中最小 id
        """
        return min(self.tokenizer_vocab.values())

    def get_last_token_index(self) -> int:
        """获取词汇表中 id 最大的 token 的 id

        Returns
            int: id 最大的 token 的 id
        """
        return max(self.tokenizer_vocab.values())

    def get_vocab_size(self) -> int:
        """返回词汇表大小（其中键的数量）

        Returns:
            int 
        """
        return len(self.tokenizer)

    def get_class_token_id(self) -> int:
        """返回 CLS token 的 id.

        Returns:
            int 
        """
        return self.token_to_id(self.class_token)

    def get_pad_token_id(self) -> int:
        """返回 PAD token 的 id.

        Returns:
            int 
        """
        return self.token_to_id(self.pad_token)

    def get_mask_token_id(self) -> int:
        """返回 MASK token 的 id.

        Returns:
            int 
        """
        return self.token_to_id(self.mask_token)

    def get_eos_token_id(self) -> int:
        """返回 end of sequence token 的 id.

        Returns:
            int
        """
        return self.token_to_id(self.end_token)

    def get_special_token_ids(self) -> List[int]:
        """获取 special token id 列表.

        Returns
            List[int]: special token id 列表.
        """
        self.special_token_ids = []

        for special_token in self.special_tokens:
            special_token_id = self.token_to_id(special_token)
            self.special_token_ids.append(special_token_id)

        return self.special_token_ids

    def save(self, save_dir: str) -> None:
        """将当前分词器的​​完整配置和词汇表​​保存到本地 JSON 文件，实现分词器状态的持久化

        Args:
            save_dir: str
        
        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)  # 目录不存在 自动创建
        tokenizer_config = {
            "pad_token": self.pad_token,
            "mask_token": self.mask_token,
            "unknown_token": self.unknown_token,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "class_token": self.class_token,
            "reg_token": self.reg_token,
            "special_tokens": self.special_tokens, # 所有特殊标记
            "tokenizer_vocab": self.tokenizer_vocab,  # 词汇表（token→id）
            "token_type_vocab": self.token_type_vocab, # 类型化词汇表 {"diagnosis":["E11.9"], ...
            "data_dir": self.data_dir,  # 溯源，用于支持动态词汇表扩展
        }
        save_path = os.path.join(save_dir, "tokenizer.json")
        with open(save_path, "w") as file:
            json.dump(tokenizer_config, file, indent=4)

    @classmethod
    def load(cls, load_path: str) -> "ConceptTokenizer":
        """
        从保存的 JSON 文件中​​完整恢复​​ ConceptTokenizer 分词器实例

        Args:
            cls: 类方法约定俗成的第一个参数名，表示当前类本身
            load_path (str): 分词器配置读取目录 (json文件)

        Returns:
            ConceptTokenizer: 恢复的 分词器对象实例
        """
        with open(load_path, "r") as file:
            tokenizer_config = json.load(file)

        tokenizer = cls(
            pad_token=tokenizer_config["pad_token"],
            mask_token=tokenizer_config["mask_token"],
            unknown_token=tokenizer_config["unknown_token"],
            start_token=tokenizer_config["start_token"],
            end_token=tokenizer_config["end_token"],
            class_token=tokenizer_config["class_token"],
            reg_token=tokenizer_config["reg_token"],
            data_dir=tokenizer_config["data_dir"],
        )

        tokenizer.special_tokens = tokenizer_config["special_tokens"]
        tokenizer.tokenizer_vocab = tokenizer_config["tokenizer_vocab"]
        tokenizer.token_type_vocab = tokenizer_config["token_type_vocab"]

        tokenizer.tokenizer_object = Tokenizer(
            models.WordPiece(
                vocab=tokenizer.tokenizer_vocab,
                unk_token=tokenizer.unknown_token,
                max_input_chars_per_word=1000,
            ),
        )
        tokenizer.tokenizer_object.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.tokenizer = tokenizer.create_tokenizer(tokenizer.tokenizer_object)

        return tokenizer

    def create_task_to_token_dict(self) -> Dict[str, str]:
        """创建一个从 任务名称 到 token 的映射词典 (构建实例时调用)

        Args:
            None (硬编码在函数中) 

        Returns:
            Dict[str, str]
        """
        task2token = {
            "mortality_1month": "[MOR_1M]",
            "los_1week": "[LOS_1W]",
            "readmission_1month": "[REA_1M]",
        }
        for i in range(5):
            task2token[f"c{i}"] = f"[C{i}]"

        return task2token

    def task_to_token(self, task: str) -> str:
        """返回 任务名称对应的 token.

        Args:
            task(str): 事先约定的 任务名称

        Returns:
            str
        """
        return self.task2token[task]

    @staticmethod
    def create_vocab_from_sequences(
        sequences: Union[List[List[str]], pd.Series],
        save_path: str,
    ) -> None:
        """从输入的 medical events token sequences 中去重, 得到的集合 保存为 json文件.

        Args:
            sequences (Union[List[List[str]], pd.Series]): 支持 List[List[str]] 和 pd.Series 两种类型的序列
            save_path (str): json 文件保存目录
        """
        # 展平, 并用 set 去重
        unique_tokens = sorted(
            set(token for sequence in sequences for token in sequence) 
        )

        # 包含空格 则抛出异常
        if any(" " in token for token in unique_tokens):
            raise ValueError("Tokens should not contain spaces.")

        with open(save_path, "w") as vocab_file:
            json.dump(unique_tokens, vocab_file, indent=4)
            