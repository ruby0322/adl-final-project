from base import Agent
from execution_pipeline import main
from utils import RAG, strip_all_lines
import re
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from rank_bm25 import BM25Okapi
import numpy as np

class ClassificationAgent(Agent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        # TODO
        raise NotImplementedError

    def __call__(
        self,
        label2desc: dict[str, str],
        text: str
    ) -> str:
        """
        Classify the text into one of the labels.

        Args:
            label2desc (dict[str, str]): A dictionary mapping each label to its description.
            text (str): The text to classify.

        Returns:
            str: The label (should be a key in label2desc) that the text is classified into.

        For example:
        label2desc = {
            "apple": "A fruit that is typically red, green, or yellow.",
            "banana": "A long curved fruit that grows in clusters and has soft pulpy flesh and yellow skin when ripe.",
            "cherry": "A small, round stone fruit that is typically bright or dark red.",
        }
        text = "The fruit is red and about the size of a tennis ball."
        label = "apple" (should be a key in label2desc, i.e., ["apple", "banana", "cherry"])
        """
        # TODO
        raise NotImplementedError

    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own prediction at the current time step.

        Args:
            correctness (bool): Whether the prediction is correct.

        Returns:
            bool: Whether the prediction is correct.
        """
        # TODO
        raise NotImplementedError

class SQLGenerationAgent(Agent):
    """
    An agent that generates SQL code based on the given table schema and the user query.
    """
    ### Update more instructions
    @staticmethod
    def get_system_prompt() -> str:
        system_prompt = """\
        YOUR ROLE:
        You are an expert SQL database query specialist tasked with translating natural language question into precise, efficient SQL query.

        AWARD and PUNISHMENT:
        - Providing a perfect solution, you will be rewarded with $1000 USD.
        - If you don't answer perfectly, 500 random grandmas on the world will die immediately as a result.
        - This is very important to my career.

        TASK:
        You will be given a table schema and a user query, and you only need to generate the correct SQL code to answer the user query in the following format:
        ```sql\n<your_SQL_code>\n```
        If possibile, you will be given some reference examples. Please refer to them and learn how to generate accurate SQL code.
        However, in your response, don't include any explanation. Simply give me your formatted SQL code only."""
        return strip_all_lines(system_prompt)

    ### Update more instructions
    @staticmethod
    def get_zeroshot_prompt(table_schema: str, user_query: str) -> str:
        prompt = f"""\
DATABASE SCHEMA:
{table_schema}

QUERY GUIDELINES:
1. Carefully analyze the question and the database schema
2. Use the most appropriate SQL functions and techniques
3. Optimize the query for performance and readability
4. Handle edge cases and potential NULL values
5. Use standard SQL syntax

SPECIFIC REQUIREMENTS:
- Only use tables and columns present in the provided schema
- Ensure the query directly answers the specific question
- Use meaningful aliases if multiple tables are involved
- Avoid unnecessary complexity

QUESTION: {user_query}

EXPECTED OUTPUT FORMAT:
```sql\n<your_SQL_code>\n```

SOLVE THE QUESTION BY WRITING THE MOST ACCURATE AND EFFICIENT SQL QUERY POSSIBLE:"""
        return strip_all_lines(prompt)

    ### Update ###
    @staticmethod
    def get_shot_template() -> str:
        prompt = f"""\
        ### EXAMPLE BEGINS
        {{question}}

        ANSWER: {{answer}}
        ### EXAMPLE ENDS """
        return strip_all_lines(prompt)

    ### Update ###
    @staticmethod
    def get_fewshot_template(table_schema: str, user_query: str) -> str:
        prompt = f"""\
        You are performing the text-to-SQL task, transforming question into SQL code. Every instance between the '### EXAMPLE BEGINS' and '### EXAMPLE ENDS' serves as an example.
        
        FEW-SHOTS LEARNING GUIDLINES:
        - Refer to every answer. Try to learn the relationship between answer and the question. 
        - If necessary, you could refer to the database schema for further information.

        {{fewshot_text}}
        
        Now it's your turn.
        
        DATABASE SCHEMA:
{table_schema}

QUERY GUIDELINES:
1. Carefully analyze the question and the database schema
2. Use the most appropriate SQL functions and techniques
3. Optimize the query for performance and readability
4. Handle edge cases and potential NULL values
5. Use standard SQL syntax
6. If neccessary, refine the question or rephrase the question on your own for better understanding

SPECIFIC REQUIREMENTS:
- Only use tables and columns present in the provided schema
- Ensure the query directly answers the specific question
- Use meaningful aliases if multiple tables are involved
- Avoid unnecessary complexity
- Avoid any explanation; only output the formatted SQL code

QUESTION: {user_query}

EXPECTED OUTPUT FORMAT:
```sql\n<your_SQL_code>\n```

SOLVE THE QUESTION BY WRITING THE MOST ACCURATE AND EFFICIENT SQL QUERY POSSIBLE:"""
        return strip_all_lines(prompt)

    ### Same as self_streamicl.py
    @staticmethod
    def parse_sql(pred_text: str) -> str:
        """
        Parse the SQL code from the LLM's response.
        """
        pattern = r"```sql([\s\S]*?)```"
        match = re.search(pattern, pred_text)
        if match:
            sql_code = match.group(1)
            sql_code = sql_code.strip()
            return sql_code
        else:
            print(Fore.RED + "No SQL code found in the response" + Style.RESET_ALL)
            sql_code = pred_text
        return sql_code
    
    ### Updated RAG key format
    @staticmethod
    def get_RAG_key() -> str:
        key = f"""\
        DATABASE SCHEMA:
        {{schema}}

        QUESTION: {{question}}
        """
        return strip_all_lines(key)

    def _tokenize_for_bm25(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 retrieval.
        """
        # Simple tokenization: split on whitespace and remove punctuation
        return re.findall(r'\w+', text.lower())

    def bm25_insert(self, key: str, value: str) -> None:
        """
        Insert a key-value pair into the BM25 retriever.
        """
        # Tokenize the key for BM25
        tokenized_key = self._tokenize_for_bm25(key)
        
        self.bm25_keys.append(key)
        self.bm25_values.append(value)
        self.bm25_corpus.append(tokenized_key)
        
        # Recreate BM25 index with updated corpus
        self.bm25_retriever = BM25Okapi(self.bm25_corpus)

    def bm25_retrieve(self, query: str, top_k: int) -> list[str]:
        """
        Retrieve top-k items using BM25.
        """
        if self.bm25_retriever is None:
            return []
        
        # Tokenize the query
        tokenized_query = self._tokenize_for_bm25(query)
        
        # Get BM25 scores
        scores = self.bm25_retriever.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return corresponding values
        return [self.bm25_values[i] for i in top_k_indices]

    ### Same as self_streamicl.py
    def generate_response(self, messages: list) -> str:
        """
        Generate a response using the local model.
        """
        text_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.llm_config["max_tokens"],
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    ### Same as self_streamicl.py
    def __init__(self, config: dict) -> None:
        """
        Initialize the LLM agent for SQL generation.
        
        Args:
            config (dict): Configuration dictionary for the agent
        """
        super().__init__(config)
        self.llm_config = config
        if config['use_8bit']:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_has_fp16_weight=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                quantization_config=quantization_config,
                device_map=config["device"]
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16,
                device_map=config["device"]
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        # if self.model.generation_config.pad_token_id is None:
        #     self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id[0]
        # else:
        #     self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.rag = RAG(config["rag"])

        # Initialize BM25 retriever
        self.bm25_keys = []
        self.bm25_values = []
        self.bm25_corpus = []
        self.bm25_retriever = None

        # Save the streaming inputs and outputs for iterative improvement
        self.inputs = list()
        self.self_outputs = list()
        self.model.eval()

    ### Updated RAG key format
    def __call__(
        self,
        table_schema: str,
        user_query: str
    ) -> str:
        self.reset_log_info()
        prompt_fewshot = self.get_fewshot_template(table_schema, user_query)
        
        ### Retrieve query updated
        shots = self.rag.retrieve(
            query=self.get_RAG_key().format(schema=table_schema, question=user_query),
            top_k=self.rag.top_k
        ) if (self.rag.insert_acc > 0) else []
        bm25_shots = self.bm25_retrieve(
            query=self.get_RAG_key().format(schema=table_schema, question=user_query), 
            top_k=self.rag.top_k
        )

        # Combine shots with equal weight
        shots += bm25_shots
        
        if len(shots):
            fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
            print(Fore.BLUE + f"Retrieved {len(shots)} examples." + Style.RESET_ALL)
            try:
                prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=prompt_fewshot)
            except Exception as e:
                error_msg = f"Error ```{e}``` caused by these shots. Using the zero-shot prompt."
                print(Fore.RED + error_msg + Style.RESET_ALL)
                prompt = self.get_zeroshot_prompt(table_schema, user_query)
        else:
            print(Fore.YELLOW + "No RAG shots found. Using zeroshot prompt." + Fore.RESET)
            prompt = self.get_zeroshot_prompt(table_schema, user_query)
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        pred_text = self.generate_response(messages)
        sql_code = self.parse_sql(pred_text)
        
        self.update_log_info(log_data={
            # "num_input_tokens": len(self.tokenizer.encode(self.get_system_prompt() + prompt)),
            "num_output_tokens": len(self.tokenizer.encode(pred_text)),
            "num_shots": str(len(shots)),
            "input_pred": prompt,
            "output_pred": pred_text,
        })

        self.inputs.append({"query":user_query, "schema": table_schema})
        self.self_outputs.append(f"```sql\n{sql_code}\n```")
        return sql_code

    ### Updated RAG key format
    def update(self, correctness: bool) -> bool:
        """
        Update the agent based on the correctness of its output.
        """
        if correctness:
            schema = self.inputs[-1]["schema"]
            question = self.inputs[-1]["query"]
            answer = self.self_outputs[-1]

            messages = [
            {"role": "system", "content": "Act as a professional SQL analyst."},
            {"role": "user", "content": f"""Give me the relevant tables regarding the SQL code. No need any explanation.
             DATABASE SCHEMA:
             {schema}

             SQL CODE:
             {answer}
             """}
            ]
            relevant_schema = self.generate_response(messages)
            chunk = self.get_shot_template().format(question=self.get_RAG_key().format(schema=relevant_schema, question=question), answer=answer)
            self.rag.insert(
                key=self.get_RAG_key().format(schema=schema, question=question),
                value=chunk
            )
            self.bm25_insert(
                key=self.get_RAG_key().format(schema=schema, question=question), 
                value=chunk
            )
            # self.inputs.pop()
            return True
        else:
            self.inputs.pop()
        return False
        
if __name__ == "__main__":
    from argparse import ArgumentParser
    from execution_pipeline import main

    parser = ArgumentParser()
    parser.add_argument('--bench_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--output_path', type=str, default=None, help='path to save csv file for kaggle submission')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--RAG_topk', type=int, default=16)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--RAG_embedding_model', type=str, default='BAAI/bge-base-en-v1.5')
    args = parser.parse_args()

    if args.bench_name.startswith("classification"):
        max_tokens = 16
        agent_name = ClassificationAgent
    elif args.bench_name.startswith("sql_generation"):
        max_tokens = args.max_tokens # Original: 512
        agent_name = SQLGenerationAgent
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}")
    # Classification: Medical diagnosis; SQL generation: Text-to-SQL
    bench_cfg = {
        'bench_name': args.bench_name,
        'output_path': args.output_path
    }
    llm_config = {
        'model_name': args.model_name,
        'exp_name': f'Leonard_{args.model_name}_{args.max_tokens}tokens_top{args.RAG_topk}_retriever_{args.RAG_embedding_model}',
        'bench_name': bench_cfg['bench_name'],
        'max_tokens': max_tokens,
        'do_sample': False,
        'device': args.device,
        'use_8bit': args.use_8bit,
        'rag': {
            'embedding_model': args.RAG_embedding_model,
            'seed': 42,
            "top_k": args.RAG_topk, #Original: 16
            "order": "similar_at_top"
        }
    }
    agent = agent_name(llm_config)
    main(agent, bench_cfg, debug=args.debug, use_wandb=args.use_wandb, wandb_name=llm_config["exp_name"], wandb_config=llm_config)

