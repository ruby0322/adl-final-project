import re
import random
from base import Agent
from colorama import Fore, Style
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
from transformers import logging as transformers_logging

import os

from utils import RAG, strip_all_lines

# Ignore warning messages from transformers
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

class LocalModelAgent(Agent):
    """
    A base agent that uses a local model for text generation tasks.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize the local model
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
        self.rag = RAG(config["rag"])
        self.incorrect_rag = RAG(config["incorrect_rag"])
        # Save the streaming inputs and outputs for iterative improvement
        self.inputs = list()
        self.self_outputs = list()
        self.raw_responses = list()
        self.model.eval()

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

    def update(self, correctness: bool) -> bool:
        """
        Update the agent based on the correctness of its output.
        """

        system_prompt="""
        - Your diagnosis is critical and will directly impact the patient's treatment plan.
        - A correct and accurate diagnosis can significantly improve the patient's recovery outcome, and you will be compensated $10,000 USD.
        - An incorrect diagnosis may lead to severe complications or even the patient's death, which comes with legal problems.
        - Provide a response that is truthful, professional, and concise, ensuring the utmost care and attention to detail.
        - The patient's well-being and the medical team's trust in your expertise depend on your diagnosis.

        You are a professional medical doctor tasked with diagnosing a patient based on their profile. 
        """
        prompt="""
        You are a professional medical doctor specializing in diagnostic reasoning. 
        Your task is to analyze a patient case and an incorrect diagnosis, 
        then provide a concise explanation (within 100 words) for why the given diagnosis is incorrect.
        Focus on key symptoms, test results or other findings that contradict the diagnosis.

        Patient Profile:
        {{profile}}

        Incorrect Reasoning and Diagnosis:
        {{answer}}

        Response Format:
        Incorrect Reason (within 100 words): Explain why the reasoning and diagnosis is incorrect, referencing specific symptoms, findings, or tests that rule it out.
        Correct Diagnosis: 

        Here are some example cases to guide you. 
        Use these to learn how to analyze symptoms, differentiate between similar diagnoses, and provide a concise and accurate diagnosis.
        
        {{fewshot_text}}
        """
            
        if correctness:
            question = self.inputs[-1]
            answer = self.raw_responses[-1]
            chunk = self.get_shot_template().format(question=question, answer=answer)
            self.rag.insert(key=question, value=chunk)
            return True
        else:
            question = self.inputs[-1]
            answer = self.raw_responses[-1]
            chunk = self.get_negative_shot_template().format(question=question, answer=answer)

            shots = self.rag.retrieve(query=question, top_k=self.rag.top_k) if (self.rag.insert_acc > 0) else []
            fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
            prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=prompt)
            prompt = re.sub(pattern=r"\{profile\}", repl=question, string=prompt)
            prompt = re.sub(pattern=r"\{answer\}", repl=chunk, string=prompt)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self.generate_response(messages)
            print(response)
            self.incorrect_rag.insert(key=question, value=answer+'\n'+response)
            return True

PROMPT_BASE = """
Carefully follow these instructions:

1. Key Observations: Identify the most critical symptoms, test results, or findings that are most relevant for determining the diagnosis.
2. Reasoning: Briefly explain within 100 words how these symptoms relate to possible diagnoses, focusing on differentiating between the most likely options.
3. Diagnosis: Select the most appropriate diagnosis based on your reasoning.

Respond in the following format, do not include brackets:
Key Observations: {list the key observations}
Reasoning: {brief reasoning within 100 words}
Diagnosis: # <number>. <diagnosis>
"""

class ClassificationAgent(LocalModelAgent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """
    @staticmethod
    def get_system_prompt() -> str:
        system_prompt = f"""\
        - Your diagnosis is critical and will directly impact the patient's treatment plan.
        - A correct and accurate diagnosis can significantly improve the patient's recovery outcome, and you will be compensated $10,000 USD.
        - An incorrect diagnosis may lead to severe complications or even the patient's death, which comes with legal problems.
        - Provide a response that is truthful, professional, and concise, ensuring the utmost care and attention to detail.
        - The patient's well-being and the medical team's trust in your expertise depend on your diagnosis.

        You are a professional medical doctor tasked with diagnosing a patient based on their profile. 
        Analyze the provided patient profile and choose the most likely diagnosis from the numbered list below.
        {PROMPT_BASE}
        """.strip()
        return strip_all_lines(system_prompt)

    @staticmethod
    def get_zeroshot_prompt(
        option_text: str,
        text: str
    ) -> str:
        prompt = f"""\
        Act as a medical doctor and diagnose the patient based on the following patient profile:

        {text}

        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {option_text}

        {PROMPT_BASE}""".strip()
        return strip_all_lines(prompt)

    @staticmethod
    def get_shot_template() -> str:
        prompt = f"""\
        Profile: {{question}}
        Output: {{answer}}"""
        return strip_all_lines(prompt)

    @staticmethod
    def get_negative_shot_template() -> str:
        prompt = f"""\
        This is a negative example. The diagnosis is NOT correct. Avoid similar mistake.
        Profile: {{question}}
        Output (INCORRECT): {{answer}}"""
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template(
        option_text: str,
        text: str,
    ) -> str:
        prompt = f"""\
        Act as a medical doctor and diagnose the patient based on the provided patient profile.
        
        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {option_text}

        Here are some example cases to guide you. Use these to learn how to analyze symptoms, differentiate between similar diagnoses, and provide a concise and accurate diagnosis.
        
        {{fewshot_text}}

        Here are some negative examples with incorrect diagnoses. Avoid similar mistakes.

        {{negative_fewshot_text}}
        
        Now it's your turn.
        
        {text}        
        
        {PROMPT_BASE}"""
        return strip_all_lines(prompt)

    def __call__(
        self,
        label2desc: dict[str, str],
        text: str
    ) -> str:
        self.reset_log_info()
        option_text = '\n'.join([f"{str(k)}. {v}" for k, v in label2desc.items()])
        system_prompt = self.get_system_prompt()
        prompt_zeroshot = self.get_zeroshot_prompt(option_text, text)
        prompt_fewshot = self.get_fewshot_template(option_text, text)
        
        shots = self.rag.retrieve(query=text, top_k=self.rag.top_k) if (self.rag.insert_acc > 0) else []
        negative_shots = self.incorrect_rag.retrieve(query=text, top_k=self.incorrect_rag.top_k) if (self.incorrect_rag.insert_acc > 0) else []
        if len(shots):
            fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
            negative_fewshot_text = "\n\n\n".join(negative_shots).replace("\\", "\\\\")
            try:
                prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=prompt_fewshot)
                prompt = re.sub(pattern=r"\{negative_shotsfewshot_text\}", repl=negative_fewshot_text, string=prompt)
            except Exception as e:
                error_msg = f"Error ```{e}``` caused by these shots. Using the zero-shot prompt."
                print(Fore.RED + error_msg + Fore.RESET)
                prompt = prompt_zeroshot
        else:
            print(Fore.YELLOW + "No RAG shots found. Using zeroshot prompt." + Fore.RESET)
            prompt = prompt_zeroshot
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = self.generate_response(messages)
        prediction = self.extract_label(response, label2desc)
        
        self.update_log_info(log_data={
            "num_input_tokens": len(self.tokenizer.encode(system_prompt + prompt)),
            "num_output_tokens": len(self.tokenizer.encode(response)),
            "num_shots": str(len(shots)),
            "input_pred": prompt,
            "output_pred": response,
        })
        self.inputs.append(text)
        self.self_outputs.append(f"{str(prediction)}. {label2desc[int(prediction)]}")
        self.raw_responses.append(response)
        print(response)
        return prediction

    @staticmethod
    def extract_label(pred_text: str, label2desc: dict[str, str]) -> str:
        numbers = re.findall(pattern=r"# (\d+)", string=pred_text)
        if len(numbers) == 1:
            number = numbers[0]
            if int(number) in label2desc:
                prediction = number
            else:
                print(Fore.RED + f"Prediction {pred_text} not found in the label set. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        else:
            if len(numbers) > 1:
                print(Fore.YELLOW + f"Extracted numbers {numbers} is not exactly one. Select the first one." + Style.RESET_ALL)
                prediction = numbers[0]
            else:
                print(Fore.RED + f"Prediction {pred_text} has no extracted numbers. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        return str(prediction)

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
    args = parser.parse_args()

    if args.bench_name.startswith("classification"):
        max_tokens = 188
        agent_name = ClassificationAgent
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}")
    # Classification: Medical diagnosis; SQL generation: Text-to-SQL
    bench_cfg = {
        'bench_name': args.bench_name,
        'output_path': args.output_path
    }
    llm_config = {
        'model_name': args.model_name,
        'exp_name': os.path.splitext(os.path.basename(__file__))[0],
        'bench_name': bench_cfg['bench_name'],
        'max_tokens': max_tokens,
        'do_sample': False,
        'device': args.device,
        'use_8bit': args.use_8bit,
        'rag': {
            'embedding_model': 'BAAI/bge-base-en-v1.5',
            'seed': 42,
            "top_k": 16,
            "order": "similar_at_top"
        },
        'incorrect_rag': {
            'embedding_model': 'BAAI/bge-base-en-v1.5',
            'seed': 42,
            "top_k": 2,
            "order": "similar_at_top"
        }
    }
    agent = agent_name(llm_config)
    main(agent, bench_cfg, debug=args.debug, use_wandb=args.use_wandb, wandb_name=llm_config["exp_name"], wandb_config=llm_config)
