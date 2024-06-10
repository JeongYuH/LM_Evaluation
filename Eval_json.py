import os
import json
import datetime
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class Evaluation_json:
    def __init__(self, file_path1, file_path2, question_index=0):
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.question_index = question_index
        self.system_prompt = '''
        Please act as an impartial judge and evaluate the quality of the responses provided by two
        AI assistants to the user question displayed below. You should choose the assistant that
        follows the user’s instructions and answers the user’s question better. Your evaluation
        should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
        and level of detail of their responses. Begin your evaluation by comparing the two
        responses and provide a short explanation. Avoid any position biases and ensure that the
        order in which the responses were presented does not influence your decision. Do not allow
        the length of the responses to influence your evaluation. Do not favor certain names of
        the assistants. Be as objective as possible. After providing your explanation, output your
        final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
        if assistant B is better, and "[[C]]" for a tie. And you should print output using 'json' format.

        [Example]
        {
          "evaluation": {
            "question": {question},
            "answer_A": {
              "model_name":{model_a}
              "output": {model_a_answer},
              "helpfulness": 4,
              "relevance": 4,
              "accuracy": 4,
              "depth": 4,
              "creativity": 4
            },
            "answer_B": {
              "model_name":{model_b}
              "output": {model_b_answer},
              "helpfulness": 4,
              "relevance": 4,
              "accuracy": 4,
              "depth": 4,
              "creativity": 3
            }
          },
          "final_verdict": "[[A]]"
        }
        '''
        self.questions, self.model1, self.answers1, self.model2, self.answers2 = self.load_answers()
        self.question = self.questions[self.question_index]
        self.answer1 = self.answers1[self.question_index]
        self.answer2 = self.answers2[self.question_index]
        self.api_key = self.load_api_key("OPENAI_API_KEY")
        self.llm = self.call_llm()
        self.prompt = self.prompt_templates()

    def load_api_key(self, env_var_name):
        """Load the API key from the environment variables."""
        load_dotenv()
        api_key = os.getenv(env_var_name)
        if not api_key:
            raise ValueError(f"API key is not set for {env_var_name}. Please check your .env file.")
        return api_key

    def call_llm(self, gpt_model='gpt-4o'):
        """Call llm settings(Default=gpt-4o)"""
        llm = ChatOpenAI(openai_api_key=self.api_key, model=gpt_model, model_kwargs={"seed": 42})
        return llm

    def prompt_templates(self):
        """Define prompt template for Prompt evaluation"""
        prompt = PromptTemplate(
            input_variables=['system_prompt', 'question', 'model_a', 'model_a_answer', 'model_b', 'model_b_answer'],
            template="""
            [system]
            {system_prompt}

            [User Question]
            {question}

            [The Start of Assistant A’s Answer]
            {model_a}
            {model_a_answer}
            [The End of Assistant A’s Answer]

            [The Start of Assistant B’s Answer]
            {model_b}
            {model_b_answer}
            [The End of Assistant B’s Answer]
            """
        )
        return prompt

    def load_answers(self):
        """Call Q&A files(Json format) and extract elements(model name, questions, answers of each models)"""
        with open(self.file_path1, 'r', encoding='utf-8') as file1:
            questions = []
            answer1 = []
            data1 = json.load(file1)
            model1, qa1 = data1["model_name"], data1["output"]
            for key, value in qa1.items():
                questions.append(value['question'])
                answer1.append(value['answer'])

        with open(self.file_path2, 'r', encoding='utf-8') as file2:
            answer2 = []
            data2 = json.load(file2)
            model2, qa2 = data2["model_name"], data2["output"]
            for key, value in qa2.items():
                answer2.append(value['answer'])

        return questions, model1, answer1, model2, answer2

    def answer_mixer(self, answer1, answer2):
        """Shake answers using randint method"""
        mixer = np.random.randint(0, high=2)
        if mixer == 1:
            answer_a, answer_b = answer1, answer2
            model_a, model_b = self.model1, self.model2
        else:
            answer_a, answer_b = answer2, answer1
            model_a, model_b = self.model2, self.model1

        return model_a, model_b, answer_a, answer_b

    def invoke_evaluation(self, model_a, model_b, answer_a, answer_b):
        """Call LLM's evaluation."""
        chain = self.prompt | self.llm | JsonOutputParser()
        response = chain.invoke({
            "system_prompt": self.system_prompt,
            "question": self.question,
            "model_a": model_a,
            "model_b": model_b,
            "model_a_answer": answer_a,
            "model_b_answer": answer_b
        })
        return response

    def response_json(self, response):
        """Convert response to json format"""
        eval_data = {
            "evaluation": response['evaluation'],
            "final_verdict": response['final_verdict']
        }
        return eval_data

    def check_file(self, filename):
        """Check evaluation log file(json format) is have, call that file, or made it."""
        if os.path.isfile(filename):
            print("update data...")
            return True
        else:
            print("saving data...")
            return False

    def add_log(self, filename, eval_data):
        with open(filename, 'r', encoding="utf-8") as f:
            data = json.load(f)

        data['eval_log'].append(eval_data)
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent="\t", ensure_ascii=False)

    def make_file(self, filename, eval_data):
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump({"eval_log": [eval_data]}, f, indent='\t', ensure_ascii=False)

    def run(self, filepath=''):
        '''Save result in filepath parameter. If you don't add file path, it will saved as time you ordered this function.
        
        If you already have same file(when you set file names), It would be added it.'''
        
        model_a, model_b, answer_a, answer_b = self.answer_mixer(self.answer1, self.answer2)
        response = self.invoke_evaluation(model_a, model_b, answer_a, answer_b)
        eval_data = self.response_json(response)
        # print(model_a, model_b)
        if filepath == '':
            filepath = 'Evaluation\\Result\\' + datetime.datetime.now().strftime('%y%m%d_%H%M%S') + '.json'
        

        if self.check_file(filepath):
            self.add_log(filename=filepath, eval_data=eval_data)
        else:
            self.make_file(filename=filepath, eval_data=eval_data)