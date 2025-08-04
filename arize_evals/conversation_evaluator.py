"""
pip install -q "arize-phoenix>=4.29.0"
pip install -q openai>=0.26.4 tiktoken
pip install -q nest_asyncio 'httpx<0.28'
"""
import json
from datetime import datetime
from typing import Any, Dict

import nest_asyncio
import pandas as pd
from phoenix.evals import (
    llm_classify,
)

from sm_llm.agents.phoenix_utility import get_aps_azure_openai


# Apply nest_asyncio to handle async operations in Jupyter
nest_asyncio.apply()

# Custom prompt templates
RESPONSE_QUALITY_TEMPLATE = """
You are an evaluation assistant analyzing the quality of responses in a conversation.
Here is the exchange:

[BEGIN DATA]
[Conversation]: {conversation}
[END DATA]

Evaluate if the assistant's response appropriately answers the user's message.
Consider:
1. Relevance to user's message
2. Clarity and coherence
3. Helpfulness
4. Professionalism

Your response must be a single word, either "good" or "bad",
and should not contain any text or characters aside from that word.
"good" means the response is high quality and meets all the criteria above.
"bad" means the response is low quality or fails to meet one or more criteria.
"""

TASK_COMPLETION_TEMPLATE = """
You are an evaluation assistant analyzing task completion in a conversation.
Here is the exchange:

[BEGIN DATA]
[Conversation]: {conversation}
[END DATA]

Evaluate if the assistant successfully completed or addressed the user's task/request.
Consider:
1. Task understanding
2. Completion status
3. Appropriate follow-up

Your response must be a single word, either "complete" or "incomplete",
and should not contain any text or characters aside from that word.
"complete" means the task was successfully addressed or completed.
"incomplete" means the task was not fully addressed or completed.
"""

FORMAT_ADHERENCE_TEMPLATE = """
You are an evaluation assistant analyzing response format and user satisfaction.
Here is the exchange:

[BEGIN DATA]
[Conversation]: {conversation}
[END DATA]

Evaluate if the response format is appropriate and professional.
Consider:
1. Clear and professional tone
2. Proper formatting of information
3. Consistent style
4. User-friendly presentation

Your response must be a single word, either "proper" or "improper",
and should not contain any text or characters aside from that word.
"proper" means the format is appropriate and professional.
"improper" means the format needs improvement.
"""
MODEL = "gpt-4o" # "gpt-4-turbo-preview"
# import os
# from openai import OpenAI
# print(os.environ['OPENAI_API_KEY'])
# client = OpenAI()
# client.api_key = os.environ['OPENAI_API_KEY']
# completion = client.completions.create(model=MODEL, prompt="Just say hello")
# print(completion.choices[0].text)
# print(dict(completion).get('usage'))
# print(completion.model_dump_json(indent=2))

import os


# Add Phoenix API Key for tracing
PHOENIX_API_KEY = "8a718e4e89f2f0cf3fb:8ae5d39"
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

class ConversationEvaluator:
    def __init__(self, model_name=MODEL, max_tokens=10000, timeout=1200):
        # openai_model = OpenAIModel(model=model_name, max_tokens=max_tokens, timeout=timeout)
        
        # Using our Azure OpenAI
        openai_model = get_aps_azure_openai(MODEL, max_tokens, timeout)
        self.model = openai_model 
        
    def split_conversation(self, conversation):
        """Split conversation into turns and roles."""
        turns = []
        for line in conversation.strip().split('\n'):
            if line.startswith('user: '):
                role = 'user'
                content = line[6:].strip()
            elif line.startswith('assistant: '):
                role = 'assistant'
                content = line[11:].strip()
            else:
                continue
            turns.append({'role': role, 'content': content})
        return turns

    def evaluate_response_quality(self, conversation_turn: Dict[str, Any]):
        """Evaluate the quality of assistant's response."""
        print("Evaluating response quality...")
        df = pd.DataFrame([conversation_turn])
        result = llm_classify(
            data=df,
            template=RESPONSE_QUALITY_TEMPLATE,
            model=self.model,
            rails=["good", "bad"]
        )
        label = result['label'].iloc[0]
        return {'score': 1.0 if label == 'good' else 0.0, 'label': label}

    def evaluate_task_completion(self, conversation_turn: Dict[str, Any]):
        """Evaluate if the assistant completed the user's task/request."""
        print("Evaluating task completion...")
        df = pd.DataFrame([conversation_turn])
        result = llm_classify(
            data=df,
            template=TASK_COMPLETION_TEMPLATE,
            model=self.model,
            rails=["complete", "incomplete"]
        )
        label = result['label'].iloc[0]
        return {'score': 1.0 if label == 'complete' else 0.0, 'label': label}

    def evaluate_format_adherence(self, conversation_turn: Dict[str, Any]):
        """Evaluate if the assistant followed the expected format."""
        print("Evaluating format adherence...")
        df = pd.DataFrame([conversation_turn])
        result = llm_classify(
            data=df,
            template=FORMAT_ADHERENCE_TEMPLATE,
            model=self.model,
            rails=["proper", "improper"]
        )
        label = result['label'].iloc[0]
        return {'score': 1.0 if label == 'proper' else 0.0, 'label': label}
    
    def evaluate_conversation(self, conversation):
        """Evaluate an entire conversation."""
        evals = {}
        evals['response_quality'] = self.evaluate_response_quality({'conversation': conversation})
        evals['task_completion'] = self.evaluate_task_completion({'conversation': conversation})
        evals['format_adherence'] = self.evaluate_format_adherence({'conversation': conversation})
        return evals
    
    def evaluate_messages(self, conversation):
        """Evaluate each message in a conversation."""
        turns = self.split_conversation(conversation)
        results = []
        
        for i in range(0, len(turns)-1, 2):
            if i+1 < len(turns):
                turn_result = {
                    'user_message': turns[i]['content'],
                    'assistant_response': turns[i+1]['content'],
                    'turn_number': i//2 + 1
                }
                
                # Evaluate different aspects
                turn_result['response_quality'] = self.evaluate_response_quality({
                    'user_message': turns[i]['content'],
                    'assistant_response': turns[i+1]['content']
                })

                turn_result['task_completion'] = self.evaluate_task_completion({
                    'user_message': turns[i]['content'],
                    'assistant_response': turns[i+1]['content']
                })
                
                turn_result['format_adherence'] = self.evaluate_format_adherence({
                    'user_message': turns[i]['content'],
                    'assistant_response': turns[i+1]['content']
                })
                results.append(turn_result)
        
        return results

    def evaluate_conversations_from_csv(self, csv_path):
        """Evaluate all conversations from a CSV file."""
        df = pd.read_csv(csv_path)
        all_results = []
        # Let's only evaluate the first 10 conversations for now
        for idx, row in df.iterrows():
            print(f"Evaluating conversation {idx + 1}/{len(df)}...")
            # print(row['conversation'])
            conversation_results = {
                'session_id': row['session_id'],
                'evaluation_time': datetime.now().isoformat(),
                'conversation': row['conversation'],
                'evals': self.evaluate_conversation(row['conversation'])
            }
            all_results.append(conversation_results)
            
        # Save intermediate results
        self.save_results(all_results, f'arize_evals/results/evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        return all_results

    def save_results(self, results, filename):
        """Save evaluation results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    def display_results_summary(self, results):
        """Display interactive summary of evaluation results."""
        total_conversations = len(results)
        total_turns = sum(len(r['evals']) for r in results)
        
        print("Evaluation Summary:")
        print(f"Total conversations evaluated: {total_conversations}")
        print(f"Total conversation turns evaluated: {total_turns}")
        
        # Calculate average scores
        quality_scores = []
        completion_scores = []
        format_scores = []
        
        for item in results:
            quality_scores.append(item['evals']['response_quality']['score'])
            completion_scores.append(item['evals']['task_completion']['score'])
            format_scores.append(item['evals']['format_adherence']['score'])
        
        print("\nAverage Scores:")
        print(f"Response Quality: {sum(quality_scores)/len(quality_scores):.2f}")
        print(f"Task Completion: {sum(completion_scores)/len(completion_scores):.2f}")
        print(f"Format Adherence: {sum(format_scores)/len(format_scores):.2f}")

if __name__ == "__main__":
    # Example usage
    evaluator = ConversationEvaluator()
    csv_path = "arize_evals/data/Production Onboarding Conversations - Anonymized 2_4+ (includes V2).csv"
    results = evaluator.evaluate_conversations_from_csv(csv_path)
    evaluator.display_results_summary(results)
