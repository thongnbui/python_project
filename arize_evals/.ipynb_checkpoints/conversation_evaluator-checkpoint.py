import os
import pandas as pd
from phoenix.evals import (
    OpenAIModel,
    llm_classify,
    RESPONSE_QUALITY_PROMPT_RAILS_MAP,
    TASK_COMPLETION_PROMPT_RAILS_MAP,
    FORMAT_ADHERENCE_PROMPT_RAILS_MAP,
)
import nest_asyncio
from datetime import datetime
import json

# Apply nest_asyncio to handle async operations in Jupyter
nest_asyncio.apply()

class ConversationEvaluator:
    def __init__(self, model_name="gpt-4", max_tokens=1000):
        self.model = OpenAIModel(model=model_name, max_tokens=max_tokens)
        
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

    def evaluate_response_quality(self, conversation_turn):
        """Evaluate the quality of assistant's response."""
        prompt = f"""
        User message: {conversation_turn['user_message']}
        Assistant response: {conversation_turn['assistant_response']}
        
        Evaluate the response quality based on:
        1. Relevance to user's message
        2. Clarity and coherence
        3. Helpfulness
        4. Professionalism
        """
        
        result = llm_classify(
            prompt=prompt,
            prompt_rails_map=RESPONSE_QUALITY_PROMPT_RAILS_MAP,
            model=self.model
        )
        return result

    def evaluate_task_completion(self, conversation_turn):
        """Evaluate if the assistant completed the user's task/request."""
        prompt = f"""
        User message: {conversation_turn['user_message']}
        Assistant response: {conversation_turn['assistant_response']}
        
        Did the assistant successfully complete the user's task or address their request?
        Consider:
        1. Task understanding
        2. Completion status
        3. Appropriate follow-up
        """
        
        result = llm_classify(
            prompt=prompt,
            prompt_rails_map=TASK_COMPLETION_PROMPT_RAILS_MAP,
            model=self.model
        )
        return result

    def evaluate_format_adherence(self, conversation_turn):
        """Evaluate if the assistant followed the expected format."""
        prompt = f"""
        Assistant response: {conversation_turn['assistant_response']}
        
        Evaluate if the response follows the expected format:
        1. Clear prefix ('assistant: ')
        2. Proper formatting of options (if present)
        3. Consistent style
        """
        
        result = llm_classify(
            prompt=prompt,
            prompt_rails_map=FORMAT_ADHERENCE_PROMPT_RAILS_MAP,
            model=self.model
        )
        return result

    def evaluate_conversation(self, conversation):
        """Evaluate an entire conversation."""
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
        
        for idx, row in df.iterrows():
            conversation_results = {
                'session_id': row['session_id'],
                'evaluation_time': datetime.now().isoformat(),
                'results': self.evaluate_conversation(row['conversation'])
            }
            all_results.append(conversation_results)
            
            # Save intermediate results
            self.save_results(all_results, f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            
            # Display progress
            print(f"Evaluated conversation {idx + 1}/{len(df)}")
        
        return all_results

    def save_results(self, results, filename):
        """Save evaluation results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    def display_results_summary(self, results):
        """Display interactive summary of evaluation results."""
        total_conversations = len(results)
        total_turns = sum(len(r['results']) for r in results)
        
        print(f"Evaluation Summary:")
        print(f"Total conversations evaluated: {total_conversations}")
        print(f"Total conversation turns evaluated: {total_turns}")
        
        # Calculate average scores
        quality_scores = []
        completion_scores = []
        format_scores = []
        
        for conv in results:
            for turn in conv['results']:
                quality_scores.append(turn['response_quality']['score'])
                completion_scores.append(turn['task_completion']['score'])
                format_scores.append(turn['format_adherence']['score'])
        
        print("\nAverage Scores:")
        print(f"Response Quality: {sum(quality_scores)/len(quality_scores):.2f}")
        print(f"Task Completion: {sum(completion_scores)/len(completion_scores):.2f}")
        print(f"Format Adherence: {sum(format_scores)/len(format_scores):.2f}")

if __name__ == "__main__":
    # Example usage
    evaluator = ConversationEvaluator()
    csv_path = "data/Production Onboarding Conversations - Anonymized 2_4+ (includes V2).csv"
    results = evaluator.evaluate_conversations_from_csv(csv_path)
    evaluator.display_results_summary(results) 