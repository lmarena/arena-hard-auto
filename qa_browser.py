import os
import json
import pandas as pd
import glob
import gradio as gr

# Cache for loaded data
data_cache = {}

# Load data functions with caching
def load_jsonl(file_path):
    """Load a JSONL file into a pandas DataFrame with caching."""
    if file_path in data_cache:
        return data_cache[file_path]
    
    if not os.path.exists(file_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_json(file_path, lines=True)
        data_cache[file_path] = df
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def get_available_benchmarks():
    """Get list of available benchmarks in data directory."""
    return [dir_name for dir_name in os.listdir("data") 
            if os.path.isdir(os.path.join("data", dir_name))]

def get_categories(benchmark):
    """Get list of categories for a given benchmark."""
    questions = load_jsonl(f"data/{benchmark}/question.jsonl")
    if questions.empty:
        return []
    return sorted(questions['category'].unique().tolist())

def get_languages(benchmark):
    """Get list of languages available in the benchmark."""
    questions = load_jsonl(f"data/{benchmark}/question.jsonl")
    if questions.empty or 'language' not in questions.columns:
        return ["English"]  # Default if no language column
    
    return sorted(questions['language'].unique().tolist())

def get_judges(benchmark):
    """Get list of available judges for a benchmark."""
    judgment_dir = f"data/{benchmark}/model_judgment"
    if not os.path.exists(judgment_dir):
        return []
    return [dir_name for dir_name in os.listdir(judgment_dir)
            if os.path.isdir(os.path.join(judgment_dir, dir_name))]

def get_models(benchmark, judge):
    """Get list of models that have judgments by the specified judge."""
    if not judge:
        return []
    
    judgment_dir = f"data/{benchmark}/model_judgment/{judge}"
    if not os.path.exists(judgment_dir):
        return []
    
    return [os.path.splitext(os.path.basename(file))[0] 
            for file in glob.glob(f"{judgment_dir}/*.jsonl")]

def get_questions(benchmark, category=None, language=None):
    """Get questions with category and language filters if provided."""
    questions = load_jsonl(f"data/{benchmark}/question.jsonl")
    if questions.empty:
        return []
    
    # Apply category filter if provided
    if category and category != "All":
        questions = questions[questions['category'] == category]
    
    # Apply language filter if provided and column exists
    if language and language != "All" and 'language' in questions.columns:
        questions = questions[questions['language'] == language]
    
    # Create list of question previews with their UIDs
    question_previews = [(row['uid'], row['prompt'][:100] + "..." if len(row['prompt']) > 100 else row['prompt']) 
                        for _, row in questions.iterrows()]
    
    return question_previews

def get_model_answer(benchmark, model, uid):
    """Get a model's answer for a specific question."""
    model_answers = load_jsonl(f"data/{benchmark}/model_answer/{model}.jsonl")
    if model_answers.empty:
        return "No answer found"
    
    answer = model_answers[model_answers['uid'] == uid]
    if answer.empty:
        return "No answer found"
    
    # Extract the actual answer from the messages
    try:
        messages = answer.iloc[0]['messages']
        if len(messages) < 2:
            return "No answer found"
        
        # The assistant's message should be the second one
        assistant_msg = messages[1]
        if 'role' in assistant_msg and assistant_msg['role'] == 'assistant':
            content = assistant_msg['content']
            
            # Handle different content formats
            if isinstance(content, dict) and 'answer' in content:
                return content['answer']
            elif isinstance(content, str):
                return content
            else:
                return str(content)
        else:
            return "Invalid message format"
    except Exception as e:
        return f"Error extracting answer: {str(e)}"

def get_judgment(benchmark, judge, model, uid):
    """Get judgment for a specific model and question."""
    judgments = load_jsonl(f"data/{benchmark}/model_judgment/{judge}/{model}.jsonl")
    if judgments.empty:
        return None, None
    
    judgment = judgments[judgments['uid'] == uid]
    if judgment.empty:
        return None, None
    
    games = judgment.iloc[0]['games']
    if len(games) < 2:
        return games[0] if games else None, None
    
    return games[0], games[1]  # First game, second game

def format_judgment(game):
    """Format judgment for display."""
    if not game:
        return "No judgment available"
    
    score = game.get('score', 'No score')
    
    # Try to get judgment text
    judgment = game.get('judgment', {})
    if isinstance(judgment, dict) and 'answer' in judgment:
        judgment_text = judgment['answer']
    else:
        judgment_text = str(judgment)
    
    return f"### Score: {score}\n\n{judgment_text}"

# Gradio interface functions
def update_categories(benchmark):
    """Update category dropdown based on selected benchmark."""
    categories = ["All"] + get_categories(benchmark)
    return gr.Dropdown(choices=categories, value="All")

def update_languages(benchmark):
    """Update language dropdown based on selected benchmark."""
    languages = ["All"] + get_languages(benchmark)
    default = "English" if "English" in languages else languages[0]
    return gr.Dropdown(choices=languages, value=default)

def update_judges(benchmark):
    """Update judge dropdown based on selected benchmark."""
    judges = get_judges(benchmark)
    default = judges[0] if judges else None
    return gr.Dropdown(choices=judges, value=default)

def update_models(benchmark, judge):
    """Update model dropdown based on selected benchmark and judge."""
    models = get_models(benchmark, judge)
    default = models[0] if models else None
    return gr.Dropdown(choices=models, value=default)

def update_questions(benchmark, category, language):
    """Update question dropdown based on selected benchmark, category and language."""
    question_list = get_questions(benchmark, category, language)
    if not question_list:
        return gr.Dropdown(choices=[], value=None), {}
    
    # Create a dictionary mapping previews to UIDs to ensure we can look up UIDs from previews
    question_dict = {q[1]: q[0] for q in question_list}
    question_options = list(question_dict.keys())
    
    default = question_options[0] if question_options else None
    return gr.Dropdown(choices=question_options, value=default), question_dict

def display_content(benchmark, category, language, judge, model, question, question_dict):
    """Display the question, answers, and judgments."""
    if not question or not question_dict or question not in question_dict:
        return "No question selected", "No baseline answer", "No model answer", "No judgment", "No judgment"
    
    uid = question_dict[question]
    
    # Load the question text
    questions_df = load_jsonl(f"data/{benchmark}/question.jsonl")
    question_row = questions_df[questions_df['uid'] == uid]
    if question_row.empty:
        return "Question not found", "No baseline answer", "No model answer", "No judgment", "No judgment"
    
    question_text = question_row.iloc[0]['prompt']
    
    # Load judgments and identify baseline model
    judgments = load_jsonl(f"data/{benchmark}/model_judgment/{judge}/{model}.jsonl")
    judgment_row = judgments[judgments['uid'] == uid]
    
    if judgment_row.empty:
        return question_text, "No baseline answer", "No model answer", "No judgment", "No judgment"
    
    baseline_model = judgment_row.iloc[0]['baseline']
    
    # Get answers
    baseline_answer = get_model_answer(benchmark, baseline_model, uid)
    model_answer = get_model_answer(benchmark, model, uid)
    
    # Get judgments
    game1, game2 = get_judgment(benchmark, judge, model, uid)
    
    judgment1 = format_judgment(game1)
    judgment2 = format_judgment(game2)
    
    return question_text, baseline_answer, model_answer, judgment1, judgment2

# Initialize app components based on selected benchmark
def init_app(benchmark):
    categories = ["All"] + get_categories(benchmark)
    default_category = "All"
    
    languages = ["All"] + get_languages(benchmark)
    default_language = "English" if "English" in languages else languages[0]
    
    judges = get_judges(benchmark)
    default_judge = judges[0] if judges else None
    
    models = get_models(benchmark, default_judge) if default_judge else []
    default_model = models[0] if models else None
    
    question_list = get_questions(benchmark, default_category, default_language)
    question_dict = {q[1]: q[0] for q in question_list}
    question_options = list(question_dict.keys())
    default_question = question_options[0] if question_options else None
    
    # Get initial display content
    if default_question and default_model and default_judge:
        question_text, baseline_ans, model_ans, judgment1, judgment2 = display_content(
            benchmark, default_category, default_language, default_judge, default_model, default_question, question_dict
        )
    else:
        question_text = "No question available"
        baseline_ans = "No baseline answer"
        model_ans = "No model answer"
        judgment1 = "No judgment"
        judgment2 = "No judgment"
    
    return (
        gr.Dropdown(choices=categories, value=default_category),
        gr.Dropdown(choices=languages, value=default_language),
        gr.Dropdown(choices=judges, value=default_judge),
        gr.Dropdown(choices=models, value=default_model),
        gr.Dropdown(choices=question_options, value=default_question),
        question_dict,
        question_text,
        baseline_ans, model_ans,
        judgment1, judgment2
    )

# Function to go to the next question
def next_question(benchmark, category, language, current_question, question_dict):
    question_list = get_questions(benchmark, category, language)
    previews = [q[1] for q in question_list]
    
    if current_question not in previews:
        return gr.Dropdown(value=previews[0] if previews else None)
            
    current_idx = previews.index(current_question)
    next_idx = (current_idx + 1) % len(previews)
    return gr.Dropdown(value=previews[next_idx])

# Create Gradio app
def create_app():
    benchmarks = get_available_benchmarks()
    default_benchmark = "arena-hard-v2.0" if "arena-hard-v2.0" in benchmarks else benchmarks[0]
    
    # Initialize data for the default benchmark
    init_data = init_app(default_benchmark)
    
    with gr.Blocks() as app:
        gr.Markdown(
            '''# Arena-Hard-Auto Benchmark Viewer
            
            Arena-Hard-Auto is an automatic evaluation tool for instruction-tuned LLMs. It has the highest correlation and separability to LMArena (Chatbot Arena) among popular open-ended LLM benchmarks. If you are curious to see how well your model might perform on LMArena before deploying, we recommend trying Arena-Hard-Auto's newest evaluation set, **Arena-Hard-v2.0-Preview**.
            
            **Repo:** https://github.com/lmarena/arena-hard-auto
            
            **Paper:** https://arxiv.org/abs/2406.11939
            '''
        )
        
        with gr.Row():
            with gr.Column():
                benchmark_dropdown = gr.Dropdown(
                    choices=benchmarks,
                    value=default_benchmark,
                    label="Benchmark"
                )
                
                category_dropdown = gr.Dropdown(
                    choices=init_data[0].choices,
                    value=init_data[0].value,
                    label="Category"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=init_data[1].choices,
                    value=init_data[1].value,
                    label="Language"
                )
            
            with gr.Column():
                judge_dropdown = gr.Dropdown(
                    choices=init_data[2].choices,
                    value=init_data[2].value,
                    label="Judge Model"
                )
                
                model_dropdown = gr.Dropdown(
                    label="Model to Evaluate",
                    choices=init_data[3].choices,
                    value=init_data[3].value,
                )
        
        question_dict = gr.State(init_data[5])
        question_dropdown = gr.Dropdown(
            choices=init_data[4].choices,
            value=init_data[4].value,
            label="Select Question"
        )
        
        # Add a next question button
        next_button = gr.Button("Next Question")
        
        # Display the question
        gr.Markdown("---")
        question_display = gr.Markdown(value="### Question\n\n" + init_data[6])
        
        with gr.Tabs():
            with gr.TabItem("Game 1: Baseline (A) vs Model (B)"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Baseline (A)")
                        baseline_answer1 = gr.Markdown(value=init_data[7])
                    with gr.Column():
                        gr.Markdown("### Model (B)")
                        model_answer1 = gr.Markdown(value=init_data[8])
                gr.Markdown("---")
                gr.Markdown("### Judgment")
                judgment1 = gr.Markdown(value=init_data[9])
            
            with gr.TabItem("Game 2: Model (A) vs Baseline (B)"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model (A)")
                        model_answer2 = gr.Markdown(value=init_data[8])
                    with gr.Column():
                        gr.Markdown("### Baseline (B)")
                        baseline_answer2 = gr.Markdown(value=init_data[7])
                gr.Markdown("---")
                gr.Markdown("### Judgment")
                judgment2 = gr.Markdown(value=init_data[10])
                
        gr.Markdown("---")
        gr.Markdown("### Citation")
        gr.Markdown("If you find this tool useful, please cite the following papers:")
        gr.Markdown(
            '''```bibtex
@article{li2024crowdsourced,
  title={From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline},
  author={Li, Tianle and Chiang, Wei-Lin and Frick, Evan and Dunlap, Lisa and Wu, Tianhao and Zhu, Banghua and Gonzalez, Joseph E and Stoica, Ion},
  journal={arXiv preprint arXiv:2406.11939},
  year={2024}
}
@misc{arenahard2024,
    title = {From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline},
    url = {https://lmsys.org/blog/2024-04-19-arena-hard/},
    author = {Tianle Li*, Wei-Lin Chiang*, Evan Frick, Lisa Dunlap, Banghua Zhu, Joseph E. Gonzalez, Ion Stoica},
    month = {April},
    year = {2024}
}
```''')
        
        # Set up event handlers
        benchmark_dropdown.change(
            fn=init_app,
            inputs=benchmark_dropdown,
            outputs=[
                category_dropdown, language_dropdown, judge_dropdown, model_dropdown, 
                question_dropdown, question_dict,
                question_display, 
                baseline_answer1, model_answer1,
                judgment1, judgment2
            ]
        ).then(
            fn=lambda model, baseline: (model, baseline),
            inputs=[model_answer1, baseline_answer1],
            outputs=[model_answer2, baseline_answer2]
        )
        
        # Update questions when category changes
        category_dropdown.change(
            fn=update_questions,
            inputs=[benchmark_dropdown, category_dropdown, language_dropdown],
            outputs=[question_dropdown, question_dict]
        ).then(
            fn=display_content,
            inputs=[benchmark_dropdown, category_dropdown, language_dropdown, judge_dropdown, model_dropdown, question_dropdown, question_dict],
            outputs=[question_display, baseline_answer1, model_answer1, judgment1, judgment2]
        ).then(
            fn=lambda model, baseline: (model, baseline),
            inputs=[model_answer1, baseline_answer1],
            outputs=[model_answer2, baseline_answer2]
        )
        
        # Update questions when language changes
        language_dropdown.change(
            fn=update_questions,
            inputs=[benchmark_dropdown, category_dropdown, language_dropdown],
            outputs=[question_dropdown, question_dict]
        ).then(
            fn=display_content,
            inputs=[benchmark_dropdown, category_dropdown, language_dropdown, judge_dropdown, model_dropdown, question_dropdown, question_dict],
            outputs=[question_display, baseline_answer1, model_answer1, judgment1, judgment2]
        ).then(
            fn=lambda model, baseline: (model, baseline),
            inputs=[model_answer1, baseline_answer1],
            outputs=[model_answer2, baseline_answer2]
        )
        
        # Update models when judge changes
        judge_dropdown.change(
            fn=update_models,
            inputs=[benchmark_dropdown, judge_dropdown],
            outputs=model_dropdown
        ).then(
            fn=display_content,
            inputs=[benchmark_dropdown, category_dropdown, language_dropdown, judge_dropdown, model_dropdown, question_dropdown, question_dict],
            outputs=[question_display, baseline_answer1, model_answer1, judgment1, judgment2]
        ).then(
            fn=lambda model, baseline: (model, baseline),
            inputs=[model_answer1, baseline_answer1],
            outputs=[model_answer2, baseline_answer2]
        )
        
        # Display content when model changes
        model_dropdown.change(
            fn=display_content,
            inputs=[benchmark_dropdown, category_dropdown, language_dropdown, judge_dropdown, model_dropdown, question_dropdown, question_dict],
            outputs=[question_display, baseline_answer1, model_answer1, judgment1, judgment2]
        ).then(
            fn=lambda model, baseline: (model, baseline),
            inputs=[model_answer1, baseline_answer1],
            outputs=[model_answer2, baseline_answer2]
        )
        
        # Display content when question changes
        question_dropdown.change(
            fn=display_content,
            inputs=[benchmark_dropdown, category_dropdown, language_dropdown, judge_dropdown, model_dropdown, question_dropdown, question_dict],
            outputs=[question_display, baseline_answer1, model_answer1, judgment1, judgment2]
        ).then(
            fn=lambda model, baseline: (model, baseline),
            inputs=[model_answer1, baseline_answer1],
            outputs=[model_answer2, baseline_answer2]
        )
        
        # Handle next question button
        next_button.click(
            fn=next_question,
            inputs=[benchmark_dropdown, category_dropdown, language_dropdown, question_dropdown, question_dict],
            outputs=question_dropdown
        )
    
    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    app = create_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
    