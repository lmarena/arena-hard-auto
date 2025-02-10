# Tag structure
# - category_tag
#     - criteria_v0.1
#         - specificity
#         - ...
#     - math_v0.1
#         - math
#     - if_v0.1
#         - if
#         - score
import ast
import re


class Category:
    def __init__(self):
        pass

    @staticmethod
    def create_category(name):
        if name == "criteria_v0.1":
            return CategoryHardPrompt()
        raise Exception(f"Category name is incorrect: {name}")

    def post_process(self):
        pass


class CategoryHardPrompt(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "criteria_v0.1"
        self.pattern = re.compile(r"(\[[1234567](?:\,\s[1234567])*\])")
        self.sys_prompt = \
"""
Your task is to evaluate how well the following input prompts can assess the capabilities of advanced AI assistants.
For the input prompt, please analyze it based on the following 7 criteria giving a 1 to 5 mark for each.
1. Specificity
    - Characteristics: whether the question requires a specific output, such as code, a mathematical solution, a logical simplification, a problem-solving strategy, or a hardware setup recommendation. 
    - 1 grade means that the question does not require any specific structure.
    - 5 grade means that the question requires a complex and specific structure such as mentioned.
2. Domain Knowledge
    - Characteristics: whether the question covers a specific domain, such as programming, mathematics, logic, problem-solving, or hardware setup. 
    - 1 grade means that the answer does not require any deep knowledge of a specific domain, common knowledge is enough.
    - 5 grade means that the question touches a range of topics or/and different domains.
3. Complexity
    - Characteristics: whether the question requires a complex, multi-step solution.
    - 1 grade means that the answer does not require any thought process or uncommon knowledge.
    - 5 grade means that the question requires a complex multi step thought.
4. Problem-Solving Skills
    - Characteristics: whether the answer should demonstrate active problem-solving skills.
    - 1 grade means that the answer does not require any thought process and is regurgitating an existing fact. 
    - 5 grade means that the question requires a complex multi step thought.
5. Creativity
    - Characteristics: assesses whether the response requires to think up a creative novel approach.
    - 1 grade means that the answer requires a straightforward or factual response with no creativity.
    - 5 grade means that the question invites a highly creative or novel approach, requiring the generation of unique ideas or solutions.
6. Technical Accuracy
    - Characteristics: assesses the levels of technical knowledge and accuracy required for technical fields.
    - 1 grade means that the answer can be very general or imprecise, with no need for specific technical accuracy.
    - 5 grade means that the response must be meticulously accurate, reflecting deep technical expertise and attention to detail.
7. Real-world Application
    - Characteristics: how much thr prompt relates to real-world applications, such as setting up a functional system or writing code for practical use.
    - 1 grade means that the question pertains to theoretical knowledge or information with no direct application.
    - 5 grade means that the question demands an actionable solution applicable to real-life situations, requiring practical implementation guidance.

You must respond with a valid json dictionary containing aspect names as values and corresponding numbers as keys such this:
{
    "specificity": int,
    "domain_knowledge": int,
    "complexity": int,
    "problem_solving": int,
    "creativity": int,
    "technical_accuracy": int,
    "real_world": int
}

Do not add anything additional to your answer.
### Input prompt
{{prompt}}
### Json output
"""
        self.tags = {
            1: "specificity",
            2: "domain_knowledge",
            3: "complexity",
            4: "problem_solving",
            5: "creativity",
            6: "technical_accuracy",
            7: "real_world",
        }

    def get_score(self, judgment):
        matches = self.pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return ['No Match']
        elif len(set(matches)) == 1:
            try:
                return ast.literal_eval(matches[0])
            except SyntaxError:
                print(matches[0])
                return ['Syntax Error']
        else:
            return ['Multiple Match']

    def pre_process(self, prompt):
        conv = [{"role": "system", "content": self.sys_prompt}]
        conv.append({"role": "user", "content": prompt})
        return conv

    def post_process(self, judgment):
        criteria = self.get_score(judgment=judgment)
        return {name: bool(i in criteria) for i, name in self.tags.items()}