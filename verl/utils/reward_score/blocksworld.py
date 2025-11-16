import os
import re

def extract_solution(solution_str):
    """
    Extracts the plan from a solution string.
    The plan is expected to start with '(plan' and be a balanced S-expression.
    """
    # Use regex to find the start of '(plan', ignoring case and allowing whitespace
    match = re.search(r'\(plan\b', solution_str, re.IGNORECASE)
    if not match:
        return ""

    start_index = match.start()
    
    # Find the matching closing parenthesis for the plan
    open_parens = 0
    for i in range(start_index, len(solution_str)):
        if solution_str[i] == '(':
            open_parens += 1
        elif solution_str[i] == ')':
            open_parens -= 1
            if open_parens == 0:
                # Found the end of the plan
                return solution_str[start_index:i+1]
    
    # If we get here, the parentheses are not balanced
    return ""

def extract_answer_from_solution(solution_str: str) -> str:
    """
    Extracts the final answer from the <answer></answer> tag of solution_str.

    Args:
        solution_str: The solution string containing the answer.

    Returns:
        The extracted answer as a string, or an empty string if not found.
    """
    try:
        return solution_str.split("<answer>")[1].split("</answer>")[0].strip()
    except IndexError:
        return ""

def validate_plan(domain, instance, plan_file):
    val_path = os.getenv("VAL")
    cmd = f"{val_path}/validate {domain} {instance} {plan_file}"
    response = os.popen(cmd).read()
    if 'Problem in domain' in response:
        raise Exception('Problem in domain: Check PDDL Writer')
    return True if "Plan valid" in response else False

def compute_score(data_source, solution_str, ground_truth, extra_info):
    """
    Compute the score for the given data source, solution string, ground truth, and extra info.
    Should return a float between 0 and 1.
    """
    # solution_str = solution_str.strip()
    final_answer = extract_answer_from_solution(solution_str)
    if final_answer == "":
        return {"score": 0.0}

    #Check if the solution string is a valid plan
    with open("temp_domain.pddl", "w") as f:
        f.write(extra_info["domain"])
    with open("temp_problem.pddl", "w") as f:
        f.write(extra_info["problem"])
    with open("temp_plan", "w") as f:
        f.write(final_answer) 
    is_valid = validate_plan("temp_domain.pddl", "temp_problem.pddl", "temp_plan")
    # os.remove("temp_domain.pddl")
    # os.remove("temp_problem.pddl")
    # os.remove("temp_plan")
    score_dict = {
        "score": 1.0 if is_valid else 0.1,
    }
    return score_dict
