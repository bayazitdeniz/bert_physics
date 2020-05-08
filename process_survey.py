import pandas as pd

# abstract dicts
q1_dict = {"Y is smaller than X": "option_a", "X is smaller than Y": "option_b"}
q2_dict = {"X is smaller than Y": "option_a", "Y is smaller than X": "option_b"}

q3_dict = {"X is heavier than Y": "option_a", "Y is heavier than X": "option_b"}
q4_dict = {"Y is heavier than X": "option_a", "X is heavier than Y": "option_b"}

q5_dict = {"X is stronger than Y": "option_a", "Y is stronger than X":  "option_b"}
q6_dict = {"Y is stronger than X": "option_a", "X is stronger than Y": "option_b"}

# concrete dicts
q7_dict = {"The boy is bigger than the ball": "option_a", "The ball is bigger than the boy": "option_b"}
q8_dict = {"The woman is heavier than the book": "option_a", "The book is heavier than the woman": "option_b"}
q9_dict = {"The horse is stronger than the bug": "option_a", "The bug is stronger than the horse": "option_b"}
q10_dict = {"The girl is bigger than the rocket": "option_a", "The rocket is bigger than the girl": "option_b"}
q11_dict = {"The boy is heavier than the car": "option_a", "The car is heavier than the boy": "option_b"}
q12_dict = {"The machine is stronger than the truck": "option_a", "The truck is stronger than the machine": "option_b"}

find_dict = {"Q1": q1_dict, "Q2": q2_dict, "Q3": q3_dict, "Q4": q4_dict,  "Q5": q5_dict,  "Q6": q6_dict,  "Q7": q7_dict,  "Q8": q8_dict,  "Q9": q9_dict, "Q10": q10_dict, "Q11": q11_dict, "Q12": q12_dict}

def main():
    df = pd.read_csv("data/survey_options.csv")
    for c in df.columns:
        curr_dict = find_dict[c]
        df[c] = df[c].apply(lambda s: curr_dict[s]) # default axis=0 aka col
    df.to_csv("data/survey_options_processed.csv", index=False)

if __name__ == '__main__':
    main()
