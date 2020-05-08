import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt

###############################################################################
# Human Agreement on each question
###############################################################################
def human_agg_per_question():
    df = pd.read_csv("data/survey_options_processed.csv")
    labels = []
    q_a_count = []
    q_b_count = []
    for col in df.columns:
        a_count = (df[col] == 'option_a').sum()
        b_count = (df[col] == 'option_b').sum()
        labels.append(col)
        q_a_count.append(a_count)
        q_b_count.append(b_count)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, q_a_count, width, label='Option A', color="#CAC0FB")
    rects2 = ax.bar(x + width/2, q_b_count, width, label='Option B', color="#BEE5FB")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count of the option')
    ax.set_title('Count of options chosen per question')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.savefig("figs/cnt_per_question.png")

###############################################################################
# Human Agreement for each task type
###############################################################################
def human_agg_per_task_type():
    concrete = ["Q7","Q8","Q9","Q10","Q11","Q12"]
    abstract = ["Q1","Q2","Q3","Q4","Q5","Q6"]
    df = pd.read_csv("data/survey_options_processed.csv")
    con_a_count = []
    con_b_count = []
    abs_a_count = []
    abs_b_count = []
    for col in df.columns:
        if col in concrete:
            con_a_count.append((df[col] == 'option_a').sum())
            con_b_count.append((df[col] == 'option_b').sum())
        elif col in abstract:
            abs_a_count.append((df[col] == 'option_a').sum())
            abs_b_count.append((df[col] == 'option_b').sum())

    labels = ["concrete", "abstract"]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    a_means = [np.mean(con_a_count), np.mean(abs_a_count)]
    a_std = [np.std(con_a_count), np.std(abs_a_count)]
    b_means = [np.mean(con_b_count), np.mean(abs_b_count)]
    b_std = [np.std(con_b_count), np.std(abs_b_count)]

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, a_means, width, yerr=a_std, label='Option A', color="#CAC0FB")
    rects2 = ax.bar(x + width/2, b_means, width, yerr=b_std, label='Option B', color="#BEE5FB")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count of the option')
    ax.set_title('Average count of options per task type')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.savefig("figs/dist_per_type.png")

###############################################################################
# BERT Perplexity
###############################################################################
def bert_perplexity():
    df1 = pd.read_csv("data/bert_result_wQ.csv")
    fig, ax = plt.subplots()
    q = df1["Q"].values
    ap = df1["a_perplexity"].values
    bp = df1["b_perplexity"].values
    ax.scatter(q, ap, color="#9690BA", label="Option A")
    ax.scatter(q, bp, color="#98B8CA", label="Option B")
    ax.set_title("BERT Perplexity for each question")
    ax.set_xticks(q)
    ax.legend()

    plt.show()

###############################################################################
# Accuracy Table
###############################################################################
def acc():
    df1 = pd.read_csv("data/bert_result.csv")
    df1["eq"] = df1["winner"].eq("a")
    accuracy = float(df1["eq"].sum()) / df1["winner"].shape[0]
    print("BERT accuracy is: ", accuracy)

    accs = []
    df2 = pd.read_csv("data/survey_options_processed.csv")
    for c in df2.columns:
        accs.append(float(df2[c].eq("option_a").sum()) / df2[c].shape[0])
    mean = np.mean(accs)
    std = np.std(accs) / sqrt(12)
    print("Human mean accuracy is: ", mean)
    print("Human accuracy std error is: ", std)

def human_abs_con_acc():
    concrete = ["Q7","Q8","Q9","Q10","Q11","Q12"]
    abstract = ["Q1","Q2","Q3","Q4","Q5","Q6"]
    abs_acc = []
    con_acc = []

    df2 = pd.read_csv("data/survey_options_processed.csv")
    for c in df2.columns:
        if c in concrete:
            con_acc.append(float(df2[c].eq("option_a").sum()) / df2[c].shape[0])
        elif c in abstract:
            abs_acc.append(float(df2[c].eq("option_a").sum()) / df2[c].shape[0])

    con_mean = np.mean(con_acc)
    con_std = np.std(con_acc)/ sqrt(6)

    abs_mean = np.mean(abs_acc)
    abs_std = np.std(abs_acc)/ sqrt(6)

    print("Human con acc: ", con_mean)
    print("Human con stderr: ", con_std)

    print("Human abs acc:  ", abs_mean)
    print("Human abs stderr: ", abs_std)

def bert_abs_con_acc():
    concrete = ["Q7","Q8","Q9","Q10","Q11","Q12"]
    abstract = ["Q1","Q2","Q3","Q4","Q5","Q6"]
    abs_acc = 0
    con_acc = 0
    df = pd.read_csv("data/bert_result_wQ.csv")
    for index, row in df.iterrows():
        if row["Q"] in concrete:
            if row["winner"] == "a":
                con_acc += 1.0
        elif row["Q"] in abstract:
            if row["winner"] == "a":
                abs_acc += 1.0

    print("BERT con acc: ", con_acc / 6.0)
    print("BERT abs acc: ", abs_acc / 6.0)

def main():
    human_agg_per_question()
    print("Done, fig saved.")
    print("_________________________________")
    human_agg_per_task_type()
    print("Done, fig saved.")
    print("_________________________________")
    bert_perplexity()
    print("Done, fig saved.")
    print("_________________________________")
    acc()
    print("_________________________________")
    human_abs_con_acc()
    print("_________________________________")
    bert_abs_con_acc()

if __name__ == '__main__':
    main()
