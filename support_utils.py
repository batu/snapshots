import smtplib
import numpy as np
import time
import re
import argparse

import matplotlib.pyplot as plt
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os.path import basename



def visualize_graph_messed_up(input_file: str, ouput_destionation: str, readme_dest: str,  experiment_num: str, run: int) -> None:
    lines = []
    with open(input_file, "r") as input:
        line = input.readline()
        commas = [m.start() for m in re.finditer(r",", line)]
        print(commas)
        last_index = 0
        for i, comma_index in enumerate(commas):
            if(i < 9):
                lines.append(line[:comma_index - last_index - 1])
                line = line[comma_index - last_index - 1:]
                last_index = comma_index - 1
            else:
                lines.append(line[:comma_index - last_index - 2])
                line = line[comma_index - last_index - 2:]
                last_index = comma_index - 2

    indexes = []
    values = []
    for line in lines[1:]:
        indexes.append(int(line.split(",")[0]))
        value = float(line.split(",")[1])
        if value > 10000:
            value = values[-1]
        values.append(value)

    fig = plt.figure()
    plt.title(f"Max Reward for Run {run} of {experiment_num}")
    plt.plot(indexes, values, color="teal", linewidth=1.5, linestyle="-", label="Max Reward")
    plt.legend(loc='upper left', frameon=False)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    plt.savefig(ouput_destionation, bbox_inches='tight')
    plt.savefig(readme_dest, bbox_inches='tight')
    # plt.show()


def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
        y[ctr] = np.sum(x[ctr:(ctr + N)])
    return y / N


def visualize_cumulative_reward(input_file: str, ouput_destionation: str, readme_dest: str, experiment_name: str, run_count: int) -> None:
    indexes = []
    values = []
    N = 4

    with open(input_file, "r") as input:
        for line in input:
            indexes.append(int(line.split(",")[0]))
            value = float(line.split(",")[1])
            if value > 10000:
                value = values[-1]
            values.append(value)

    running_average = runningMean(values, N)

    fig = plt.figure()
    plt.title(f"Cumulative Reward for Run {run_count} of {experiment_name}")
    plt.plot(indexes, values, color="teal", linewidth=1.5, linestyle="-", label="Cumulative Reward")
    plt.plot(indexes, running_average, color="gray", linewidth=1, linestyle=":", label="Running Average of 4")
    plt.legend(loc='upper left', frameon=False)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    plt.savefig(ouput_destionation, bbox_inches='tight')
    plt.savefig(readme_dest, bbox_inches='tight')
    # plt.show()


def visualize_max_reward(input_file: str, ouput_destionation: str, readme_dest: str, experiment_name: str, run_count: int) -> None:
    indexes = []
    values = []
    N = 4

    with open(input_file, "r") as input:
        for line in input:
            indexes.append(int(line.split(",")[0]))
            value = float(line.split(",")[1])
            if value > 10000:
                value = values[-1]
            values.append(value)

    running_average = runningMean(values, N)

    fig = plt.figure()
    plt.title(f"Max Reward for Run {run_count} of {experiment_name}")
    plt.plot(indexes, values, color="orange", linewidth=1.5, linestyle="-", label="Max Reward")
    plt.plot(indexes, running_average, color="gray", linewidth=1, linestyle=":", label="Running Average of 4")
    plt.legend(loc='upper left', frameon=False)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    plt.savefig(ouput_destionation, bbox_inches='tight')
    plt.savefig(readme_dest, bbox_inches='tight')
    # plt.show()


def send_email(msg_body: str, run_path: str, experiment_name: str, run_number: int) -> None:
    """
    Sends the email to me!
    """
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("kafkabot9000@gmail.com", "thisisnotimportant")

    msg = MIMEMultipart()

    files = [f"{run_path}/cumulative_reward_{experiment_name}_R{run_number}_plot.png",
             f"{run_path}/max_reward_{experiment_name}_R{run_number}_plot.png"]

    msg_body += "\n"
    msg_body += time.asctime(time.localtime(time.time()))
    msg.attach(MIMEText(msg_body))

    for f in files:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)

    server.sendmail("kafkabot9000@gmail.com", "baytemiz@ucsc.edu", msg.as_string())
    server.quit()

def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', action='store', dest='save_prob', default=None,
                        help='The snapshot save probability [0,1]')

    parser.add_argument('-l', action='store', dest='load_prob',default=None,
                        help='The snapshot load probability [0,1]')

    parser.add_argument('-e', action='store', dest='experiment_name',default=None,
                        help='The name of the experiment')
    results = parser.parse_args()

    assert results.save_prob and results.load_prob and results.experiment_name, "\nPlease specify the save and load probablities with the flags -s and -l and the experiment name with the -e flag."
    print(f"\n\n\nTHE SAVE PROBABILITY: {results.save_prob}")
    print(f"THE LOAD PROBABILITY: {results.load_prob}\n\n\n")
    return (float(results.save_prob), float(results.load_prob), results.experiment_name)

def save_hyperparameters(filenames: list, path_to_file: str, save_prob: float, load_prob: float, experiment_name:str, breadcrumb="# BREADCRUMBS") -> None:
    """
    Saves the lines in between breadcrumbs in all the given filenames. This is used for saving hyperparameters for RL training.
    Parameters
    ----------
    breadcrumb: Writes the lines between {breadcrumb}_START and {breadcrumb}_END.
    """
    with open(path_to_file, "a") as dest:
        dest.write(f"Experiment name: {experiment_name} \n")
        dest.write(f"Save prob:{save_prob} \n")
        dest.write(f"Load prob:{load_prob} \n")
        for filename in filenames:
            with open(filename, "r") as source:
                saving = False
                for line in source:
                    if line.strip() == f"{breadcrumb}_START":
                        dest.write("\n")
                        saving = True
                        continue
                    if line.strip() == f"{breadcrumb}_END":
                        saving = False
                        continue
                    if saving:
                        dest.write(line)
            print(f"{filename} hyperparameters have been saved!")
        print("Information saving is complete!")
