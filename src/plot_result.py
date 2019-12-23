import matplotlib.pyplot as plt
import matplotlib as mpl

# Acceptance rate between algorithms
# greedy, icc, worst, dqn

throughput = []




def plot_acceptance_rate_between_algorithms():
    ANS = [0.40905729255141887, 0.29516296075934945, 0.25974669929907995]
    AboveNet = [0.37363002324809036, 0.2554794520547945, 0.25749741468459153]
    BSO = [0.4, 0.30480798306622314, 0.2961441213653603]
    labels = ["ANS", "AboveNet", "BSO"]
    dqn_index = [0.4, 1.2, 2.0]
    greedy_index = [0.6, 1.4, 2.2]
    icc_index = [0.8, 1.6, 2.4]
    label_index = [0.6, 1.4, 2.2]
    width = 0.18
    fig = plt.figure()
    ax = fig.add_subplot(111)

    dqn_rects = ax.bar(dqn_index, [ANS[0], AboveNet[0], BSO[0]], width, color='#1b77aa', label="DDQN")
    for rect in dqn_rects:
        height = rect.get_height()
        ax.annotate('{:.2}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    greedy_rects = ax.bar(greedy_index, [ANS[1], AboveNet[1], BSO[1]], width, color="#ef8935", label="Greedy")
    for rect in greedy_rects:
        height = rect.get_height()
        ax.annotate('{:.2}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    icc_rects = ax.bar(icc_index, [ANS[2], AboveNet[2], BSO[2]], width, color="#0d7263", label="ICC")
    for rect in icc_rects:
        height = rect.get_height()
        ax.annotate('{:.2}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_xticks(label_index)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Acceptance rate')
    # title
    ax.set_title("Acceptance rate between different algorithms")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig("bar.jpg")

def plot_running_time_cost_between_algorithms():
    ANS = [3.0, 14.0, 125.0]
    AboveNet = [3.0, 25.0, 176.0]
    BSO = [3.0, 14.0, 90.0]
    labels = ["ANS", "AboveNet", "BSO"]
    dqn_index = [0.4, 1.2, 2.0]
    greedy_index = [0.6, 1.4, 2.2]
    icc_index = [0.8, 1.6, 2.4]
    label_index = [0.6, 1.4, 2.2]
    width = 0.18
    fig = plt.figure()
    ax = fig.add_subplot(111)

    dqn_rects = ax.bar(dqn_index, [ANS[0], AboveNet[0], BSO[0]], width, color='#1b77aa', label="DDQN")
    for rect in dqn_rects:
        height = rect.get_height()
        ax.annotate('{:.2}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    greedy_rects = ax.bar(greedy_index, [ANS[1], AboveNet[1], BSO[1]], width, color="#ef8935", label="Greedy")
    for rect in greedy_rects:
        height = rect.get_height()
        ax.annotate('{:.2}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    icc_rects = ax.bar(icc_index, [ANS[2], AboveNet[2], BSO[2]], width, color="#0d7263", label="ICC")
    for rect in icc_rects:
        height = rect.get_height()
        ax.annotate('{:.2}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_xticks(label_index)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Running time cost')
    # title
    ax.set_title("Running time cost between different algorithms")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig("bar.jpg")

if __name__ == "__main__":
    plot_acceptance_rate_between_algorithms()
    plot_running_time_cost_between_algorithms()


