import matplotlib.pyplot as plt
from generate_topo import *
import matplotlib as mpl
from matplotlib.lines import Line2D

# Vertical
# Acceptance rate, Throughput, Service time, Running time cost, Average total reward
# Acceptance rate, Real fail rate, Service time

# Acceptance rate between algorithms
# greedy, icc, worst, dqn

path_prefix = "..\\result\\" if pf == "Windows" else "../result/"  # file name


def plot_acceptance_rate_between_algorithms():
    ANS = [0.4608141131095045, 0.3404969778374748, 0.2932129722500836]
    AboveNet = [0.4402557615719327, 0.3044816968867602, 0.316382362407467]
    BSO = [0.4405448843799856, 0.33715170278637774, 0.3622867972204674]
    higher_RG = (ANS[0] + AboveNet[0] + BSO[0]) / (ANS[1] + AboveNet[1] + BSO[1]) - 1
    higher_BFG = (ANS[0] + AboveNet[0] + BSO[0]) / (ANS[2] + AboveNet[2] + BSO[2]) - 1
    print("acceptance_rate: avg higher than RG: {}, avg higher than BFG: {}".format(higher_RG, higher_BFG))

    plt.ylabel('Acceptance rate')
    plt.title("Acceptance rate between different algorithms")
    plt.ylim(0, 0.55)

    labels = ["ANS", "AboveNet", "BSO"]
    ddqp_index = [0.4, 1.2, 2.0]
    rg_index = [0.6, 1.4, 2.2]
    bfg_index = [0.8, 1.6, 2.4]
    label_index = [0.6, 1.4, 2.2]
    width = 0.18
    dqn_rects = plt.bar(ddqp_index, [ANS[0], AboveNet[0], BSO[0]], width, color='#1b77aa', label="DDQP", lw=1, edgecolor="black")
    for rect in dqn_rects:
        height = rect.get_height()
        plt.annotate('{:.2f}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    greedy_rects = plt.bar(rg_index, [ANS[1], AboveNet[1], BSO[1]], width, color="#ef8935", label="RG", lw=1, edgecolor="black")
    for rect in greedy_rects:
        height = rect.get_height()
        plt.annotate('{:.2f}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    icc_rects = plt.bar(bfg_index, [ANS[2], AboveNet[2], BSO[2]], width, color="#0d7263", label="BFG", lw=1, edgecolor="black")
    for rect in icc_rects:
        height = rect.get_height()
        plt.annotate('{:.2f}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.xticks(label_index, labels=labels)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend()
    plt.savefig(path_prefix + "acceptance.jpg")
    plt.show()


def plot_throughput_between_algorithms():
    ANS = [102015.8, 76041.0, 66293.0]
    AboveNet = [95993.4, 67158.0, 73581.0]
    BSO = [93395.9, 81429.0, 84932.0]
    higher_RG = (ANS[0] + AboveNet[0] + BSO[0]) / (ANS[1] + AboveNet[1] + BSO[1]) - 1
    higher_BFG = (ANS[0] + AboveNet[0] + BSO[0]) / (ANS[2] + AboveNet[2] + BSO[2]) - 1
    print("throughput: avg higher than RG: {}, avg higher than BFG: {}".format(higher_RG, higher_BFG))
    plt.ylabel('Throughput')
    plt.title("Throughput between different algorithms")
    plt.ylim(0, 120000)

    labels = ["ANS", "AboveNet", "BSO"]
    ddqp_index = [0.4, 1.2, 2.0]
    rg_index = [0.6, 1.4, 2.2]
    bfg_index = [0.8, 1.6, 2.4]
    label_index = [0.6, 1.4, 2.2]
    width = 0.18
    dqn_rects = plt.bar(ddqp_index, [ANS[0], AboveNet[0], BSO[0]], width, color='#1b77aa', label="DDQP", lw=1, edgecolor="black")
    for rect in dqn_rects:
        height = rect.get_height()
        plt.annotate('{:.0}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    greedy_rects = plt.bar(rg_index, [ANS[1], AboveNet[1], BSO[1]], width, color="#ef8935", label="RG", lw=1, edgecolor="black")
    for rect in greedy_rects:
        height = rect.get_height()
        plt.annotate('{:.0}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    icc_rects = plt.bar(bfg_index, [ANS[2], AboveNet[2], BSO[2]], width, color="#0d7263", label="BFG", lw=1, edgecolor="black")
    for rect in icc_rects:
        height = rect.get_height()
        plt.annotate('{:.0}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.xticks(label_index, labels=labels)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend()
    plt.savefig(path_prefix + "throughput.jpg")
    plt.show()


def plot_service_time_between_algorithms():
    ANS = [8404.366501803534, 6250.818839877734, 5080.296956371169]
    AboveNet = [8399.882424619322, 5367.621422329118, 6069.682253995072]
    BSO = [6621.5742377654915, 5581.148640142848, 5763.347038424634]
    higher_RG = (ANS[0] + AboveNet[0] + BSO[0]) / (ANS[1] + AboveNet[1] + BSO[1]) - 1
    higher_BFG = (ANS[0] + AboveNet[0] + BSO[0]) / (ANS[2] + AboveNet[2] + BSO[2]) - 1
    print("service_time: avg higher than RG: {}, avg higher than BFG: {}".format(higher_RG, higher_BFG))
    plt.ylabel('Service time')
    plt.title("Service time between different algorithms")
    plt.ylim(0, 11000)

    labels = ["ANS", "AboveNet", "BSO"]
    ddqp_index = [0.4, 1.2, 2.0]
    rg_index = [0.6, 1.4, 2.2]
    bfg_index = [0.8, 1.6, 2.4]
    label_index = [0.6, 1.4, 2.2]
    width = 0.18
    dqn_rects = plt.bar(ddqp_index, [ANS[0], AboveNet[0], BSO[0]], width, color='#1b77aa', label="DDQP", lw=1, edgecolor="black")
    for rect in dqn_rects:
        height = rect.get_height()
        plt.annotate('{:.0}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    greedy_rects = plt.bar(rg_index, [ANS[1], AboveNet[1], BSO[1]], width, color="#ef8935", label="RG", lw=1, edgecolor="black")
    for rect in greedy_rects:
        height = rect.get_height()
        plt.annotate('{:.0}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    icc_rects = plt.bar(bfg_index, [ANS[2], AboveNet[2], BSO[2]], width, color="#0d7263", label="BFG", lw=1, edgecolor="black")
    for rect in icc_rects:
        height = rect.get_height()
        plt.annotate('{:.0}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.xticks(label_index, labels=labels)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend()

    plt.savefig(path_prefix + "service.jpg")
    plt.show()


def plot_running_time_cost_between_algorithms():
    ANS = [3.0, 14.0, 125.0]
    AboveNet = [3.0, 25.0, 176.0]
    BSO = [3.0, 14.0, 90.0]
    higher_RG = 1 - (ANS[0] + AboveNet[0] + BSO[0]) / (ANS[1] + AboveNet[1] + BSO[1])
    higher_BFG = 1 - (ANS[0] + AboveNet[0] + BSO[0]) / (ANS[2] + AboveNet[2] + BSO[2])
    print("running_time_cost: avg higher than RG: {}, avg higher than BFG: {}".format(higher_RG, higher_BFG))
    plt.ylabel('Running time cost')
    plt.title("Running time cost between different algorithms")
    plt.ylim(0, 200)

    labels = ["ANS", "AboveNet", "BSO"]
    ddqp_index = [0.4, 1.2, 2.0]
    rg_index = [0.6, 1.4, 2.2]
    bfg_index = [0.8, 1.6, 2.4]
    label_index = [0.6, 1.4, 2.2]
    width = 0.18
    dqn_rects = plt.bar(ddqp_index, [ANS[0], AboveNet[0], BSO[0]], width, color='#1b77aa', label="DDQP", lw=1, edgecolor="black")
    for rect in dqn_rects:
        height = rect.get_height()
        plt.annotate('{:.2f}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    greedy_rects = plt.bar(rg_index, [ANS[1], AboveNet[1], BSO[1]], width, color="#ef8935", label="RG", lw=1, edgecolor="black")
    for rect in greedy_rects:
        height = rect.get_height()
        plt.annotate('{:.2f}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    icc_rects = plt.bar(bfg_index, [ANS[2], AboveNet[2], BSO[2]], width, color="#0d7263", label="BFG", lw=1, edgecolor="black")
    for rect in icc_rects:
        height = rect.get_height()
        plt.annotate('{:.2f}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.xticks(label_index, labels=labels)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend()

    plt.savefig(path_prefix + "runningtime.jpg")
    plt.show()


def plot_reward_trace():
    if pf == "Windows":
        TRACE_FILE = "model\\trace.pkl"
    elif pf == "Linux":
        TRACE_FILE = "model/trace.pkl"
    with open(TRACE_FILE, 'rb') as f:
        reward_trace = pickle.load(f)  # read file and build object
        index_len = len(reward_trace)
        ddqp_height = 1210
        rg_height = 835
        bfg_height = 934
        line_ddqp = [(0, ddqp_height), (index_len, ddqp_height)]
        line_rg = [(0, rg_height), (index_len, rg_height)]
        line_bfg = [(0, bfg_height), (index_len, bfg_height)]
        (line_tgt_xs, line_tgt_ys) = zip(*line_ddqp)
        (line_RG_xs, line_RG_ys) = zip(*line_rg)
        (line_BFG_xs, line_BFG_ys) = zip(*line_bfg)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel('Total reward')
        ax.set_title("Total reward of training network on AboveNet")
        plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
        plt.ylim(0, 1500)
        ax.add_line(Line2D(line_tgt_xs, line_tgt_ys, color="#1b77aa", linestyle="--"))
        ax.add_line(Line2D(line_RG_xs, line_RG_ys, color="#ef8935", linestyle="--"))
        ax.add_line(Line2D(line_BFG_xs, line_BFG_ys, color="#0d7263", linestyle="--"))
        ax.annotate('DDQP Target Agent',
                    xy=(index_len - 280, ddqp_height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color="#1b77aa")
        ax.annotate('RG',
                    xy=(index_len - 90, rg_height),
                    xytext=(0, -15),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color="#ef8935")
        ax.annotate('BFG',
                    xy=(index_len - 90, bfg_height),
                    xytext=(0, -15),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color="#0d7263")
        ax.set_xlabel("Episode")
        plt.plot(reward_trace, color="#1b77aa", linewidth=0.7, label="DDQP Online Agent")
        plt.legend()
        plt.savefig(path_prefix + "training.jpg")
        plt.show()


def plot_acceptance_rate_between_configs():
    AboveNet = [0.5687950609528223, 0.4952037313026484, 0.48626878336136004, 0.4402557615719327, 0.34137601746496715]
    plt.ylabel('Acceptance rate')
    plt.title("Acceptance rate between different configurations")
    plt.ylim(0, 0.7)

    labels = ["NoBackup", "Aggressive", "Normal", "MaxR", "FullyR"]
    index = [0.4, 0.7, 1.0, 1.3, 1.6]

    width = 0.18
    dqn_rects = plt.bar(index, [AboveNet[0], AboveNet[1], AboveNet[2], AboveNet[3], AboveNet[4]], width, color='#1b77aa', label="DDQP", lw=1, edgecolor="black")
    for rect in dqn_rects:
        height = rect.get_height()
        plt.annotate('{:.2f}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.xticks(index, labels=labels)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend()
    plt.savefig(path_prefix + "acceptance_c.jpg")
    plt.show()


def plot_real_fail_rate_between_configs():
    AboveNet = [1.0, 0.07998642380359769, 0.04666666666666667, 0.009208772567551194, 0.0]
    plt.ylabel('Fail rate')
    plt.title("Fail rate between different configurations")
    plt.ylim(0, 1.2)

    labels = ["NoBackup", "Aggressive", "Normal", "MaxR", "FullyR"]
    index = [0.4, 0.7, 1.0, 1.3, 1.6]

    width = 0.18
    dqn_rects = plt.bar(index, [AboveNet[0], AboveNet[1], AboveNet[2], AboveNet[3], AboveNet[4]], width, color='#1b77aa', label="DDQP", lw=1, edgecolor="black")
    for rect in dqn_rects:
        height = rect.get_height()
        plt.annotate('{:.2f}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.xticks(index, labels=labels)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend()
    plt.savefig(path_prefix + "fail_rate_c.jpg")
    plt.show()


def plot_service_time_between_configs():
    AboveNet = [8124.12508656571, 9478.485801782732, 9448.347064335872, 8558.482112733094, 6842.192849871526]
    plt.ylabel('Service time')
    plt.title("Service time between different configurations")
    plt.ylim(0, 11000)

    labels = ["NoBackup", "Aggressive", "Normal", "MaxR", "FullyR"]
    index = [0.4, 0.7, 1.0, 1.3, 1.6]

    width = 0.18
    dqn_rects = plt.bar(index, [AboveNet[0], AboveNet[1], AboveNet[2], AboveNet[3], AboveNet[4]], width, color='#1b77aa', label="DDQP", lw=1, edgecolor="black")
    for rect in dqn_rects:
        height = rect.get_height()
        plt.annotate('{:.2f}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.xticks(index, labels=labels)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend()
    plt.savefig(path_prefix + "service_time_c.jpg")
    plt.show()


if __name__ == "__main__":
    # plot_acceptance_rate_between_algorithms()
    # plot_throughput_between_algorithms()
    # plot_service_time_between_algorithms()
    # plot_running_time_cost_between_algorithms()
    plot_reward_trace()
    # plot_acceptance_rate_between_configs()
    # plot_real_fail_rate_between_configs()
    # plot_service_time_between_configs()
