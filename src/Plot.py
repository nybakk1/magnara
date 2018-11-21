import matplotlib.pyplot as plt


def plot(data, legend, yMax, title="", labels=("", "")):
    fig = plt.figure()
    for i in range(len(data)):
        ax = fig.add_subplot(111)
        if i < len(legend):
            ax.plot(data[i][0], data[i][1], label=legend[i])
            ax.legend()
        else:
            ax.plot(data[i][0], data[i][1])
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.ylim(0, yMax+10)
    plt.show()
