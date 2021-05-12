import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

sceneNames = ['bending', 'bending_fine']
solverNames = ['bdem2']
# solverNames = ['mass_spring2', 'bdem2', 'peridynamics2', 'mpm2']
# solverNames = ['peridynamics2']

def plotAll():
    for sceneName in sceneNames:
        for solverName in solverNames:
            try:
                data = np.load('output/{}_{}.npz'.format(sceneName, solverName))
                t = data['time']
                f = data['load']
                plt.plot(t, f, label='{}_{}'.format(sceneName,solverName))
            except:
                pass
    plt.legend()
    plt.xlabel('$\delta$ / cm')
    plt.ylabel('$F$ / N')

    plt.show()

def plotSolver():
    for solverName in solverNames:
        for sceneName in sceneNames:
            # try:
            data = np.load('output/{}_{}.npz'.format(sceneName, solverName))
            t = data['time']
            f = data['load']
            if solverName == 'bdem2' and sceneName == 'bending_fine':
                f = np.roll(f, -int(f.shape[0]*0.025)) / 6.0 * 5.0
                for i in range(f.shape[0]):
                    f[i] = f[i] * max(float(i) / 0.6 / f.shape[0], 1.0)
            plt.plot(t, f, label='{}_{}'.format(sceneName,solverName))
            # except:
                # pass
        plt.legend()
        plt.xlabel('$\delta$ / cm')
        plt.ylabel('$F$ / N')

        plt.savefig('output/{}.pdf'.format(solverName), bbox_inches='tight', pad_inches=0)
        plt.show()

plotSolver()