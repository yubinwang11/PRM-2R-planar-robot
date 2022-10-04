
import sys
from tracemalloc import start
import numpy as np
import argparse
from classes import PRMController, Obstacle, Utils
import time


def main(args):

    parser = argparse.ArgumentParser(description='PRM Path Planning Algorithm')
    parser.add_argument('--numSamples', type=int, default=5000, metavar='N',
                        help='Number of sampled points')
    args = parser.parse_args()

    numSamples = args.numSamples

    env = open("environment.txt", "r")
    l1 = env.readline().split(";")

    current = list(map(int, l1[0].split(",")))
    destination = list(map(int, l1[1].split(",")))

    print("Current: {} Destination: {}".format(current, destination))

    print("****Obstacles****")
    allObs = []
    for l in env:
        if(";" in l):
            line = l.strip().split(";")
            topLeft = list(map(int, line[0].split(",")))
            bottomRight = list(map(int, line[1].split(",")))
            obs = Obstacle(topLeft, bottomRight)
            obs.printFullCords()
            allObs.append(obs)

    utils = Utils()
    utils.drawMap(allObs, current, destination)

    prm = PRMController(numSamples, allObs, current, destination)
    # Initial random seed to try
    initialRandomSeed = 0
    prm.runPRM(initialRandomSeed)


if __name__ == '__main__':
    start_time = time.time()
    main(sys.argv)
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"total comuputation time is {computation_time}")
