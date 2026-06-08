import csv, sys
import os,glob
import time
import argparse
import numpy as np
from numpy import *
import multiprocessing

wd = "path_to_RevBayes_scripts"
rev_dir = "path_to_RevBayes_executable"
n_of_runs = 5 # number of parallel jobs

def runRev(arg):
    # run with phyloRNN rates
    cmd="cd %s; ./rb  %s/ali%s_DL" % (rev_dir, wd, arg)
    os.system(cmd)
    # run with Gamma rates
    cmd="cd %s; ./rb  %s/ali%s_G" % (rev_dir, wd, arg)
    os.system(cmd)

list_args = list(range(n_of_runs))

if __name__ =="__main__":
    pool = multiprocessing.Pool(len(list_args))
    pool.map(runRev, list_args)
    pool.close()
