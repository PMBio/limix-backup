# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.

import subprocess, sys, os
import time
import h5py
import scipy as sp
#import limix
from optparse import OptionParser
import pandas as pd

sys.path.append("./../../")

class LIMIX_runner(object):
    '''
    A class for running a LIMIX pipeline.
    '''
    __slots__=["options","args","data", "result"] 

    def __init__(self):
        '''
        nothing to initialize
        '''
        self.options=None
        self.args=None
        self.data=None
        self.result=None
        pass    

    def parse_args(self):
        parser = OptionParser()
        parser.add_option('--data_script', metavar='data_script', type=str, help='A python script for loading the data', default=r"example_scripts/example_data.py")
        parser.add_option('--experiment_script', metavar='experiment_script', type=str, help='A python script for running the experiment', default=r"example_scripts/gwas_example.py")
        parser.add_option('--outfile', metavar='outfile', type=str, help='The output file', default="example_out.txt")
        parser.add_option('--seed', metavar='seed', type=int, help='The random seed', default=123123)
        parser.add_option('--delimiter', metavar='delimiter',type=str, help="The delimiter between output values", default="\t")
        (self.options, self.args) = parser.parse_args()
        return (self.options,self.args)

    def load_data(self):
        """
        Run the job specified in data_script
        """
        options=self.options
        command = open(self.options.data_script).read()
        t0=time.time()
        data=None       #fallback data
        exec(command)   #creates variable data
        t1=time.time()
        print ("Elapsed time for data reading is %.2f seconds" % (t1-t0))
        self.data=data
        return self.data
    
    def run_experiment(self):
        """
        Run the job specified in experiment_script
        """
        data=self.data
        options=self.options
        command = open(self.options.experiment_script).read()
        t0=time.time()
        result=None     #fallback result
        exec(command)   #creates variable result
        t1=time.time()
        print ("Elapsed time for running the experiment is %.2f seconds" % (t1-t0)) 
        self.result=result
        return self.result

    def write_resultfile(self):
        """
        Write the outout to disk
        """
        #data=self.data
        #options=self.options
        #result=self.result
        #command = open(self.options.output_script).read()
        t0=time.time()
        self.result.to_csv(self.options.outfile,sep=self.options.delimiter,index=True,header=True,na_rep="NAN")
        #output=None     #fallback output
        #exec(command)   #creates variable output
        t1=time.time()
        print ("Elapsed time for running the experiment is %.2f seconds" % (t1-t0)) 
        
        

if __name__ == "__main__":
    print ("last modified: %s" % time.ctime(os.path.getmtime(__file__)))
    
    runner = LIMIX_runner()
    (options,args) =runner.parse_args()
    data = runner.load_data()
    result = runner.run_experiment()
    runner.write_resultfile()

