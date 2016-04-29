# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tables
import numpy as np
import time, os
import pandas as pd
from datetime import datetime

def _depth_first_hdf5(dictionary, outfile, root=None, filters=None):
    """
    performs a depth first search on a dict object and creates a HDF5 outfile 
    structure that mirrors the dictionary.
    Supported types for leaf nodes are numpy.ndarray, scalars, regular lists, and pandas.DataFrame
    
    Args:
        dictionary:     dict
        outfile:        pytables HDF5 outfile
        root:           the current root node (default "/")
        filters:        filters for chunked storage
    """
    if root is None:
        root=outfile.root
    for child_key in list(dictionary.keys()):
        if isinstance(dictionary[child_key],dict):
            child_group = outfile.create_group(root, child_key)
            _depth_first_hdf5(dictionary[child_key], outfile=outfile, root=child_group, filters=filters)
        elif np.isscalar(dictionary[child_key]):
            leaf = outfile.create_array(where=root, name=child_key, obj=dictionary[child_key])
        elif isinstance(dictionary[child_key],list):
            leaf = outfile.create_array(where=root, name=child_key, obj=dictionary[child_key])
        elif isinstance(dictionary[child_key],np.ndarray):
            atom = tables.Atom.from_dtype(dictionary[child_key].dtype)
            if filters is None:
                leaf = outfile.create_array(where=root, name=child_key, atom=atom, shape=dictionary[child_key].shape, obj=dictionary[child_key])
            else:
                leaf = outfile.create_array(where=root, name=child_key, atom=atom, shape=dictionary[child_key].shape, obj=dictionary[child_key], filters=filters)
        elif isinstance(dictionary[child_key],pd.core.frame.DataFrame):
            #raise NotImplementedError("to columscome: DataFrames to HDF")
            child_group = outfile.create_group(root, child_key) #The DataFrame is stored in a group of its own

            index = np.array(dictionary[child_key].index,dtype=type(dictionary[child_key].index[0]))
            leaf = outfile.create_array(where=child_group, name="index", obj=index) #Store the index
            
            columns = np.array(dictionary[child_key].columns,dtype=type(dictionary[child_key].columns[0]))
            leaf = outfile.create_array(where=child_group, name="columns", obj=columns) #store the columns
            
            values = dictionary[child_key].values
            leaf = outfile.create_array(where=child_group, name="values", obj=columns)  #store the values
        else:
            raise IOError("unsupported IO type in output dictionary: "+str(type(dictionary)))

def _depth_first_text(dictionary, outdir=".", delimiter=" ",float_format="%.6e"):
    """
    performs a depth first search on a dict object and creates a directory 
    structure that mirrors the dictionaries. The leafs of the dictionaries are text files
    Supported types for leaf nodes are numpy.ndarray, scalars, regular lists and pandas.DataFrame
    
    Args:
        dictionary:     dict
        outdir:         directory to write in
        delimiter:      delimiter for root node text files
        float_format:   default format for floating point outputs.   
    """
    
    for child_key in list(dictionary.keys()):
        if isinstance(dictionary[child_key],dict):
            child_path = os.path.join(outdir, child_key)
            _depth_first_text(dictionary[child_key], outdir=child_path,delimiter=delimiter,float_format=float_format)
        else:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            filename = os.path.join(outdir, child_key)+".txt"
            if np.isscalar(dictionary[child_key]):
                file = open(filename, "w")
                if type(dictionary[child_key])==float:
                    outstr=float_format % dictionary[child_key]
                else:
                    outstr=str(dictionary[child_key])
                file.write(outstr)
                file.close()
            elif isinstance(dictionary[child_key],list):
                outarray=np.array(dictionary[child_key])
                _write_txt_array(filename,array=outarray,delimiter=delimiter,float_format=float_format)
            elif isinstance(dictionary[child_key],np.ndarray):
                _write_txt_array(filename,array=dictionary[child_key],delimiter=delimiter,float_format=float_format)
            elif isinstance(dictionary[child_key],pd.core.frame.DataFrame):
                dictionary[child_key].to_csv(filename,sep=delimiter,index=True,header=True,na_rep="nan",float_format=float_format)
            else:
                raise IOError("unsupported IO type in output dictionary: "+str(type(dictionary)))

def _write_txt_array(filename, array, delimiter=" ",float_format="%.6f"):
    """
    stores an array with unknown dtype to a text file.
    
    Args:
        filename:       name of the text file
        array:          array to be saved in the text file
        delimiter:      delimiter seperating values in the text file
        float_format:   formating string for floating point values
    """
    if array.dtype==np.float:
        np.savetxt(filename,array,fmt=float_format,delimiter=delimiter)
    elif array.dtype==np.integer:
        np.savetxt(filename,array,fmt="%i",delimiter=delimiter)
    elif array.dtype.type==np.string_:
        np.savetxt(filename,array,fmt="%s",delimiter=delimiter)
    else:
        raise IOError("unsupported format")

class output_writer(object):
    """writes an output dictionary to disk"""
    
    def __init__(self, output_dictionary,timestamp=None):
        """
        Args:
            output_dictionary:  a dictionary holding either other dictionaries, numpy arrays, or scalars as members
            timestamp:          a timestamp from time.time() If None, the current time is used. 
        """
        self.output_dict = output_dictionary
        self.filters = None
        if timestamp is None:
            self.timestamp = time.time()
        else:
            self.timestamp = timestamp
        #here we could add filters for saving compressed chunked arrays:
        #self.filters = tables.Filters(complib='blosc', complevel=5)

    def get_timestamp(self):
        """
        create a timestamp
        Returns:
            a string 
        """
        return str(datetime.fromtimestamp(self.timestamp))[0:10]+"_"+str(datetime.fromtimestamp(self.timestamp))[11:13]+"-"+str(datetime.fromtimestamp(self.timestamp))[14:16]+"-"+str(datetime.fromtimestamp(self.timestamp))[17:19]

    def write_hdf5(self, filename, timestamp=False):
        """
        Creates a HDF5 file that mirrors the dictionary.
        Supported types for leaf nodes are numpy.ndarray, scalars, and regular lists
    
        Args:
            filename:       name of the HDF5 output file
            timestamp:      Boolean indicator whether to append a timestap to the filename
        """
        if timestamp:
            stamp=self.get_timestamp()
            if len(filename)>4 and filename[-4] == ".":
                filename=filename[0:-4]+"_"+self.get_timestamp()+filename[-4:]
            elif len(filename)>3 and filename[-3] == ".":
                filename=filename[0:-3]+"_"+self.get_timestamp()+filename[-3:]
            else:
                filename=filename+self.get_timestamp()
        outfile = tables.open_file(filename, mode = "w", title = "Output")
        _depth_first_hdf5(self.output_dict,outfile=outfile,root=outfile.root,filters=self.filters)
        outfile.close()
        pass
    def write_txt(self, outdir=".", delimiter=" ",float_format="%.6f", timestamp=False):
        """
        Creates a directory structure that mirrors the dictionary.
        The leaf nodes are text files
        Supported types for leaf nodes are numpy.ndarray, scalars, and regular lists
    
        Args:
            outdir:         name of the HDF5 output file
            delimiter:      delimiter for values in text files
            float_format:   formating string for floating point values
            timestamp:      Boolean indicator whether to append a timestap to the filename
        """
        if timestamp:
            stamp=self.get_timestamp()
            if outdir.endswith(".") or outdir=="" or outdir.endswith(".."):
                outdir=os.path.join(outdir,self.get_timestamp())
            else:
                outdir=outdir+"_"+self.get_timestamp()
        _depth_first_text(dictionary=self.output_dict, outdir=outdir, delimiter=delimiter,float_format=float_format)
        pass


if __name__ == "__main__":
    print(("last modified: %s" % time.ctime(os.path.getmtime(__file__))))
    di={"a":{"B":np.ones((5,5)),"C":1},"B":["dfd","f"]}
    writer = output_writer(output_dictionary=di)
    writer.write_hdf5("test.h5",timestamp=True)
    writer.write_txt("test",timestamp=True)