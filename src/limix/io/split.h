// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

#ifndef __SPLIT_H
#define __SPLIT_H

// functions to split a string by a specific delimiter
#include <string>
#include <vector>
#include <sstream>
#include <string.h>

// thanks to Evan Teran, http://stackoverflow.com/questions/236129/how-to-split-a-string/236803#236803

// split a string on a single delimiter character (delim)
std::vector<std::string>& split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<std::string>  split(const std::string &s, char delim);

// split a string on any character found in the string of delimiters (delims)
std::vector<std::string>& split(const std::string &s, const std::string& delims, std::vector<std::string> &elems);
std::vector<std::string>  split(const std::string &s, const std::string& delims);

// from Marius, http://stackoverflow.com/a/1493195/238609
template < class ContainerT >
void tokenize(const std::string& str, ContainerT& tokens,
              const std::string& delimiters = " ", const bool trimEmpty = false)
{

    std::string::size_type pos, lastPos = 0;
    while(true)
    {
	pos = str.find_first_of(delimiters, lastPos);
	if(pos == std::string::npos)
	{

	    pos = str.length();

	    if(pos != lastPos || !trimEmpty) {
		tokens.push_back(typename ContainerT::value_type(str.data()+lastPos, (typename ContainerT::value_type::size_type)pos-lastPos));
	    }

	    break;
	}
	else
	{
	    if(pos != lastPos || !trimEmpty) {
		tokens.push_back(typename ContainerT::value_type(str.data()+lastPos, (typename ContainerT::value_type::size_type)pos-lastPos));
	    }
	}

	lastPos = pos + 1;
    }
};


#endif
