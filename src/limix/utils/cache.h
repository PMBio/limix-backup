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

#ifndef _CACHE_H_
#define _CACHE_H_

#include "limix/types.h"
#include <map>



namespace limix {


//container with of const MatrixXd
typedef std::map<std::string,PCVoid> PCVoidContainer;

/*!
 * \brief Named caching function for Matrix Object
 *
 */
class CNamedCache : public CParamObject
{
protected:

	PCVoidContainer container;
public:
	CNamedCache();
	virtual ~CNamedCache();

	// Cache access functions
	void set(std::string name,PCVoid m)
	{
		container[name] = m;
	}

	PCVoid get(const std::string& name)
	{
		return container[name];
	}

};
typedef sptr<CNamedCache> PNamedCache;







} /* namespace limix */

#endif /* _CACHE_H_ */


