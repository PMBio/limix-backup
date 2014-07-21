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

#include "genotype.h"
//#include "limix/io/vcf/Variant.h"
//#include "limix/io/vcf/split.h"

//using namespace vcf;

namespace limix {

/* CheaderMap implementation*/

void CHeaderMap::resize(muint_t n)
{
	for(CHeaderMap::iterator iter = this->begin(); iter!=this->end();iter++)
	{
		std::string name = (*iter).first;
		CFlexVector value = (*iter).second;
		//resize
		value.conservativeResize(n);
	}
}

void CHeaderMap::setStr(std::string name, muint_t n, std::string value)
{
	CFlexVector vector;
	vector = (*this)[name];
	CFlexVector::PStringMatrix strMatrix;
	strMatrix = vector;
	(*strMatrix)[n] = value;
};

/*
std::string CHeaderMap::get(std::string name, muint_t n)
{
	return (*(*this)[name])[n];
};
*/

PHeaderMap CHeaderMap::copy(muint_t i_start, muint_t n_elements)
{
	PHeaderMap RV = PHeaderMap(new CHeaderMap());
	for(CHeaderMap::iterator iter = this->begin(); iter!=this->end();iter++)
	{
		std::string key = (*iter).first;
		CFlexVector::PStringMatrix value = (*iter).second;
		CFlexVector::PStringMatrix copy = CFlexVector::PStringMatrix(new CFlexVector::StringMatrix(*value));
		(*RV)[key].setM(copy);
	}
	return RV;
}

} //end ::limix
