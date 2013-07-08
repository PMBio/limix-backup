/*
 * dataframe.cpp
 *
 *  Created on: May 16, 2013
 *      Author: stegle
 */

#include "genotype.h"
#include "limix/io/vcf/Variant.h"
#include "limix/io/vcf/split.h"

using namespace vcf;

namespace limix {

/* CheaderMap implementation*/

void CHeaderMap::resize(muint_t n)
{
	for(CHeaderMap::iterator iter = this->begin(); iter!=this->end();iter++)
	{
		std::string name = (*iter).first;
		PstringVec value = (*iter).second;
		//resize
		value->resize(n);
	}
}

void CHeaderMap::set(std::string name, muint_t n, std::string value)
{

	(*(*this)[name])[n] = value;
};

std::string CHeaderMap::get(std::string name, muint_t n)
{
	return (*(*this)[name])[n];
};

PHeaderMap CHeaderMap::copy(muint_t i_start, muint_t n_elements)
{
	PHeaderMap RV = PHeaderMap(new CHeaderMap());
	for(CHeaderMap::iterator iter = this->begin(); iter!=this->end();iter++)
	{
		std::string key = (*iter).first;
		PstringVec value = (*iter).second;
		(*RV)[key] = PstringVec(new stringVec((*this)[key]->begin(),(*this)[key]->begin()+n_elements));
	}
}

} //end ::limix
