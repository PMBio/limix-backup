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


